"""Run frozen method logic on newly handwritten synthetic fixtures, without a model.

This is a software smoke test, not a benchmark run or a reproduction of paper scores.
The ten candidate IDs below are test doubles sharing one elementary BMI function.
No calculator-library content or MIMIC-derived example is used.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

PACKAGE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PACKAGE / 'src'))

from medtool_fine.runner import execute_episode
from medtool_fine.scoring import Scorer
from medtool_fine.retrieval import lexical_rank, rrf_rank, token_windows

METHODS = ('Direct', 'CoT', 'Static', 'ReAct', 'AgentMD', 'CLEAR')
DISPLAY = {}
IDS = [f'SYNTHETIC-BMI-{i:02d}' for i in range(1, 11)]
PARAMETERS = {'weight_kg': 81.0, 'height_m': 1.8}
PROVIDER = {'model': 'gpt-5.4', 'temperature': 0, 'max_output_tokens': 12000,
            'reasoning_effort': 'none', 'timeout_seconds': 1,
            'endpoint': 'https://example.invalid/v1/chat/completions'}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def clear_action(action):
    return {'working_state': {'candidate_paths': [], 'remaining_information': []},
            'next_action': action}


def react_action(action, args):
    return {'thought': 'Scripted fixture response; no model inference.',
            'action': action, 'action_input': args}


class ScriptedTransport:
    """A deterministic in-memory responder; it performs no network operations."""
    def __init__(self, responses):
        self.responses = copy.deepcopy(responses)
        self.requests = []

    def __call__(self, wire):
        self.requests.append(copy.deepcopy(wire))
        if not self.responses:
            raise AssertionError('Unexpected additional model request')
        return json.dumps({'model': 'gpt-5.4',
            'choices': [{'finish_reason': 'stop', 'message': {
                'content': json.dumps(self.responses.pop(0))}}],
            # Artificial API metadata for testing the frozen mode checker.
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2,
                      'completion_tokens_details': {'reasoning_tokens': 0}}})


def fixtures(base):
    """Create temporary, openly specified test inputs; no source dataset is read."""
    prep, snapshot = base / 'prepared', base / 'snapshot'
    prep.mkdir()
    snapshot.mkdir()
    shutil.copytree(PACKAGE / 'methods', prep / 'methods')
    implementation = snapshot / 'synthetic_bmi.py'
    implementation.write_text('def calculate(p):\n    return {"bmi": p["weight_kg"] / p["height_m"] ** 2}\n', encoding='utf-8')
    param_schema = {'type': 'object', 'required': ['weight_kg', 'height_m'],
        'additionalProperties': False, 'properties': {
            'weight_kg': {'type': 'number', 'exclusiveMinimum': 0},
            'height_m': {'type': 'number', 'exclusiveMinimum': 0}}}
    output_schema = {'type': 'object', 'required': ['bmi'], 'additionalProperties': False,
                     'properties': {'bmi': {'type': 'number'}}}
    tools = {tid: {'tool_id': tid,
        'tool_card': {'name': 'Synthetic BMI fixture ' + str(i),
                     'clinical_purpose': 'Software fixture: divide weight in kg by squared height in m.'},
        'parameter_schema': copy.deepcopy(param_schema), 'output_schema': copy.deepcopy(output_schema),
        'implementation': {'path': implementation.name,
            'sha256': hashlib.sha256(implementation.read_bytes()).hexdigest(), 'entrypoint': 'calculate'}}
        for i, tid in enumerate(IDS, 1)}
    tol = {'abs_tol': 1e-8, 'rel_tol': 1e-8}
    call_node = {'status': 'FROZEN_RESEARCH_GOLD', 'initial_action': 'CALL',
        'references': {IDS[0]: {'parameters': PARAMETERS, 'output': {'bmi': 25.0},
                              'output_tolerance': tol}}, 'parameter_tolerance': tol,
        'acceptable_tool_set': [IDS[0]], 'ask_transitions': {}}
    # FROZEN_RESEARCH_GOLD is a required legacy schema literal, not a provenance claim.
    ask_node = {'initial_action': 'ASK', 'references': {}, 'ask_transitions': {
        'height': {'answer_text': 'The synthetic height is 1.8 m.', 'next_state_id': 'ready'}}}
    stop_node = {'initial_action': 'ABSTAIN', 'references': {}, 'ask_transitions': {},
                 'acceptable_abstain_reasons': ['REQUIRED_INFORMATION_UNAVAILABLE']}
    cases = {
        'synthetic_complete': 'A fictional adult weighs 81 kg and is 1.8 m tall.',
        'synthetic_ask': 'A fictional adult weighs 81 kg. Height is not initially supplied.',
        'synthetic_stop': 'A fictional adult weighs 81 kg. Height cannot be obtained.'}
    jobs = {name: {'episode_id': name, 'k': 10, 'initial_candidate_ids': IDS,
        'case': {'clinical_goal': 'Calculate body mass index for this synthetic software test.',
                 'case_text': text}} for name, text in cases.items()}
    gold = {
        'synthetic_complete': {'status': 'FROZEN_RESEARCH_GOLD', 'initial_state_id': 'ready', 'states': {'ready': call_node}},
        'synthetic_ask': {'status': 'FROZEN_RESEARCH_GOLD', 'initial_state_id': 'need', 'states': {'need': ask_node, 'ready': call_node}},
        'synthetic_stop': {'status': 'FROZEN_RESEARCH_GOLD', 'initial_state_id': 'stop', 'states': {'stop': stop_node}}}
    answer_env = {name: {'aliases': {'height': 'height'}, 'record': {
        'initial_state': 'need' if name == 'synthetic_ask' else 'ready',
        'states': {'need': {'height': {'text': 'The synthetic height is 1.8 m.', 'next_state': 'ready'}},
                   'ready': {}}}} for name in cases}
    protocol = {'global_ask_concepts': ['height'], 'alignment': {}}
    # clinical is only the frozen scorer's compatibility discriminator.
    # All values here are handwritten fixture policy; no actual dataset is loaded.
    precision = {'dataset': 'clinical', 'policy_id': 'synthetic_1_percent_fixture',
        'absolute_floor': 1e-8, 'relative_tolerance': .01,
        'discrete_output_keys': [], 'discrete_tool_outputs': {}, 'exact_input_units': ['years','count'],
        'exact_input_keys': ['age'], 'conversion_parameters': {}, 'conversions': {}, 'derived_origins': []}
    field = {'key': 'bmi', 'kind': 'number', 'unit': 'dimensionless', 'aliases': ['bmi'],
             'required': True, 'expected': 25.0, 'tolerance': tol}
    references = {'episodes': {'synthetic_complete': {'scope': 'FINAL_ANSWER',
        'branches': [{'tool_id': IDS[0], 'scorable': True, 'fields': [field]}]}},
        'tool_catalog': {IDS[0]: {'fields': [field]}}}
    for name, content in {
        'config.json': {'numeric_hold_tool_ids': []},
        'private_gold.json': gold, 'numeric_precision.json': precision,
        'answer_reference_private.json': references,
        'answer_policy.json': {'version': 'synthetic_fixture', 'renal_continuous': {
            'kinds': [], 'units': [], 'absolute_tolerance': 1e-8, 'relative_tolerance': .01}},
        'vocabulary.json': {IDS[0]: {'labels': ['BMI']}}
    }.items():
        write(prep / name, content)
    return prep, snapshot, tools, jobs, answer_env, protocol


def run_demo(base):
    prep, snapshot, tools, jobs, answer_env, protocol = fixtures(base)
    scorer = Scorer(prep, tools, snapshot)
    call = {'action': 'CALL', 'tool_id': IDS[0], 'parameters': PARAMETERS}
    plans = {
        'Direct': [{'answer': '25'}],
        'CoT': [{'reasoning': '81 / (1.8 * 1.8) = 25.', 'answer': '25'}],
        'Static': [{'tool_id': IDS[0], 'parameters': PARAMETERS}],
        'CLEAR': [clear_action(call)],
        'ReAct': [react_action(IDS[0], PARAMETERS), react_action('Finish', {'answer': '25', 'result_step': 1})],
        'AgentMD': [react_action('Select', {'tool_id': IDS[0]}), react_action(IDS[0], PARAMETERS),
                              react_action('Finish', {'answer': '25', 'result_step': 2})]}
    results = []
    for method in METHODS:
        transport = ScriptedTransport(plans[method])
        record = execute_episode(method, jobs['synthetic_complete'], base / ('run_' + method), prep,
            PROVIDER, '', tools, snapshot, [], answer_env, protocol, transport)
        score = scorer.assess(method, jobs['synthetic_complete'], record)
        assert score['qualified_success'], (method, record, score)
        assert not transport.responses
        results.append({'method': DISPLAY.get(method, method), 'status': record['status'],
                        'mock_requests': len(transport.requests), 'synthetic_check_passed': True})
    for name, actions, expected in [
        ('synthetic_ask', [clear_action({'action': 'ASK', 'concepts': ['height'], 'question': 'What is the synthetic height?'}),
                           clear_action(call)], 'EXECUTED'),
        ('synthetic_stop', [clear_action({'action': 'ABSTAIN', 'reason': 'REQUIRED_INFORMATION_UNAVAILABLE',
            'justification': 'The fixture explicitly states that height cannot be obtained.',
            'evidence': [{'source': 'case', 'concept': 'height', 'quote': 'Height cannot be obtained.'}]})], 'ABSTAINED')]:
        transport = ScriptedTransport(actions)
        record = execute_episode('CLEAR', jobs[name], base / ('run_' + name), prep,
            PROVIDER, '', tools, snapshot, [], answer_env, protocol, transport)
        score = scorer.assess('CLEAR', jobs[name], record)
        assert record['status'] == expected and score['qualified_success'], (record, score)
        results.append({'method': 'CLEAR', 'scenario': name, 'status': record['status'],
                        'mock_requests': len(transport.requests), 'synthetic_check_passed': True})
    ranking = lexical_rank('height weight body mass index', {'a': 'body mass index weight height', 'b': 'unrelated words'})
    assert ranking[0]['tool_id'] == 'a'
    assert rrf_rank([ranking, ranking])[0]['tool_id'] == 'a'
    assert sum(w['weight'] for w in token_windows(list(range(1300)))) == 1300
    return {'purpose': 'Offline synthetic software smoke test, not paper-result reproduction',
        'fixture_provenance': 'Newly handwritten elementary formula and fictional inputs; no source cases or original library used.',
        'network_requests': 0, 'model_api_requests': 0, 'dense_encoder_executed': False,
        'checks': results, 'retrieval_checks': ['BM25', 'RRF', 'window coverage']}


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='medtool_fine_synthetic_') as scratch:
        result = run_demo(Path(scratch))
    print(json.dumps(result, indent=2))
