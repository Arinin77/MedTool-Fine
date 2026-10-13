"""Final MedTool-Fine research implementation."""
import copy
import re
from .precision_clinical import score_episode
from .answer_scoring import extract, score as score_prose
from .io import read_json as read
from . import clear_scoring

def no_tool_stop(text, parsed, allowed):
    if 'REQUIRED_INFORMATION_UNAVAILABLE' in allowed and parsed.get('stop_declaration') == 'INSUFFICIENT_INFORMATION':
        return (True, 'REQUIRED_INFORMATION_UNAVAILABLE')
    no_tool = bool(re.search('\\bno\\s+(?:(?:clinically|currently)\\s+)?(?:appropriate|applicable|suitable|validated)\\s+(?:(?:clinical|risk|prediction)\\s+)?(?:calculator|tool|model|score)\\b', text, re.I))
    concrete = [a for a in parsed.get('atoms', []) if not any((a.get(k) for k in ('qualified', 'component', 'range_endpoint')))]
    if 'NO_APPLICABLE_TOOL' in allowed and no_tool and (not concrete):
        return (True, 'NO_APPLICABLE_TOOL')
    return (False, 'NO_MATCHING_STOP_DECLARATION')

class Scorer:

    def __init__(self, prepared, tools, snapshot):
        self.prepared = prepared
        self.tools = tools
        self.snapshot = snapshot
        self.config = read(prepared / 'config.json')
        self.gold = read(prepared / 'private_gold.json')
        self.precision = read(prepared / 'numeric_precision.json')
        self.references = read(prepared / 'answer_reference_private.json')
        self.answer_policy = read(prepared / 'answer_policy.json')
        self.vocabulary = read(prepared / 'vocabulary.json')
        self.clear = clear_scoring
        self.available = {k: v for k, v in tools.items() if k not in self.config['numeric_hold_tool_ids']}

    def assess(self, method, job, record):
        ep = self.gold[job['episode_id']]
        initial = ep['states'][ep['initial_state_id']]
        if method in ('Direct', 'CoT'):
            valid = record['status'] == 'ANSWER_PARSED' and len(record['turns']) == 1
            text = record['answer']['final_answer'] if valid else ''
            reference = self.references['episodes'][job['episode_id']]
            scored = score_prose(text, reference, self.vocabulary, self.answer_policy, self.references['tool_catalog']) if valid else {'status': record['status']}
            stop, reason = (False, None)
            if initial['initial_action'] == 'ABSTAIN' and valid:
                stop, reason = no_tool_stop(text, scored.get('extracted', {}), initial['acceptable_abstain_reasons'])
            success = bool(valid and (stop if initial['initial_action'] == 'ABSTAIN' else scored.get('answer_success_type_aware') if initial['initial_action'] == 'CALL' else False))
            return {'qualified_success': success, 'protocol_valid': valid, 'score': scored, 'strict_trace_success': None, 'answer_replay_matches': None, 'answer_scope': reference['scope'], 'stop_success': stop, 'stop_reason': reason, 'metric_kind': 'ANSWER_ONLY_1_PERCENT_NOT_TOOL_TASK_SUCCESS'}
        if method == 'CLEAR':
            result = self.clear.score_record(ep, record, self.tools, self.snapshot, self.precision, job['initial_candidate_ids'], self.config['numeric_hold_tool_ids'])
            result['strict_trace_success'] = bool(result['qualified_success'] and result['score']['strict_trace_success'])
            result['metric_kind'] = 'TOOL_TASK_SUCCESS'
            return result
        actions = []
        observed = []
        valid = True
        adapter_error = None
        turns = record.get('turns', [])
        cap = 1 if method == 'Static' else 5
        valid &= 1 <= len(turns) <= cap and record.get('automatic_retries') == 0
        for i, t in enumerate(turns):
            expected = [record.get('selected_tool_id')] if method == 'AgentMD' and i > 0 else job['initial_candidate_ids']
            valid &= t.get('candidate_ids') == expected
            for answer in t.get('controlled_answers', []):
                actions.append({'action': 'ASK', 'concept': answer['concept']})
                observed.append(answer)
        if method == 'Static':
            terminal = record.get('actions', [])[-1] if record.get('actions') else {}
            if terminal.get('action') not in ('CALL', 'ABSTAIN'):
                valid = False
                adapter_error = 'NO_TERMINAL_ACTION'
            else:
                actions.append(copy.deepcopy(terminal))
                valid &= terminal['action'] == 'CALL' and record['status'] == 'EXECUTED' and (terminal['tool_id'] in job['initial_candidate_ids']) or (terminal['action'] == 'ABSTAIN' and record['status'] == 'ABSTAINED')
        else:
            valid &= record['status'] == 'FINISHED'
            final = record.get('final_submission') or {}
            if 'result_step' not in final:
                valid = False
                adapter_error = 'NO_VALID_FINISH'
            elif final['result_step'] is not None:
                ref = final['result_step']
                chosen = next((t for t in turns if t['step'] == ref), {})
                obs = chosen.get('observation') or {}
                valid &= ref < turns[-1]['step'] and obs == record.get('selected_observation') and (obs.get('execution_status') == 'PASS') and (obs.get('tool_id') in chosen.get('candidate_ids', [])) and (obs.get('tool_id') in self.available)
                if method == 'AgentMD':
                    valid &= obs.get('tool_id') == record.get('selected_tool_id')
                if {'tool_id', 'parameters', 'output'} <= obs.keys():
                    actions.append({'action': 'CALL', 'tool_id': obs['tool_id'], 'parameters': obs['parameters'], 'output': obs['output']})
                else:
                    valid = False
                    adapter_error = 'INVALID_FINAL_REFERENCE'
            else:
                text = final.get('answer', '')
                matched, reason = no_tool_stop(text, extract(text, self.vocabulary), {'NO_APPLICABLE_TOOL', 'REQUIRED_INFORMATION_UNAVAILABLE'})
                valid &= matched
                if matched:
                    actions.append({'action': 'ABSTAIN', 'reason': reason})
                else:
                    adapter_error = 'UNRESOLVED_FINAL_STOP'
        result = score_episode(ep, actions, self.available, self.snapshot, self.precision, max_actions=max(cap, len(actions)))
        replay = result['controlled_answers'] == observed
        success = bool(valid and replay and result['final_task_success'])
        return {'qualified_success': success, 'protocol_valid': bool(valid), 'score': result, 'answer_replay_matches': replay, 'adapter_error': adapter_error, 'strict_trace_success': bool(success and result['strict_trace_success']), 'metric_kind': 'TOOL_TASK_SUCCESS', 'adapted_actions': actions}

