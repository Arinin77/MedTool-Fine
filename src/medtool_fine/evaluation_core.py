"""Final MedTool-Fine research implementation."""
import copy
import math
from .execution import _check_basic as checked, checked as _check_public_contract, _execute_verified as execute

def equivalent(actual, expected, *, abs_tol=1e-08, rel_tol=1e-08):
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and (not isinstance(actual, bool)) and math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all((equivalent(actual[k], v, abs_tol=abs_tol, rel_tol=rel_tol) for k, v in expected.items()))
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all((equivalent(a, b, abs_tol=abs_tol, rel_tol=rel_tol) for a, b in zip(actual, expected)))
    return False

def _score_exact_call(gold, submission, tools, snapshot):
    """Replay submitted arguments, never replace them with reference parameters.

    Caller must load the pinned, verified registry and Gold release. Unknown IDs
    and malformed responses are scored failures, not dropped observations.
    """
    if gold.get('status') != 'FROZEN_RESEARCH_GOLD' or gold.get('initial_action') != 'CALL':
        raise ValueError('requires frozen CALL Gold')
    score = dict.fromkeys(['action_correct', 'tool_correct', 'parameters_correct', 'execution_pass', 'output_correct', 'task_success'], False)
    score['error'] = None
    if not isinstance(submission, dict):
        score['error'] = 'MALFORMED_RESPONSE'
        return score
    score['action_correct'] = submission.get('action') == 'CALL'
    tid = submission.get('tool_id')
    if not isinstance(tid, str) or tid not in gold['references'] or tid not in tools:
        score['error'] = 'UNKNOWN_OR_UNACCEPTABLE_TOOL'
        return score
    score['tool_correct'] = True
    if not score['action_correct']:
        score['error'] = 'WRONG_ACTION'
        return score
    ref, tool = (gold['references'][tid], tools[tid])
    params = submission.get('parameters')
    try:
        checked(params, tool['parameter_schema'])
        expected = {k: ref['parameters'][k] for k in params if k in ref['parameters']}
        score['parameters_correct'] = equivalent(params, expected, **gold['parameter_tolerance'])
        actual = execute(snapshot, tool, params)
        checked(actual, tool['output_schema'])
        score['execution_pass'] = True
        score['output_correct'] = equivalent(actual, ref['output'], **ref['output_tolerance'])
        if 'output' in submission:
            score['output_correct'] &= equivalent(submission['output'], ref['output'], **ref['output_tolerance'])
        score['task_success'] = all((score[k] for k in ['action_correct', 'tool_correct', 'parameters_correct', 'execution_pass', 'output_correct']))
    except Exception as exc:
        score['error'] = f'{type(exc).__name__}: {exc}'
    return score

def score_call(gold, submission, tools, snapshot):
    result = _score_exact_call(gold, submission, tools, snapshot)
    result['scoring_version'] = 'tool_call'
    result['execution_status'] = 'NOT_ASSESSED'
    result['silent_wrong_call'] = False
    if not isinstance(submission, dict):
        result['execution_pass'] = None
        return result
    tid = submission.get('tool_id')
    if not isinstance(tid, str) or tid not in tools:
        result['error'] = 'UNKNOWN_TOOL_ID'
        result['execution_pass'] = None
        return result
    if submission.get('action') != 'CALL':
        result['execution_pass'] = None
        result['error'] = 'WRONG_ACTION'
        return result
    if tid in gold['references']:
        result['execution_status'] = 'PASS' if result['execution_pass'] else 'CALL_FAILED'
        return result
    result['error'] = 'UNACCEPTABLE_KNOWN_TOOL'
    result['parameters_correct'] = None
    result['output_correct'] = None
    result['diagnostic_error'] = None
    tool = tools[tid]
    try:
        checked(submission.get('parameters'), tool['parameter_schema'])
    except Exception as exc:
        result['execution_status'] = 'PARAMETER_VALIDATION_FAILED'
        result['diagnostic_error'] = f'{type(exc).__name__}: {exc}'
        return result
    try:
        output = execute(snapshot, tool, submission['parameters'])
        checked(output, tool['output_schema'])
        result['execution_status'] = 'PASS'
        result['execution_pass'] = True
        result['diagnostic_execution_output'] = output
        result['silent_wrong_call'] = True
    except Exception as exc:
        result['execution_status'] = 'EXECUTION_FAILED'
        result['diagnostic_error'] = f'{type(exc).__name__}: {exc}'
    return result

def _replay_calculation(episode, actions, tools, snapshot):
    """Actions use CALL(tool_id, parameters) or ASK(concept); no natural-text judge.

    Gold/transition access belongs to the evaluator or answer server, not model
    inference. The server emits only the selected answer text after a valid ASK.
    """
    if episode.get('status') != 'FROZEN_RESEARCH_GOLD':
        raise ValueError('episode is not frozen')
    result = {'task_success': False, 'initial_action_correct': False, 'ask_count': 0, 'question_correct': None, 'terminal_call': None, 'error': None, 'scoring_version': 'strict_calculation_trace', 'controlled_answers': []}
    if not isinstance(actions, list) or not actions:
        result['error'] = 'MALFORMED_OR_EMPTY_TRACE'
        return result
    state = episode['initial_state_id']
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            result['error'] = 'MALFORMED_ACTION'
            return result
        node = episode['states'][state]
        if index == 0:
            result['initial_action_correct'] = action.get('action') == node['initial_action']
        if action.get('action') == 'ASK':
            result['ask_count'] += 1
            concept = action.get('concept')
            transition = node.get('ask_transitions', {}).get(concept) if isinstance(concept, str) else None
            result['question_correct'] = transition is not None
            if transition is None:
                result['error'] = 'UNACCEPTABLE_QUESTION_CONCEPT'
                return result
            result['controlled_answers'].append({'concept': concept, 'text': transition['answer_text']})
            state = transition['next_state_id']
        elif action.get('action') == 'CALL':
            if node['initial_action'] != 'CALL':
                result['error'] = 'CALL_BEFORE_REQUIRED_INFORMATION'
                return result
            result['terminal_call'] = score_call(node, action, tools, snapshot)
            if index != len(actions) - 1:
                result['error'] = 'ACTIONS_AFTER_TERMINAL_CALL'
                return result
            result['task_success'] = result['initial_action_correct'] and result['terminal_call']['task_success']
            return result
        else:
            result['error'] = 'UNACCEPTABLE_ACTION'
            return result
    result['error'] = 'NO_TERMINAL_CALL'
    return result

CONCEPT_ALIASES = {'wbc': 'white_blood_cell_count', 'white blood cell count': 'white_blood_cell_count', 'white_blood_cell_count': 'white_blood_cell_count', '白细胞计数': 'white_blood_cell_count', '白细胞总数': 'white_blood_cell_count', 'leukocytosis': 'leukocytosis', '白细胞增多': 'leukocytosis', '是否白细胞增多': 'leukocytosis'}

def _replay_aliases(episode, actions, tools, snapshot):
    if isinstance(actions, list):
        actions = [{**action, 'concept': CONCEPT_ALIASES.get(action['concept'].strip().lower(), action['concept'])} if isinstance(action, dict) and action.get('action') == 'ASK' and isinstance(action.get('concept'), str) else action for action in actions]
    return _replay_calculation(episode, actions, tools, snapshot)

ABSTAIN_REASONS = {'NO_APPLICABLE_TOOL', 'REQUIRED_INFORMATION_UNAVAILABLE'}

def _replay_strict(episode, actions, tools, snapshot):
    if not any((n.get('initial_action') == 'ABSTAIN' for n in episode.get('states', {}).values())):
        return _replay_aliases(episode, actions, tools, snapshot)
    if episode.get('status') != 'FROZEN_RESEARCH_GOLD':
        raise ValueError('episode is not frozen')
    result = {'task_success': False, 'initial_action_correct': False, 'ask_count': 0, 'question_correct': None, 'terminal_call': None, 'terminal_abstain': None, 'error': None, 'scoring_version': 'strict_trajectory', 'controlled_answers': []}
    if not isinstance(actions, list) or not actions:
        result['error'] = 'MALFORMED_OR_EMPTY_TRACE'
        return result
    state = episode['initial_state_id']
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            result['error'] = 'MALFORMED_ACTION'
            return result
        node = episode['states'][state]
        kind = action.get('action')
        if index == 0:
            result['initial_action_correct'] = kind == node['initial_action']
        if kind == 'ASK':
            result['ask_count'] += 1
            raw = action.get('concept')
            concept = CONCEPT_ALIASES.get(raw.strip().lower(), raw.strip().lower()) if isinstance(raw, str) else None
            tr = node.get('ask_transitions', {}).get(concept)
            result['question_correct'] = tr is not None
            if tr is None:
                result['error'] = 'UNACCEPTABLE_QUESTION_CONCEPT'
                return result
            result['controlled_answers'].append({'concept': concept, 'text': tr['answer_text']})
            state = tr['next_state_id']
        elif kind in ['CALL', 'ABSTAIN']:
            if index != len(actions) - 1:
                result['error'] = 'ACTIONS_AFTER_TERMINAL_ACTION'
                return result
            if kind == 'CALL':
                if node['initial_action'] != 'CALL':
                    result['error'] = 'CALL_BEFORE_REQUIRED_INFORMATION'
                    return result
                result['terminal_call'] = score_call(node, action, tools, snapshot)
                success = result['terminal_call']['task_success']
            else:
                reason = action.get('reason')
                success = node['initial_action'] == 'ABSTAIN' and isinstance(reason, str) and (reason in ABSTAIN_REASONS) and (reason in node.get('acceptable_abstain_reasons', []))
                result['terminal_abstain'] = {'reason': reason, 'task_success': success}
            result['task_success'] = result['initial_action_correct'] and success
            return result
        else:
            result['error'] = 'UNACCEPTABLE_ACTION'
            return result
    result['error'] = 'NO_TERMINAL_ACTION'
    return result

def score_episode(episode, actions, tools, snapshot, max_actions=8):
    strict = _replay_strict(episode, actions, tools, snapshot)
    result = {'scoring_version': 'trajectory', 'strict_trace': strict, 'strict_trace_success': strict['task_success'], 'final_task_success': False, 'initial_action_correct': strict['initial_action_correct'], 'ask_count': 0, 'redundant_question_count': 0, 'unrelated_question_count': 0, 'overask_count': 0, 'controlled_answers': [], 'terminal_call': None, 'error': None, 'max_actions': max_actions}
    if not isinstance(actions, list) or not actions:
        result['error'] = 'MALFORMED_OR_EMPTY_TRACE'
        return result
    if len(actions) > max_actions:
        result['error'] = 'ACTION_BUDGET_EXCEEDED'
        return result
    state = episode['initial_state_id']
    answered = {}
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            result['error'] = 'MALFORMED_ACTION'
            return result
        node = episode['states'][state]
        if action.get('action') == 'ASK':
            result['ask_count'] += 1
            result['overask_count'] += bool(node.get('references'))
            raw = action.get('concept')
            if not isinstance(raw, str) or not raw.strip():
                result['error'] = 'MALFORMED_QUESTION_CONCEPT'
                return result
            concept = CONCEPT_ALIASES.get(raw.strip().lower(), raw.strip().lower())
            if concept in answered:
                result['redundant_question_count'] += 1
                answer = answered[concept]
                kind = 'REPEATED_KNOWN_ANSWER'
            else:
                transition = node.get('ask_transitions', {}).get(concept)
                if transition:
                    answer = transition['answer_text']
                    state = transition['next_state_id']
                    kind = 'CONTROLLED_NEW_INFORMATION'
                else:
                    result['unrelated_question_count'] += 1
                    answer = 'No additional information is available for this question.'
                    kind = 'NO_INFORMATION'
                answered[concept] = answer
            result['controlled_answers'].append({'concept': concept, 'text': answer, 'kind': kind})
        elif action.get('action') == 'CALL':
            tid = action.get('tool_id')
            missing_tools = set(node.get('applicable_tools', [])) | set(node.get('applicable_but_missing_tools', {}))
            if isinstance(tid, str) and tid in missing_tools and (tid not in node.get('references', {})):
                result['error'] = 'CURRENTLY_UNCALLABLE_UNOBSERVED_PARAMETERS'
                result['tool_applicability'] = 'APPLICABLE_BUT_REQUIRED_INFORMATION_UNOBSERVED'
                return result
            if node['initial_action'] != 'CALL':
                result['error'] = 'CALL_BEFORE_REQUIRED_INFORMATION'
                return result
            result['terminal_call'] = score_call(node, action, tools, snapshot)
            if index != len(actions) - 1:
                result['error'] = 'ACTIONS_AFTER_TERMINAL_CALL'
                return result
            result['final_task_success'] = result['terminal_call']['task_success']
            return result
        elif action.get('action') == 'ABSTAIN':
            reason = action.get('reason')
            if index != len(actions) - 1:
                result['error'] = 'ACTIONS_AFTER_TERMINAL_ABSTAIN'
                return result
            if node['initial_action'] != 'ABSTAIN':
                result['error'] = 'PREMATURE_OR_UNJUSTIFIED_ABSTAIN'
                return result
            if not isinstance(reason, str) or reason not in ABSTAIN_REASONS:
                result['error'] = 'MALFORMED_ABSTAIN_REASON'
                return result
            success = reason in node.get('acceptable_abstain_reasons', [])
            result['terminal_abstain'] = {'reason': reason, 'task_success': success}
            result['final_task_success'] = success
            if not success:
                result['error'] = 'INCORRECT_ABSTAIN_REASON'
            return result
        else:
            result['error'] = 'UNACCEPTABLE_ACTION'
            return result
    result['error'] = 'NO_TERMINAL_CALL'
    return result

def finite_number(value):
    return isinstance(value, (int, float)) and (not isinstance(value, bool)) and math.isfinite(value)

def unit_key(value):
    if value is None:
        return None
    return value.replace('µ', 'u').replace('μ', 'u').replace(' ', '').lower()

def contract_check(result, submission, tools):
    result = copy.deepcopy(result)
    result['scoring_version'] = 'contract_checked'
    if not result.get('tool_correct') or not result.get('action_correct'):
        return result
    try:
        _check_public_contract(submission['parameters'], tools[submission['tool_id']]['parameter_schema'])
    except ValueError as exc:
        result.update(parameters_correct=False, execution_pass=False, output_correct=False, task_success=False, execution_status='PARAMETER_VALIDATION_FAILED', error='PUBLIC_CONTRACT_VIOLATION: ' + str(exc))
    return result

