"""Final MedTool-Fine research implementation."""
import copy
from .precision_clinical import score_episode

def score_record(ep, record, tools, snapshot, precision, expected_candidates, holds=()):
    actions, observed, submitted = ([], [], [])
    consistent = True
    for turn in record['turns']:
        decision = turn['model_result'].get('decision')
        if not decision or 'rejected_proposal' in turn:
            continue
        action = turn.get('submitted_action')
        if not isinstance(action, dict):
            consistent = False
            continue
        consistent &= {k: v for k, v in action.items() if k != 'output'} == decision
        submitted.append(action)
        if decision['action'] == 'ASK':
            for answer in turn.get('controlled_answers', []):
                actions.append({'action': 'ASK', 'concept': answer['concept']})
                observed.append(answer)
        elif decision['action'] == 'ABSTAIN':
            actions.append({k: action[k] for k in ('action', 'reason')})
        elif decision['action'] == 'CALL':
            actions.append(copy.deepcopy(action))
        else:
            consistent = False
    consistent &= submitted == record['actions']
    available = {k: v for k, v in tools.items() if k not in holds}
    result = score_episode(ep, actions, available, snapshot, precision, max_actions=max(5, len(actions)))
    terminal = actions[-1] if actions else {}
    expected = list(expected_candidates)
    pool_ok = len(expected) == len(set(expected)) == 10 and record.get('initial_candidate_ids') == expected and all((t['candidate_ids'] == expected for t in record['turns']))
    turn_count = len(record['turns'])
    valid = bool(consistent and pool_ok and (record.get('protocol') == 'clear-fixed10') and (record.get('max_model_turns') == 5) and (record.get('max_ask_concepts') == 8) and (1 <= turn_count <= 5) and (record['model_request_count'] == turn_count) and (record.get('fallback_retrieval_count') == 0) and (record.get('post_initial_retrieval_count') == 0) and (not record.get('retrieval_actions')) and (record.get('automatic_retries') == 0) and (not any(('rejected_proposal' in t for t in record['turns']))) and (record['status'] == 'EXECUTED' and terminal.get('action') == 'CALL' and (terminal['tool_id'] in expected) or (record['status'] == 'ABSTAINED' and terminal.get('action') == 'ABSTAIN')))
    replay_ok = result['controlled_answers'] == observed
    success = bool(valid and replay_ok and result['final_task_success'])
    return {'qualified_success': success, 'answer_replay_matches': replay_ok, 'protocol_valid': valid, 'fixed_candidate_pool': pool_ok, 'score': result, 'model_turns': turn_count, 'post_initial_retrievals': record.get('post_initial_retrieval_count'), 'interaction_protocol_success': bool(success and result['strict_trace_success'] and (not result['overask_count']) and (not result['redundant_question_count']) and (not result['unrelated_question_count']))}

