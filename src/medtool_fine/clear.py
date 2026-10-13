"""Final MedTool-Fine research implementation."""
import copy
import json
from pathlib import Path
from . import runtime as shared
from .interaction import _BatchAnswerSession as BatchAnswerSession
from .io import write_json
from .execution import checked, execute, public_schema

LEGACY_NO_MATCH = 'No additional information is available for this question.'

PUBLIC_NO_MATCH = 'NO_MATCHING_ANSWER: This question did not match an answer in the current information state. This is not evidence that the clinical fact is absent or impossible to obtain. Review the supplied record, prior answers and the question vocabulary.'

def public_answers(answer_events):
    """Disambiguate the transport fallback; retain canonical events for unchanged replay."""
    result = copy.deepcopy(answer_events)
    for event in result:
        if event['kind'] == 'NO_INFORMATION' or (event['kind'] == 'REPEATED_KNOWN_ANSWER' and event['text'] == LEGACY_NO_MATCH):
            event['text'] = PUBLIC_NO_MATCH
    return result

def explicit_unavailability(text):
    return any((x in text.lower() for x in ('cannot be recovered', 'unavailable', 'not available', 'cannot be obtained')))

def decode(data):
    if not isinstance(data, dict) or set(data) != {'working_state', 'next_action'}:
        raise ValueError('output fields')
    state, action = (data['working_state'], data['next_action'])
    if not isinstance(state, dict) or set(state) != {'candidate_paths', 'remaining_information'}:
        raise ValueError('working state fields')
    if not isinstance(state['candidate_paths'], list) or not isinstance(state['remaining_information'], list):
        raise ValueError('working state lists')
    if not isinstance(action, dict):
        raise ValueError('action object')
    kind = action.get('action')
    if kind == 'ASK':
        if set(action) not in ({'action', 'concepts'}, {'action', 'concepts', 'question'}):
            raise ValueError('ASK fields')
        concepts = action['concepts']
        if not isinstance(concepts, list) or not 1 <= len(concepts) <= 8:
            raise ValueError('ASK requires 1-8 concepts')
        if any((not isinstance(c, str) or not c.strip() for c in concepts)):
            raise ValueError('empty concept')
        if 'question' in action and (not isinstance(action['question'], str)):
            raise ValueError('question')
    elif kind == 'CALL':
        if set(action) != {'action', 'tool_id', 'parameters'}:
            raise ValueError('CALL fields')
        if not isinstance(action['tool_id'], str) or not isinstance(action['parameters'], dict):
            raise ValueError('CALL types')
    elif kind == 'ABSTAIN':
        if set(action) != {'action', 'reason', 'justification', 'evidence'}:
            raise ValueError('ABSTAIN requires reason, justification and evidence')
        if action['reason'] not in ('NO_APPLICABLE_TOOL', 'REQUIRED_INFORMATION_UNAVAILABLE'):
            raise ValueError('ABSTAIN reason')
        if not isinstance(action['justification'], str) or not action['justification'].strip():
            raise ValueError('ABSTAIN justification')
        if not isinstance(action['evidence'], list):
            raise ValueError('ABSTAIN evidence list')
        for item in action['evidence']:
            if not isinstance(item, dict) or set(item) != {'source', 'concept', 'quote'}:
                raise ValueError('evidence fields')
            if not all((isinstance(v, str) and v.strip() for v in item.values())):
                raise ValueError('evidence strings')
            if item['source'] not in ('case', 'answer'):
                raise ValueError('evidence source')
    else:
        raise ValueError('only CALL, ASK and ABSTAIN are allowed')
    return (action, state)

def parse(raw, model):
    try:
        env = json.loads(raw, object_pairs_hook=shared._unique_object, parse_constant=shared._reject_constant)
        if not isinstance(env, dict) or 'error' in env or env.get('model') != model:
            raise ValueError('model/envelope')
        choices = env['choices']
        if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
            raise ValueError('incomplete response')
        data = json.loads(choices[0]['message']['content'], object_pairs_hook=shared._unique_object, parse_constant=shared._reject_constant)
        action, state = decode(data)
        return {'status': 'ACTION_READY', 'decision': action, 'working_state': state}
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        return {'status': 'ACTION_SCHEMA_ERROR', 'decision': None, 'schema_error': str(exc)}

def check_stop(action, case, answer_events):
    """Verify evidence provenance; clinical correctness remains in offline Gold scoring."""
    if action['action'] != 'ABSTAIN' or action['reason'] != 'REQUIRED_INFORMATION_UNAVAILABLE':
        return None
    if not action['evidence']:
        return 'UNSUPPORTED_UNAVAILABILITY'
    for item in action['evidence']:
        quote = item['quote']
        if len(quote) < 12 or not explicit_unavailability(quote):
            return 'UNSUPPORTED_UNAVAILABILITY'
        if item['source'] == 'case':
            valid = quote in case['case_text'] or quote in case['clinical_goal']
        else:
            valid = any((a['kind'] == 'CONTROLLED_NEW_INFORMATION' and a['concept'] == item['concept'] and (quote in a['text']) for a in answer_events))
        if not valid:
            return 'UNSUPPORTED_UNAVAILABILITY'
    return None

def answer_state(answer_events):
    """Deduplicate already received canonical concepts; never inspect answer-table states."""
    facts = {}
    for event in answer_events:
        concept = event['concept']
        if concept in facts:
            continue
        if event['kind'] == 'NO_INFORMATION':
            status = 'NO_MATCHING_ANSWER'
        elif event['kind'] == 'CONTROLLED_NEW_INFORMATION':
            terminal = any((p in event['text'].lower() for p in ('cannot be recovered', 'cannot be obtained')))
            status = 'EXPLICITLY_UNAVAILABLE' if terminal else 'ANSWER_RECEIVED'
        else:
            raise ValueError('Repeated answer has no original event')
        facts[concept] = {'concept': concept, 'status': status, 'text': event['text']}
    return list(facts.values())

def question_aliases(aliases):
    """Add only unambiguous space/underscore forms of declared answer aliases.

    No semantic synonyms, tool parameters, future answers or Gold are consulted.
    Existing exact aliases take precedence; ambiguous new forms stay unmatched.
    """
    groups = {}
    for name, target in aliases.items():
        key = ' '.join(name.strip().lower().replace('_', ' ').split())
        groups.setdefault(key, set()).add(target)
    expanded = dict(aliases)
    for name, targets in groups.items():
        if len(targets) == 1:
            target = next(iter(targets))
            for variant in (name, name.replace(' ', '_')):
                expanded.setdefault(variant, target)
    return expanded

def load_prompt():
    return (Path(__file__).resolve().parents[2] / 'methods/CLEAR/prompt.txt').read_text('utf-8')

def execute_call(action, payload, tools, snapshot, provider_model, holds):
    tid = action['tool_id']
    if tid in tools and tid in {x['tool_id'] for x in payload['candidate_tools']}:
        try:
            checked(action['parameters'], tools[tid]['parameter_schema'])
        except ValueError as exc:
            return {'status': 'PARAMETER_SCHEMA_ERROR', 'execution_status': 'NOT_RUN', 'execution_output': None, 'errors': [str(exc)]}
    raw = json.dumps({'model': provider_model, 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'tool_id': tid, 'parameters': action['parameters']})}}]})
    return shared.evaluate(raw, payload, tools, snapshot, provider_model, holds, executor=execute)

def run_episode(job, folder, provider, key, tools, snapshot, holds, answer_record, aliases, *, transport=None, concepts=(), alignment=None, max_actions=5):
    candidates = tuple(job['initial_candidate_ids'])
    if job['k'] != 10 or len(candidates) != 10 or len(set(candidates)) != 10 or any((t not in tools for t in candidates)):
        raise ValueError('Require ten distinct known initial candidates')
    if max_actions != 5:
        raise ValueError('Require the five-model-decision budget')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    record = {'episode_id': job['episode_id'], 'started_at': shared.now(), 'protocol': 'clear-fixed10', 'initial_candidate_ids': list(candidates), 'actions': [], 'retrieval_actions': [], 'turns': [], 'status': 'RUNNING', 'retrieval_k': 10, 'visible_k': 10, 'automatic_retries': 0, 'max_model_turns': 5, 'max_ask_concepts': 8}
    write_json(folder / 'start.json', record)
    session = BatchAnswerSession(answer_record, question_aliases(aliases))
    working, history, events, repeat_streak = (None, [], [], 0)
    prompt = load_prompt()
    try:
        for turn in range(max_actions):
            payload = shared.payload_for(job['case'], candidates, tools, prompt, history, max_actions - turn, concepts)
            for card in payload['candidate_tools']:
                card['parameter_schema'] = public_schema(card['parameter_schema'])
            public_events = public_answers(events)
            payload.update(previous_model_working_state=working, ask_alignment=alignment, received_answer_events=public_events, question_state=answer_state(public_events))
            turn_dir = folder / f'turn_{turn + 1:02d}'
            response = shared.model_attempt(payload, turn_dir, provider, key, transport)
            raw = turn_dir / 'raw_response.txt'
            if response['status'] != 'PROVIDER_ERROR' and raw.exists():
                response.update(parse(raw.read_text('utf-8'), provider['model']))
                write_json(turn_dir / 'model_result.json', response)
            entry = {'turn': turn + 1, 'candidate_ids': list(candidates), 'model_result': response}
            record['turns'].append(entry)
            action = response.get('decision')
            if action is None:
                record['status'] = response['status']
                write_json(turn_dir / 'turn.json', entry)
                break
            working = response['working_state']
            rejected = check_stop(action, job['case'], events)
            if rejected:
                entry['rejected_proposal'] = copy.deepcopy(action)
                record.update(status='CONTROL_PROTOCOL_ERROR', protocol_error=rejected)
                write_json(turn_dir / 'turn.json', entry)
                break
            submitted = copy.deepcopy(action)
            if action['action'] == 'ASK':
                answers = session.answer_batch(action['concepts'])
                entry['controlled_answers'] = answers
                entry['replay_concepts'] = [a['concept'] for a in answers]
                events.extend(answers)
                displayed = public_answers(answers)
                history.append({'question': action.get('question', ', '.join(action['concepts'])), 'answer': '\n'.join((a['concept'] + ': ' + a['text'] for a in displayed))})
                repeated = bool(answers) and all((a['kind'] == 'REPEATED_KNOWN_ANSWER' for a in answers))
                repeat_streak = repeat_streak + 1 if repeated else 0
                record['status'] = 'REPEATED_NO_PROGRESS' if repeat_streak >= 2 else 'MAX_TURNS'
            elif action['action'] == 'CALL':
                execution = execute_call(action, payload, tools, snapshot, provider['model'], holds)
                entry['execution'] = execution
                if execution['execution_status'] == 'PASS':
                    submitted['output'] = execution['execution_output']
                record['status'] = execution['status']
            else:
                record['status'] = 'ABSTAINED'
            entry['submitted_action'] = submitted
            record['actions'].append(submitted)
            write_json(turn_dir / 'turn.json', entry)
            if record['status'] != 'MAX_TURNS':
                break
    except Exception as exc:
        record.update(status='ENGINEERING_ERROR', error_type=type(exc).__name__, error=str(exc))
    record.update(finished_at=shared.now(), fallback_retrieval_count=0, post_initial_retrieval_count=0, model_request_count=len(record['turns']))
    write_json(folder / 'result.json', record)
    return record

