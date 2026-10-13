"""Final MedTool-Fine research implementation."""
from . import react_io as io
from .interaction import BatchAnswerSession, public_answers
from .io import write_json as write

def selection_view(job, tools, prompt):
    return {'clinical_goal': job['case']['clinical_goal'], 'case_text': job['case']['case_text'], 'candidate_tools': [{'tool_id': tid, 'name': tools[tid]['tool_card'].get('name', ''), 'purpose': tools[tid]['tool_card'].get('clinical_purpose', '')} for tid in job['initial_candidate_ids'][:10]], 'output_instruction': prompt}

def select_tool(response, candidate_ids):
    d = response.get('decision')
    if d and d['action'] == 'Select' and (set(d['action_input']) == {'tool_id'}) and isinstance(d['action_input']['tool_id'], str) and (d['action_input']['tool_id'] in candidate_ids):
        return (d['action_input']['tool_id'], False)
    return (candidate_ids[0], True)

def question_valid(args):
    return set(args) == {'concepts', 'question'} and isinstance(args['question'], str) and isinstance(args['concepts'], list) and (1 <= len(args['concepts']) <= 8) and all((isinstance(c, str) and c.strip() for c in args['concepts']))

def run_episode(job, folder, provider, key, tools, snapshot, holds, selection_prompt, computation_prompt, halted, answer_record, aliases, concepts, max_turns=5, transport=None, executor=io.execute, alignment=None):
    folder.mkdir(parents=True, exist_ok=False)
    record = {'episode_id': job['episode_id'], 'k': job['k'], 'started_at': io.now(), 'status': 'RUNNING', 'turns': [], 'automatic_retries': 0, 'selected_tool_id': None, 'selection_fallback': False, 'final_submission': None, 'selected_observation': None}
    write(folder / 'start.json', record)
    candidates = job['initial_candidate_ids'][:10]
    history = []
    session = BatchAnswerSession(answer_record, aliases)
    repeat_streak = 0
    try:
        response = io.model_attempt(selection_view(job, tools, selection_prompt), folder / 'turn_01', provider, key, halted, transport)
        first = {'step': 1, 'phase': 'selection', 'candidate_ids': list(candidates), 'model_result': response}
        record['turns'].append(first)
        if response['status'] not in ('ACTION_READY', 'OUTPUT_PROTOCOL_ERROR'):
            record['status'] = response['status']
        else:
            selected, fallback = select_tool(response, candidates)
            record.update(selected_tool_id=selected, selection_fallback=fallback)
            first['selected_tool_id'] = selected
            first['selection_fallback'] = fallback
            write(folder / 'turn_01/selection.json', {'selected_tool_id': selected, 'fallback_to_top1': fallback})
            tool = tools[selected]
            public_tool = {'tool_id': selected, 'card': tool['tool_card'], 'parameter_schema': tool['parameter_schema']}
            for step in range(2, max_turns + 1):
                payload = {'clinical_goal': job['case']['clinical_goal'], 'case_text': job['case']['case_text'], 'candidate_tools': [public_tool], 'output_instruction': computation_prompt, 'trajectory': history, 'step': step, 'model_turns_remaining': max_turns - step + 1, 'global_ask_concepts': list(concepts)}
                payload['ask_alignment'] = alignment
                response = io.model_attempt(payload, folder / f'turn_{step:02d}', provider, key, halted, transport)
                entry = {'step': step, 'phase': 'computation', 'candidate_ids': [selected], 'model_result': response}
                record['turns'].append(entry)
                d = response.get('decision')
                if d is None:
                    record['status'] = response['status']
                    break
                if d['action'] == 'Finish':
                    record['final_submission'] = d['action_input']
                    ref = d['action_input']['result_step']
                    obs = next((h['observation'] for h in history if h['step'] == ref), None)
                    if ref is not None and (obs is None or 'tool_id' not in obs):
                        record['status'] = 'INVALID_FINAL_REFERENCE'
                    else:
                        record.update(status='FINISHED', selected_observation=obs)
                    break
                if d['action'] == 'ask_patient':
                    if not question_valid(d['action_input']):
                        obs = {'step': step, 'status': 'QUESTION_SCHEMA_ERROR', 'error': 'Use concepts (1 to 8 nonempty strings) and question (string).'}
                        repeat_streak = 0
                    else:
                        answers = session.answer_batch(d['action_input']['concepts'])
                        entry['controlled_answers'] = answers
                        obs = {'step': step, 'status': 'PATIENT_RESPONSE', 'answers': [{'concept': a['concept'], 'text': a['text']} for a in public_answers(answers)]}
                        repeated = bool(answers) and all((a['kind'] == 'REPEATED_KNOWN_ANSWER' for a in answers))
                        repeat_streak = repeat_streak + 1 if repeated else 0
                else:
                    repeat_streak = 0
                    obs = io.observe(step, d, [selected], tools, snapshot, holds, executor)
                entry['observation'] = obs
                write(folder / f'turn_{step:02d}' / 'observation.json', obs)
                history.append({'step': step, 'thought': d['thought'], 'action': d['action'], 'action_input': d['action_input'], 'observation': obs})
                if repeat_streak >= 2:
                    record['status'] = 'REPEATED_NO_PROGRESS'
                    break
            else:
                record['status'] = 'MAX_TURNS'
    except Exception as exc:
        halted.set()
        record.update(status='ENGINEERING_ERROR', error_type=type(exc).__name__)
    record['finished_at'] = io.now()
    write(folder / 'result.json', record)
    return record

