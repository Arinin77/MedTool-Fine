"""Final MedTool-Fine research implementation."""
from . import react_io as io
from .interaction import BatchAnswerSession, public_answers
from .io import write_json as write

def question_valid(args):
    return set(args) == {'concepts', 'question'} and isinstance(args['question'], str) and isinstance(args['concepts'], list) and (1 <= len(args['concepts']) <= 8) and all((isinstance(c, str) and c.strip() for c in args['concepts']))

def run_episode(job, folder, provider, key, tools, snapshot, holds, prompt, halted, answer_record, aliases, retrieval, concepts, max_turns=5, transport=None, executor=io.execute, alignment=None):
    folder.mkdir(parents=True, exist_ok=False)
    record = {'episode_id': job['episode_id'], 'started_at': io.now(), 'k': 10, 'status': 'RUNNING', 'turns': [], 'automatic_retries': 0, 'final_submission': None, 'selected_observation': None}
    write(folder / 'start.json', {k: v for k, v in record.items() if k != 'turns'})
    session = BatchAnswerSession(answer_record, aliases)
    history, answers_text = ([], [])
    retrieved_candidates = list(job['initial_candidate_ids'])
    candidates = retrieved_candidates[:10]
    repeat_streak = 0
    refresh = False
    try:
        for step in range(1, max_turns + 1):
            ranking = None
            payload = {'clinical_goal': job['case']['clinical_goal'], 'case_text': job['case']['case_text'], 'candidate_tools': [{'tool_id': tid, 'card': tools[tid]['tool_card'], 'parameter_schema': tools[tid]['parameter_schema']} for tid in candidates], 'output_instruction': prompt, 'trajectory': history, 'step': step, 'model_turns_remaining': max_turns - step + 1, 'global_ask_concepts': list(concepts)}
            payload['ask_alignment'] = alignment
            dest = folder / f'turn_{step:02d}'
            response = io.model_attempt(payload, dest, provider, key, halted, transport)
            entry = {'step': step, 'candidate_ids': list(candidates), 'retrieved_candidate_ids': list(retrieved_candidates), 'model_result': response}
            record['turns'].append(entry)
            if ranking is not None:
                write(dest / 'retrieval.json', ranking)
            d = response.get('decision')
            if d is None:
                record['status'] = response['status']
                break
            if d['action'] == 'Finish':
                record['final_submission'] = d['action_input']
                ref = d['action_input']['result_step']
                selected = next((h['observation'] for h in history if h['step'] == ref), None)
                if ref is not None and (selected is None or 'tool_id' not in selected):
                    record['status'] = 'INVALID_FINAL_REFERENCE'
                else:
                    record.update(status='FINISHED', selected_observation=selected)
                break
            if d['action'] == 'ask_patient':
                if not question_valid(d['action_input']):
                    obs = {'step': step, 'status': 'QUESTION_SCHEMA_ERROR', 'error': 'Use concepts (1 to 8 nonempty strings) and question (string).'}
                else:
                    answers = session.answer_batch(d['action_input']['concepts'])
                    entry['controlled_answers'] = answers
                    obs = {'step': step, 'status': 'PATIENT_RESPONSE', 'answers': [{'concept': a['concept'], 'text': a['text']} for a in public_answers(answers)]}
                    answers_text.append('\n'.join((a['concept'] + ': ' + a['text'] for a in answers)))
                    repeated = bool(answers) and all((a['kind'] == 'REPEATED_KNOWN_ANSWER' for a in answers))
                    repeat_streak = repeat_streak + 1 if repeated else 0
                    refresh = False
            else:
                obs = io.observe(step, d, candidates, tools, snapshot, holds, executor)
            entry['observation'] = obs
            write(dest / 'observation.json', obs)
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

