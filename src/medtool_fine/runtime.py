"""Final MedTool-Fine research implementation."""
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from .io import now, write_json
from .provider import wire_for
from .execution import FORMAT_INSTRUCTION, _unique_object, _reject_constant, evaluate

def make_payload(case, ids, tools):
    return {'clinical_goal': case['clinical_goal'], 'case_text': case['case_text'], 'candidate_tools': [{'tool_id': tid, 'card': tools[tid]['tool_card'], 'parameter_schema': tools[tid]['parameter_schema']} for tid in ids], 'output_instruction': FORMAT_INSTRUCTION}

def parse_action(raw, expected_model):

    def fail(status):
        return {'status': status, 'decision': None}
    try:
        envelope = json.loads(raw)
        if not isinstance(envelope, dict) or 'error' in envelope:
            return fail('INVALID_RESPONSE_ENVELOPE')
        if envelope.get('model') != expected_model:
            return fail('RESPONSE_MODEL_MISMATCH')
        choices = envelope.get('choices')
        if not isinstance(choices, list) or len(choices) != 1:
            return fail('INVALID_RESPONSE_ENVELOPE')
        if choices[0].get('finish_reason') != 'stop':
            return fail('INCOMPLETE_FINAL_OUTPUT')
        content = choices[0].get('message', {}).get('content')
        if not isinstance(content, str) or not content.strip():
            return fail('EMPTY_FINAL_CONTENT')
        action = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, TypeError, AttributeError, KeyError):
        return fail('INVALID_FINAL_JSON')
    if not isinstance(action, dict):
        return fail('ACTION_SCHEMA_ERROR')
    kind = action.get('action')
    fields = {'CALL': {'action', 'tool_id', 'parameters'}, 'ASK': {'action', 'concept', 'question'}, 'ABSTAIN': {'action', 'reason'}}
    if not isinstance(kind, str) or kind not in fields:
        return fail('ACTION_SCHEMA_ERROR')
    if set(action) != fields[kind] and (not (kind == 'ASK' and set(action) == {'action', 'concept'})):
        return fail('ACTION_SCHEMA_ERROR')
    if kind == 'CALL' and (not isinstance(action['tool_id'], str) or not isinstance(action['parameters'], dict)):
        return fail('ACTION_SCHEMA_ERROR')
    if kind == 'ASK' and any((not isinstance(action[f], str) or not action[f].strip() for f in ('concept', 'question') if f in action)):
        return fail('ACTION_SCHEMA_ERROR')
    if kind == 'ABSTAIN' and action['reason'] not in ('NO_APPLICABLE_TOOL', 'REQUIRED_INFORMATION_UNAVAILABLE'):
        return fail('ACTION_SCHEMA_ERROR')
    return {'status': 'ACTION_READY', 'decision': action}

def payload_for(case, candidates, tools, prompt, history, remaining, concepts=()):
    payload = make_payload(case, candidates, tools)
    payload['output_instruction'] = prompt
    payload['interaction_history'] = [{'question': h['question'], 'answer': h['answer']} for h in history]
    payload['model_turns_remaining'] = remaining
    payload['global_ask_concepts'] = list(concepts)
    return payload

def model_attempt(payload, folder, provider, key, transport=None):
    folder.mkdir(parents=True, exist_ok=False)
    wire = wire_for(payload, provider)
    write_json(folder / 'request.json', wire)
    start = time.monotonic()
    result = {'started_at': now(), 'automatic_retries': 0}
    try:
        if transport:
            raw = transport(wire)
        else:
            request = Request(provider['endpoint'], data=json.dumps(wire).encode('utf-8'), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
            with urlopen(request, timeout=provider['timeout_seconds']) as response:
                raw = response.read().decode('utf-8')
        (folder / 'raw_response.txt').write_text(raw, encoding='utf-8')
        result.update(parse_action(raw, provider['model']))
        try:
            envelope = json.loads(raw)
            result['usage'] = envelope.get('usage', {})
            result['response_model'] = envelope.get('model')
        except (ValueError, AttributeError):
            pass
    except Exception as exc:
        if isinstance(exc, HTTPError):
            (folder / 'raw_response.txt').write_bytes(exc.read())
            result['http_status'] = exc.code
        result.update(status='PROVIDER_ERROR', decision=None, error_type=type(exc).__name__)
    result.update(elapsed_seconds=round(time.monotonic() - start, 3), finished_at=now())
    write_json(folder / 'model_result.json', result)
    return result

