"""Final MedTool-Fine research implementation."""
import json
import time
from .io import now, write_json as write
from .provider import wire_for, check_wire, check_response
from .execution import _unique_object, _reject_constant, check_schema, execute, _execute_verified

def parse(raw, expected):
    try:
        env = json.loads(raw)
        choices = env['choices']
        if env.get('model') != expected or len(choices) != 1:
            return {'status': 'INVALID_RESPONSE_ENVELOPE', 'decision': None}
        if choices[0].get('finish_reason') != 'stop':
            return {'status': 'INCOMPLETE_OUTPUT', 'decision': None}
        content = choices[0]['message']['content']
        d = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if not isinstance(d, dict) or set(d) != {'thought', 'action', 'action_input'}:
            raise ValueError('Envelope schema')
        if not isinstance(d['thought'], str) or not isinstance(d['action'], str) or (not d['action']):
            raise ValueError('Action schema')
        if not isinstance(d['action_input'], dict):
            raise ValueError('Arguments schema')
        if d['action'] == 'Finish':
            a = d['action_input']
            if set(a) != {'answer', 'result_step'} or not isinstance(a['answer'], str) or (not a['answer'].strip()):
                raise ValueError('Finish schema')
            step = a['result_step']
            if step is not None and (type(step) is not int or step < 1):
                raise ValueError('Finish reference')
        return {'status': 'ACTION_READY', 'decision': d}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return {'status': 'OUTPUT_PROTOCOL_ERROR', 'decision': None}

def observe(step, decision, candidate_ids, tools, snapshot, holds, executor=_execute_verified):
    """Only public Schema and actual frozen execution feedback; no correctness feedback."""
    tid, params = (decision['action'], decision['action_input'])
    result = {'step': step, 'tool_id': tid, 'parameters': params, 'execution_status': 'NOT_RUN', 'output': None}
    if tid not in candidate_ids or tid not in tools:
        return {**result, 'status': 'UNKNOWN_OR_UNAVAILABLE_TOOL', 'error': 'This action is not a listed calculator tool.'}
    errors = check_schema(params, tools[tid]['parameter_schema'])
    if errors:
        return {**result, 'status': 'PARAMETER_SCHEMA_ERROR', 'errors': errors}
    if tid in holds:
        return {**result, 'status': 'TOOL_EXECUTION_UNAVAILABLE', 'error': 'Execution is unavailable for this tool.'}
    try:
        output = executor(snapshot, tools[tid], params)
        return {**result, 'status': 'EXECUTED', 'execution_status': 'PASS', 'output': output}
    except Exception as exc:
        return {**result, 'status': 'CALCULATOR_EXECUTION_ERROR', 'execution_status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}'}

def model_attempt(payload, folder, provider, key, halted, transport=None):
    folder.mkdir(parents=True, exist_ok=False)
    wire = wire_for(payload, provider)
    check_wire(wire, provider)
    write(folder / 'request.json', wire)
    result = {'started_at': now(), 'automatic_retries': 0}
    start = time.monotonic()
    try:
        if transport is None:
            raise RuntimeError('Explicit guarded transport required')
        raw = transport(wire)
        envelope = json.loads(raw)
        result.update(usage=envelope.get('usage', {}), response_model=envelope.get('model'), finish_reason=envelope.get('choices', [{}])[0].get('finish_reason'))
        check = check_response(raw, provider['model'])
        if not check['accepted']:
            halted.set()
            result.update(status='PROVIDER_MODE_ERROR', decision=None)
        else:
            result.update(parse(raw, provider['model']))
    except Exception as exc:
        halted.set()
        result.update(status='PROVIDER_ERROR', decision=None, error_type=type(exc).__name__)
    finally:
        result.update(finished_at=now(), elapsed_seconds=round(time.monotonic() - start, 3))
        write(folder / 'model_result.json', result)
    return result

