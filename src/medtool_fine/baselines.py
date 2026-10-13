"""Final MedTool-Fine research implementation."""
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from .io import now, write_json
from .provider import wire_for
from .execution import evaluate

def answer_wire_for(payload, provider):
    wire = {'model': provider.get('request_model', provider['model']), 'messages': [{'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}], 'temperature': provider['temperature'], 'max_completion_tokens': provider['max_output_tokens'], 'response_format': {'type': 'json_object'}, 'stream': False}
    for k in ('reasoning_effort', 'enable_thinking', 'thinking'):
        if k in provider:
            wire[k] = provider[k]
    return wire

def parse_answer(raw, baseline):
    envelope = json.loads(raw)
    content = envelope['choices'][0]['message'].get('content')
    if not isinstance(content, str) or not content.strip():
        raise ValueError('EMPTY_FINAL_CONTENT')
    content = content.strip()
    if content.startswith('```') and content.endswith('```'):
        content = '\n'.join(content.splitlines()[1:-1])
    answer = json.loads(content)
    required = {'answer', 'reasoning'} if baseline == 'no_tool_cot' else {'answer'}
    if not isinstance(answer, dict) or not required.issubset(answer):
        raise ValueError('MISSING_ANSWER_FIELDS')
    if any((not isinstance(answer[k], str) or not answer[k].strip() for k in required)):
        raise ValueError('EMPTY_OR_NONSTRING_ANSWER_FIELDS')
    return {'final_answer': answer['answer'], 'reasoning': answer.get('reasoning', ''), 'extra_fields': sorted(set(answer) - required), 'field_order': list(answer)}

def answer_attempt(job, folder, provider, key, baseline, transport=None):
    folder.mkdir(parents=True, exist_ok=False)
    wire = answer_wire_for(job['payload'], provider)
    write_json(folder / 'request.json', wire)
    start = time.monotonic()
    row = {'request_id': job['request_id'], 'episode_id': job['episode_id'], 'baseline': baseline, 'started_at': now(), 'model': provider['model'], 'automatic_retries': 0, 'tool_calls': 0, 'answer': None}
    try:
        if transport:
            raw = transport(wire)
        else:
            request = Request(provider['endpoint'], data=json.dumps(wire).encode('utf-8'), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
            with urlopen(request, timeout=provider['timeout_seconds']) as response:
                raw = response.read().decode('utf-8')
        (folder / 'raw_response.txt').write_text(raw, encoding='utf-8')
        try:
            envelope = json.loads(raw)
            row['usage'] = envelope.get('usage', {})
            row['response_model'] = envelope.get('model')
            row['finish_reason'] = envelope['choices'][0].get('finish_reason')
            row['answer'] = parse_answer(raw, baseline)
            row['status'] = 'ANSWER_PARSED'
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            row.update(status='OUTPUT_PROTOCOL_ERROR', error=str(exc)[:160])
    except Exception as exc:
        if isinstance(exc, HTTPError):
            (folder / 'raw_response.txt').write_bytes(exc.read())
            row['http_status'] = exc.code
        row.update(status='PROVIDER_ERROR', error_type=type(exc).__name__)
    row.update(elapsed_seconds=round(time.monotonic() - start, 3), finished_at=now())
    write_json(folder / 'result.json', row)
    return row

def static_attempt(job, folder, provider, key, tools, snapshot, numeric_holds, transport=None):
    folder.mkdir(parents=True, exist_ok=False)
    wire = wire_for(job['payload'], provider)
    write_json(folder / 'request.json', wire)
    started = time.monotonic()
    record = {'request_id': job['request_id'], 'episode_id': job['episode_id'], 'k': job['k'], 'started_at': now(), 'model': provider['model'], 'automatic_retries': 0}
    try:
        if transport is None:
            request = Request(provider['endpoint'], data=json.dumps(wire).encode('utf-8'), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
            with urlopen(request, timeout=provider['timeout_seconds']) as response:
                raw = response.read().decode('utf-8')
        else:
            raw = transport(wire)
        (folder / 'raw_response.txt').write_text(raw, encoding='utf-8')
        from .execution import execute
        result = evaluate(raw, job['payload'], tools, snapshot, provider['model'], numeric_holds, executor=execute)
        try:
            envelope = json.loads(raw)
            record['usage'] = envelope.get('usage', {})
            record['response_model'] = envelope.get('model')
        except (ValueError, AttributeError):
            pass
        record.update(result)
    except Exception as exc:
        if isinstance(exc, HTTPError):
            (folder / 'raw_response.txt').write_bytes(exc.read())
            record['http_status'] = exc.code
        record.update({'status': 'PROVIDER_ERROR', 'error_type': type(exc).__name__, 'execution_status': 'NOT_RUN', 'decision': None})
    record.update({'elapsed_seconds': round(time.monotonic() - started, 3), 'finished_at': now()})
    write_json(folder / 'result.json', record)
    return record

