"""Final MedTool-Fine research implementation."""
import json

def wire_for(payload, provider):
    wire = {'model': provider.get('request_model', provider['model']), 'messages': [{'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, allow_nan=False)}], 'temperature': provider['temperature'], 'max_completion_tokens': provider['max_output_tokens'], 'response_format': {'type': 'json_object'}, 'stream': False}
    for key in ('reasoning_effort', 'enable_thinking', 'thinking'):
        if key in provider:
            wire[key] = provider[key]
    return wire

MODEL = 'gpt-5.4'

def check_wire(wire, provider):
    expected = {'model', 'messages', 'temperature', 'max_completion_tokens', 'reasoning_effort', 'response_format', 'stream'}
    if set(wire) != expected or wire['model'] != provider.get('request_model', provider['model']) or wire['model'] != 'gpt-5.4' or (provider['model'] != MODEL) or (wire['reasoning_effort'] != 'none') or (wire['temperature'] != 0) or (wire['max_completion_tokens'] != provider['max_output_tokens']) or (wire['max_completion_tokens'] != 12000) or (wire['stream'] is not False) or (wire['response_format'] != {'type': 'json_object'}):
        raise ValueError('GPT wire/model/endpoint mismatch')
    if not isinstance(wire['messages'], list) or not wire['messages']:
        raise ValueError('Missing messages')

def check_response(raw, expected):
    try:
        obj = json.loads(raw)
        choices = obj['choices']
        message = choices[0]['message']
        usage = obj.get('usage') or {}
        reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
        numeric = all((type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')))
        valid_usage = numeric and usage['prompt_tokens'] > 0 and (usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'])
        valid = 'error' not in obj and expected == MODEL and (obj.get('model') in (MODEL, 'gpt-5.4-2026-03-05')) and (len(choices) == 1) and isinstance(message.get('content'), str) and (not message.get('reasoning_content')) and (type(reasoning) is int) and (reasoning == 0) and valid_usage
        return {'accepted': valid, 'reason': 'OK' if valid else 'MODEL_MODE_OR_USAGE_MISMATCH', 'response_model': obj.get('model'), 'reasoning_tokens': reasoning, 'reasoning_tokens_field_present': reasoning is not None, 'reasoning_content_nonempty': bool(message.get('reasoning_content')), 'usage_verified': valid_usage, 'provider_metadata_not_upstream_attestation': True}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return {'accepted': False, 'reason': 'SUPPLIER_ENVELOPE_UNVERIFIED'}

