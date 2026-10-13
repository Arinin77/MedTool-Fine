"""Final MedTool-Fine research implementation."""
import copy
import json
import math
import re
from typing import Any, Mapping
from .io import canonical_digest, inside, digest

def _is_type(value: Any, expected: str) -> bool:
    if expected == 'null':
        return value is None
    if expected == 'object':
        return isinstance(value, dict)
    if expected == 'array':
        return isinstance(value, list)
    if expected == 'string':
        return isinstance(value, str)
    if expected == 'boolean':
        return isinstance(value, bool)
    if expected == 'integer':
        return isinstance(value, int) and (not isinstance(value, bool))
    if expected == 'number':
        return isinstance(value, (int, float)) and (not isinstance(value, bool))
    return False

def _schema_errors(value: Any, schema: Mapping[str, Any], path: str='$') -> list[str]:
    errors: list[str] = []
    expected = schema.get('type')
    if expected is not None:
        expected_types = [expected] if isinstance(expected, str) else list(expected)
        if not any((_is_type(value, item) for item in expected_types)):
            return [f'{path}: expected type {expected_types}, got {type(value).__name__}']
    if 'const' in schema and value != schema['const']:
        errors.append(f"{path}: expected constant {schema['const']!r}")
    if 'enum' in schema and value not in schema['enum']:
        errors.append(f'{path}: value {value!r} not in enum')
    if 'anyOf' in schema:
        branches = schema['anyOf']
        if not isinstance(branches, list) or not branches:
            errors.append(f'{path}: anyOf must contain at least one schema')
        elif not any((isinstance(branch, Mapping) and (not _schema_errors(value, branch, path)) for branch in branches)):
            errors.append(f'{path}: value does not satisfy anyOf')
    if isinstance(value, str) and 'pattern' in schema:
        if re.fullmatch(str(schema['pattern']), value) is None:
            errors.append(f"{path}: string does not match pattern {schema['pattern']!r}")
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if 'minimum' in schema and value < schema['minimum']:
            errors.append(f"{path}: value is below minimum {schema['minimum']}")
    if isinstance(value, dict):
        required = schema.get('required', [])
        for key in required:
            if key not in value:
                errors.append(f'{path}: missing required property {key!r}')
        properties = schema.get('properties', {})
        if schema.get('additionalProperties') is False:
            for key in value:
                if key not in properties:
                    errors.append(f'{path}: unexpected property {key!r}')
        for key, child_schema in properties.items():
            if key in value:
                errors.extend(_schema_errors(value[key], child_schema, f'{path}.{key}'))
    if isinstance(value, list):
        if 'minItems' in schema and len(value) < int(schema['minItems']):
            errors.append(f"{path}: expected at least {schema['minItems']} items")
        item_schema = schema.get('items')
        if item_schema:
            for index, item in enumerate(value):
                errors.extend(_schema_errors(item, item_schema, f'{path}[{index}]'))
    return errors

def check_schema(value, schema, path='$'):
    """Builder's structural checker plus all bounds used by this frozen registry."""
    errors = _schema_errors(value, schema, path)
    if isinstance(value, (float, int)) and (not isinstance(value, bool)):
        if not math.isfinite(value):
            errors.append(f'{path}: nonfinite number')
        for key, invalid in (('maximum', lambda x: value > x), ('exclusiveMaximum', lambda x: value >= x), ('exclusiveMinimum', lambda x: value <= x)):
            if key in schema and invalid(schema[key]):
                errors.append(f'{path}: violates {key}')
    if isinstance(value, dict):
        for key, item in value.items():
            if key in schema.get('properties', {}):
                errors.extend(check_schema(item, schema['properties'][key], path + '.' + key))
    if isinstance(value, list):
        if schema.get('uniqueItems') and len({canonical_digest(v) for v in value}) != len(value):
            errors.append(f'{path}: duplicate items')
        for i, item in enumerate(value):
            errors.extend(check_schema(item, schema.get('items', {}), f'{path}[{i}]'))
    return sorted(set(errors))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def _check_basic(value, schema):
    errors = check_schema(value, schema)
    require(not errors, '; '.join(errors))

def _execute_verified(snapshot, tool, parameters):
    _check_basic(parameters, tool['parameter_schema'])
    path = inside(snapshot, tool['implementation']['path'])
    require(digest(path) == tool['implementation']['sha256'], 'frozen implementation hash mismatch')
    namespace = {'__name__': 'verified_calculator', 'math': math}
    exec(compile(path.read_text('utf-8'), str(path), 'exec'), namespace)
    output = namespace[tool['implementation']['entrypoint']](parameters)
    _check_basic(output, tool['output_schema'])
    return output

def checked(value, schema):
    _check_basic(value, schema)
    if 'if' in schema:
        try:
            _check_basic(value, schema['if'])
        except ValueError:
            branch = schema.get('else', {})
        else:
            branch = schema.get('then', {})
        checked(value, branch)

def execute(snapshot, tool, parameters):
    checked(parameters, tool['parameter_schema'])
    return _execute_verified(snapshot, tool, parameters)

def public_schema(schema):
    scalar = {'type', 'enum', 'const', 'required', 'additionalProperties', 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'minItems', 'maxItems', 'uniqueItems', 'pattern', 'description', 'title', 'default'}
    result = {key: copy.deepcopy(value) for key, value in schema.items() if key in scalar and (not isinstance(value, dict))}
    if 'properties' in schema:
        result['properties'] = {key: public_schema(value) for key, value in schema['properties'].items()}
    for key in ('items', 'if', 'then', 'else'):
        if key in schema:
            result[key] = public_schema(schema[key])
    for key in ('anyOf', 'oneOf', 'allOf'):
        if key in schema:
            result[key] = [public_schema(value) for value in schema[key]]
    return result

OUTPUT_SCHEMA = {'type': 'object', 'required': ['tool_id', 'parameters'], 'additionalProperties': False, 'properties': {'tool_id': {'type': 'string'}, 'parameters': {'type': 'object'}}}

INSTRUCTION = 'Select one tool from the candidates for the clinical goal using the patient information. Return only a JSON object with tool_id and parameters.'

FORMAT_INSTRUCTION = INSTRUCTION + ' The final answer must contain one non-empty JSON object with exactly the keys tool_id and parameters. Example of the output structure: {"tool_id":"<candidate_tool_id>","parameters":{"<parameter_name>":"<schema-typed value>"}}. The example illustrates structure only; replace placeholders with the selected candidate ID, its parameter names, and values of the types specified by its schema. Return the JSON object as the final answer without Markdown fences or surrounding prose.'

def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result

def _reject_constant(value):
    raise ValueError('nonfinite JSON number')

def parse_final(raw, expected_model):
    """Never interpret reasoning_content as a final answer or manufacture missing JSON."""

    def fail(status, **extra):
        return {'status': status, 'decision': None, **extra}
    try:
        envelope = json.loads(raw)
    except (ValueError, TypeError):
        return fail('INVALID_RESPONSE_ENVELOPE')
    if not isinstance(envelope, dict):
        return fail('INVALID_RESPONSE_ENVELOPE')
    if 'error' in envelope:
        return fail('PROVIDER_ERROR')
    if envelope.get('model') != expected_model:
        return fail('RESPONSE_MODEL_MISMATCH')
    choices = envelope.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or (not isinstance(choices[0], dict)):
        return fail('INVALID_RESPONSE_ENVELOPE')
    choice = choices[0]
    if choice.get('finish_reason') != 'stop':
        return fail('INCOMPLETE_FINAL_OUTPUT', finish_reason=choice.get('finish_reason'))
    message = choice.get('message')
    if not isinstance(message, dict):
        return fail('INVALID_RESPONSE_ENVELOPE')
    content = message.get('content')
    if content is None or (isinstance(content, str) and (not content.strip())):
        return fail('EMPTY_FINAL_CONTENT', reasoning_present=bool(message.get('reasoning_content')))
    if not isinstance(content, str):
        return fail('INVALID_FINAL_CONTENT_TYPE')
    try:
        decision = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, TypeError):
        return fail('INVALID_FINAL_JSON')
    errors = check_schema(decision, OUTPUT_SCHEMA)
    if errors:
        return fail('OUTPUT_SCHEMA_ERROR', errors=errors)
    return {'status': 'FINAL_JSON_READY', 'decision': decision}

def evaluate(raw, payload, tools, snapshot, expected_model, numeric_holds=(), executor=_execute_verified):
    result = parse_final(raw, expected_model)
    result.update({'execution_status': 'NOT_RUN', 'missing_parameters': [], 'execution_output': None})
    if result['status'] != 'FINAL_JSON_READY':
        return result
    decision = result['decision']
    tid = decision['tool_id']
    params = decision['parameters']
    candidates = {x['tool_id'] for x in payload['candidate_tools']}
    if tid not in candidates or tid not in tools:
        result['status'] = 'OUT_OF_CANDIDATES'
        return result
    schema = tools[tid]['parameter_schema']
    missing = sorted(set(schema.get('required', [])) - set(params))
    if missing:
        result.update({'status': 'MISSING_PARAMETERS', 'missing_parameters': missing})
        return result
    errors = check_schema(params, schema)
    if errors:
        result.update({'status': 'PARAMETER_SCHEMA_ERROR', 'errors': errors})
        return result
    if tid in numeric_holds:
        result.update({'status': 'NUMERIC_REFERENCE_HOLD', 'execution_status': 'SKIPPED_NUMERIC_HOLD'})
        return result
    try:
        result['execution_output'] = executor(snapshot, tools[tid], params)
        result.update({'status': 'EXECUTED', 'execution_status': 'PASS'})
    except Exception as exc:
        result.update({'status': 'CALCULATOR_EXECUTION_ERROR', 'execution_status': 'FAIL', 'error_type': type(exc).__name__})
    return result

