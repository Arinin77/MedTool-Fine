"""Final MedTool-Fine research implementation."""
import copy
import math
from .evaluation_core import contract_check, finite_number, unit_key, equivalent, score_call as legacy_call, score_episode as legacy_episode
from .execution import _execute_verified as execute

def within(actual, expected, policy):
    if not finite_number(actual) or not finite_number(expected):
        return False
    allowance = max(policy['absolute_floor'], policy['relative_tolerance'] * abs(expected))
    error = abs(actual - expected)
    return error <= allowance or math.isclose(error, allowance, rel_tol=1e-12, abs_tol=0)

def continuous_output_keys(tool, policy):
    """Only scalar continuous results; integer, categorical and array outputs stay exact."""
    excluded = set(policy['discrete_output_keys'])
    excluded.update(policy['discrete_tool_outputs'].get(tool['tool_id'], []))
    result = set()
    for key, schema in tool['output_schema'].get('properties', {}).items():
        types = schema.get('type', [])
        types = [types] if isinstance(types, str) else types
        if 'number' in types and key not in excluded and ('enum' not in schema):
            result.add(key)
    return result

def compare_output(actual, expected, tool, policy, *, discrete_only=False):
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        return False
    continuous = continuous_output_keys(tool, policy)
    for key, value in expected.items():
        if key in continuous and finite_number(value) and finite_number(actual[key]):
            if not discrete_only and (not within(actual[key], value, policy)):
                return False
        elif not equivalent(actual[key], value, abs_tol=policy['absolute_floor'], rel_tol=0):
            return False
    return True

def parameter_check(gold, reference, tool, key, actual, policy):
    expected = reference['parameters'].get(key)
    if not finite_number(actual) or not finite_number(expected):
        return {'accepted': False, 'reason': 'NON_FINITE_OR_NON_NUMERIC'}
    schema = tool['parameter_schema']['properties'].get(key, {})
    binding = reference.get('parameter_bindings', {}).get(key)
    facts = [f for f in gold.get('facts', []) if f.get('fact_id') == binding]
    if len(facts) != 1 or not finite_number(facts[0].get('value')):
        return {'accepted': False, 'reason': 'MISSING_OR_AMBIGUOUS_SOURCE_BINDING'}
    fact = facts[0]
    if schema.get('type') != 'number' or 'enum' in schema or unit_key(fact.get('unit')) in policy['exact_input_units'] or (key in policy['exact_input_keys']):
        return {'accepted': False, 'reason': 'DISCRETE_OR_EXACT_INPUT'}
    name = policy['conversion_parameters'].get(tool['tool_id'], {}).get(key)
    conversion = policy['conversions'].get(name, {})
    converts = bool(conversion and conversion['source_unit'] != conversion['target_unit'] and (unit_key(fact.get('unit')) == unit_key(conversion['source_unit'])))
    derived = fact.get('origin') in policy['derived_origins'] and fact.get('model_visible') is not True
    if not converts and (not derived):
        return {'accepted': False, 'reason': 'DIRECT_FACT_REQUIRES_FAITHFUL_EXTRACTION'}
    canonical = fact['value'] * conversion['canonical_factor'] if converts else fact['value']
    if not equivalent(canonical, expected, abs_tol=1e-08, rel_tol=1e-08):
        return {'accepted': False, 'reason': 'SOURCE_DOES_NOT_REPRODUCE_GOLD'}
    return {'accepted': within(actual, expected, policy), 'reason': 'CONTINUOUS_CONVERSION_OR_DERIVATION_1_PERCENT', 'source_fact_id': binding, 'relative_error': abs(actual - expected) / abs(expected) if expected else None}

def score_call(gold, submission, tools, snapshot, policy):
    if policy.get('dataset') != 'calc':
        raise ValueError('This scoring entry point requires the calc policy')
    result = contract_check(legacy_call(gold, submission, tools, snapshot), submission, tools)
    result['scoring_version'] = policy['policy_id']
    result['precision_checks'] = {}
    if result.get('error') or not result['execution_pass']:
        return result
    tid, params = (submission['tool_id'], submission['parameters'])
    tool, ref = (tools[tid], gold['references'][tid])
    checks = result['precision_checks']
    for key, value in params.items():
        if key not in ref['parameters']:
            checks[key] = {'accepted': False, 'reason': 'UNBOUND_PARAMETER'}
        elif not equivalent(value, ref['parameters'][key], **gold['parameter_tolerance']):
            checks[key] = parameter_check(gold, ref, tool, key, value, policy)
            if checks[key]['accepted']:
                probe = dict(ref['parameters'], **{key: value})
                try:
                    guard = execute(snapshot, tool, probe)
                    checks[key]['accepted'] = compare_output(guard, ref['output'], tool, policy, discrete_only=True)
                    if not checks[key]['accepted']:
                        checks[key]['reason'] = 'SCORING_THRESHOLD_OR_CATEGORY_CROSSED'
                except Exception:
                    checks[key] = {'accepted': False, 'reason': 'THRESHOLD_PROBE_EXECUTION_FAILED'}
    result['parameters_correct'] = all((x['accepted'] for x in checks.values()))
    actual = execute(snapshot, tool, params)
    result['output_correct'] = compare_output(actual, ref['output'], tool, policy)
    if 'output' in submission:
        result['output_correct'] &= equivalent(submission['output'], actual, **ref['output_tolerance'])
    result['task_success'] = bool(result['parameters_correct'] and result['output_correct'])
    result['executed_submitted_output'] = actual
    result['reference_output'] = ref['output']
    return result

def score_episode(episode, actions, tools, snapshot, policy, max_actions=8):
    result = copy.deepcopy(legacy_episode(episode, actions, tools, snapshot, max_actions))
    result['scoring_version'] = policy['policy_id']
    if result['error'] is not None or result['terminal_call'] is None:
        return result
    state = episode['initial_state_id']
    for answer in result['controlled_answers']:
        if answer['kind'] == 'CONTROLLED_NEW_INFORMATION':
            state = episode['states'][state]['ask_transitions'][answer['concept']]['next_state_id']
    call = score_call(episode['states'][state], actions[-1], tools, snapshot, policy)
    result['terminal_call'] = call
    result['final_task_success'] = call['task_success']
    strict = result['strict_trace']
    if strict.get('terminal_call') is not None and strict.get('error') is None:
        strict['terminal_call'] = copy.deepcopy(call)
        strict['task_success'] = bool(strict['initial_action_correct'] and call['task_success'])
        result['strict_trace_success'] = strict['task_success']
    return result

