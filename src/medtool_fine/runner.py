"""Final MedTool-Fine research implementation."""
import copy
from pathlib import Path
import threading
from .io import write_json as write
from .execution import public_schema
from . import baselines, clear, react, agentmd

def public_tools(tools):
    result = copy.deepcopy(tools)
    for tool in result.values():
        tool['parameter_schema'] = public_schema(tool['parameter_schema'])
    return result

def execute_episode(method, job, folder, prepared, provider, key, tools, snapshot, holds, answer_env, protocol, transport, halted=None):
    halted = halted or threading.Event()
    methods = Path(prepared) / 'methods'
    if method in ('Direct', 'CoT'):
        folder.mkdir(parents=True, exist_ok=False)
        prompt = (methods / method / 'prompt.txt').read_text('utf-8')
        public_job = {'episode_id': job['episode_id'], 'request_id': job['episode_id'], 'payload': {**job['case'], 'output_instruction': prompt}}
        kind = 'no_tool_direct' if method == 'Direct' else 'no_tool_cot'
        row = baselines.answer_attempt(public_job, folder / 'turn_01', provider, key, kind, transport)
        result = {**row, 'turns': [{'turn': 1, 'model_result': row}]}
        write(folder / 'result.json', result)
        return result
    display = public_tools(tools)
    if method == 'Static':
        folder.mkdir(parents=True, exist_ok=False)
        prompt = (methods / method / 'prompt.txt').read_text('utf-8')
        payload = {**job['case'], 'candidate_tools': [{'tool_id': tid, 'card': display[tid]['tool_card'], 'parameter_schema': display[tid]['parameter_schema']} for tid in job['initial_candidate_ids']], 'output_instruction': prompt}
        public_job = {'episode_id': job['episode_id'], 'request_id': job['episode_id'], 'k': 10, 'payload': payload}
        row = baselines.static_attempt(public_job, folder / 'turn_01', provider, key, display, snapshot, holds, transport=transport)
        actions = []
        if row.get('decision'):
            action = {'action': 'CALL', **row['decision']}
            if row.get('execution_status') == 'PASS':
                action['output'] = row['execution_output']
            actions.append(action)
        result = {'episode_id': job['episode_id'], 'status': row['status'], 'actions': actions, 'started_at': row['started_at'], 'finished_at': row['finished_at'], 'automatic_retries': 0, 'turns': [{'turn': 1, 'candidate_ids': list(job['initial_candidate_ids']), 'model_result': row, 'execution': row}]}
        write(folder / 'result.json', result)
        return result
    env = answer_env[job['episode_id']]
    common = (job, folder, provider, key, display, snapshot, holds)
    if method == 'CLEAR':
        return clear.run_episode(*common, env['record'], env['aliases'], transport=transport, concepts=protocol['global_ask_concepts'], alignment=protocol['alignment'], max_actions=5)
    if method == 'ReAct':
        return react.run_episode(*common, (methods / method / 'prompt.txt').read_text('utf-8'), halted, env['record'], env['aliases'], None, protocol['global_ask_concepts'], max_turns=5, transport=transport, alignment=protocol['alignment'])
    if method == 'AgentMD':
        return agentmd.run_episode(*common, (methods / method / 'selection_prompt.txt').read_text('utf-8'), (methods / method / 'computation_prompt.txt').read_text('utf-8'), halted, env['record'], env['aliases'], protocol['global_ask_concepts'], max_turns=5, transport=transport, alignment=protocol['alignment'])
    raise ValueError('Unconfigured method')

