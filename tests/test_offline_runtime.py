"""Behavioral checks on transparent synthetic fixtures; no model or research data."""
import copy
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
spec = importlib.util.spec_from_file_location('synthetic_demo', PACKAGE / 'scripts/run_offline_demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
from medtool_fine.runner import execute_episode
from medtool_fine import clear as clear_policy
from medtool_fine.scoring import Scorer
from medtool_fine.retrieval import token_windows
from medtool_fine.execution import execute
from medtool_fine.precision_calc import score_episode as score_calc


class OfflineRuntimeTests(unittest.TestCase):
    def test_six_methods_and_clear_transitions_without_network(self):
        with tempfile.TemporaryDirectory(prefix='medtool_fine_test_') as folder:
            with patch.object(socket, 'create_connection', side_effect=AssertionError('Network is forbidden')):
                result = demo.run_demo(Path(folder))
        self.assertEqual(len(result['checks']), 8)
        self.assertEqual(result['model_api_requests'], 0)

    def test_invalid_pool_and_missing_information_do_not_execute(self):
        with tempfile.TemporaryDirectory(prefix='medtool_fine_test_') as folder:
            base = Path(folder)
            prep, snapshot, tools, jobs, env, protocol = demo.fixtures(base)
            job = copy.deepcopy(jobs['synthetic_complete'])
            job['initial_candidate_ids'] = job['initial_candidate_ids'][:9]
            with self.assertRaises(ValueError):
                execute_episode('CLEAR', job, base / 'bad_pool', prep, demo.PROVIDER, '',
                                tools, snapshot, [], env, protocol, demo.ScriptedTransport([]))
            transport = demo.ScriptedTransport([demo.clear_action({'action': 'CALL',
                'tool_id': demo.IDS[0], 'parameters': {'weight_kg': 81.0}})])
            record = execute_episode('CLEAR', jobs['synthetic_complete'], base / 'missing', prep,
                demo.PROVIDER, '', tools, snapshot, [], env, protocol, transport)
            self.assertEqual(record['status'], 'PARAMETER_SCHEMA_ERROR')
            self.assertEqual(record['turns'][0]['execution']['execution_status'], 'NOT_RUN')

    def test_unmatched_answer_is_not_evidence_for_abstention(self):
        policy = clear_policy
        events = [{'concept': 'height', 'kind': 'NO_INFORMATION', 'text': policy.LEGACY_NO_MATCH}]
        self.assertEqual(policy.answer_state(events)[0]['status'], 'NO_MATCHING_ANSWER')
        self.assertIn('not evidence', policy.public_answers(events)[0]['text'])
        decision = {'action': 'ABSTAIN', 'reason': 'REQUIRED_INFORMATION_UNAVAILABLE',
            'justification': 'This should be rejected.', 'evidence': [
                {'source': 'answer', 'concept': 'height', 'quote': 'Height cannot be obtained.'}]}
        self.assertEqual(policy.check_stop(decision, {'case_text': '', 'clinical_goal': ''}, events),
                         'UNSUPPORTED_UNAVAILABILITY')
        with self.assertRaises(ValueError):
            policy.decode(demo.clear_action({'action': 'ASK', 'concepts': ['height'] * 9}))

    def test_wrong_parameters_fail_scoring_and_tampered_calculator_fails_hash(self):
        with tempfile.TemporaryDirectory(prefix='medtool_fine_test_') as folder:
            base = Path(folder)
            prep, snapshot, tools, jobs, env, protocol = demo.fixtures(base)
            transport = demo.ScriptedTransport([demo.clear_action({'action': 'CALL', 'tool_id': demo.IDS[0],
                'parameters': {'weight_kg': 82.0, 'height_m': 1.8}})])
            record = execute_episode('CLEAR', jobs['synthetic_complete'], base / 'wrong', prep,
                demo.PROVIDER, '', tools, snapshot, [], env, protocol, transport)
            self.assertEqual(record['status'], 'EXECUTED')
            self.assertFalse(Scorer(prep, tools, snapshot).assess('CLEAR', jobs['synthetic_complete'], record)['qualified_success'])
            with (snapshot / 'synthetic_bmi.py').open('a', encoding='utf-8') as handle:
                handle.write('# changed\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                execute(snapshot, tools[demo.IDS[0]], demo.PARAMETERS)

    def test_window_bounds(self):
        with self.assertRaises(ValueError):
            token_windows([])
        with self.assertRaises(ValueError):
            token_windows([1], capacity=64, overlap=64)

    def test_calc_evaluation_adapter_on_synthetic_state(self):
        with tempfile.TemporaryDirectory(prefix='medtool_fine_test_') as folder:
            prep, snapshot, tools, jobs, env, protocol = demo.fixtures(Path(folder))
            policy = json.loads((prep / 'numeric_precision.json').read_text('utf-8'))
            policy['dataset'] = 'calc'  # Legacy compatibility key, not actual benchmark data.
            episode = json.loads((prep / 'private_gold.json').read_text('utf-8'))['synthetic_complete']
            result = score_calc(episode, [{'action': 'CALL', 'tool_id': demo.IDS[0],
                'parameters': demo.PARAMETERS, 'output': {'bmi': 25.0}}], tools, snapshot, policy)
            self.assertTrue(result['final_task_success'])
            self.assertTrue(result['strict_trace_success'])


if __name__ == '__main__':
    unittest.main()
