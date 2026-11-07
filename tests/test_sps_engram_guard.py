#!/usr/bin/env python3
"""A profiled SPS table must not reach the Engram engine by default."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))
spec = importlib.util.spec_from_file_location('boot', ROOT / 'boot.py')
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)


class SpsEngramGuardTests(unittest.TestCase):
    def setUp(self):
        fd, self.table = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        self.addCleanup(os.unlink, self.table)

    def call(self, table, env, default=None):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            return boot.sps_table_args(table, env, table if default is None else default), output.getvalue()

    def test_existing_table_is_ignored_by_default(self):
        args, message = self.call(self.table, {})
        self.assertEqual(args, [])
        self.assertIn('Engram', message)
        self.assertIn('WARNING', message)

    def test_explicit_table_refuses_instead_of_being_ignored(self):
        with self.assertRaisesRegex(RuntimeError, 'set explicitly'):
            self.call(self.table, {}, default='/state/dspark_sps.json')
        with self.assertRaisesRegex(RuntimeError, 'does not exist'):
            self.call('/nonexistent/sps.json', {'DSPARK_SPS_ALLOW_ENGRAM': '1'},
                      default='/state/dspark_sps.json')

    def test_sts_table(self):
        self.assertEqual(boot.sts_table_args(self.table), ['--speculative-dspark-confidence-sts-path', self.table])
        self.assertEqual(boot.sts_table_args(''), [])
        with self.assertRaisesRegex(RuntimeError, 'does not exist'):
            boot.sts_table_args('/nonexistent/sts.json')

    def test_explicit_fixed_engine_override_restores_flags(self):
        args, _ = self.call(self.table, {'DSPARK_SPS_ALLOW_ENGRAM': '1'})
        self.assertEqual(
            args[:2], ['--speculative-dspark-sps-table-path', self.table])
        self.assertIn(
            '--speculative-dspark-align-verify-tokens-to-graph-tier', args)

    def test_missing_or_unset_table_stays_safe(self):
        self.assertEqual(self.call('', {})[0], [])
        args, message = self.call(
            '/nonexistent/sps.json', {'DSPARK_SPS_ALLOW_ENGRAM': '1'})
        self.assertEqual(args, [])
        self.assertIn('not found', message)


if __name__ == '__main__':
    unittest.main()
