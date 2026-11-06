"""CPU regression tests for required MXFP8 launch arguments."""
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))
spec = importlib.util.spec_from_file_location('boot', ROOT / 'boot.py')
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)


class RequiredArgsTests(unittest.TestCase):
    def resolve(self, args, backend='b12x'):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = boot.required_sglang_args(args, backend)
        return result, out.getvalue()

    def test_custom_args_keep_required_backend(self):
        args, warning = self.resolve(['--watchdog-timeout', '1800'])
        self.assertEqual(args[-2:], ['--fp8-gemm-backend', 'flashinfer_cutlass'])
        self.assertIn('WARNING', warning)

    def test_present_backend_is_not_duplicated(self):
        original = ['--fp8-gemm-backend', 'flashinfer_cutlass']
        args, warning = self.resolve(original)
        self.assertEqual(args, original)
        self.assertEqual(warning, '')

    def test_explicit_conflicting_backend_refuses(self):
        for original in (['--fp8-gemm-backend', 'auto'],
                         ['--fp8-gemm-backend=auto'],
                         ['--fp8-gemm-backend', 'flashinfer_cutlass', '--fp8-gemm-backend=auto']):
            with self.subTest(original=original):
                with self.assertRaisesRegex(RuntimeError, 'set explicitly'):
                    self.resolve(original)

    def test_overridden_explicit_conflict_is_kept(self):
        args, warning = self.resolve(['--fp8-gemm-backend=auto', '--fp8-gemm-backend', 'flashinfer_cutlass'])
        self.assertEqual(args, ['--fp8-gemm-backend', 'flashinfer_cutlass'])
        self.assertEqual(warning, '')

    def test_disabled_modes_leave_args_unchanged(self):
        original = ['--fp8-gemm-backend=auto']
        for backend in ('', 'cutlass', 'off', '0', ' CUTLASS ', 'OFF'):
            with self.subTest(backend=backend):
                args, warning = self.resolve(original, backend)
                self.assertEqual(args, original)
                self.assertEqual(warning, '')

    def test_boot_calls_resolver_after_extra_args(self):
        import inspect
        source = inspect.getsource(boot.serve)
        self.assertLess(source.index('args += extra.split()'), source.index('args = required_sglang_args('))

    def test_boot_imports_with_image_style_adapter_pythonpath(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'adapter'))
        result = subprocess.run(
            [sys.executable, '-c',
             'import importlib.util; '
             f's=importlib.util.spec_from_file_location("boot", {str(ROOT / "boot.py")!r}); '
             'm=importlib.util.module_from_spec(s); s.loader.exec_module(m)'],
            cwd='/', env=env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == '__main__':
    unittest.main()
