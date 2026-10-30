import subprocess
import sys
import textwrap


def test_missing_optional_sdks_keep_core_importable_and_explain_installation():
    script = textwrap.dedent("""
        import importlib.abc
        import sys

        class MissingOptionalSDKs(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'lab_core', 'inspect_ai', 'phoenix'}:
                    raise ModuleNotFoundError(f'No module named {fullname!r}', name=fullname)

        sys.meta_path.insert(0, MissingOptionalSDKs())
        from trains.datasets import (
            chat_dataset_from_harvey_run,
            chat_dataset_from_inspect_log,
            chat_dataset_from_records,
        )
        from trains.tracing import ArizePhoenixTracingProvider

        assert len(chat_dataset_from_records([])) == 0
        calls = [
            (lambda: chat_dataset_from_harvey_run('missing'), 'lab-core', 'harvey'),
            (lambda: chat_dataset_from_inspect_log('missing'), 'inspect-ai', 'inspect'),
            (lambda: ArizePhoenixTracingProvider(object(), experiment='experiment'),
             'arize-phoenix-client', 'arize-phoenix'),
        ]
        for call, package, extra in calls:
            try:
                call()
            except ImportError as error:
                assert package in str(error), error
                assert f"pip install 'trains-sdk[{extra}]'" in str(error), error
                assert isinstance(error.__cause__, ModuleNotFoundError), error
            else:
                raise AssertionError(f'{extra} did not report the missing SDK')
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
