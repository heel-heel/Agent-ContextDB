import importlib.util
from pathlib import Path


def _native_exec_hook():
    path = Path(__file__).parents[1] / 'tools' / 'contextdb_native_exec_hook.py'
    spec = importlib.util.spec_from_file_location('contextdb_native_exec_hook', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_exec_hook_decodes_utf8_and_gb18030_output():
    hook = _native_exec_hook()
    message = '找不到指定的路径'
    assert hook._decode_process_output(message.encode('utf-8')) == message
    assert hook._decode_process_output(message.encode('gb18030')) == message
