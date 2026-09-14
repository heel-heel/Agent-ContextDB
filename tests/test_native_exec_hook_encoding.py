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


def test_native_exec_hook_classifies_common_cli_tools():
    hook = _native_exec_hook()
    assert hook._infer_tool_name(['git', 'status']) == 'git'
    assert hook._infer_tool_name(['python.exe', '--version']) == 'python'
    assert hook._infer_tool_name(['powershell', '-NoProfile']) == 'powershell'
    assert hook._infer_tool_name([
        'powershell', '-NoProfile', '-Command',
        '$env:LC_ALL="C"; $env:LANG="C"; git show HEAD:assets/settings.json',
    ]) == 'git'
    assert hook._infer_tool_name([
        'powershell', '-NoProfile', '-Command',
        "Get-ChildItem .; git status --short",
    ]) == 'powershell'
    assert hook._infer_tool_name(['cmd', '/c', 'echo ok']) == 'shell'
