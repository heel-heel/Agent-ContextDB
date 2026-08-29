# ContextDB Live Hook Mode

When a user explicitly asks to test or run live ContextDB integration, use the
provided `source` and `session_id` consistently for the entire Agent session.
Run shell commands through the native-exec hook wrapper, not through a mocked
tool result:

```powershell
& 'D:\software\Anaconda\ProgramFile\envs\contextdb_env\python.exe' `
  'D:\software\Pycharm\SelfCode\Agent-ContextDB\tools\contextdb_native_exec_hook.py' `
  --source codex --session-id <session_id> -- <real command and arguments>
```

The wrapper runs the real child command, returns its original output and exit
code, records the result in ContextDB, and may append a
`CONTEXTDB_RECOMMENDATION` object. Treat that object as advisory. If applying a
recommendation, execute the real repair through the same wrapper with
`--apply-skill`; do not claim it succeeded without observing its result.

Outside an explicit live integration request, use normal project commands.
