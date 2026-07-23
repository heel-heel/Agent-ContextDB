def trajectory_meta(tid: str) -> str:
    return f"trajectories/{tid}/meta"

def event_key(tid: str, event_id: str) -> str:
    return f"trajectories/{tid}/events/{event_id}"

def branch_key(tid: str, branch_id: str) -> str:
    return f"trajectories/{tid}/branches/{branch_id}"

def snapshot_key(tid: str, snapshot_id: str) -> str:
    return f"trajectories/{tid}/snapshots/{snapshot_id}"

def view_key(tid: str, branch_id: str, view_name: str) -> str:
    return f"trajectories/{tid}/views/{branch_id}/{view_name}"

def artifact_key(tid: str, artifact_id: str) -> str:
    return f"trajectories/{tid}/artifacts/{artifact_id}"
