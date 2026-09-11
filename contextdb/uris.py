from hashlib import sha256


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


def summary_view_key(tid: str, branch_id: str, profile_identity: str) -> str:
    """Keep semantic-summary materializations isolated by LLM configuration."""
    identity = str(profile_identity or "default").encode("utf-8")
    profile_hash = sha256(identity).hexdigest()[:16]
    return f"trajectories/{tid}/views/{branch_id}/summary_profiles/{profile_hash}"


def semantic_judgments_view_key(tid: str, profile_identity: str) -> str:
    """Keep trajectory-wide semantic judgments isolated by LLM configuration."""
    identity = str(profile_identity or "default").encode("utf-8")
    profile_hash = sha256(identity).hexdigest()[:16]
    return f"trajectories/{tid}/views/all_branches/semantic_judgments_profiles/{profile_hash}"

def artifact_key(tid: str, artifact_id: str) -> str:
    return f"trajectories/{tid}/artifacts/{artifact_id}"
