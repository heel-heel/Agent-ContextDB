#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-http://127.0.0.1:8765}
curl -s "$BASE/health"
printf '\n'
TRAJ=$(curl -s -X POST "$BASE/api/v1/trajectories" -d '{"title":"HTTP demo","agent_id":"codex","source_id":"curl"}' | python -c 'import sys,json; print(json.load(sys.stdin)["trajectory_id"])')
curl -s -X POST "$BASE/api/v1/events" -d "{\"trajectory_id\":\"$TRAJ\",\"event_type\":\"user_message\",\"payload\":{\"text\":\"hello contextdb\"}}"
printf '\n'
curl -s -X POST "$BASE/api/v1/query_view" -d "{\"trajectory_id\":\"$TRAJ\",\"view_name\":\"current_prompt\"}"
printf '\n'
