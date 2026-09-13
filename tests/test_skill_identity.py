from tempfile import TemporaryDirectory

from contextdb import uris
from contextdb.service import ContextDB


def test_learned_skills_use_stable_ids_and_migrate_live_trace_references():
    with TemporaryDirectory() as root:
        db = ContextDB(root)
        try:
            trajectory_id = db.create_trajectory("Stable skill identity")['trajectory_id']
            trigger = {
                'failed_tool': 'shell',
                'failed_command_pattern': 'powershell -NoProfile -Command "Get-Content -Raw settings.json"',
                'likely_cause': 'missing file at specified path',
                'normalized_signature': 'exit code get-content bad-encoding-tokens',
            }
            legacy_skill_id = 'skill_file_not_found_specified_path_exit_code_get-content'
            db.store.put_object(uris.view_key(trajectory_id, 'main', 'learned_skills'), {
                'view_name': 'learned_skills',
                'trajectory_id': trajectory_id,
                'branch_id': 'main',
                'created_at': '2026-01-01T00:00:00+00:00',
                'content': [{'skill_id': legacy_skill_id, 'trigger': trigger}],
            })
            match = db.append_event(trajectory_id, 'skill_match', {
                'matches': [{'matched_skill_id': legacy_skill_id, 'matched_trigger': trigger}],
            }, actor='contextdb')
            application = db.append_event(
                trajectory_id,
                'tool_result',
                {'status': 'ok', 'preview': 'settings loaded'},
                refs={'skill_match_event_id': match['event_id'], 'skill_id': legacy_skill_id},
                metadata={'operation': 'skill_application'},
            )

            db.semantic_repair_judgments = lambda _trajectory_id: [{
                'failure_event_id': 'evt_failure',
                'success_event_id': 'evt_success',
                'failure': {
                    'failure_event_id': 'evt_failure',
                    'tool': 'shell',
                    'command': trigger['failed_command_pattern'],
                    'likely_cause': 'file not found at expected path',
                    'normalized_signature': 'pathnotfound bad-encoding-tokens',
                },
                'repair_candidate': {
                    'success_event_id': 'evt_success',
                    'branch_id': 'main',
                    'tool': 'shell',
                    'command': 'powershell -NoProfile -Command "Get-Content -Raw fixed-settings.json"',
                    'outcome': 'ok',
                },
                'judgment': {
                    'enabled': True,
                    'label': 'likely_repair',
                    'recommended_for_skill': True,
                    'confidence': 0.9,
                },
                'highlight_event_ids': ['evt_failure', 'evt_success'],
            }]

            skills = db.learned_skills(trajectory_id)
            assert [skill['skill_id'] for skill in skills] == ['skill_shell_file_not_found_get_content']
            assert all('锟' not in str(value) for value in skills[0]['trigger'].values())

            migrated_match = db.store.get_object(uris.event_key(trajectory_id, match['event_id']))
            migrated_application = db.store.get_object(uris.event_key(trajectory_id, application['event_id']))
            assert migrated_match['payload']['matches'][0]['matched_skill_id'] == 'skill_shell_file_not_found_get_content'
            assert migrated_application['refs']['skill_id'] == 'skill_shell_file_not_found_get_content'
        finally:
            db.vector_index.conn.close()
            db.store.conn.close()
