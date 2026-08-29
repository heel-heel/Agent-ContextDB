# Clean Live Path Skill Demo

This fixture is for the clean live ContextDB demonstration. It intentionally
uses a hyphenated directory name so copied Agent prompts do not transform
underscores into path separators.

The intentional failing lookup is `./assets/settings.json` from the project
root. The successful repair reads
`./examples/live-path-demo/assets/settings.json`.
