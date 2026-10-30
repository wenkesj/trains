# Container entrypoint: configure native harness plugins in its disposable profile.

import json
import os
import subprocess
from pathlib import Path

base_url = os.environ["TRAINS_MODEL_URL"]
config = {
    "$schema": "https://opencode.ai/config.json",
    "plugin": ["@langchain/langsmith-opencode@0.1.0"],
    "model": "local/policy",
    "provider": {
        "local": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Local Tinker policy",
            "options": {"baseURL": base_url + "/v1", "apiKey": os.environ["TINKER_API_KEY"]},
            "models": {
                "policy": {
                    "name": "Published Tinker policy",
                    "limit": {"context": 32768, "output": 2048},
                }
            },
        }
    },
    "permission": "allow",
    "agent": {"build": {"steps": 20, "temperature": 1}},
}
# Outside /workspace so the candidate contains only benchmark solution files.
path = Path("/tmp/opencode.json")
path.write_text(json.dumps(config))
os.environ["OPENCODE_CONFIG"] = str(path)
command = [
    "opencode",
    "run",
    "--format",
    "json",
    "Read INSTRUCTIONS.md and the existing Python starter files in /workspace. Implement those solution files in place, preserving their filenames and required interfaces. Check one representative example, then finish with a concise summary. Hidden upstream tests run separately. Complete the work without asking for input.",
]
raise SystemExit(subprocess.run(command, check=False).returncode)
