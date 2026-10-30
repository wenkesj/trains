# Container entrypoint: configure native harness plugins in its disposable profile.

import json
import os
import subprocess
import sys
from pathlib import Path

base_url = os.environ["TRAINS_MODEL_URL"]
subprocess.run([sys.executable, "-m", "tracing.codex.install", "install"], check=True)
from core.setup import merge_harness_entry

merge_harness_entry("codex", os.environ["ARIZE_PROJECT_NAME"])
config = Path("/root/.codex/config.toml")
with config.open("a") as stream:
    stream.write('\nmodel = "policy"\nmodel_provider = "local"\nmodel_reasoning_effort = "none"\n')
    stream.write('model_catalog_json = "/opt/codex-models.json"\nweb_search = "disabled"\n')
    stream.write("[features]\ngoals = false\nmulti_agent = false\nmulti_agent_v2 = false\n")
    stream.write('[model_providers.local]\nname = "Local Tinker policy"\n')
    stream.write("base_url = " + json.dumps(base_url + "/v1") + "\n")
    stream.write('env_key = "TINKER_API_KEY"\nwire_api = "responses"\n')
command = [
    "codex",
    "exec",
    "--skip-git-repo-check",
    "--dangerously-bypass-approvals-and-sandbox",
    "--json",
    "Read INSTRUCTIONS.md and the existing Python starter files in /workspace. Implement those solution files in place, preserving their filenames and required interfaces. Check one representative example, then finish with a concise summary. Hidden upstream tests run separately. Complete the work without asking for input.",
]
raise SystemExit(subprocess.run(command, check=False).returncode)
