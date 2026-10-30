import sys

import httpx
from baseten.client.managementapi import ApiClient, DeployLoopsCheckpointRequest

with httpx.Client(base_url=sys.argv[1], timeout=30) as http:
    deployment = ApiClient(http).post_loops_checkpoints_deploy(
        request=DeployLoopsCheckpointRequest(
            checkpoint_ids=[sys.argv[2]],
            model_name="sdk-policy",
            instance_type_id="local",
            hf_secret_name="unused-local",
        )
    )
    status = ApiClient(http).get_models_deployments_deployment_id(
        model_id=deployment.model_id,
        deployment_id=deployment.deployment_id,
    )
    assert status.status == "ACTIVE"
    response = http.post(
        f"/deployment/{deployment.deployment_id}/sync/v1/chat/completions",
        json={"model": "standalone", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200, response.text
    assert http.delete(f"/v1/models/{deployment.model_id}").status_code == 200
