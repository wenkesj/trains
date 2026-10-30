import tinker
from tinker import types


def test_native_target_choice_logprobs(training_server):
    service = tinker.ServiceClient(base_url=training_server, api_key="tml-local")
    sampler = service.create_sampling_client(base_model="tiny")
    result = sampler.sample(
        types.ModelInput.from_ints([1, 4, 5]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1),
        target_prompt_logprobs=types.TensorData(data=[-1, -1, 6, 7], dtype="int64", shape=[2, 2]),
    ).result(timeout=20)
    assert result.target_prompt_logprobs.shape == [2, 2]
    assert all(value < 0 for value in result.target_prompt_logprobs.data[-2:])
    service.close("success").result()
