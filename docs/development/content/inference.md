Use your inference provider's native client to run the agent and collect model calls. After training, publish the saved weights with that provider's deployment interface and use the resulting model ID for the next collection.

## Call your inference endpoint

Start a [local training service and vLLM](https://github.com/wenkesj/trains/tree/main/train#publish-and-serve)
with your chosen model. These examples use its default OpenAI-compatible gateway
at `http://127.0.0.1:8382/v1`. For hosted inference, use your provider's URL and credentials.

```bash
curl --fail-with-body \
  http://127.0.0.1:8382/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "LiquidAI/LFM3-Omni-2B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

The response contains the assistant message in `choices[0].message`. The local training service accepts dummy credentials.

## Use the OpenAI client

Install the client in your application:

```bash
uv add openai
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8382/v1",
    api_key="local",
)
response = client.chat.completions.create(
    model="LiquidAI/LFM3-Omni-2B",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=128,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

The dummy key satisfies the client constructor. Use the endpoint and credentials required by your own inference provider when adapting this example.

## Use an agent SDK

Choose a model interface supported by your agent framework and endpoint. For OpenAI Agents, an OpenAI-compatible Chat Completions model can use the endpoint above. Anthropic-based agents require an endpoint supporting the Anthropic Messages protocol.

Retain the exact messages, tools, and rendering settings in the collected trace. See [Traces and rewards](#/docs/guides/traces) for the data needed to reproduce a call.

## Publish trained weights

Save a sampler checkpoint through the native training SDK. Hosted providers expose their own sampling or deployment interfaces; the [training recipes](#/docs/guides/training) show the supported client patterns.

For the local training service, publish through the native SDK; the service handles adapter loading into vLLM. Preserve optimizer state for resumption. Collect the next dataset with the published model identity and the same rendering configuration.
