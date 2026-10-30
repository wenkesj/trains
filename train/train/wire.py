import base64

import numpy as np
from tinker.proto import tinker_public_pb2 as pb

from train.models import Batch


def decode_forward(data: bytes) -> dict:
    """Normalize provider-native tensors and chunks into a training batch."""
    message = pb.ForwardBackwardRequest.FromString(data)
    datums = []
    dtypes = {pb.DTYPE_FLOAT32: np.float32, pb.DTYPE_INT64: np.int64}
    for datum in message.data:
        chunks = []
        for chunk in datum.model_input:
            kind = chunk.WhichOneof("chunk")
            if kind == "encoded_text":
                chunks.append(
                    {
                        "type": "encoded_text",
                        "tokens": np.frombuffer(chunk.encoded_text.tokens, dtype=np.int32).tolist(),
                    }
                )
            elif kind == "image":
                chunks.append(
                    {
                        "type": "image",
                        "data": base64.b64encode(chunk.image.data).decode(),
                        "format": chunk.image.format,
                        "expected_tokens": chunk.image.expected_tokens,
                    }
                )
            elif kind == "dmel":
                chunks.append({"type": "dmel", "dmel": base64.b64encode(chunk.dmel.dmel).decode()})
            else:
                raise ValueError("Unsupported native input chunk")
        inputs = {}
        for name, tensor in datum.loss_fn_inputs.items():
            if tensor.dtype not in dtypes or tensor.WhichOneof("encoding") != "dense":
                raise ValueError("Only dense float32/int64 tensors are supported")
            dtype = dtypes[tensor.dtype]
            inputs[name] = {
                "data": np.frombuffer(tensor.dense, dtype=dtype).tolist(),
                "dtype": "float32" if tensor.dtype == pb.DTYPE_FLOAT32 else "int64",
                "shape": list(tensor.shape),
            }
        datums.append({"model_input": {"chunks": chunks}, "loss_fn_inputs": inputs})
    config: dict[str, float | str] = dict(message.loss_fn_config)
    for name, value in message.loss_fn_config_v2.items():
        config[name] = value.text if value.WhichOneof("value") == "text" else value.number
    batch = Batch.model_validate(
        {"data": datums, "loss_fn": message.loss_fn, "loss_fn_config": config}
    )
    return {
        "model_id": message.model_id,
        "seq_id": message.seq_id,
        "forward_backward_input": batch.model_dump(),
        "forward_only": message.forward_only,
    }


def encode_forward(result: dict) -> bytes:
    """Encode native SDK tensors with per-datum offsets measured in bytes."""
    message = pb.ForwardBackwardOutput(
        loss_fn_output_type=result["loss_fn_output_type"], metrics=result["metrics"]
    )
    for output in result["loss_fn_outputs"]:
        record = message.loss_fn_outputs.add(num_datums=1)
        for key, tensor in output.items():
            target = record.fields[key]
            target.dtype = pb.DTYPE_FLOAT32
            target.trailing_shape.extend(tensor.get("shape", [len(tensor["data"])])[1:])
            target.offsets = np.asarray([0, len(tensor["data"]) * 4], dtype=np.int64).tobytes()
            target.data = np.asarray(tensor["data"], dtype=np.float32).tobytes()
    return message.SerializeToString()


def encode_sample(result: dict) -> bytes:
    """Encode local token samples into a native SDK response."""
    message = pb.SampleResponse()
    for sequence in result["sequences"]:
        target = message.sequences.add()
        target.tokens = np.asarray(sequence["tokens"], dtype=np.int32).tobytes()
        target.logprobs = np.asarray(sequence["logprobs"], dtype=np.float32).tobytes()
        target.stop_reason = (
            pb.STOP_REASON_STOP if sequence["stop_reason"] == "stop" else pb.STOP_REASON_LENGTH
        )
        if sequence["topk_logprobs"]:
            pairs = np.asarray(sequence["topk_logprobs"])
            topk = target.topk_sampled_logprobs
            topk.length, topk.k = pairs.shape[:2]
            topk.token_ids = pairs[:, :, 0].astype(np.int32).tobytes()
            topk.logprobs = pairs[:, :, 1].astype(np.float32).tobytes()
    if result.get("prompt_logprobs"):
        message.prompt_logprobs = np.asarray(
            [float("nan") if value is None else value for value in result["prompt_logprobs"]],
            dtype=np.float32,
        ).tobytes()
    if result.get("target_prompt_logprobs"):
        value = result["target_prompt_logprobs"]
        message.target_prompt_logprobs.dtype = pb.DTYPE_FLOAT32
        message.target_prompt_logprobs.shape.extend(value["shape"])
        message.target_prompt_logprobs.dense = np.asarray(value["data"], dtype=np.float32).tobytes()
    return message.SerializeToString()
