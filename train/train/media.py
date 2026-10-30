import base64
import io

import torch
from PIL import Image
from transformers import AutoImageProcessor, PretrainedConfig

from train.models import ModelInput, ModelSpec


def model_inputs(
    model_input: ModelInput,
    spec: ModelSpec,
    device: torch.device,
    config: PretrainedConfig,
    *,
    trust_remote_code: bool = False,
) -> dict[str, torch.Tensor]:
    """Process native image chunks with the registered model's exact HF processor."""
    images, tokens, counts = [], [], []
    image_id = getattr(config, "image_token_id", None)
    for chunk in model_input.chunks:
        if chunk.type == "encoded_text":
            tokens.extend(chunk.tokens)
        elif chunk.type == "image":
            if image_id is None:
                raise ValueError("The registered model has no image input capability")
            raw = base64.b64decode(chunk.data, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                images.append(image.convert("RGB"))
            counts.append(chunk.expected_tokens)
            tokens.extend([image_id] * chunk.expected_tokens)
        else:
            raise ValueError(
                "Local DMel decoding is not supported by this model backend; use a compatible hosted provider"
            )
    result = {"input_ids": torch.tensor([tokens], dtype=torch.long, device=device)}
    if images:
        processor = AutoImageProcessor.from_pretrained(
            spec.base_path, local_files_only=True, trust_remote_code=trust_remote_code
        )
        encoded = processor(images=images, return_tensors="pt")
        grid = encoded.get("image_grid_thw")
        if grid is None:
            raise ValueError("This local backend requires an image processor with image_grid_thw")
        merge = processor.merge_size
        if [int(row.prod()) // (merge * merge) for row in grid] != counts:
            raise ValueError("Native image token counts differ from the registered processor")
        result.update({key: value.to(device) for key, value in encoded.items()})
        result["mm_token_type_ids"] = torch.tensor(
            [[int(token == image_id) for token in tokens]], dtype=torch.int32, device=device
        )
    return result
