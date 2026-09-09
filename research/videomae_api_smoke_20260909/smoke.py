import json
import subprocess
import sys
from pathlib import Path

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "transformers>=4.48.0,<5",
])
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

model_id = "MCG-NJU/videomae-base-finetuned-kinetics"
processor = VideoMAEImageProcessor.from_pretrained(model_id)
model = VideoMAEForVideoClassification.from_pretrained(
    model_id, num_labels=5, ignore_mismatched_sizes=True,
)
pixel_values = torch.zeros(1, 16, 3, 224, 224)
with torch.inference_mode():
    hidden = model.videomae(pixel_values).last_hidden_state
    logits = model(pixel_values=pixel_values).logits
result = {
    "processor": type(processor).__name__,
    "hidden_shape": list(hidden.shape),
    "logits_shape": list(logits.shape),
    "encoder_blocks": len(model.videomae.encoder.layer),
    "has_layernorm": hasattr(model.videomae, "layernorm"),
}
Path("/kaggle/working/videomae_smoke.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
