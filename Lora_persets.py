from diffusers import StableDiffusionXLControlNetPipeline
import torch
import os

MODELS_DIR = "/home/ubuntu/Artistic-QR-Code-generation/models"
OUTPUTS_DIR = "Outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Define your LoRA presets
LORA_PRESETS = {
    "general":   None,
    "cinematic": ("cinematic_style.safetensors",  0.85),
    "cars":      ("hypercar_style.safetensors",   0.80),
    "food":      ("food_photography.safetensors", 0.75),
    "nature":    ("nature_landscape.safetensors", 0.80),
    "anime":     ("anime_style.safetensors",      0.80),
}

QR_LORA = ("qr_code_monster.safetensors", 0.55)  # Base QR LoRA always loaded

def load_lora_preset(pipeline, preset_name: str):
    """Stack QR base LoRA + style LoRA."""
    adapters, weights = [], []

    # Always load QR monster as structural base
    qr_file, qr_scale = QR_LORA
    qr_path = os.path.join(MODELS_DIR, "loras", qr_file)
    if os.path.exists(qr_path):
        pipeline.load_lora_weights(
            os.path.join(MODELS_DIR, "Lora_weights"),
            weight_name=qr_file,
            adapter_name="qr"
        )
        adapters.append("qr")
        weights.append(qr_scale)
    else:
        print(f"⚠️  QR LoRA not found at {qr_path}, skipping.")

    # Load style LoRA if preset has one
    preset = LORA_PRESETS.get(preset_name)
    if preset:
        style_file, style_scale = preset
        style_path = os.path.join(MODELS_DIR, "Lora_weights", style_file)
        if os.path.exists(style_path):
            pipeline.load_lora_weights(
                os.path.join(MODELS_DIR, "Lora_weights"),
                weight_name=style_file,
                adapter_name="style"
            )
            adapters.append("style")
            weights.append(style_scale)
        else:
            print(f"⚠️  Style LoRA '{style_file}' not found, skipping.")

    if adapters:
        pipeline.set_adapters(adapters, adapter_weights=weights)

    return pipeline


def unload_loras(pipeline):
    """Free LoRA weights from memory after generation."""
    try:
        pipeline.unload_lora_weights()
    except Exception:
        pass