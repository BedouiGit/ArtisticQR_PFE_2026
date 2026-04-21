import cv2
from qreader import QReader
from PIL import Image
import torch
from compel import Compel
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pyngrok import ngrok
import nest_asyncio
import uvicorn
import os
import time
import qrcode
from dotenv import load_dotenv
from generate_QR import generateBasicQR, generateQRArt, conditioning_gen, nconditioning_gen
from pipeline import init_pipeline
from Lora_persets import load_lora_preset, unload_loras, LORA_PRESETS
from scannability_check import check_scannable

# Setup

nest_asyncio.apply()
load_dotenv()

MODELS_DIR = "/home/ubuntu/Artistic-QR-Code-generation/models"
OUTPUTS_DIR = "Outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)


print("🚀 Initializing pipeline...")

pipeline,compel = init_pipeline()

# QReader for scanability check
qreader = QReader()

print("✅ Pipeline initialized!")

# ─────────────────────────────────────────────
# Core Generation Function
# ─────────────────────────────────────────────
def run_generation(pipeline, compel, qr_code_image, prompt: str, style: str):
    """Full generation: load LoRA → build prompts → run pipe → unload."""

    # Load LoRAs for this style
    pipeline = load_lora_preset(pipeline, style)

    # Build conditionings
    pos = conditioning_gen(compel, p=prompt, style_preset=style)
    neg = nconditioning_gen(compel, style="qr_art")

    # Run diffusion
    image = pipeline(
        prompt_embeds=pos,
        negative_prompt_embeds=neg,
        image=qr_code_image,
        num_inference_steps=25,
        controlnet_conditioning_scale=0.8,
        control_guidance_start=0.0,
        control_guidance_end=0.8,
        cross_attention_kwargs={"scale": 1},
        width=512,
        height=512,
    ).images[0]

    # Unload LoRAs to free VRAM for next call
    unload_loras(pipeline)

    return image


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="🎨 Artistic QR Generator",
    description="Generate scannable artistic QR codes using Stable Diffusion + ControlNet.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.mount("/Outputs", StaticFiles(directory=OUTPUTS_DIR), name="Outputs")


def load_image(image_path: str) -> Image.Image:
    """Load image from file path."""
    return Image.open(image_path).convert("RGB")


@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse(url="/docs")


@app.post("/generate_qr_art")
def generate_qr_art(
    request: Request,
    prompt: str,
    urlOrText: str,
    style: str = "cinematic",
):
    """
    Generate an artistic QR code.

    - **prompt**: Describe the visual style (e.g. 'red Ferrari on mountain road')
    - **urlOrText**: The URL or text to encode in the QR code
    - **style**: One of: `general`, `cinematic`, `cars`, `food`, `nature`, `anime`, `digital_art`
    """

    # Validate style
    valid_styles = list(LORA_PRESETS.keys()) + ["digital_art", "photorealistic", "anime"]
    if style not in valid_styles:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid style '{style}'. Choose from: {valid_styles}"}
        )

    try:
        # 1. Generate base QR code
        qr_path = generateBasicQR(urlOrText)
        qr_code_image = load_image(qr_path)

        # 2. Run artistic generation
        image = run_generation(pipeline, compel, qr_code_image, prompt, style)

        if image is None:
            raise ValueError("Generation returned None")

        # 3. Save output with unique timestamp
        timestamp = int(time.time())
        image_name = f"qr_{style}_{timestamp}.png"
        image_path = os.path.join(OUTPUTS_DIR, image_name)

        image_resized = image.resize((512, 512))
        image_resized.save(image_path)

        # 4. Scanability check
        scannable = check_scannable(image_path)

        # 5. Build response
        base_url = str(request.base_url)
        image_url = f"{base_url}Outputs/{image_name}"

        return {
            "prompt": prompt,
            "style": style,
            "url_encoded": urlOrText,
            "image_url": image_url,
            "scannable": scannable is not None,
            "decoded_text": scannable,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/styles")
def list_styles():
    """List all available style presets."""
    return {
        "styles": list(LORA_PRESETS.keys()),
        "description": {
            "general":    "No style LoRA, clean output",
            "cinematic":  "Dramatic lighting, film-grade color",
            "cars":       "Automotive photography, chrome and speed",
            "food":       "Studio food photography, warm tones",
            "nature":     "Golden hour, botanical, lush greens",
            "anime":      "Cel shading, vibrant, clean lineart",
        }
    }


@app.get("/health")
def health():
    """Check GPU and pipeline status."""
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    }


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Kill old ngrok tunnels
    ngrok.kill()

    # Start ngrok tunnel
    public_url = ngrok.connect(8000)
    print(f"🌐 Public URL: {public_url}")
    print(f"📖 Swagger UI: {public_url}/docs")
    print(f"🎨 Styles:     {public_url}/styles")
    print(f"💊 Health:     {public_url}/health")

    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")