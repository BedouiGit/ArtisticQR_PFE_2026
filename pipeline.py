import os
import nest_asyncio
import torch   
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline, DPMSolverMultistepScheduler
from compel import Compel
from PIL import Image
from IPython.display import clear_output

MODELS_DIR = "/home/ubuntu/Artistic-QR-Code-generation/models"
OUTPUTS_DIR = "Outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def init_pipeline():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load ControlNet directly in fp16 
    controlnet = ControlNetModel.from_pretrained(
        "monster-labs/control_v1p_sd15_qrcode_monster",
        torch_dtype=torch.float16
    )

    # Load pipeline (SDXL base)
    #pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.bfloat16, device_map="cuda")



# The DreamShaper checkpoint (SD 1.5 based)
    pipeline =StableDiffusionControlNetPipeline.from_pretrained(
    "Lykon/dreamshaper-8", 
    controlnet=controlnet,
    torch_dtype=torch.float16
)

    pipeline.to("cuda")

    # Load textual inversions
    try:
        pipeline.load_textual_inversion(
            os.path.join(MODELS_DIR, "prompt_optimizers/easynegative.safetensors")
    )
        pipeline.load_textual_inversion(
            os.path.join(MODELS_DIR, "prompt_optimizers/ng_deepnegative_v1_75t.pt")
    )
        print("✅ Textual inversions loaded.")
    except Exception as e:
        print(f"⚠️  Textual inversions skipped: {e}")

    # Load LoRA
    pipeline.load_lora_weights("OedoSoldier/detail-tweaker-lora")

    clear_output()

    # Scheduler
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True
    )


    compel = Compel(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        device=pipeline.device   # <-- ensures CUDA
    )

    pipeline.enable_attention_slicing()

    # CPU offload (AFTER everything is on CUDA)
    pipeline.enable_model_cpu_offload()

    return pipeline, compel
    
pipeline,compel = init_pipeline()
