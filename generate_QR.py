import os
import qrcode
import torch

# ─────────────────────────────────────────────
# Prompt Engineering
# ─────────────────────────────────────────────
def nconditioning_gen(compel, style: str = "qr_art"):
    """Generate negative conditioning tensor."""

    base_negative = """
    (worst quality)++, (low quality)++, (normal quality)+,
    (blurry)++, (out of focus)++, jpeg artifacts, compression artifacts,
    (watermark)++, (text)++, (logo)++, (signature)++,
    (bad anatomy)++, (bad proportions)++, (extra limbs)+, (missing limbs)+,
    (deformed)++, (disfigured)++, (mutation)+, (mutated)+,
    (duplicate)++, overexposed, underexposed, oversaturated,
    monochrome, grayscale, sketch, unfinished, ugly, amateur,
    (long fingers)+, (extra fingers)+, (bad hands)++, (fused fingers)+,
    horror, disturbing, nsfw, nude,
    """

    style_negatives = {
        "qr_art": """
        (cluttered background)++, (busy background)++,
        (noisy pattern)++, (conflicting patterns)++,
        dark corners, heavy vignette, heavy shadows,
        low contrast, washed out, flat lighting,
        """,
        "portrait": """
        (asymmetric eyes)+, (crossed eyes)+,
        (bad face)++, (ugly face)++, (deformed face)++,
        double chin, acne, bad teeth,
        """,
        "landscape": """
        people, human, person, crowd,
        urban, city, buildings,
        artificial, plastic, fake,
        """,
        "general": "",
    }

    embeddings_block = """
    easynegative, ng_deepnegative_v1_75t,
    bad_prompt_version2, bad-artist, bad-artist-anime,
    """

    full_negative = base_negative + style_negatives.get(style, "") + embeddings_block
    nconditioning = compel.build_conditioning_tensor(full_negative)
    nconditioning = nconditioning.half()
    return nconditioning


def conditioning_gen(compel, p: str, style_preset: str = "cinematic"):
    """Generate positive conditioning tensor."""

    base_quality = """
    (masterpiece)++, (best quality)++, (ultra detailed)++,
    (high resolution)++, sharp focus,
    """

    style_presets = {
        "cinematic": """
        cinematic lighting, dramatic atmosphere, professional photography,
        (volumetric lighting)+, depth of field, film grain,
        color grading, movie still, epic composition,
        """,
        "digital_art": """
        digital art, concept art, (artstation quality)++,
        trending on artstation, illustration,
        vivid colors, dynamic composition, detailed brushwork,
        """,
        "photorealistic": """
        photorealistic, hyperrealistic, RAW photo,
        DSLR, 85mm lens, natural lighting,
        true to life colors, skin texture,
        """,
        "anime": """
        (anime style)++, manga, cel shading,
        vibrant colors, clean lineart,
        studio ghibli quality, expressive eyes,
        """,
        "nature": """
        (lush nature)++, golden hour lighting,
        macro photography, botanical detail,
        soft bokeh background, morning dew, vibrant greens,
        """,
        "food": """
        (food photography)++, studio lighting,
        appetizing, fresh ingredients, shallow depth of field,
        warm tones, professional food styling, macro detail,
        """,
        "cars": """
        (automotive photography)++, dramatic lighting,
        sleek reflections, chrome details,
        race track or studio setting, speed lines,
        """,
    }

    # QR-specific enhancers always appended
    qr_enhancers = """
    seamless integration, visually coherent,
    balanced contrast, clear focal point,
    """

    full_prompt = (
        base_quality
        + style_presets.get(style_preset, style_presets["cinematic"])
        + f"({p})++"
        + qr_enhancers
    )

    conditioning = compel.build_conditioning_tensor(full_prompt)
    conditioning = conditioning.half()
    return conditioning



def generateBasicQR(data:str):

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=8
    )

    qr.add_data(data)
    qr.make(fit=True)


    img_qr = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img_path="basic_qr.png"
    img_qr.save(img_path)

    return img_path

def generateQRArt(pipe,compel,qr_code_image,prompt):

    conditioning = conditioning_gen(compel,prompt)
    nconditioning=nconditioning_gen(compel)
    device = pipe.device
    dtype = pipe.unet.dtype
    conditioning = conditioning.to(device=device, dtype=dtype)
    nconditioning = nconditioning.to(device=device, dtype=dtype)
    if hasattr(qr_code_image, "to"):
        qr_code_image = qr_code_image.to(device=device, dtype=dtype)
    image = pipe(prompt_embeds=conditioning, negative_prompt_embeds =nconditioning,
                 image=qr_code_image,
                 num_inference_steps=25,
                 controlnet_conditioning_scale=1.0, # Increased this value
                 control_guidance_start=0,
                 control_guidance_end=0.8,
                 cross_attention_kwargs={"scale": 1},
                 width=1024,height=1024).images[0]
    return image
