"""
FLUX.2-klein-4B RunPod Serverless Handler (SFW Only)

Loads FLUX.2-klein-4B model (~13GB VRAM, 4-step generation)
at container start, then handles inference requests returning base64-encoded images.
"""

import base64
import io
import os
import sys
import traceback

print("=" * 60)
print("FLUX.2-klein-4B Handler Starting...")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Import torch first and check CUDA
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
except Exception as e:
    print(f"ERROR importing torch: {e}")
    traceback.print_exc()
    sys.exit(1)

# Import runpod
try:
    import runpod
    print("RunPod SDK loaded")
except Exception as e:
    print(f"ERROR importing runpod: {e}")
    traceback.print_exc()
    sys.exit(1)

# Import diffusers
try:
    from diffusers import Flux2KleinPipeline
    print("Flux2KleinPipeline imported successfully")
except Exception as e:
    print(f"ERROR importing diffusers: {e}")
    traceback.print_exc()
    sys.exit(1)

# Model paths (set during Docker build)
MODEL_PATH = os.environ.get("MODEL_PATH", "black-forest-labs/FLUX.2-klein-4B")
HF_CACHE = os.environ.get("HF_HOME", "/models/hf_cache")

print(f"Model: {MODEL_PATH}")
print(f"HF cache: {HF_CACHE}")

# Default generation settings
DEFAULTS = {
    "width": 512,
    "height": 512,
    "steps": 4,  # FLUX.2-klein default (4 steps for distilled models)
    "quality": 85,  # WebP quality (1-100)
}

# Configuration limits
LIMITS = {
    "min_dimension": 64,
    "max_dimension": 2048,
    "min_steps": 1,
    "max_steps": 50,
    "min_quality": 1,
    "max_quality": 100,
    "max_pixels": 1024 * 1024,  # 1MP max to prevent OOM
}


def load_pipeline():
    """Load the FLUX.2-klein-4B pipeline."""
    print("Loading FLUX.2-klein-4B pipeline...")

    # Clear any existing CUDA memory
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load the FLUX.2-klein-4B pipeline
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
    )

    # Move to GPU
    print("Moving pipeline to CUDA...")
    pipe.to("cuda")

    # Clear any cached memory
    torch.cuda.empty_cache()

    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print("Pipeline loaded successfully!")
    return pipe


# Load pipeline at container start (stays warm)
try:
    pipe = load_pipeline()
    print("=" * 60)
    print("Handler ready to accept requests!")
    print("=" * 60)
except Exception as e:
    print(f"FATAL ERROR loading pipeline: {e}")
    traceback.print_exc()
    sys.exit(1)


def handler(job):
    """
    Handle image generation request.

    Input:
        prompt (str): Image description
        width (int, optional): Image width (default: 512)
        height (int, optional): Image height (default: 512)
        steps (int, optional): Inference steps (default: 4)
        seed (int, optional): Random seed (default: random)
        quality (int, optional): WebP quality 1-100 (default: 85)

    Output:
        image (str): Base64-encoded WebP image
        format (str): "webp"
        width (int): Image width
        height (int): Image height
        seed (int|null): Random seed used
        gpu (str): GPU name used

    Note: FLUX.2-klein-4B uses guidance_scale=1.0
    """
    job_input = job["input"]

    # Extract and validate prompt
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing required parameter: prompt"}

    if not isinstance(prompt, str):
        return {"error": "prompt must be a string"}

    if len(prompt) > 10000:
        return {"error": "prompt too long (max 10000 characters)"}

    # Validate and sanitize numeric parameters
    try:
        width = int(job_input.get("width", DEFAULTS["width"]))
        height = int(job_input.get("height", DEFAULTS["height"]))
        steps = int(job_input.get("steps", DEFAULTS["steps"]))
        quality = int(job_input.get("quality", DEFAULTS["quality"]))

        # Range validation
        width = max(LIMITS["min_dimension"], min(width, LIMITS["max_dimension"]))
        height = max(LIMITS["min_dimension"], min(height, LIMITS["max_dimension"]))
        steps = max(LIMITS["min_steps"], min(steps, LIMITS["max_steps"]))
        quality = max(LIMITS["min_quality"], min(quality, LIMITS["max_quality"]))

        # Validate seed if provided
        seed = job_input.get("seed")
        if seed is not None:
            seed = int(seed)
            if seed < 0 or seed > 2147483647:
                return {"error": "seed must be between 0 and 2147483647"}

    except (ValueError, TypeError) as e:
        return {"error": f"Invalid parameter type: {str(e)}"}

    # Cap size to prevent OOM
    if width * height > LIMITS["max_pixels"]:
        scale = (LIMITS["max_pixels"] / (width * height)) ** 0.5
        width = int(width * scale)
        height = int(height * scale)
        print(f"Resized to {width}x{height} to prevent OOM")

    print(f"Generating image: {width}x{height}, steps={steps}, seed={seed}, quality={quality}")
    print(f"CUDA memory before generation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Prompt: {prompt[:100] if len(prompt) > 100 else prompt}")

    # Create generator with seed
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(int(seed))
    else:
        generator = None

    try:
        # Generate image with inference mode for memory efficiency
        # Add timeout to prevent stuck workers
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Image generation timed out after 30 seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout

        with torch.inference_mode():
            # FLUX.2-klein with CFG=1 (guidance_scale=1.0 for distilled models)
            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=1.0,
                generator=generator,
            )

        image = result.images[0]

        # Convert to base64 WebP (optimized for web)
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=quality, method=6)
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Cancel timeout
        signal.alarm(0)

        # Clean up to prevent memory buildup
        del result
        torch.cuda.empty_cache()

        print(f"Image generated successfully: {len(image_b64)} bytes")
        print(f"CUDA memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        return {
            "image": image_b64,
            "format": "webp",
            "width": width,
            "height": height,
            "seed": seed,
            "gpu": torch.cuda.get_device_name(0),
        }

    except TimeoutError as e:
        signal.alarm(0)
        torch.cuda.empty_cache()
        print(f"TIMEOUT ERROR: {e}")
        return {"error": "Image generation timed out after 30 seconds"}

    except torch.cuda.OutOfMemoryError as e:
        signal.alarm(0)
        torch.cuda.empty_cache()
        print(f"OOM ERROR: {e}")
        return {"error": f"Out of memory - try smaller image size (current: {width}x{height})"}

    except Exception as e:
        signal.alarm(0)
        torch.cuda.empty_cache()
        print(f"ERROR during generation: {e}")
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
