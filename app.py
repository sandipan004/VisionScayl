import os
import io
import base64
import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import RRDBNet_arch as arch

app = FastAPI(title="VisionScayl - ESRGAN Image Upscaler")

# Mount static files and setup templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'RRDB_ESRGAN_x4.pth'
model = None
model_load_error = None

# Store the latest output image in memory for the download route
latest_output_image_bytes = None


MODEL_URL = "https://huggingface.co/databuzzword/esrgan/resolve/main/RRDB_ESRGAN_x4.pth"


def load_model():
    """Load the ESRGAN model weights, automatically downloading them if missing or LFS pointer."""
    global model, model_load_error
    import urllib.request

    # Check if file is missing or just an LFS pointer (<100KB)
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 100000:
        try:
            print(f"[INFO] Downloading ESRGAN weights (~67 MB) from Hugging Face...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("[INFO] Model weights downloaded successfully.")
        except Exception as e:
            model_load_error = f"Failed to auto-download model weights: {str(e)}"
            print(f"[ERROR] {model_load_error}")
            return None

    try:
        net = arch.RRDBNet(3, 3, 64, 23, gc=32)
        net.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        net.eval()
        net = net.to(device)
        model = net
        model_load_error = None
        print(f"[INFO] ESRGAN model loaded successfully on {device}")
        return model
    except Exception as e:
        model_load_error = f"Failed to load model '{model_path}': {str(e)}"
        print(f"[ERROR] {model_load_error}")
        return None


# Attempt initial model load
load_model()


def tile_process(img_tensor: torch.Tensor, net_model: torch.nn.Module, scale: int = 4, tile_size: int = 384, tile_pad: int = 16, dev: torch.device = device):
    """
    Process image in overlapping tiles to prevent CUDA Out-Of-Memory (OOM) on GPUs with <= 4GB VRAM.
    Seamlessly merges tiles with padding to prevent boundary artifacts.
    """
    b, c, h, w = img_tensor.shape
    if h <= tile_size and w <= tile_size:
        with torch.inference_mode():
            return net_model(img_tensor.to(dev)).cpu()

    output_h, output_w = h * scale, w * scale
    output = torch.zeros((b, c, output_h, output_w), dtype=torch.float32)

    tiles_x = int(np.ceil(w / tile_size))
    tiles_y = int(np.ceil(h / tile_size))

    for y in range(tiles_y):
        for x in range(tiles_x):
            input_start_x = x * tile_size
            input_end_x = min(input_start_x + tile_size, w)
            input_start_y = y * tile_size
            input_end_y = min(input_start_y + tile_size, h)

            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, w)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, h)

            input_tile = img_tensor[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            with torch.inference_mode():
                output_tile = net_model(input_tile.to(dev)).cpu()

            if dev.type == 'cuda':
                torch.cuda.empty_cache()

            crop_start_x = (input_start_x - input_start_x_pad) * scale
            crop_end_x = crop_start_x + (input_end_x - input_start_x) * scale
            crop_start_y = (input_start_y - input_start_y_pad) * scale
            crop_end_y = crop_start_y + (input_end_y - input_start_y) * scale

            dest_start_x = input_start_x * scale
            dest_end_x = input_end_x * scale
            dest_start_y = input_start_y * scale
            dest_end_y = input_end_y * scale

            output[:, :, dest_start_y:dest_end_y, dest_start_x:dest_end_x] = output_tile[:, :, crop_start_y:crop_end_y, crop_start_x:crop_end_x]

    return output


def process_image(image_bytes: bytes):
    """Process and upscale image using ESRGAN with memory-safe tiled execution."""
    global model, device
    if model is None:
        if not load_model():
            raise RuntimeError(
                model_load_error or "Model is not loaded. Please ensure RRDB_ESRGAN_x4.pth weights are downloaded."
            )

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid JPG or PNG file.")

    # Convert BGR -> RGB normalized float tensor
    img = img * 1.0 / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_lr = img.unsqueeze(0)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    try:
        # Attempt GPU/configured device inference using tiled processing
        with torch.inference_mode():
            output_tensor = tile_process(img_lr, model, scale=4, tile_size=384, tile_pad=16, dev=device)
    except (torch.cuda.OutOfMemoryError if hasattr(torch.cuda, 'OutOfMemoryError') else RuntimeError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower() or (hasattr(torch.cuda, 'OutOfMemoryError') and isinstance(e, torch.cuda.OutOfMemoryError)):
            print("[WARNING] CUDA Out of Memory on GPU. Clearing cache and falling back to CPU...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Fallback to CPU execution
            cpu_dev = torch.device('cpu')
            model.to(cpu_dev)
            with torch.inference_mode():
                output_tensor = tile_process(img_lr, model, scale=4, tile_size=256, tile_pad=16, dev=cpu_dev)
            # Re-mount model to GPU for future requests
            if device.type == 'cuda':
                model.to(device)
        else:
            raise e

    output = output_tensor.data.squeeze().float().clamp(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    success, buffer = cv2.imencode('.jpg', output)
    if not success:
        raise RuntimeError("Failed to encode upscaled image.")

    raw_bytes = buffer.tobytes()
    base64_str = base64.b64encode(raw_bytes).decode('utf-8')
    return base64_str, raw_bytes


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render main homepage."""
    warning = model_load_error if model is None else None
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"img_data": None, "error_message": warning}
    )


@app.post("/", response_class=HTMLResponse)
async def upload_and_upscale(request: Request, file: UploadFile = File(...)):
    """Handle image upload, run ESRGAN upscaling, and render result."""
    global latest_output_image_bytes
    img_data = None
    error_message = None

    if not file or not file.filename:
        error_message = "Please select an image file to upload."
    else:
        try:
            content = await file.read()
            if not content:
                raise ValueError("Uploaded file is empty.")
            img_data, raw_bytes = process_image(content)
            latest_output_image_bytes = raw_bytes
        except Exception as e:
            error_message = str(e)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"img_data": img_data, "error_message": error_message}
    )


@app.get("/download")
async def download_image():
    """Download the latest upscaled image."""
    global latest_output_image_bytes
    if latest_output_image_bytes:
        return Response(
            content=latest_output_image_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=output_image.jpg"}
        )
    return Response(content="Output image is not available for download.", media_type="text/plain", status_code=404)


@app.post("/api/upscale")
async def api_upscale(file: UploadFile = File(...)):
    """REST API endpoint to directly upscale and return image binary."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        _, raw_bytes = process_image(content)
        return Response(content=raw_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run("app:app", host=host, port=port, reload=True)
