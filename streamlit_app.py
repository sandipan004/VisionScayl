"""
VisionScayl — AI-Powered 4x Image Super-Resolution
Streamlit Web Application
Powered by ESRGAN (RRDBNet) & PyTorch
"""

import os
import io
import time
import zipfile
import base64
from pathlib import Path
from typing import Tuple, Optional, Callable

import cv2
import numpy as np
from PIL import Image
import torch
import streamlit as st

import RRDBNet_arch as arch

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "RRDB_ESRGAN_x4.pth")
MODEL_URL = "https://huggingface.co/databuzzword/esrgan/resolve/main/RRDB_ESRGAN_x4.pth"
SAMPLE_IMG_PATH = BASE_DIR / "readme-images" / "VisionScayl-sample.jpg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionScayl — AI Super-Resolution",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling (Dark Glassmorphic Theme with Compact Layout)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Compact hero box */
    .hero-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    
    /* Sleek gradient badges */
    .badge-card {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-cuda {
        background: rgba(118, 185, 0, 0.15);
        border-color: rgba(118, 185, 0, 0.5);
        color: #76B900;
        font-weight: 600;
    }
    .badge-cpu {
        background: rgba(59, 130, 246, 0.15);
        border-color: rgba(59, 130, 246, 0.5);
        color: #60a5fa;
    }
    .badge-model {
        background: rgba(168, 85, 247, 0.15);
        border-color: rgba(168, 85, 247, 0.5);
        color: #c084fc;
    }
    
    /* Metric container cards */
    .metric-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.75rem;
        text-align: center;
        backdrop-filter: blur(8px);
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.1rem;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
    }

    /* Result Action Bar */
    .result-banner {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(56, 189, 248, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Preview card container */
    .preview-card {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(0, 0, 0, 0.2);
        padding: 0.75rem;
        margin-bottom: 1rem;
    }

    /* Constrained height for comparison previews to avoid infinite scrolling */
    .preview-image-container img {
        max-height: 520px;
        object-fit: contain;
        width: 100%;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Model Management & Cached Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_esrgan_model():
    """
    Loads and caches the pre-trained ESRGAN model on the available device.
    Automatically downloads the weights file from Hugging Face if absent or incomplete.
    """
    import urllib.request

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
        with st.spinner("Downloading ESRGAN weights (~67 MB) from Hugging Face..."):
            req = urllib.request.Request(
                MODEL_URL,
                headers={"User-Agent": "Mozilla/5.0 (VisionScayl Streamlit App)"}
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as response, open(MODEL_PATH, "wb") as out_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out_file.write(chunk)
            except Exception as exc:
                raise RuntimeError(f"Failed to auto-download ESRGAN weights: {exc}")

    net = arch.RRDBNet(3, 3, 64, 23, gc=32)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    net = net.to(DEVICE)
    return net

# -----------------------------------------------------------------------------
# Memory-Safe Tiled Super-Resolution Engine
# -----------------------------------------------------------------------------
def tile_process(
    img_tensor: torch.Tensor,
    net_model: torch.nn.Module,
    scale: int = 4,
    tile_size: int = 384,
    tile_pad: int = 16,
    dev: torch.device = DEVICE,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> torch.Tensor:
    """
    Upscales an image tensor using overlapping tiles to guarantee zero GPU/CPU memory overflow.
    Overlapping pads are automatically cropped and stitched to eliminate boundary seams.
    """
    b, c, h, w = img_tensor.shape

    if h <= tile_size and w <= tile_size:
        with torch.inference_mode():
            res = net_model(img_tensor.to(dev)).cpu()
            if progress_callback:
                progress_callback(1, 1)
            return res

    output_h, output_w = h * scale, w * scale
    output = torch.zeros((b, c, output_h, output_w), dtype=torch.float32)

    tiles_x = int(np.ceil(w / tile_size))
    tiles_y = int(np.ceil(h / tile_size))
    total_tiles = tiles_x * tiles_y
    tile_idx = 0

    for y in range(tiles_y):
        for x in range(tiles_x):
            tile_idx += 1
            if progress_callback:
                progress_callback(tile_idx, total_tiles)

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

            if dev.type == "cuda":
                torch.cuda.empty_cache()

            crop_start_x = (input_start_x - input_start_x_pad) * scale
            crop_end_x = crop_start_x + (input_end_x - input_start_x) * scale
            crop_start_y = (input_start_y - input_start_y_pad) * scale
            crop_end_y = crop_start_y + (input_end_y - input_start_y) * scale

            dest_start_x = input_start_x * scale
            dest_end_x = input_end_x * scale
            dest_start_y = input_start_y * scale
            dest_end_y = input_end_y * scale

            output[:, :, dest_start_y:dest_end_y, dest_start_x:dest_end_x] = output_tile[
                :, :, crop_start_y:crop_end_y, crop_start_x:crop_end_x
            ]

    return output


def enhance_image(
    image_input: Image.Image,
    model: torch.nn.Module,
    tile_size: int = 384,
    tile_pad: int = 16,
    sharpen_amount: float = 0.0,
    contrast: float = 1.0,
    brightness: int = 0,
    output_format: str = "PNG",
    quality: int = 95,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[Image.Image, bytes, float]:
    """
    Main processing pipeline:
    1. Converts input PIL image to float tensor
    2. Runs tiled ESRGAN 4x super-resolution
    3. Applies optional post-processing filters
    4. Encodes to requested format
    """
    start_time = time.time()

    img_rgb = np.array(image_input.convert("RGB"))
    img_float = (img_rgb * 1.0 / 255.0).astype(np.float32)
    img_tensor = torch.from_numpy(np.transpose(img_float, (2, 0, 1))).float().unsqueeze(0)

    try:
        output_tensor = tile_process(
            img_tensor,
            model,
            scale=4,
            tile_size=tile_size,
            tile_pad=tile_pad,
            dev=DEVICE,
            progress_callback=progress_callback
        )
    except Exception as e:
        if "out of memory" in str(e).lower() and DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            cpu_dev = torch.device("cpu")
            model.to(cpu_dev)
            output_tensor = tile_process(
                img_tensor,
                model,
                scale=4,
                tile_size=256,
                tile_pad=tile_pad,
                dev=cpu_dev,
                progress_callback=progress_callback
            )
            model.to(DEVICE)
        else:
            raise e

    out_arr = output_tensor.data.squeeze().float().clamp(0, 1).numpy()
    out_arr = np.transpose(out_arr, (1, 2, 0))
    out_arr = (out_arr * 255.0).round().astype(np.uint8)

    # Post-processing filters
    if sharpen_amount > 0.0:
        blurred = cv2.GaussianBlur(out_arr, (0, 0), sigmaX=1.5)
        out_arr = cv2.addWeighted(out_arr, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)
        out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)

    if contrast != 1.0 or brightness != 0:
        out_arr = cv2.convertScaleAbs(out_arr, alpha=contrast, beta=brightness)

    upscaled_pil = Image.fromarray(out_arr)

    buf = io.BytesIO()
    fmt_upper = output_format.upper()
    if fmt_upper in ["JPG", "JPEG"]:
        upscaled_pil.save(buf, format="JPEG", quality=quality, optimize=True)
    elif fmt_upper == "WEBP":
        upscaled_pil.save(buf, format="WEBP", quality=quality)
    else:
        upscaled_pil.save(buf, format="PNG", optimize=True)

    elapsed = time.time() - start_time
    return upscaled_pil, buf.getvalue(), elapsed


def generate_difference_map(orig_pil: Image.Image, upscaled_pil: Image.Image) -> Image.Image:
    """Computes a colorized high-frequency difference map (ESRGAN detail synthesis)."""
    w_up, h_up = upscaled_pil.size
    bicubic_pil = orig_pil.resize((w_up, h_up), Image.BICUBIC)
    
    arr_bicubic = np.array(bicubic_pil, dtype=np.float32)
    arr_upscaled = np.array(upscaled_pil, dtype=np.float32)
    
    diff = np.abs(arr_upscaled - arr_bicubic)
    diff_mag = np.mean(diff, axis=2)
    
    norm_diff = np.clip(diff_mag * 3.5, 0, 255).astype(np.uint8)
    diff_colored = cv2.applyColorMap(norm_diff, cv2.COLORMAP_MAGMA)
    diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(diff_colored)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Status")

    if DEVICE.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "NVIDIA GPU"
        st.markdown(f"<div class='badge-card badge-cuda'>⚡ GPU: {gpu_name}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-card badge-cpu'>🖥️ Engine: CPU Mode</div>", unsafe_allow_html=True)

    st.markdown("<div class='badge-card badge-model'>🧠 Model: ESRGAN 4x RRDBNet</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("#### 🧩 Tiling Configuration")
    tile_size = st.select_slider(
        "Tile Size (Pixels)",
        options=[192, 256, 384, 512],
        value=384,
        help="Controls patch size. 384 is optimal for RTX 3050 (4 GB VRAM)."
    )
    tile_pad = st.slider("Overlap Padding", min_value=8, max_value=32, value=16, step=4)

    st.divider()

    st.markdown("#### 🎨 Clarity & Fine-Tuning")
    enable_postproc = st.checkbox("Enable Fine-Tuning Filters", value=False)
    if enable_postproc:
        sharpen_amount = st.slider("Micro-Sharpening", 0.0, 0.8, 0.15, 0.05)
        contrast = st.slider("Contrast", 0.8, 1.3, 1.0, 0.05)
        brightness = st.slider("Brightness", -25, 25, 0, 5)
    else:
        sharpen_amount = 0.0
        contrast = 1.0
        brightness = 0

    st.divider()

    st.markdown("#### 💾 Default Export Format")
    output_format = st.selectbox("Format", options=["PNG", "JPEG", "WEBP"], index=0)
    quality = 95
    if output_format in ["JPEG", "WEBP"]:
        quality = st.slider("Export Quality", 60, 100, 95, 5)

    st.divider()
    st.caption("VisionScayl v2.0 • GPU Accelerated")

# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <h1 style="margin: 0; font-size: 2.0rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🌟 VisionScayl — AI Image Super-Resolution
        </h1>
        <p style="margin: 0.35rem 0 0 0; color: #94a3b8; font-size: 0.98rem;">
            Restore ultra-fine details, eliminate noise, and upscale images by <b>400% (4x)</b> with Deep Residual-in-Residual Dense Networks (ESRGAN).
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load model once with cache
try:
    with st.spinner("Initializing ESRGAN weights..."):
        model = load_esrgan_model()
except Exception as model_err:
    st.error(f"Error loading ESRGAN model: {model_err}")
    st.stop()

# -----------------------------------------------------------------------------
# Application Tabs
# -----------------------------------------------------------------------------
tab_single, tab_batch, tab_info = st.tabs([
    "⚡ Single Image Upscale",
    "📦 Batch Upscaling",
    "ℹ️ Architecture & Diagnostics"
])

# =============================================================================
# TAB 1: SINGLE IMAGE UPSCALE
# =============================================================================
with tab_single:
    col_upload, col_sample = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload an image to upscale (PNG, JPG, WEBP, BMP):",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="single_file_uploader"
        )

    with col_sample:
        st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
        if st.button("🖼️ Load Demo Sample", use_container_width=True):
            st.session_state["use_sample_trigger"] = True

    # Active Image Determination
    active_img: Optional[Image.Image] = None
    input_filename = "image"

    if uploaded_file is not None:
        try:
            active_img = Image.open(uploaded_file)
            input_filename = Path(uploaded_file.name).stem
            # Reset sample trigger if new upload provided
            st.session_state.pop("use_sample_trigger", None)
        except Exception as e:
            st.error(f"Could not open uploaded file: {e}")
    elif st.session_state.get("use_sample_trigger", False):
        if SAMPLE_IMG_PATH.exists():
            active_img = Image.open(str(SAMPLE_IMG_PATH))
            input_filename = "VisionScayl-sample"
        else:
            arr = np.zeros((128, 128, 3), dtype=np.uint8)
            arr[20:100, 20:100] = [64, 128, 255]
            cv2.putText(arr, "VisionScayl", (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            active_img = Image.fromarray(arr)
            input_filename = "demo_sample"

    if active_img is not None:
        orig_w, orig_h = active_img.size
        orig_mp = (orig_w * orig_h) / 1_000_000
        target_w, target_h = orig_w * 4, orig_h * 4
        target_mp = (target_w * target_h) / 1_000_000

        # Compact Metric Display & Input Preview Row
        col_preview_in, col_metrics_in = st.columns([1, 2])

        with col_preview_in:
            st.markdown("**📸 Input Image Preview:**")
            st.image(active_img, caption=f"{input_filename} ({orig_w}×{orig_h} px)", use_container_width=True)

        with col_metrics_in:
            st.markdown("**📊 Upscale Analysis & Specifications:**")
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    f"<div class='metric-container'><div class='metric-value'>{orig_w} × {orig_h}</div><div class='metric-label'>Current Dimensions</div></div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='metric-container'><div class='metric-value'>{orig_mp:.2f} MP</div><div class='metric-label'>Current Megapixels</div></div>",
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    f"<div class='metric-container'><div class='metric-value' style='color:#4ade80;'>{target_w} × {target_h}</div><div class='metric-label'>Target 4x Dimensions</div></div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='metric-container'><div class='metric-value' style='color:#4ade80;'>{target_mp:.2f} MP (+400%)</div><div class='metric-label'>Target Megapixels</div></div>",
                    unsafe_allow_html=True
                )

            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            btn_upscale = st.button("🚀 Upscale Image Now (4x Super-Resolution)", type="primary", use_container_width=True)

        # Upscale Processing Trigger
        if btn_upscale:
            progress_bar = st.progress(0, text="Initializing neural upscaling...")

            def on_progress(current: int, total: int):
                pct = int((current / total) * 100)
                progress_bar.progress(pct, text=f"Processing image tiles: {current} / {total} ({pct}%)")

            try:
                upscaled_img, raw_bytes, elapsed = enhance_image(
                    image_input=active_img,
                    model=model,
                    tile_size=tile_size,
                    tile_pad=tile_pad,
                    sharpen_amount=sharpen_amount,
                    contrast=contrast,
                    brightness=brightness,
                    output_format=output_format,
                    quality=quality,
                    progress_callback=on_progress
                )
                progress_bar.progress(100, text=f"✨ Complete in {elapsed:.2f}s!")

                # Cache results in session state
                st.session_state["last_upscaled"] = upscaled_img
                st.session_state["last_raw_bytes"] = raw_bytes
                st.session_state["last_elapsed"] = elapsed
                st.session_state["last_filename"] = f"{input_filename}_4x.{output_format.lower()}"
                st.session_state["last_orig"] = active_img
                st.session_state["last_format"] = output_format

            except Exception as up_err:
                progress_bar.empty()
                st.error(f"Inference error during upscaling: {up_err}")

        # =====================================================================
        # PROMINENT RESULTS SECTION (Preview & Download Immediately Visible)
        # =====================================================================
        if "last_upscaled" in st.session_state and st.session_state.get("last_orig") is not None:
            res_img = st.session_state["last_upscaled"]
            res_bytes = st.session_state["last_raw_bytes"]
            res_elapsed = st.session_state["last_elapsed"]
            res_filename = st.session_state["last_filename"]
            orig_view = st.session_state["last_orig"]
            res_fmt = st.session_state.get("last_format", output_format)

            st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.1); margin: 1.5rem 0;'>", unsafe_allow_html=True)
            
            # --- PROMINENT TOP DOWNLOAD & ACTION BAR ---
            st.markdown("### 🎉 Super-Resolution Complete!")
            
            action_col1, action_col2, action_col3 = st.columns([2, 1, 1])
            
            mime_type = "image/png" if res_fmt == "PNG" else ("image/webp" if res_fmt == "WEBP" else "image/jpeg")

            with action_col1:
                st.download_button(
                    label=f"⬇️ DOWNLOAD ENHANCED IMAGE ({res_fmt.upper()} • {len(res_bytes) / 1024:.1f} KB)",
                    data=res_bytes,
                    file_name=res_filename,
                    mime=mime_type,
                    type="primary",
                    use_container_width=True
                )

            with action_col2:
                st.info(f"⚡ Time: **{res_elapsed:.2f}s** • {res_img.size[0]}×{res_img.size[1]} px")

            with action_col3:
                if st.button("🔄 Clear Result", use_container_width=True):
                    for k in ["last_upscaled", "last_raw_bytes", "last_elapsed", "last_filename", "last_orig", "last_format"]:
                        st.session_state.pop(k, None)
                    st.rerun()

            # --- INTERACTIVE PREVIEW & INSPECTION TABS ---
            st.markdown("#### 🔍 Visual Comparison & Inspection Preview")

            prev_tab1, prev_tab2, prev_tab3, prev_tab4 = st.tabs([
                "↔️ Side-by-Side Comparison",
                "🌟 Upscaled Output (Full Preview)",
                "📷 Original Input",
                "🔥 Detail Difference Heatmap"
            ])

            with prev_tab1:
                c_orig, c_upscaled = st.columns(2)
                with c_orig:
                    st.markdown(f"**Original Image ({orig_view.size[0]} × {orig_view.size[1]} px)**")
                    st.image(orig_view, use_container_width=True)
                with c_upscaled:
                    st.markdown(f"**VisionScayl 4x Output ({res_img.size[0]} × {res_img.size[1]} px)**")
                    st.image(res_img, use_container_width=True)

            with prev_tab2:
                st.markdown(f"**High-Resolution 4x Output ({res_img.size[0]} × {res_img.size[1]} px)**")
                st.image(res_img, use_container_width=True)

            with prev_tab3:
                st.markdown(f"**Original Input ({orig_view.size[0]} × {orig_view.size[1]} px)**")
                st.image(orig_view, use_container_width=True)

            with prev_tab4:
                st.markdown("**High-Frequency Detail Heatmap (ESRGAN Reconstructed Edges & Textures)**")
                diff_map = generate_difference_map(orig_view, res_img)
                c_res, c_diff = st.columns(2)
                with c_res:
                    st.image(res_img, caption="Upscaled Output", use_container_width=True)
                with c_diff:
                    st.image(diff_map, caption="Synthesized Texture Heatmap", use_container_width=True)

            # Secondary download button at bottom of review
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.download_button(
                label=f"⬇️ Download {res_filename} ({len(res_bytes) / 1024:.1f} KB)",
                data=res_bytes,
                file_name=res_filename,
                mime=mime_type,
                key="download_button_bottom",
                use_container_width=True
            )

    else:
        st.info("👆 Upload an image or click **🖼️ Load Demo Sample** to preview and upscale your image.")

# =============================================================================
# TAB 2: BATCH UPSCALING (Persistent State & ZIP Download)
# =============================================================================
with tab_batch:
    st.markdown("### 📦 Multi-Image Batch Super-Resolution")
    st.markdown(
        "Upload multiple low-resolution images to process them sequentially with the tiled GPU engine and export all enhanced results in a single ZIP archive."
    )

    batch_files = st.file_uploader(
        "Select multiple images:",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
        key="batch_files_uploader"
    )

    if batch_files:
        st.write(f"📁 **{len(batch_files)} image(s) queued for processing:**")
        summary_data = [{"File Name": f.name, "File Size (KB)": round(f.size / 1024, 1)} for f in batch_files]
        st.dataframe(summary_data, use_container_width=True)

        if st.button("🚀 Process Batch & Upscale All", type="primary", use_container_width=True):
            batch_progress = st.progress(0, text="Starting batch upscaling...")
            completed_items = []
            total_count = len(batch_files)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for idx, file_item in enumerate(batch_files):
                    stem_name = Path(file_item.name).stem
                    out_name = f"{stem_name}_4x.{output_format.lower()}"

                    batch_progress.progress(
                        int((idx / total_count) * 100),
                        text=f"Enhancing ({idx + 1}/{total_count}): {file_item.name}..."
                    )

                    try:
                        pil_img = Image.open(file_item)
                        upscaled_pil, raw_bytes, elapsed = enhance_image(
                            image_input=pil_img,
                            model=model,
                            tile_size=tile_size,
                            tile_pad=tile_pad,
                            sharpen_amount=sharpen_amount,
                            contrast=contrast,
                            brightness=brightness,
                            output_format=output_format,
                            quality=quality
                        )
                        zip_file.writestr(out_name, raw_bytes)
                        completed_items.append({
                            "original_name": file_item.name,
                            "output_name": out_name,
                            "img": upscaled_pil,
                            "elapsed": elapsed,
                            "orig_size": f"{pil_img.size[0]}×{pil_img.size[1]}",
                            "upscaled_size": f"{upscaled_pil.size[0]}×{upscaled_pil.size[1]}"
                        })
                    except Exception as b_err:
                        st.error(f"Failed to process '{file_item.name}': {b_err}")

            batch_progress.progress(100, text="🎉 Batch upscaling complete!")
            zip_buffer.seek(0)

            # Store in session state so it persists across reruns!
            st.session_state["batch_completed"] = completed_items
            st.session_state["batch_zip_data"] = zip_buffer.getvalue()

    # Display persistent batch results & download button
    if "batch_completed" in st.session_state and st.session_state["batch_completed"]:
        items = st.session_state["batch_completed"]
        zip_bytes = st.session_state["batch_zip_data"]

        st.success(f"🎉 Successfully enhanced **{len(items)} images**!")

        st.download_button(
            label=f"⬇️ DOWNLOAD ALL ENHANCED IMAGES (.ZIP • {len(zip_bytes) / 1024:.1f} KB)",
            data=zip_bytes,
            file_name="VisionScayl_Batch_Upscaled.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )

        st.markdown("#### 🖼️ Enhanced Image Gallery")
        gallery_cols = st.columns(min(3, len(items)) or 1)
        for i, it in enumerate(items):
            with gallery_cols[i % len(gallery_cols)]:
                st.image(
                    it["img"],
                    caption=f"{it['output_name']} ({it['upscaled_size']} in {it['elapsed']:.1f}s)",
                    use_container_width=True
                )

# =============================================================================
# TAB 3: ARCHITECTURE & SYSTEM DIAGNOSTICS
# =============================================================================
with tab_info:
    st.markdown("### 🧠 Neural Architecture & Runtime Diagnostics")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### 🔬 ESRGAN / RRDBNet Specifications")
        st.markdown(
            """
            - **Model:** Enhanced Super-Resolution Generative Adversarial Network (ESRGAN)
            - **Backbone Architecture:** Residual-in-Residual Dense Block (`RRDBNet`)
            - **Residual Blocks:** 23 RRDB Blocks
            - **Growth Channels (gc):** 32
            - **Internal Features:** 64
            - **Scaling Factor:** 4x (400% linear enlargement / 16x pixel expansion)
            - **Parameters:** ~16.7 Million
            - **Checkpoint:** `RRDB_ESRGAN_x4.pth` (~67 MB)
            """
        )

    with col_d2:
        st.markdown("#### 🖥️ Runtime & Hardware Acceleration")
        cuda_status = "Available (Accelerated)" if torch.cuda.is_available() else "Unavailable (CPU Mode)"
        st.markdown(
            f"""
            - **PyTorch Version:** `{torch.__version__}`
            - **Active Computing Device:** `{DEVICE.type.upper()}`
            - **CUDA GPU Acceleration:** `{cuda_status}`
            - **Streamlit Version:** `{st.__version__}`
            - **Virtual Environment:** `.visionscayl`
            """
        )

    st.divider()

    st.markdown("#### 🛡️ Memory-Safe Tiled Super-Resolution Engine")
    st.markdown(
        """
        Deep super-resolution generative networks require substantial activation tensors during execution. 
        A 1000×1000 input expanding to 4000×4000 can consume multiple gigabytes of memory.
        
        **VisionScayl Safeguards:**
        1. **Patch Slicing:** Automatically splits images into optimal tiles (e.g. 384×384 px).
        2. **Border Padding:** Overlaps 16 px border pads to completely prevent convolution edge seams.
        3. **Dynamic Reassembly:** Seamlessly writes enhanced patches back to the output tensor.
        4. **Automatic Fallback:** Gracefully recovers from GPU memory pressure by clearing cache and routing to CPU.
        """
    )
