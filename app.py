%%writefile app.py
import torch
from diffusers import StableDiffusionPipeline
import streamlit as st
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import time
import io
import os
import datetime

# --- Configuration & Model Selection ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# --- Storage Setup ---
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Loading---
@st.cache_resource
def load_model():
    """
    Loads the Stable Diffusion pipeline using memory optimizations.
    
    Uses enable_model_cpu_offload() to ensure the large model fits into low RAM
    by swapping components to the disk when not in use.
    """
    st.info(f"Loading model ({MODEL_ID}) on {DEVICE.upper()} with memory offload...")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
        )
        
        pipe.enable_model_cpu_offload() 
        
        if DEVICE == "cuda":
            pipe.to(DEVICE)
            
        st.success("Model configuration and loading successful!")
        return pipe
        
    except Exception as e:
        st.error(f"Error during model download/loading: {e}")
        st.warning(f"This error typically indicates insufficient RAM/Disk space for the {MODEL_ID} model. Please try the Google Colab link for a guaranteed functional demo.")
        return None

def enhance_prompt(prompt, style):
    """Adds professional quality descriptors based on user style selection."""
    style_descriptors = {
        "Photorealistic": ", highly detailed, 4K, professional photography, cinematic lighting, dramatic volumetric light, hyper-realistic, sharp focus",
        "Artistic (Van Gogh)": ", in the style of Van Gogh, thick impasto brushstrokes, swirling patterns, vibrant colors",
        "Cartoon/Vector": ", vector art, smooth, high contrast, clean lines, cell-shading, minimal detail background",
        "Default/None": "",
    }
    return prompt + style_descriptors.get(style, "")

def generate_image(pipe, prompt, negative_prompt, steps, scale, num_images):
    """Handles the core image generation process."""
    if DEVICE == "cuda":
        with torch.autocast(DEVICE):
            images = pipe(
                prompt, 
                negative_prompt=negative_prompt,
                num_inference_steps=steps, 
                guidance_scale=scale,
                num_images_per_prompt=num_images
            ).images
    else:
        images = pipe(
            prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=steps, 
            guidance_scale=scale,
            num_images_per_prompt=num_images
        ).images
    return images

def save_and_export_image(image, prompt, negative_prompt, steps, scale, filename, index):
    """Saves image with metadata and returns bytes for download."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metadata = {
        "Prompt": prompt,
        "Negative Prompt": negative_prompt,
        "Inference Steps": str(steps),
        "Guidance Scale": str(scale),
        "Model": MODEL_ID,
        "Device": DEVICE,
        "Timestamp": timestamp,
        "AI_Watermark": "Generated using Stable Diffusion v1.5",
    }
    
    pnginfo_obj = PngInfo()
    for key, value in metadata.items():
        pnginfo_obj.add_text(key, str(value))
    
    safe_filename = filename.replace(" ", "_").replace("/", "").strip()[:50] 
    base_name = f"{timestamp}_{safe_filename}_{index}"

    png_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
    image.save(png_path, format="PNG", pnginfo=pnginfo_obj)
    
    jpeg_path = os.path.join(OUTPUT_DIR, f"{base_name}.jpeg")
    image.save(jpeg_path, format="JPEG", quality=90)
    
    png_bytes = io.BytesIO()
    image.save(png_bytes, format="PNG", pnginfo=pnginfo_obj)
    return png_bytes.getvalue(), base_name

# --- Streamlit ---
def main():
    st.set_page_config(page_title="AI Image Generator", layout="wide")
    st.title("**AI Image Generator**")
    
    st.sidebar.markdown("## Hardware Status")
    st.sidebar.info(f"**Device:** {DEVICE.upper()}\n")
    
    pipe = load_model()
    if pipe is None:
        return 

    st.subheader("Text Prompt & Style Selection")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area("Describe your desired image:", "A majestic lion standing on a cliff edge, facing a massive thunder storm, hyper detailed, 8K render", height=100)
    with col2:
        style = st.selectbox("Style Guidance", ["Photorealistic", "Artistic (Van Gogh)", "Cartoon/Vector", "Default/None"])
        num_images = st.slider("Number of Images", min_value=1, max_value=4, value=1, step=1)

    st.sidebar.markdown("---")    
    negative_prompt = "blurry, ugly, deformed, noisy, jpeg artifacts, bad anatomy, worst quality, low resolution, extra fingers, malformed hands"   
    steps = st.sidebar.slider("Inference Steps", min_value=10, max_value=100, value=25, step=5)
    scale = st.sidebar.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Export Settings")
    filename_input = st.sidebar.text_input("Custom Filename Base:", "ai_art_output")
    
    if st.button("**Generate Image(s)**"):
        if not prompt:
            st.error("Please enter a text prompt.")
            return

        final_prompt = enhance_prompt(prompt, style)
        st.info(f"Using Final Prompt: {final_prompt}")

        start_time = time.time()
        
        progress_text = f"Generating {num_images} image(s)... Time estimate: ~{num_images * 2} minutes (CPU) / ~{num_images * 20} seconds (GPU)"
        progress_bar = st.progress(0, text=progress_text)
        
        try:
            images = generate_image(pipe, final_prompt, negative_prompt, steps, scale, num_images)
        except Exception as e:
            st.error(f"Generation failed during inference: {e}")
            return
            
        end_time = time.time()
        duration = end_time - start_time
        
        progress_bar.progress(100, text="Generation Complete!")

        st.subheader("🖼️ Generated Results")
        
        cols = st.columns(num_images)
        for i, image in enumerate(images):
            png_data, base_name = save_and_export_image(image, final_prompt, negative_prompt, steps, scale, filename_input, i)
            
            with cols[i]:
                st.image(image, caption=f"Result {i+1} | {style} Style", use_column_width="stretch")
                st.markdown(f"**AI Watermark:** {base_name}.png")
                
                # Download Buttons (PNG, JPEG is saved to disk)
                st.download_button(
                    label="⬇️ Download PNG", 
                    data=png_data, 
                    file_name=f"{base_name}.png", 
                    mime="image/png"
                )
                st.download_button(
                    label="⬇️ Download JPEG (Saved to Disk)", 
                    data=open(os.path.join(OUTPUT_DIR, f"{base_name}.jpeg"), 'rb').read(), 
                    file_name=f"{base_name}.jpeg", 
                    mime="image/jpeg"
                )

        st.markdown(f"---")
        st.success(f"**Total Generation Time:** {duration:.2f} seconds ({num_images} images)")

if __name__ == "__main__":
    main()