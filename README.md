# Text-to-Image Generator

## Project Overview and Architecture
This project implements a powerful **Text-to-Image Generation System** built using **Diffusion Models** via the Hugging Face `diffusers` library, wrapped inside a clean and interactive Streamlit web UI.

It is optimized to run on **High-end GPU systems** AND **Low-RAM / CPU-only laptops** using model offloading techniques.

### Features
* **Stable Diffusion v1.5** powered text-to-image generator
* Runs on low-end hardware using **CPU/GPU offload**
* Multiple image styles (Photorealistic, Artistic, Van Gogh, etc.)
* **Automatic prompt enhancement** for higher quality images
* **Universal Negative Prompt** support (fixed in code)
* Automatic saving of images (PNG + JPEG) with metadata
* **AI Watermarked** PNG metadata for transparency
* Highly optimized model loading with **Streamlit cache**

### Technology Stack
* **Model Core:** Stable Diffusion v1.5
* **Framework:** PyTorch
* **Frontend/UI:** Streamlit
* **Model Tools:** `diffusers`, `transformers`, `accelerate`

### Core Architecture
The system uses a standard client-server pattern implemented within Streamlit:

1.  **Model Loading:** The `load_model()` function uses **`@st.cache_resource`**. This ensures the $\sim5\text{GB}$ model loads only once, even after UI interactions.
2.  **Memory Optimization:** To support low-end laptops, **`pipe.enable_model_cpu_offload()`** is used.  This technique dynamically moves components (**UNet, VAE, Text Encoder**) between GPU/CPU on demand, allowing generation on systems with **limited physical memory**.
3.  **Prompt Enhancement:** The `enhance_prompt()` function automatically adds high-quality keywords (e.g., `hyper-realistic`, `4K`, `cinematic lighting`, `Van Gogh brush strokes`) based on the style chosen. This increases output quality drastically.

## Setup and Installation

### A. Prerequisites

* **Python 3.9+**
* (Optional) GPU with CUDA

### B. Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ThangakumarC/Image_gen.git
    cd Image_gen
    ```

2.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate   # Windows
    # source venv/bin/activate  # Linux/macOS
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run application:**
    ```bash
    streamlit run app.py
    ```
    The UI opens at: `http://localhost:8501`. The Stable Diffusion model will download automatically on first run.

## Usage Instruction

1.  **Select Style:** Choose one of the preset styles (e.g., *Photorealistic*) from the sidebar dropdown. This triggers the automatic prompt enhancement.
2.  **Enter Base Prompt:** Provide a simple, descriptive subject and scene.
3.  **Set Parameters:** Adjust the **Inference Steps** (25 is recommended) and **Guidance Scale** (7.5 is recommended) in the sidebar.
4.  **Generate:** Click **Generate Image(s)**.

### Prompt Engineering Tips and Best Practices

The system employs two core prompt engineering techniques:

* **Automatic Enhancement:** The `enhance_prompt()` function automatically appends quality tokens like `hyper-realistic, 4K, cinematic lighting`.
* **Universal Negative Prompt:** A fixed negative prompt is hardcoded to filter out common flaws (e.g., `blurry, bad anatomy, deformed`).

| Best Practice Tip | Description |
| :--- | :--- |
| **Be Descriptive** | Always specify the subject, scene, action, and key aesthetic (e.g., `wet fur`, `golden hour lighting`). |
| **Adjust Guidance Scale** | Lowering the **Guidance Scale** (e.g., from 7.5 to 5.0) makes the image more creative but less strictly tied to the prompt. |

##  Hardware Requirements & Implementation Paths

The system is designed to run efficiently across various hardware configurations, prioritizing resource management for low-end systems.

| Configuration | Recommended RAM | Recommended VRAM/GPU | Generation Time (512x512) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **GPU Path (Fastest)** | 16 GB | 8 GB+ NVIDIA/AMD | **5 - 20 seconds** | PyTorch with CUDA/ROCm. **Highly recommended for practical use.** |
| **Optimized CPU Path** | **8 GB Minimum** | N/A | **5 - 20+ minutes** | Relies on `pipe.enable_model_cpu_offload()` for stability on systems with **limited physical memory**. |

## Image Quality Enhancement & Prompt Engineering

The system implements prompt engineering techniques to ensure consistently high-quality output.

* **Automatic Quality Tokens:** The `enhance_prompt()` function automatically appends quality-boosting terms (e.g., `highly detailed`, `4K`).
* **Universal Negative Prompt:** A robust negative prompt is embedded directly in the code, filtering out flaws like: `blurry, deformed, extra limbs, low resolution, distorted face, jpeg artifacts`.
* **Configurable Parameters:** Users can adjust **Inference Steps** and **Guidance Scale** for fine-tuning quality and style adherence.

## Storage and Export

Generated images are saved to the local **`generated_images/`** directory.

* **PNG Format:** Includes **metadata** (prompt, parameters, timestamp, AI watermark) for complete provenance.
* **JPEG Format:** A lightweight version is also saved for easy sharing.
* **AI Watermark:** Every PNG includes the ethical transparency metadata: `AI_Watermark: Generated using Stable Diffusion v1.5 (AI Origin)`.

## Sample Generated Outputs

Below are example generations showcasing the effectiveness of the automatic prompt enhancement and style guidance.
### 1. Style - Photorealistic  |  Base Prompt  -  A futuristic city at sunset
###   Image Result 

<img width="512" height="512" alt="futuristic city at sunset" src="https://github.com/user-attachments/assets/b5752212-4e33-4e88-8e47-f9f5a7709994"/>
    
### 2. Style - Artistic (Van Gogh) | Base Prompt - Portrait of a robot
###   Image Result

<img width="512" height="512" alt="20251126_184951_robos_1" src="https://github.com/user-attachments/assets/1b02777c-59c0-4e78-b779-da97d0c6a862" />

## Limitations

* **CPU Speed:** While the app can run on CPU, generation time is excessively slow (tens of minutes per image), making the GPU path essential for practical use.
* **VRAM Optimization:** The system primarily optimizes for low **system RAM** (via CPU offload), not deep optimization for extremely low **VRAM** (GPU memory).

## Future Improvements

* **Tuning:** Implement a fine-tuning module (e.g., LoRA) to allow training the model on custom datasets (e.g., company branding).
* **Performance:** Integrate **ONNX/OpenVINO** optimizations to significantly speed up CPU inference for the non-GPU path.
* **Advanced Safety:** Implement an external CLIP-based filter to flag inappropriate *prompts* before generation begins, enhancing ethical compliance.
