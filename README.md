# Text-to-Image Generative System

## Overview

This project implements a text-to-image generation system based on **Diffusion Models** using the Hugging Face `diffusers` library and a Streamlit web interface. It is optimized to run on both **GPU** and severely **low-RAM/CPU-only** environments.

### Technology Stack

* **Model Core:** Stable Diffusion v1.5 (Diffusion Model)
* **Framework:** PyTorch
* **UI/Interface:** Streamlit
* **Core Libraries:** `diffusers`, `transformers`, `accelerate`

---
### Core Architecture

The system uses a standard client-server pattern implemented within Streamlit:

1.  **Model Loading:** The `load_model()` function is decorated with `@st.cache_resource` to ensure the 5GB model is only loaded **once**, even across user interactions.
2.  **Memory Optimization (Low-End Hardware):** The model is loaded with `pipe.enable_model_cpu_offload()`. This is a crucial technique that dynamically loads model components (UNet, VAE, Text Encoder) to the GPU only when needed, moving them back to **system RAM/Disk (Pagefile)** when idle. This allows the 5GB model to function on systems with **limited physical memory**. 
3.  **Prompt Enhancement:** User input is processed by the `enhance_prompt()` function, which prepends or appends high-quality descriptors ("highly detailed," "cinematic lighting") based on the selected style.
---

## Setup and Installation

### A. Prerequisites

You must have **Python 3.9+** installed.

### B. Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR GITHUB REPO URL]
    cd [YOUR REPO NAME]
    ```

2.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate # On Windows
    # source venv/bin/activate # On Linux/macOS
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your browser (`http://localhost:8501`). The **model will automatically download** to your Hugging Face cache upon the first run.

---

## Image Quality Enhancement & Prompt Engineering

### Implementation:
* **Prompt Enhancement:** Implemented in `enhance_prompt()` to add specific style and quality tokens (e.g., `hyper-realistic, 4K, cinematic lighting`) based on user selection.
* **Negative Prompt:** A dedicated sidebar input is included to filter unwanted features (`blurry, deformed, jpeg artifacts`).

## Storage and Export

* **Storage:** Generated images are saved to the local `generated_images/` directory.
* **Metadata:** A comprehensive dictionary of metadata (Prompt, Timestamp, Parameters, **AI Watermark**) is saved within the **PNG file format** upon export.
* **Export Formats:** Both **PNG (with metadata)** and **JPEG** formats are automatically saved to disk and provided for browser download.
* **Watermarking (Ethical AI):** A mandatory metadata field `AI_Watermark: Generated using Stable Diffusion v1.5 (AI Origin Watermark)` is added to every saved image file.

---
