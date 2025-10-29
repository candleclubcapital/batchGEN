Python GUI for batch Stable Diffusion XL image-to-image generation with optional refiner support. Built with PySide6 and Diffusers, it processes folders of input images into stylized outputs using SDXL pipelines, complete with logging, progress tracking, and real-time control.
<img width="1171" height="678" alt="Screenshot 2025-10-29 at 2 19 24 PM" src="https://github.com/user-attachments/assets/9f79271f-22ba-46ab-8577-d6f99b96d4f5" />


---

# batchGEN

**batchGEN** is a Python application for batch processing images through **Stable Diffusion XL (SDXL)** using an intuitive, neon-terminal-inspired GUI. It supports both the base and refiner pipelines, allowing rapid stylized re-generation of large image sets.

---

### ✨ Features

* Batch **img2img** processing using SDXL base and optional refiner
* Adjustable parameters for **strength**, **guidance**, **steps**, and **dimensions**
* Real-time **log output** and **progress bar**
* Automatic device detection (`cuda`, `mps`, or `cpu`)
* Stylish **cyberpunk interface** built with PySide6
* Safe multithreading with interruptible generation

---

### 🧩 Requirements

```bash
pip install torch diffusers transformers accelerate safetensors Pillow PySide6
```

---

### 🚀 Usage

1. Run the application:

   ```bash
   python batchgen.py
   ```
2. Select input and output folders.
3. Enter your **prompt** and **negative prompt**.
4. Adjust generation parameters.
5. Click **START** to begin processing.

Output images will be saved in the selected output folder as `<filename>_out.png`.

---

### 🖥️ Interface Overview

* **Directories**: choose input/output folders
* **Models**: define base and optional refiner checkpoints
* **Parameters**: control strengths, guidance, steps, and size
* **Prompts**: text prompts and negative prompts
* **Progress/Log**: live feedback and processing updates

---

### ⚙️ Notes

* Works best with GPUs or Apple Silicon (MPS).
* Refiner usage increases detail but doubles memory use.
* Automatically saves each result as PNG.

