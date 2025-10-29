# pip install torch diffusers transformers accelerate safetensors Pillow PySide6

import os, sys
from pathlib import Path
from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QPushButton, QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFileDialog, QProgressBar, QLabel
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor

# ============================================================
#                   Worker Thread
# ============================================================
class Img2ImgWorker(QThread):
    log = Signal(str)
    progress = Signal(int)
    finished = Signal()

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stop_flag = False

    def stop(self): self.stop_flag = True

    def run(self):
        c = self.cfg
        try:
            self.log.emit(f"[INIT] Loading base model: {c['base_model']} ({c['device']})")
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                c["base_model"],
                torch_dtype=torch.float16 if c["device"] != "cpu" else torch.float32,
                use_safetensors=True
            ).to(c["device"])
            pipe.set_progress_bar_config(disable=True)

            refiner = None
            if c["use_refiner"]:
                self.log.emit(f"[INIT] Loading refiner: {c['refiner_model']}")
                refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    c["refiner_model"],
                    torch_dtype=torch.float16 if c["device"] != "cpu" else torch.float32,
                    use_safetensors=True
                ).to(c["device"])
                refiner.set_progress_bar_config(disable=True)

            files = [f for f in Path(c["input"]).iterdir()
                     if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
            total = len(files)
            if not total:
                self.log.emit("[ERROR] No images found.")
                self.finished.emit()
                return
            Path(c["output"]).mkdir(parents=True, exist_ok=True)

            for i, fpath in enumerate(files, 1):
                if self.stop_flag: break
                self.log.emit(f"[PROCESS] ({i}/{total}) {fpath.name}")
                try:
                    img = Image.open(fpath).convert("RGB").resize((c["width"], c["height"]))
                    base_out = pipe(
                        prompt=c["prompt"], negative_prompt=c["neg"],
                        image=img, strength=c["strength_base"],
                        guidance_scale=c["guidance"], num_inference_steps=c["steps_base"],
                        output_type="latent" if c["use_refiner"] else "pil"
                    )
                    if c["use_refiner"]:
                        self.log.emit("  ↳ [REFINER] Enhancing details…")
                        out = refiner(
                            prompt=c["prompt"], negative_prompt=c["neg"],
                            image=base_out.images[0],
                            strength=c["strength_refiner"],
                            guidance_scale=c["guidance"],
                            num_inference_steps=c["steps_refiner"]
                        ).images[0]
                    else:
                        out = base_out.images[0]
                    out.save(Path(c["output"]) / f"{fpath.stem}_out.png")
                    self.log.emit(f"  ✓ Saved {fpath.stem}_out.png")
                except Exception as e:
                    self.log.emit(f"[ERROR] {fpath.name}: {e}")
                self.progress.emit(int(i / total * 100))

            if refiner: del refiner
            del pipe
            torch.cuda.empty_cache()
            self.log.emit("[DONE] All images processed.")
        except Exception as e:
            self.log.emit(f"[FATAL] {e}")
        self.finished.emit()

# ============================================================
#                   GUI
# ============================================================
class BatchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("batchGEN")
        self.setMinimumSize(1200, 700)
        self.device = self.detect_device()
        self.worker = None
        self.build_ui()
        self.apply_style()

    def detect_device(self):
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    # ---------------- UI ----------------
    def build_ui(self):
        root = QHBoxLayout(self)          # two columns side-by-side

        # ==== LEFT COLUMN (controls) ====
        left = QVBoxLayout()

        # --- Directories ---
        dirs = QGroupBox("Directories")
        f = QFormLayout()
        self.in_dir, self.out_dir = QLineEdit(), QLineEdit()
        binp, bout = QPushButton("Browse"), QPushButton("Browse")
        binp.clicked.connect(lambda: self.pick_folder(self.in_dir))
        bout.clicked.connect(lambda: self.pick_folder(self.out_dir))
        r1 = QHBoxLayout(); r1.addWidget(self.in_dir); r1.addWidget(binp)
        r2 = QHBoxLayout(); r2.addWidget(self.out_dir); r2.addWidget(bout)
        f.addRow("Input:", r1); f.addRow("Output:", r2)
        dirs.setLayout(f)
        left.addWidget(dirs)

        # --- Models ---
        models = QGroupBox("Models")
        fm = QFormLayout()
        self.base_model = QLineEdit("stabilityai/stable-diffusion-xl-base-1.0")
        self.refiner_model = QLineEdit("stabilityai/stable-diffusion-xl-refiner-1.0")
        self.use_refiner = QCheckBox("Use Refiner"); self.use_refiner.setChecked(True)
        fm.addRow("Base:", self.base_model)
        fm.addRow("Refiner:", self.refiner_model)
        fm.addRow("", self.use_refiner)
        models.setLayout(fm)
        left.addWidget(models)

        # --- Params ---
        params = QGroupBox("Parameters")
        fp = QFormLayout()
        self.str_base = QDoubleSpinBox(); self.str_base.setRange(0,1); self.str_base.setValue(0.7)
        self.str_ref = QDoubleSpinBox(); self.str_ref.setRange(0,1); self.str_ref.setValue(0.25)
        self.guid = QDoubleSpinBox(); self.guid.setRange(1,30); self.guid.setValue(7.5)
        self.steps_base = QSpinBox(); self.steps_base.setRange(1,150); self.steps_base.setValue(30)
        self.steps_ref = QSpinBox(); self.steps_ref.setRange(1,150); self.steps_ref.setValue(10)
        self.width = QSpinBox(); self.width.setRange(256,2048); self.width.setValue(1024)
        self.height = QSpinBox(); self.height.setRange(256,2048); self.height.setValue(1024)
        fp.addRow("Base Strength:", self.str_base)
        fp.addRow("Refiner Strength:", self.str_ref)
        fp.addRow("Guidance:", self.guid)
        fp.addRow("Base Steps:", self.steps_base)
        fp.addRow("Refiner Steps:", self.steps_ref)
        fp.addRow("Width:", self.width)
        fp.addRow("Height:", self.height)
        params.setLayout(fp)
        left.addWidget(params)

        # --- Controls ---
        ctl = QHBoxLayout()
        self.start = QPushButton("START"); self.stop = QPushButton("STOP")
        self.stop.setEnabled(False)
        self.start.clicked.connect(self.start_batch)
        self.stop.clicked.connect(self.stop_batch)
        ctl.addWidget(self.start); ctl.addWidget(self.stop)
        left.addLayout(ctl)

        root.addLayout(left, 1)

        # ==== RIGHT COLUMN (prompt + log) ====
        right = QVBoxLayout()

        # --- Prompts ---
        prompts = QGroupBox("Prompts")
        pf = QFormLayout()
        self.prompt = QTextEdit("EVIL")
        self.prompt.setMaximumHeight(80)
        self.neg = QTextEdit("blurry, lowres, distorted, bad anatomy")
        self.neg.setMaximumHeight(60)
        pf.addRow("Prompt:", self.prompt)
        pf.addRow("Negative:", self.neg)
        prompts.setLayout(pf)
        right.addWidget(prompts)

        # --- Progress + Log ---
        self.progress = QProgressBar()
        self.log = QTextEdit(); self.log.setReadOnly(True)
        right.addWidget(self.progress)
        right.addWidget(self.log, 5)

        root.addLayout(right, 1)
        self.setLayout(root)

    # ---------------- STYLE ----------------
    def apply_style(self):
        self.setStyleSheet("""
            QWidget { background-color:#0a0f0a; color:#00ff99; font-family:'Courier New'; font-size:12px; }
            QGroupBox { border:1px solid #00ff99; border-radius:6px; margin-top:6px; padding:6px; font-weight:bold; }
            QPushButton { background-color:#001a0f; border:1px solid #00ff99; border-radius:6px; padding:6px 10px; }
            QPushButton:hover { background-color:#00331a; }
            QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color:#001a0f; border:1px solid #004d26; border-radius:4px; color:#00ff99;
            }
            QProgressBar { border:1px solid #00ff99; border-radius:4px; height:16px; text-align:center; color:black; }
            QProgressBar::chunk { background-color:#00ff99; margin:0.5px; }
            QCheckBox::indicator { width:14px; height:14px; }
            QCheckBox::indicator:checked { background-color:#00ff99; }
            QCheckBox::indicator:unchecked { border:1px solid #00ff99; }
        """)

    # ---------------- Helpers ----------------
    def logmsg(self, msg):
        self.log.append(msg)
        try: self.log.moveCursor(QTextCursor.End)
        except Exception:
            cur = self.log.textCursor(); cur.movePosition(QTextCursor.End); self.log.setTextCursor(cur)
        QApplication.processEvents()

    def pick_folder(self, target):
        p = QFileDialog.getExistingDirectory(self, "Select Folder")
        if p: target.setText(p)

    # ---------------- Batch ----------------
    def start_batch(self):
        if not self.in_dir.text() or not self.out_dir.text():
            self.logmsg("[ERROR] Select input/output folders."); return
        cfg = dict(
            base_model=self.base_model.text().strip(),
            refiner_model=self.refiner_model.text().strip(),
            input=self.in_dir.text().strip(),
            output=self.out_dir.text().strip(),
            prompt=self.prompt.toPlainText().strip(),
            neg=self.neg.toPlainText().strip(),
            strength_base=self.str_base.value(),
            strength_refiner=self.str_ref.value(),
            guidance=self.guid.value(),
            steps_base=self.steps_base.value(),
            steps_refiner=self.steps_ref.value(),
            width=self.width.value(),
            height=self.height.value(),
            use_refiner=self.use_refiner.isChecked(),
            device=self.device
        )
        self.worker = Img2ImgWorker(cfg)
        self.worker.log.connect(self.logmsg)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.finished_batch)
        self.start.setEnabled(False); self.stop.setEnabled(True)
        self.progress.setValue(0)
        self.logmsg(f"[START] Device: {self.device} | Refiner: {'ON' if cfg['use_refiner'] else 'OFF'}")
        self.worker.start()

    def stop_batch(self):
        if self.worker: self.worker.stop(); self.logmsg("[STOP] Stop signal sent.")

    def finished_batch(self):
        self.logmsg("[SYSTEM] Finished or stopped.")
        self.start.setEnabled(True); self.stop.setEnabled(False)

# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = BatchApp(); w.show()
    sys.exit(app.exec())
