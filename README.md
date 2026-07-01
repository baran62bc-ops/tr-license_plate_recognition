# Turkish License Plate OCR Pipeline

An end-to-end pipeline for detecting and reading Turkish license plates from car images. Built from scratch using a fine-tuned YOLOv11 detector and a custom CNN → Transformer → CTC text recognition model, deployed as a Streamlit app.

---

## Architecture

### Detection
- **YOLOv11 nano** fine-tuned on Turkish license plate images
- Outputs bounding boxes, crops the plate region, corrects EXIF rotation

### Recognition (OCR)
- **CNN backbone** (CRNN-style): 7-block VGG-inspired network with BatchNorm, asymmetric pooling `(2,1)` to preserve the width dimension as the time axis
- **Transformer encoder**: multi-head self-attention (no causal masking — bidirectional context), positional encoding, layer norm, feed-forward blocks
- **CTC loss + greedy decoder**: handles variable-length character sequences without explicit segmentation
- **Vocab**: 64 characters — blank, space, `a-z`, `A-Z`, `0-9`

```
Car Image
    ↓
YOLOv11 (plate detection) → bounding box crop
    ↓
CNN (B, 1, 64, 128) → (B, T, 512)
    ↓
Linear projection → (B, T, d_model)
    ↓
Positional Encoding
    ↓
Transformer Encoder (N layers)
    ↓
Linear → (B, T, vocab_size)
    ↓
CTC decode → plate string
```

---

## Project Structure

```
├── model.py                  # CNN, Transformer, OCRModel definitions
├── app.py                    # Streamlit demo app
├── ocr_train.py                  # Training loop with AMP, epoch batching
├── retrain_oxf/              # Saved model checkpoints
└── README.md
```

---

## Training Journey

This project went through several distinct stages before arriving at a working model:

**Stage 1 — General OCR pretraining**
Trained on the MJSynth/Oxford OCR dataset (~200k word images). The model learned letter shapes but had no exposure to digits or alphanumeric sequences, and performed poorly on real plates due to the synthetic-to-real domain gap.

**Stage 2 — Synthetic plate fine-tuning**
Generated 50k+ synthetic plate images using Pillow with randomized fonts (Windows system fonts), backgrounds, noise, blur, and contrast variation. Plates followed the Turkish format `NN LLL NNN` with a configurable space ratio. Despite diverse augmentation, the model still failed to generalize to real photographed plates — confirming the domain gap was the fundamental blocker, not data quantity.

**Stage 3 — Real data collection and labeling**
Built a semi-automated labeling pipeline: YOLO detects the plate in each car image, the annotated frame is displayed, and the user types the plate number. ~2,000 real Turkish plate crops were collected and hand-labeled this way, with EXIF rotation correction and multi-plate detection handling built in.

**Stage 4 — Real-data fine-tuning**
Fine-tuning on 1,800 real plate crops (with heavy augmentation: rotation, color jitter, Gaussian blur, perspective distortion) from the best general pretraining checkpoint produced a dramatic improvement. Val loss dropped from ~4.0 to ~0.26 and char accuracy on real plates reached **0.76+**. Best checkpoint saved at the epoch before val loss began slowly rising.

**Key lessons learned along the way:**
- Synthetic-to-real domain gap is the primary failure mode in OCR, not architecture
- CTC blank collapse (outputting all blanks) is caused by learning rate being too low or model being too deep for the data size — solved by increasing lr and reducing layers during initial training
- Augmentation is more valuable than dataset size when real data is scarce
- Val set must be evaluated sequentially (not with random sampling with replacement) to get a true loss estimate
- Inference transforms must never include random augmentations — only training does

---

## Setup

```bash
pip install torch torchvision ultralytics streamlit pillow pandas numpy
```

Place your YOLO weights and OCR checkpoint at the paths referenced in `app.py`, then:

```bash
streamlit run app.py
```

---

## Model Checkpoints

| Checkpoint | Dataset | Train Loss | Val Loss | Notes |
|---|---|---|---|---|
| `oxf_trplate_00006` | Oxford + Synthetic | 0.8 | 0.99 | Best general text checkpoint |
| `realtrplate_2k_00027` | 2k Real Plates | 0.04 | 0.26 | Best plate checkpoint — use this |

---

## Results

| Metric | Value |
|---|---|
| Char accuracy (val set) | 0.96+ |
| Char accuracy (train set) | ~1.00 |
| Val CTC loss (best checkpoint) | ~0.26 |

---

## Limitations

- Performs best on clean, well-lit, roughly frontal plate crops
- Rare letters (`I`, `O`, `M`, `J`) underrepresented in real training data due to Turkish plate registration rules — may underperform on these
- Does not yet handle skewed, night, or heavily occluded plates
- Single-plate assumption — multi-plate images use highest-confidence detection

---

## Tech Stack

`PyTorch` · `YOLOv11 (Ultralytics)` · `Streamlit` · `Pillow` · `torchvision` · `NumPy` · `pandas`
