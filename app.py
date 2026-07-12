
import uuid
import sqlite3
import streamlit as st
from model import *
import tempfile
from PIL import ImageOps
from ultralytics import YOLO

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plate Reader",
    page_icon="🚗",
    layout="centered",
)

# ── minimal dark styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0f0f0f; color: #e8e8e8; }

    h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        letter-spacing: -0.02em;
        color: #f0f0f0;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 2rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .plate-result {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-left: 3px solid #f5c518;
        border-radius: 6px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.5rem;
    }
    .plate-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #666;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 0.4rem;
    }
    .plate-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #f5c518;
        letter-spacing: 0.15em;
    }
    .error-box {
        background: #1a0f0f;
        border: 1px solid #3a1a1a;
        border-left: 3px solid #e74c3c;
        border-radius: 6px;
        padding: 1rem 1.4rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #e74c3c;
        font-family: 'JetBrains Mono', monospace;
    }
    .info-box {
        background: #0f1a1a;
        border: 1px solid #1a3a3a;
        border-left: 3px solid #2ecc71;
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.8rem;
        color: #2ecc71;
        font-family: 'JetBrains Mono', monospace;
    }
    div[data-testid="stFileUploader"] {
        background: #141414;
        border: 1px dashed #333;
        border-radius: 8px;
        padding: 1rem;
    }
    div[data-testid="stFileUploader"]:hover { border-color: #555; }
    img { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>🚗 Plate Reader</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">YOLO detection · CNN→Transformer→CTC recognition</p>', unsafe_allow_html=True)

# ── model loading ─────────────────────────────────────────────────────────────
YOLO_PATH = "models/best.pt"
OCR_PATH  = "models/best_model.pth"
img_save_path = "detected_plates"

@st.cache_resource
def load_models():
    from model import OCRModel, device
    yolo = YOLO(YOLO_PATH)
    ocr  = OCRModel().to(device)
    ocr.load_state_dict(torch.load(OCR_PATH, map_location=device))
    ocr.eval()
    return yolo, ocr

@st.cache_resource
def get_transform():
    return v2.Compose([
        v2.Resize((64, 128)),
        v2.Grayscale(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

@st.cache_resource
def get_vocab():
    chars    = ['<blank>'] + list(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    idx2char = {i: c for i, c in enumerate(chars)}
    return idx2char

def greedy_decode(logits, idx2char):
    pred_ids = logits.argmax(dim=-1).squeeze(0)
    result, prev = [], None
    for idx in pred_ids:
        idx = idx.item()
        if idx != 0 and idx != prev:
            result.append(idx2char[idx])
        prev = idx
    return ''.join(result)

# ── file upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a car photo",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file:
    # show original
    pil_img = ImageOps.exif_transpose(Image.open(uploaded_file).convert("RGB"))
    unique_id = str(uuid.uuid4().hex)
    curr_image_save_path = os.path.join(img_save_path,f"{unique_id}.jpg")
    pil_img.save(curr_image_save_path)
    st.image(pil_img, caption="Uploaded image", use_container_width=True)

    with st.spinner("Detecting plate…"):
        try:
            yolo_model, ocr_model = load_models()
            transform = get_transform()
            idx2char  = get_vocab()
            from model import device
        except Exception as e:
            st.markdown(f'<div class="error-box">⚠ Model load failed: {e}</div>', unsafe_allow_html=True)
            st.stop()

        # run yolo on PIL image directly
        results = yolo_model(pil_img)

    if len(results[0].boxes) == 0:
        st.markdown('<div class="error-box">⚠ No plate detected in this image.</div>', unsafe_allow_html=True)
        st.stop()

    if len(results[0].boxes) > 1:
        st.markdown('<div class="info-box">ℹ Multiple plates detected — using the highest-confidence one.</div>', unsafe_allow_html=True)

    # pick highest confidence box
    boxes = results[0].boxes
    best  = boxes.conf.argmax().item()
    best_conf_score = boxes[best].conf.item()
    x1, y1, x2, y2 = boxes.xyxy[best].int().tolist()

    # crop plate
    img_np = np.array(pil_img)
    crop   = img_np[y1:y2, x1:x2]
    crop_pil = Image.fromarray(crop).convert("RGB")

    # show annotated frame and crop side by side
    col1, col2 , ...?= st.columns(2) # this is not a  fixed number now.
    with col1:
        annotated = results[0].plot()
        st.image(annotated, caption="Detection", use_container_width=True)
    with col2:
        st.image(crop_pil, caption="Plate crop", use_container_width=True)
    
    # save crop to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        crop_pil.save(tmp_path)

    # run OCR
    with st.spinner("Reading plate…"):
        try:
            plate_img = Image.open(tmp_path).convert("L")
            tensor    = transform(plate_img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = ocr_model(tensor, None)

            prediction = greedy_decode(logits, idx2char).strip().upper()
        except Exception as e:
            st.markdown(f'<div class="error-box">⚠ OCR failed: {e}</div>', unsafe_allow_html=True)
            st.stop()
        finally:
            os.unlink(tmp_path)  # clean up temp file

    # result
    if prediction:
        st.markdown(f"""
        <div class="plate-result">
            <div class="plate-label">Detected plate number</div>
            <div class="plate-text">{prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        try:
            with sqlite3.connect("plates.db") as conn:
                conn.execute("""
                    INSERT INTO plates(plate_number, confidence_score, image_path, source)
                    VALUES (?, ?, ?, ?)
                """, (prediction, best_conf_score, curr_image_save_path, 'upload'))
                conn.commit()
            st.markdown('<div class="info-box">✓ Saved to database</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error-box">⚠ DB error: {e}</div>', unsafe_allow_html=True)
            conn.close()
    else:
        st.markdown('<div class="error-box">⚠ OCR returned empty result — plate may be unreadable.</div>', unsafe_allow_html=True)
