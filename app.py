import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from rapidocr_onnxruntime import RapidOCR

CODE_LEN = 20          # build codes are 20 characters
MAX_IMG_SIDE = 1600    # resize big photos so OCR stays fast

st.set_page_config(page_title="Build Code Scanner", layout="wide")
st.title("🔍 Build Code Scanner")


# ====================================================== helpers
def clean(text):
    """Upper-case, letters and digits only. Used for the real code."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def fold(text):
    """Used ONLY to compare OCR text with codes: O/0 and I/1 look alike to OCR."""
    return clean(text).replace("O", "0").replace("I", "1")


@st.cache_resource
def get_ocr():
    return RapidOCR()          # load the OCR model once, not on every rerun


@st.cache_data
def load_data(file):
    df = pd.read_excel(file, dtype=str, engine="openpyxl").fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ====================================================== sidebar
st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Excel file (optional)", type=["xlsx"])
max_err = st.sidebar.slider(
    "Max OCR character errors allowed", 0, 3, 1,
    help="0 = code must be read perfectly. 1 = forgives one misread character.")
show_other = st.sidebar.checkbox("Also outline NON-selected codes in red", value=False)
input_mode = st.sidebar.radio("Photo source", ["Camera", "Upload / phone camera (sharper)"])

# ====================================================== load Excel
default_file = Path(__file__).parent / "codes.xlsx"
if uploaded:
    df = load_data(uploaded)
elif default_file.exists():
    df = load_data(default_file)
else:
    st.error("No Excel file found. Upload one in the sidebar or put codes.xlsx next to app.py.")
    st.stop()

if df.shape[1] < 27:
    st.error("Excel needs columns A-F (description), G-Z (positions) and AA (full code).")
    st.stop()

cols = list(df.columns)
desc_cols = cols[0:6]      # A-F
code_col = cols[26]        # AA

df["_code"] = df[code_col].map(clean)          # real code (no O/I changes)
df = df[df["_code"] != ""].copy()
df["_fold"] = df["_code"].map(fold)            # comparison version
df["_desc"] = df[desc_cols].apply(
    lambda r: " | ".join(str(v).strip() for v in r if str(v).strip()), axis=1)

# ====================================================== dropdowns
st.subheader("1. Select build code characters (leave blank = any)")
selected = {}
PER_ROW = 5
for start in range(0, CODE_LEN, PER_ROW):
    row_cols = st.columns(PER_ROW)          # a new row of columns for each group
    for j, i in enumerate(range(start, min(start + PER_ROW, CODE_LEN))):
        values = sorted({c[i] for c in df["_code"] if len(c) > i})
        with row_cols[j]:
            selected[i] = st.selectbox(f"Pos {i + 1}", [""] + values, key=f"pos_{i}")

any_selected = any(selected.values())

mask = pd.Series(True, index=df.index)
for i, v in selected.items():
    if v:
        # compare with the REAL code, so choosing "I" or "O" works
        mask &= df["_code"].str[i] == v
filtered = df[mask]

st.subheader(f"2. Intended build codes ({len(filtered)})")
st.dataframe(
    filtered[[code_col, "_desc"]].rename(columns={"_desc": "Description"}),
    use_container_width=True, hide_index=True)

# ====================================================== lookup tables
fold_to_row = {}
for idx, f in df["_fold"].items():
    fold_to_row.setdefault(f, idx)
fold_list = list(fold_to_row.keys())                       # ALL known codes
target_folds = set(filtered["_fold"]) if any_selected else set()


def candidates(text):
    """Possible 20-char code strings inside one OCR text line."""
    t = clean(text)
    n = len(t)
    if n < CODE_LEN - 2:
        return []
    if n <= CODE_LEN + 2:
        return [t]
    return [t[i:i + CODE_LEN] for i in range(n - CODE_LEN + 1)]


def match_code(text):
    """Return (folded known code, errors) or None.
    Matches against ALL codes and accepts only a clear, unique best match."""
    best = None
    for cand in candidates(text):
        f = cand.replace("O", "0").replace("I", "1")
        res = process.extract(f, fold_list, scorer=Levenshtein.distance, limit=2)
        if not res:
            continue
        d1 = res[0][1]
        d2 = res[1][1] if len(res) > 1 else 99
        if d1 <= max_err and (d1 == 0 or d2 > d1):
            if best is None or d1 < best[1]:
                best = (res[0][0], d1)
    return best


# ====================================================== photo
st.subheader("3. Scan")
if not any_selected:
    st.info("Pick at least one character above. Nothing is highlighted green until you do.")

if input_mode == "Camera":
    photo = st.camera_input("Take photo")
else:
    photo = st.file_uploader("Take or choose a photo", type=["jpg", "jpeg", "png"])

if photo:
    img = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
    scale = MAX_IMG_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    arr = np.array(img)
    display = arr.copy()
    font = max(0.6, arr.shape[1] / 1600)

    with st.spinner("Reading text..."):
        result, _ = get_ocr()(arr)

    hits = {}          # folded code -> errors
    seen_other = False

    for box, text, _score in (result or []):
        m = match_code(text)
        if not m:
            continue
        f, errs = m
        ok = f in target_folds
        pts = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        x, y = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min())

        if ok:
            hits[f] = min(errs, hits.get(f, 99))
            cv2.polylines(display, [pts], True, (0, 0, 200), 6)
            # cv2.putText(display, df.loc[fold_to_row[f], "_code"], (x, max(25, y - 8)),
            #            cv2.FONT_HERSHEY_SIMPLEX, font, (0, 200, 0), 2)
        else:
            seen_other = True
            if show_other:
                cv2.polylines(display, [pts], True, (220, 0, 0), 2)

    st.subheader("Detected build codes")
    st.image(display, use_container_width=True)

    if hits:
        rows = []
        for f, errs in hits.items():
            r = df.loc[fold_to_row[f]]
            rows.append({
                "Build code": r["_code"],
                "Description": r["_desc"],
                "Read accuracy": f"{100 * (1 - errs / CODE_LEN):.0f}%",
            })
        st.success(f"{len(rows)} selected build code(s) found")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif seen_other:
        st.warning("Known build codes were seen, but none match your selection.")
    else:
        st.error("No known build code read. Move closer, avoid glare, keep text sharp.")

    with st.expander("OCR debug: what the camera read"):
        for _b, t, s in (result or []):
            st.write(f"{t}  (score {float(s):.2f})")
