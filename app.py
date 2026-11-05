import re
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from rapidfuzz import process

# ========================================================
# PAGE
# ========================================================

st.set_page_config(
    page_title="Build Code Scanner",
    layout="wide"
)

st.title("🔍 Build Code Scanner")

# ========================================================
# HELPERS
# ========================================================

def norm(text):

    s = re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper()
    )

    s = (
        s.replace("O", "0")
         .replace("I", "1")
    )

    return s


# ========================================================
# LOAD EXCEL
# ========================================================

@st.cache_data
def load_data(file):

    df = pd.read_excel(
        file,
        dtype=str,
        engine="openpyxl"
    ).fillna("")

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


uploaded = st.sidebar.file_uploader(
    "Excel File",
    type=["xlsx"]
)

if uploaded:
    df = load_data(uploaded)
else:
    df = load_data("codes.xlsx")


# ========================================================
# COLUMN MAPPING
# ========================================================

cols = list(df.columns)

desc_cols = cols[0:6]      # A-F
pos_cols = cols[6:26]      # G-Z
code_col = cols[26]        # AA

df["_code"] = df[code_col].apply(norm)

df["_description"] = (
    df[desc_cols]
    .astype(str)
    .agg(" | ".join, axis=1)
)


# ========================================================
# DROPDOWNS
# ========================================================

st.subheader("1. Select Build Code Characters")

selected = {}

grid = st.columns(5)

for i, col in enumerate(pos_cols):

    values = sorted(
        [
            v
            for v in df[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
            if v
        ]
    )

    options = [""] + values

    with grid[i % 5]:

        selected[col] = st.selectbox(
            f"Pos {i+1}",
            options,
            key=col
        )


# ========================================================
# FILTER USING ACTUAL POSITION IN BUILD CODE
# ========================================================

filtered = df.copy()

for idx, col in enumerate(pos_cols):

    value = selected[col]

    if value:

        filtered = filtered[
            filtered["_code"].str[idx] == value
        ]


# ========================================================
# SHOW POSSIBLE CODES
# ========================================================

st.subheader(
    f"2. Intended Build Codes ({len(filtered)})"
)

st.dataframe(
    filtered[
        [code_col, "_description"]
    ].rename(
        columns={
            "_description": "Description"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# ========================================================
# BUILD CODE LISTS
# ========================================================

target_codes = filtered["_code"].tolist()

all_codes = df["_code"].tolist()


# ========================================================
# CAMERA
# ========================================================

st.subheader("3. Scan Build Code")

photo = st.camera_input(
    "Take Photo"
)


# ========================================================
# OCR
# ========================================================

if photo:

    image = Image.open(photo)

    image_np = np.array(image)

    display_img = image_np.copy()

    ocr = RapidOCR()

    with st.spinner(
        "Scanning build code..."
    ):

        result, _ = ocr(image_np)

    detected_matches = []

    if result:

        for box, text, score in result:

            clean_text = norm(text)

            # Build code must be exactly 20 chars
            if len(clean_text) != 20:
                continue

            search_pool = (
                target_codes
                if len(target_codes) > 0
                else all_codes
            )

            match = process.extractOne(
                clean_text,
                search_pool
            )

            if not match:
                continue

            matched_code = match[0]
            confidence = match[1]

            if confidence < 85:
                continue

            pts = np.array(
                box,
                dtype=np.int32
            )

            # GREEN RECTANGLE ONLY
            cv2.polylines(
                display_img,
                [pts],
                True,
                (0, 255, 0),
                6
            )

            detected_matches.append(
                (
                    matched_code,
                    confidence
                )
            )

    # ====================================================
    # IMAGE WITH RECTANGLES
    # ====================================================

    st.subheader("Detected Build Codes")

    st.image(
        display_img,
        channels="RGB",
        use_container_width=True
    )

    # ====================================================
    # RESULTS
    # ====================================================

    shown = set()

    found_match = False

    for matched_code, confidence in detected_matches:

        if matched_code in shown:
            continue

        shown.add(matched_code)

        found_match = True

        row = df[
            df["_code"] == matched_code
        ].iloc[0]

        st.success(
            f"Matched Build Code: {matched_code} | Confidence: {confidence:.1f}%"
        )

        st.markdown(
            "### Test Configuration"
        )

        for col in desc_cols:

            st.write(
                f"**{col}:** {row[col]}"
            )

        st.info(
            row["_description"]
        )

    if not found_match:

        st.error(
            "No matching build code found."
        )
