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
    return re.sub(
        r'[^A-Z0-9]',
        '',
        str(text).upper()
    )

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

desc_cols = cols[0:6]
pos_cols = cols[6:26]
code_col = cols[26]

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

    options = [""]

    values = (
        df[col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    options.extend(sorted(values))

    with grid[i % 5]:

        selected[col] = st.selectbox(
            col,
            options,
            key=col
        )

# ========================================================
# FILTER DATA
# ========================================================

filtered = df.copy()

for col, value in selected.items():

    if value:

        filtered = filtered[
            filtered[col] == value
        ]

# ========================================================
# SHOW MATCHING CODES
# ========================================================

st.subheader(
    f"2. Matching Build Codes ({len(filtered)})"
)

st.dataframe(
    filtered[
        [code_col, "_description"]
    ].rename(
        columns={
            "_description":
            "Description"
        }
    ),
    use_container_width=True,
    hide_index=True
)

# ========================================================
# TARGET CODES
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

    st.image(
        image,
        caption="Captured Image"
    )

    image_np = np.array(image)

    ocr = RapidOCR()

    with st.spinner(
        "Reading build code..."
    ):

        result, _ = ocr(image_np)

    detected = []

    if result:

        for box, text, score in result:

            clean_text = norm(text)

            if len(clean_text) >= 10:

                detected.append(clean_text)

    st.subheader("Detected Text")

    if detected:

        for txt in detected:

            st.write(txt)

    else:

        st.warning(
            "No build code detected."
        )

    # ====================================================
    # FIND BEST MATCH
    # ====================================================

    st.subheader("Best Match")

    best_match_found = False

    search_pool = (
        target_codes
        if len(target_codes) > 0
        else all_codes
    )

    for text in detected:

        match = process.extractOne(
            text,
            search_pool
        )

        if match:

            matched_code = match[0]
            confidence = match[1]

            if confidence > 75:

                best_match_found = True

                st.success(
                    f"Matched Build Code: {matched_code}"
                )

                st.write(
                    f"Confidence: {confidence:.1f}%"
                )

                row = df[
                    df["_code"]
                    == matched_code
                ].iloc[0]

                st.markdown("### Test Configuration")

                st.write(
                    f"Vehicle: {row[desc_cols[0]]}"
                )

                st.write(
                    f"Front/Rear: {row[desc_cols[1]]}"
                )

                st.write(
                    f"RH/LH: {row[desc_cols[2]]}"
                )

                st.write(
                    f"Disc Size: {row[desc_cols[3]]}"
                )

                st.write(
                    f"Caliper Colour: {row[desc_cols[4]]}"
                )

                st.write(
                    f"Extra Features: {row[desc_cols[5]]}"
                )

                st.info(
                    row["_description"]
                )

    if not best_match_found:

        st.error(
            "No matching build code found."
        )
