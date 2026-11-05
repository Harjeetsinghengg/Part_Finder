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

    image_np = np.array(image)

    display_img = image_np.copy()

    ocr = RapidOCR()

    with st.spinner(
        "Reading build code..."
    ):

        result, _ = ocr(image_np)

    detected_matches = []

    if result:

        for box, text, score in result:

            clean_text = norm(text)

            # OCR common mistakes
            clean_text = (
                clean_text
                .replace("O", "0")
                .replace("I", "1")
                .replace("S", "5")
            )

            if len(clean_text) < 18:
                continue

            pts = np.array(
                box,
                dtype=np.int32
            )

            x = int(pts[:, 0].min())
            y = int(pts[:, 1].min())

            match = process.extractOne(
                clean_text,
                all_codes
            )

            color = (0, 0, 255)
            label = "UNKNOWN"

            if match:

                matched_code = match[0]
                confidence = match[1]

                if confidence > 75:

                    if matched_code in target_codes:

                        color = (0, 255, 0)
                        label = matched_code

                        detected_matches.append(
                            (
                                matched_code,
                                confidence
                            )
                        )

                    else:

                        color = (0, 165, 255)
                        label = matched_code

            cv2.polylines(
                display_img,
                [pts],
                True,
                color,
                4
            )

            cv2.putText(
                display_img,
                label,
                (
                    x,
                    max(25, y - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    # ====================================================
    # SHOW IMAGE
    # ====================================================

    st.subheader("Detected Build Codes")

    st.image(
        display_img,
        channels="RGB",
        use_container_width=True
    )

    # ====================================================
    # SHOW MATCHES
    # ====================================================

    st.subheader("Match Results")

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
            f"Matched Build Code: {matched_code}"
        )

        st.write(
            f"Confidence: {confidence:.1f}%"
        )

        st.markdown(
            "### Test Configuration"
        )

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

    if not found_match:

        st.error(
            "No matching build code found."
        )

    st.markdown("---")
    st.markdown(
        """
        **Colour Legend**

        🟢 Green = Selected build code match

        🟠 Orange = Build code exists but not in current selection

        🔴 Red = Unknown text
        """
    )
