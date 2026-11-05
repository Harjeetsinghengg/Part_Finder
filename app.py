import re
import threading
import time
import difflib

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

try:
    from pyzbar import pyzbar
except Exception:
    pyzbar = None
from rapidocr_onnxruntime import RapidOCR

st.set_page_config(page_title="Build Code Scanner", layout="wide")

# ------------------------------------------------------------------ data
@st.cache_data
def load_data(file):
    df = pd.read_excel(file, dtype=str).fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


st.title("🔍 Build Code Scanner")

uploaded = st.sidebar.file_uploader("Excel file (optional)", type=["xlsx"])
df = load_data(uploaded if uploaded else "codes.xlsx")

cols = list(df.columns)
desc_cols = cols[0:6]        # A..F  -> description
pos_cols = cols[6:26]        # G..Z  -> character positions 1..20
code_col = cols[26]          # AA    -> full code

df["_code"] = df[code_col].map(norm)
df["_desc"] = df[desc_cols].apply(
    lambda r: " | ".join(v.strip() for v in r if v.strip()), axis=1
)

# ------------------------------------------------------------ selections
st.subheader("1. Choose characters (leave blank for any)")
selected = {}
grid = st.columns(5)
for i, c in enumerate(pos_cols):
    options = [""] + sorted(v for v in df[c].unique() if v != "")
    with grid[i % 5]:
        selected[c] = st.selectbox(f"Position {c}", options, key=f"sel_{c}")

mask = pd.Series(True, index=df.index)
for c, v in selected.items():
    if v:
        mask &= df[c] == v
filtered = df[mask]

st.subheader(f"2. Intended build codes ({len(filtered)})")
st.dataframe(filtered[[code_col, "_desc"]].rename(columns={"_desc": "Description"}),
             use_container_width=True, hide_index=True)

target = dict(zip(filtered["_code"], filtered["_desc"]))   # selected
known = dict(zip(df["_code"], df["_desc"]))                # everything

# --------------------------------------------------------------- scanner
class Scanner(VideoProcessorBase):
    def __init__(self):
        self.ocr = RapidOCR()
        self.lock = threading.Lock()
        self.target, self.known = {}, {}
        self.latest = None
        self.dets = []          # (box, code, ok, timestamp)
        self.found = {}         # code -> description (selected only)
        self.running = True
        threading.Thread(target=self.worker, daemon=True).start()

    def update(self, target, known):
        with self.lock:
            self.target, self.known = target, known

    def match(self, text):
        t = norm(text)
        with self.lock:
            known, target = dict(self.known), dict(self.target)
        for code in known:
            if code and code in t:
                return code, code in target
        if len(t) >= 16:
            for code in known:
                if code and difflib.SequenceMatcher(None, t, code).ratio() >= 0.9:
                    return code, code in target
        return None, False

    def worker(self):
        while self.running:
            frame = self.latest
            if frame is None:
                time.sleep(0.05)
                continue
            self.latest = None
            h, w = frame.shape[:2]
            scale = 960 / max(h, w) if max(h, w) > 960 else 1.0
            small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame
            new = []
            try:  # barcodes
                if pyzbar:
                    for b in pyzbar.decode(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)):
                        code, ok = self.match(b.data.decode(errors="ignore"))
                        if code:
                            x, y, bw, bh = b.rect
                            box = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]) / scale
                            new.append((box.astype(int), code, ok, time.time()))
                res, _ = self.ocr(small)   # printed text
                for box, text, _score in (res or []):
                    code, ok = self.match(text)
                    if code:
                        new.append((np.array(box) / scale, code, ok, time.time()))
            except Exception:
                pass
            with self.lock:
                self.dets = new
                for _, code, ok, _t in new:
                    if ok:
                        self.found[code] = self.target.get(code, "")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest = img.copy()
        with self.lock:
            dets = [d for d in self.dets if time.time() - d[3] < 1.5]
            known = dict(self.known)
        for box, code, ok, _ in dets:
            color = (0, 200, 0) if ok else (0, 0, 255)
            pts = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, color, 3)
            x, y = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min())
            label = known.get(code, "") if ok else "NOT SELECTED"
            cv2.putText(img, label[:60], (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def on_ended(self):
        self.running = False


st.subheader("3. Scan")
st.caption("Press START and allow camera access. Hold the label steady about 15-30 cm away.")

ctx = webrtc_streamer(
    key="scanner",
    video_processor_factory=Scanner,
    media_stream_constraints={"video": {"facingMode": "environment",
                                        "width": {"ideal": 1280}}, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)

placeholder = st.empty()
if ctx.video_processor:
    while ctx.state.playing:
        ctx.video_processor.update(target, known)
        with ctx.video_processor.lock:
            found = dict(ctx.video_processor.found)
        if found:
            placeholder.dataframe(
                pd.DataFrame({"Build code": found.keys(), "Description": found.values()}),
                use_container_width=True, hide_index=True)
        else:
            placeholder.info("No selected build code detected yet.")
        time.sleep(0.5)
