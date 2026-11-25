
import os
import sys
import cv2
import av
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode
)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(0, BASE_DIR)

from src.hand_detector import HandDetector
from src.feature_extractor import FeatureExtractor
from src.classifier import SignClassifier
from src.prediction_buffer import PredictionBuffer


# =====================================================
# PATHS
# =====================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sign_classifier.pkl"
)

COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.pkl"
)

HAND_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)


# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="SignVision AI",
    page_icon="🤟",
    layout="wide"
)


# =====================================================
# DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background:
    linear-gradient(
        135deg,
        #07111f 0%,
        #101c35 50%,
        #07111f 100%
    );
}

.title {
    text-align:center;
    font-size:50px;
    font-weight:800;
    color:white;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#94a3b8;
    margin-bottom:30px;
}

.card {
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.10);
    border-radius:20px;
    padding:20px;
}

.status {
    color:#22c55e;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="title">🤟 SignVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Real-Time Sign Language Recognition using
    Computer Vision & Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# HAND CONNECTIONS
# =====================================================

HAND_CONNECTIONS = [

    (0,1),(1,2),(2,3),(3,4),

    (0,5),(5,6),(6,7),(7,8),

    (5,9),(9,10),(10,11),(11,12),

    (9,13),(13,14),(14,15),(15,16),

    (13,17),(17,18),(18,19),(19,20),

    (0,17)
]


# =====================================================
# VIDEO PROCESSOR
# =====================================================

class SignProcessor(VideoProcessorBase):

    def __init__(self):

        self.detector = HandDetector(
            HAND_MODEL_PATH
        )

        self.classifier = SignClassifier(
            MODEL_PATH,
            COLUMNS_PATH,
            confidence_threshold=0.40
        )

        self.buffer = PredictionBuffer(8)


    def draw_hand(
        self,
        image,
        landmarks
    ):

        h, w, _ = image.shape

        points = []

        for landmark in landmarks:

            point = (
                int(landmark.x * w),
                int(landmark.y * h)
            )

            points.append(point)

        # Skeleton

        for start, end in HAND_CONNECTIONS:

            cv2.line(
                image,
                points[start],
                points[end],
                (0,255,0),
                3
            )

        # Landmark points

        for point in points:

            cv2.circle(
                image,
                point,
                5,
                (255,0,255),
                -1
            )


    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        image = cv2.flip(
            image,
            1
        )

        result = self.detector.detect(
            image
        )

        if result.hand_landmarks:

            landmarks = (
                result.hand_landmarks[0]
            )

            self.draw_hand(
                image,
                landmarks
            )

            features = (
                FeatureExtractor.extract(
                    landmarks
                )
            )

            sign, confidence = (
                self.classifier.predict(
                    features
                )
            )

            self.buffer.add(sign)

            stable = (
                self.buffer.get_stable()
            )

            display_sign = (
                stable
                if stable
                else sign
            )

            # Overlay

            cv2.rectangle(
                image,
                (15,15),
                (355,125),
                (15,23,42),
                -1
            )

            cv2.putText(
                image,
                f"SIGN: {display_sign}",
                (30,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0,255,0),
                3
            )

            cv2.putText(
                image,
                f"Confidence: {confidence*100:.1f}%",
                (30,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

        else:

            self.buffer.clear()

            cv2.putText(
                image,
                "Show your hand",
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# =====================================================
# MAIN DASHBOARD
# =====================================================

camera_col, info_col = st.columns(
    [2,1]
)


with camera_col:

    st.subheader("📷 Live Recognition")

    webrtc_ctx = webrtc_streamer(
        key="signvision",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=signProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal":640},
                "height": {"ideal":480},
                "framerate": {"ideal":20}
            }.
            "audio": False
        },
        rtc_configuration={
            "iceServers":[
                {
                    "urls":[
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },
        async_processing=True
    )


with info_col:

    st.subheader("🧠 AI Recognition")

    st.success(
        "System Ready"
    )

    st.markdown("""
### Supported Signs

**A · B · C · D · E · F**

**I · L · V · W · Y**

---

### How it works

**1.** Camera captures your hand

**2.** MediaPipe detects 21 landmarks

**3.** 63 normalized features are generated

**4.** Random Forest classifies the gesture

**5.** Prediction stabilization reduces flickering

---

### Model

**Algorithm:** Random Forest

**Input:** 63 landmark features

**Classes:** 11

**Dataset:** 550 samples
""")


# =====================================================
# PIPELINE
# =====================================================

st.divider()

st.subheader("⚙️ Recognition Pipeline")

st.markdown("""
**Webcam → MediaPipe → 21 Hand Landmarks → Feature
Normalization → Random Forest → Prediction**
""")


st.divider()

st.markdown(
    """
    <div style="
    text-align:center;
    color:#64748b;
    ">
    SignVision AI • Computer Vision Project
    </div>
    """,
    unsafe_allow_html=True
)
