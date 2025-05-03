import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
import datetime

# Load Haar Cascade
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# App UI
st.set_page_config(page_title="Face Detection App")
st.title("📸 Real-time Face Detection with Streamlit")
st.markdown("""
Welcome! This app detects faces in real-time using your webcam and the Viola-Jones algorithm (Haar Cascade).

**Instructions:**
- Use the sliders to tune face detection settings.
- Pick a color for face rectangle.
- Press the save button to download the current frame with detected faces.
""")

# Sidebar controls
st.sidebar.header("🎛️ Detection Settings")
scaleFactor = st.sidebar.slider("Scale Factor", 1.1, 2.0, 1.1, 0.1)
minNeighbors = st.sidebar.slider("Min Neighbors", 1, 10, 3, 1)
rect_color_hex = st.sidebar.color_picker("Rectangle Color", "#00FF00")

# Convert hex to BGR
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

bgr_color = hex_to_bgr(rect_color_hex)

# For saving image
captured_frame = st.empty()
save_button_clicked = st.button("📸 Save Current Frame")

# Global frame store
global_frame = {"image": None}

# Custom video processor class
class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        for (x, y, w, h) in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), bgr_color, 2)

        # Store current frame
        global_frame["image"] = frm.copy()
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Stream from webcam
webrtc_streamer(
    key="key",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)

# Handle save button
if save_button_clicked and global_frame["image"] is not None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"face_capture_{timestamp}.png"
    cv2.imwrite(filename, global_frame["image"])
    st.success(f"Saved image as {filename}")
    st.image(cv2.cvtColor(global_frame["image"], cv2.COLOR_BGR2RGB), caption="Captured Frame", use_column_width=True)
