"""
Rock Paper Scissors vs AI — Hand Gesture Controlled
Built with MediaPipe (hand tracking) + Streamlit + streamlit-webrtc (live webcam in browser)

Run locally:  streamlit run app.py
Deploy free:  Streamlit Community Cloud (share.streamlit.io) or Hugging Face Spaces
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import random
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Rock Paper Scissors — Hand Gesture AI", page_icon="✊", layout="centered")

st.title("✊ ✋ ✌️ Rock Paper Scissors vs AI")
st.caption("Show your hand gesture to the webcam. The AI reads it live using MediaPipe hand tracking.")

# ----------------------------
# MediaPipe setup
# ----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ----------------------------
# Session state for score + game flow
# ----------------------------
if "user_score" not in st.session_state:
    st.session_state.user_score = 0
if "ai_score" not in st.session_state:
    st.session_state.ai_score = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = ""
if "last_user_move" not in st.session_state:
    st.session_state.last_user_move = ""
if "last_ai_move" not in st.session_state:
    st.session_state.last_ai_move = ""

MOVES = ["Rock", "Paper", "Scissors"]


def classify_gesture(hand_landmarks):
    """
    Classify Rock / Paper / Scissors from MediaPipe hand landmarks.
    Logic: check whether each finger is 'extended' (tip above pip joint)
    Thumb handled separately since it moves sideways, not up/down.
    """
    lm = hand_landmarks.landmark

    # Landmark indices for fingertips and lower joints
    tips = [8, 12, 16, 20]   # index, middle, ring, pinky tip
    pips = [6, 10, 14, 18]   # corresponding lower joints

    fingers_up = []
    for tip, pip in zip(tips, pips):
        fingers_up.append(lm[tip].y < lm[pip].y)  # extended if tip is above joint (smaller y)

    # Thumb: compare x position relative to its own joint (works for either hand roughly)
    thumb_up = lm[4].x < lm[3].x if lm[17].x < lm[5].x else lm[4].x > lm[3].x

    total_up = sum(fingers_up) + (1 if thumb_up else 0)

    if total_up <= 1:
        return "Rock"
    elif total_up >= 4:
        return "Paper"
    elif fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
        return "Scissors"
    else:
        return None  # uncertain gesture, ignore this frame


def decide_winner(user_move, ai_move):
    if user_move == ai_move:
        return "Draw"
    beats = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}
    if beats[user_move] == ai_move:
        return "You Win!"
    return "AI Wins!"


# ----------------------------
# Video frame callback
# ----------------------------
class HandGestureProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.detected_move = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        self.detected_move = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                move = classify_gesture(hand_landmarks)
                if move:
                    self.detected_move = move
                    cv2.putText(img, move, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------------------
# Layout: webcam + controls
# ----------------------------
col1, col2 = st.columns([1.3, 1])

with col1:
    ctx = webrtc_streamer(
        key="rps-game",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=HandGestureProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Scoreboard")
    st.metric("You", st.session_state.user_score)
    st.metric("AI", st.session_state.ai_score)

    st.divider()

    if st.button("🎲 Play Round", use_container_width=True, type="primary"):
        if ctx.video_processor and ctx.video_processor.detected_move:
            user_move = ctx.video_processor.detected_move
            ai_move = random.choice(MOVES)
            result = decide_winner(user_move, ai_move)

            st.session_state.last_user_move = user_move
            st.session_state.last_ai_move = ai_move
            st.session_state.last_result = result

            if result == "You Win!":
                st.session_state.user_score += 1
            elif result == "AI Wins!":
                st.session_state.ai_score += 1
        else:
            st.session_state.last_result = "No clear gesture detected — try again!"

    if st.session_state.last_result:
        st.divider()
        st.write(f"**You showed:** {st.session_state.last_user_move}")
        st.write(f"**AI showed:** {st.session_state.last_ai_move}")
        st.subheader(st.session_state.last_result)

    if st.button("Reset Score", use_container_width=True):
        st.session_state.user_score = 0
        st.session_state.ai_score = 0
        st.session_state.last_result = ""

st.divider()
with st.expander("How it works (for your Instagram caption)"):
    st.markdown("""
    1. **MediaPipe Hands** detects 21 landmark points on your hand in real time.
    2. We check which fingers are *extended* (fingertip above its lower joint) to classify the gesture.
    3. **Rock** = no fingers up, **Paper** = all fingers up, **Scissors** = only index + middle fingers up.
    4. The AI picks randomly, then standard RPS rules decide the winner.
    5. Everything runs live in the browser via `streamlit-webrtc` — no server-side video storage.
    """)
