import streamlit as st
from deepface import DeepFace
import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import Image
# Initialize Mediapipe face detection and face mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
# use this background theme: #[theme]
    #base="light"
    #backgroundColor="#0af5f9"

# Define emotions and colors
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_colors = {
    'angry': (0, 0, 255),       # Red
    'disgust': (0, 255, 0),     # Green
    'fear': (255, 0, 0),        # Blue
    'happy': (0, 255, 255),     # Yellow
    'sad': (255, 255, 0),       # Cyan
    'surprise': (255, 0, 255),  # Magenta
    'neutral': (128, 128, 128)  # Gray
}

# Streamlit App
st.title("FacAi")

# ---------- App background gradient animated design -------------------
# Add content to the web app
st.markdown('<div class="content">', unsafe_allow_html=True)

# Custom CSS for gradient background, white text, and dark gray buttons
gradient_css = """
<style>
    .stApp {
        background: linear-gradient(55deg, #6fffe9, #780000, #00b4d8, #ffffff);
        background-size: 300% 300%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% {
            background-position: 0% 70%;
        }
        50% {
            background-position: 100% 70%;
        }
        100% {
            background-position: 0% 70%;
        }
    }
    
    /* Set text color to white for all elements */
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label {
        color: white !important;
    }
    
    /* Ensure input text remains visible */
    .stTextInput>div>div>input {
        color: black !important;
    }
    
    /* Change button color to dark gray */
    .stButton > button {
        background-color: #333333 !important;
        color: white !important;
    }
    
    /* Hover effect for buttons */
    .stButton > button:hover {
        background-color: #444444 !important;
    }
</style>
"""

# Inject custom CSS
st.markdown(gradient_css, unsafe_allow_html=True)

# -------------------- App body with two columns for iamge and text  --------------------------

col1, col2 = st.columns(2)
with col1:
    fc_img = Image.open('face_ai.jpg')
    st.image(fc_img)
with col2:
    st.write(""" 
             MAi is a cutting-edge facial emotion detection app that leverages advanced AI and computer vision technologies 
             to provide real-time emotional analysis. By simply pointing your smartphone's camera at a face, MAi swiftly delivers 
             an in-depth emotional breakdown, revealing nuanced insights into human expressions.
             The app can recognize core emotions like happiness, sadness, anger, surprise, and neutral states with impressive accuracy. 
             Using sophisticated machine learning algorithms, MAi transforms facial expressions into actionable emotional insights, 
             making it a powerful tool for understanding human sentiment in various contexts.
             
             """)

st.write("Click the button below to detect facial expressions.")

def detect_emotion_and_landmarks():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    st_frame = st.empty()  # Placeholder for the video frame

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam. Please check your camera settings.")
            break

        # Convert the frame to RGB for Mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and bounding boxes using Mediapipe
        detection_results = face_detection.process(rgb_frame)
        face_mesh_results = face_mesh.process(rgb_frame)
        face_detected = False
        current_time = time.time()

        if detection_results.detections:
            face_detected = True
            for detection in detection_results.detections:
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Draw corner shapes instead of full square
                corner_length = 30  # Length of each corner line
                color = (0, 255, 255)  # Yellow

                # Top-left corner
                cv2.line(frame, (x, y), (x + corner_length, y), color, 2)
                cv2.line(frame, (x, y), (x, y + corner_length), color, 2)

                # Top-right corner
                cv2.line(frame, (x + width, y), (x + width - corner_length, y), color, 2)
                cv2.line(frame, (x + width, y), (x + width, y + corner_length), color, 2)

                # Bottom-left corner
                cv2.line(frame, (x, y + height), (x + corner_length, y + height), color, 2)
                cv2.line(frame, (x, y + height), (x, y + height - corner_length), color, 2)

                # Bottom-right corner
                cv2.line(frame, (x + width, y + height), (x + width - corner_length, y + height), color, 2)
                cv2.line(frame, (x + width, y + height), (x + width, y + height - corner_length), color, 2)

        # Display geometrical shapes and points for the first 10 seconds
        if face_mesh_results.multi_face_landmarks and (current_time - start_time < 4):
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                # Draw facial landmarks as points
                for landmark in face_landmarks.landmark:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)  # White points

                # Draw connections between landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, 
                    face_landmarks, 
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                )

        try:
            # Analyze emotions using DeepFace
            emotion_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion_scores = emotion_result[0]['emotion']
            dominant_emotion = emotion_result[0]['dominant_emotion']

            # Create color-coded bar chart for emotions
            bar_width = 200
            bar_height = 20
            max_score = max(emotion_scores.values())
            for i, emotion in enumerate(emotions):
                score = emotion_scores[emotion]
                bar_length = int((score / max_score) * bar_width)
                start_point = (10, 50 + i * 30)
                end_point = (10 + bar_length, 50 + i * 30 + bar_height)
                color = emotion_colors[emotion]  # Use the color-coded dictionary
                cv2.rectangle(frame, start_point, end_point, color, -1)
                cv2.putText(frame, f"{emotion}: {score:.2f}", (10 + bar_width + 20, 50 + i * 30 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display dominant emotion
            cv2.putText(frame, f"Detected Dominant Expression: {dominant_emotion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        except Exception as e:
            st.error(f"Error in emotion analysis: {e}")

        # Convert frame to RGB for Streamlit display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(rgb_frame, channels="RGB")

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    st.success("Webcam stopped.")

if st.button("Detect Expression"):
    #st.info("Starting webcam... Press 'q' in the video window to exit detection.")
    detect_emotion_and_landmarks()
