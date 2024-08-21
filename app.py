import os
import cv2
import subprocess
import streamlit as st
import matplotlib.pyplot as plt
from os.path import join
import test_app

# Set the default dataset and test video directory paths
dataset_name = 'UCSD_Ped2'
test_video_dir = "C:/Users/hp/Documents/Thesis/Abnormal_Event_Detection-master/Abnormal_Event_Detection-master/output/"
model_dir = ("C:/Users/hp/Documents/Thesis/Abnormal_Event_Detection-master/Abnormal_Event_Detection-master"
             "/Abnormal_Event_Detection/output/AnomalyDetector_UCSD_Ped2_tds.h5")


# Convert video files to mp4 format
def convert_to_mp4(video_path):
    """Convert a video file to mp4 format using ffmpeg."""
    mp4_path = video_path.rsplit('.', 1)[0] + '.mp4'
    command = f'ffmpeg -i "{video_path}" -vcodec libx264 -crf 23 "{mp4_path}"'
    subprocess.run(command, shell=True)
    return mp4_path


# Streamlit title and header
st.title('Real-Time Dynamic Anomaly Detection in Video Surveillance Systems')
st.header("Test the anomaly detection system by selecting a video from the options on the left.")

# Test video selection
tests = ['Regularity Score']
selected_test = st.sidebar.selectbox("Choose the test:", tests)

test_videos = [f for f in os.listdir(test_video_dir) if join(test_video_dir, f)]
selected_video = st.sidebar.selectbox('Select a test video:', test_videos)

if selected_video:
    col1, col2 = st.columns(2)

    # Convert video to mp4 if necessary
    if selected_video.endswith('.avi'):
        selected_video = convert_to_mp4(os.path.join(test_video_dir, selected_video))
    else:
        selected_video = os.path.join(test_video_dir, selected_video)

    # Display the selected video
    video_file = open(selected_video, 'rb')
    video_bytes = video_file.read()
    col1.video(video_bytes)

    if col1.button('Plot Regularity Score'):
        # Test and plot the regularity score
        regularity_score = test_app.test(selected_video, dataset_name, compile_model=False)

        fig, ax = plt.subplots()
        ax.plot(regularity_score)
        ax.set(xlabel='Frame', ylabel='Regularity Score', title='Regularity Score Over Time')
        ax.grid()

        col2.pyplot(fig)
        col2.write('A high score suggests normal events, while a low score indicates anomalies.')
