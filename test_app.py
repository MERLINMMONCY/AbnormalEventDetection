import numpy as np
import os
from tensorflow.keras.models import load_model
from skimage.transform import resize


def video_to_clips(X_test, t):
    """Convert video frames to clips of length t."""
    sz = X_test.shape[0] - t + 1
    X_test = np.expand_dims(X_test, axis=-1)  # Add channel dimension if not present
    sequences = np.zeros((sz, 227, 227, t, 1))

    for i in range(sz):
        clip = np.zeros((227, 227, t, 1))
        for j in range(t):
            frame = X_test[i + j]
            if frame.shape[:2] != (227, 227):  # Resize frame if necessary
                frame = resize(frame, (227, 227), preserve_range=True, anti_aliasing=True)
            if frame.shape[-1] != 1:  # Ensure channel dimension exists
                frame = np.expand_dims(frame, axis=-1)
            clip[:, :, j, :] = frame
        sequences[i] = clip

    return sequences, sz


def t_predict_video(model, X_test, t=10):
    """Predict regularity scores for the video."""
    sequences, sz = video_to_clips(X_test, t)
    reconstructed_sequences = model.predict(sequences)
    sa = np.array([np.linalg.norm(sequences[i] - reconstructed_sequences[i]) for i in range(sz)])
    sa_normalized = (sa - np.min(sa)) / (np.max(sa) - np.min(sa))
    return sa_normalized


def test(test_video, dataset_name, compile_model=True):
    """Test the model on the provided video and return regularity scores."""
    t = 10  # Time length of each clip

    # Paths provided in app.py
    test_video_dir = "C:/Users/hp/Documents/Thesis/Abnormal_Event_Detection-master/Abnormal_Event_Detection-master/output/"
    model_path = "C:/Users/hp/Documents/Thesis/Abnormal_Event_Detection-master/Abnormal_Event_Detection-master/trained_models/AnomalyDetector.h5"

    # Load the model
    model = load_model(model_path, compile=compile_model)

    # Replace video extension with `.npy`
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        if test_video.endswith(ext):
            test_video = test_video.replace(ext, '.npy')
            break

    filepath = os.path.join(test_video_dir, test_video)

    # Ensure the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test video file not found: {filepath}")

    # Load the test video data (in .npy format)
    X_test = np.load(filepath, allow_pickle=True)

    # Calculate and return regularity score
    score_vid = t_predict_video(model, X_test, t)

    return score_vid
