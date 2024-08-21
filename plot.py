import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.models import Sequential
from keras.layers import Conv3D, Conv3DTranspose, ConvLSTM2D, BatchNormalization, Cropping3D, Input
from keras.losses import MeanSquaredError

# Define the loss function
mse = MeanSquaredError()


# Build the anomaly detection model
def build_model(input_shape):
    model = Sequential()

    model.add(Input(shape=input_shape))

    # First Conv3D layer
    model.add(Conv3D(128, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='same'))
    model.add(BatchNormalization())

    # Second Conv3D layer
    model.add(Conv3D(64, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='same'))
    model.add(BatchNormalization())

    # ConvLSTM2D layers for capturing temporal features
    model.add(ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())

    # Conv3DTranspose layers for upsampling
    model.add(Conv3DTranspose(128, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv3DTranspose(1, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='same'))

    # Cropping layer to maintain correct sequence length
    model.add(Cropping3D(cropping=((3, 3), (0, 0), (0, 0))))

    return model


# Load and preprocess video or image data
def get_single_test(video_path):
    frames = []
    if os.path.isdir(video_path):
        image_files = sorted(
            [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(('.tif', '.jpg', '.png'))])
        for img_file in image_files:
            frame = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            frame_resized = cv2.resize(frame, (256, 256))  # Resize to 256x256
            frames.append(frame_resized)
    else:
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(frame_gray, (256, 256))  # Resize to 256x256
            frames.append(frame_resized)
        cap.release()

    frames_array = np.array(frames, dtype=np.float32)
    frames_array = np.expand_dims(frames_array, axis=-1)  # Add channel dimension
    return frames_array


# Evaluate the model and generate plots
def evaluate_and_plot(video_path):
    # Initialize the model
    input_shape = (10, 256, 256, 1)
    model = build_model(input_shape)
    model.compile(optimizer='adam', loss=mse)
    print("Model built and compiled.")

    # Load test data
    test = get_single_test(video_path)
    print("Test data loaded.")

    # Prepare sequences of 10 frames for prediction
    sz = test.shape[0] - 10
    sequences = np.zeros((sz, 10, 256, 256, 1))

    for i in range(sz):
        sequences[i] = test[i:i + 10]

    print("Sequences prepared.")

    # Predict and compute reconstruction errors
    reconstructed_sequences = model.predict(sequences, batch_size=4)

    sequences_reconstruction_cost = np.array(
        [np.linalg.norm(sequences[i] - reconstructed_sequences[i]) for i in range(sz)])

    # Calculate regularity and anomaly scores
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.ptp(sequences_reconstruction_cost)
    sr = 1.0 - sa

    # Plot regularity score over time
    plt.figure(figsize=(10, 4))
    plt.plot(sr, color='b', linewidth=2.0)
    plt.ylabel('Regularity Score')
    plt.xlabel('Frame Number')
    plt.title('Regularity Score over Time')
    plt.tight_layout(pad=2.0)
    plt.savefig('/content/drive/MyDrive/Anomaly_Detection/Anomaly__Detection/regularity_score_plot.png')
    plt.close()

    # Plot reconstruction error over time
    plt.figure(figsize=(10, 4))
    plt.plot(sa, color='r', linewidth=2.0)
    plt.ylabel('Reconstruction Error')
    plt.xlabel('Frame Number')
    plt.title('Reconstruction Error over Time')
    plt.tight_layout(pad=2.0)
    plt.savefig('/content/drive/MyDrive/Anomaly_Detection/Anomaly__Detection/reconstruction_error_plot.png')
    plt.close()


# Example video path for evaluation
video_path = '/content/drive/MyDrive/Anomaly_Detection/Anomaly__Detection/test_video.mp4'

# Run the evaluation and plotting
evaluate_and_plot(video_path)
