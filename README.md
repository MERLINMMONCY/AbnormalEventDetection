# Real-Time Dynamic Anomaly Detection in Video Surveillance Systems

This repository contains the code for my thesis project, **"Real-Time Dynamic Anomaly Detection in Video Surveillance Systems"**. The project aims to improve surveillance efficiency by detecting unusual events in real-time using a deep learning model that combines 2D Convolutional and ConvLSTM layers.

## Dataset

The project uses the **UCSD Ped1 and Ped2 datasets** for training and evaluation, providing a mix of normal and abnormal activities in pedestrian areas.

- Dataset link: [UCSD Ped1 and Ped2 Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)

## Setup and Running the Code

To use this project, follow the steps below:

1. **Preprocess the Data**: 
   - Run `pre_process.py` to preprocess the video frames, which includes tasks like grayscale conversion, normalization, and grouping frames into sequences.
   
2. **Train the Model**: 
   - Execute `train.py` to train the model on the preprocessed data. The training code uses 2D Convolutional layers and ConvLSTM to learn temporal patterns in the video data.

3. **Test the Model**: 
   - Use `test.py` to evaluate the model's performance on the test dataset.

4. **Evaluate the Results**: 
   - Run `evaluate.py` to calculate performance metrics and analyze the model's effectiveness.

5. **Plot Results**:
   - The `plot.py` file provides visualization for the results, showing detection accuracy and regularity scores over time.

6. **Real-Time Anomaly Detection**:
   - Run `app.py` using Streamlit to perform real-time anomaly detection on uploaded videos. The app provides real-time plotting of regularity scores for each frame, making it easy to identify anomalies in video streams.


## Acknowledgments

This project builds upon research in video anomaly detection, leveraging insights from ConvLSTM architectures to effectively capture spatial-temporal patterns in surveillance data. Special thanks to open-source contributors for providing foundational frameworks and resources that supported this implementation.

   
