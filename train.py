import numpy as np
import argparse
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from model import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision

# Enable mixed precision for faster training
mixed_precision.set_global_policy('mixed_float16')

# Optimize TensorFlow threading for performance
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Parse command-line arguments for the number of epochs
parser = argparse.ArgumentParser()
parser.add_argument('n_epochs', type=int)
args = parser.parse_args()

time_length = 10  # Sequence length

print("Loading training data...")
X_train = np.load('/content/drive/MyDrive/Anomaly_Detection/Anomaly__Detection/Processed_Data/train.npy')
frames = X_train.shape[1]
print("Training data loaded, frames shape:", X_train.shape)

# Pad sequences if necessary to match the required time length
if frames < time_length:
    pad_size = time_length - frames
    padding = np.zeros((X_train.shape[0], pad_size, 256, 256, 1), dtype=X_train.dtype)
    X_train = np.concatenate((X_train, padding), axis=1)
    frames = time_length

X_train = X_train[:, :frames, :, :, :]
Y_train = X_train.copy()

print("Training data reshaped, new shape:", X_train.shape)

# Perform train-validation split
if X_train.shape[0] < 5:
    print("Not enough samples to split into train and validation sets.")
    X_val, Y_val = X_train, Y_train
else:
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    print("Train/Test split done: ", X_train.shape, X_val.shape)

# Prepare datasets for training and validation
batch_size = 16

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Custom callback to log metrics during training
class MetricsLogger(Callback):
    def on_train_begin(self, logs=None):
        self.history = {'loss': [], 'val_loss': []}

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))

if __name__ == "__main__":
    print("Loading model...")
    model = load_model(time_length=time_length)
    print("Model loaded")

    # Define callbacks for training
    callback_save = ModelCheckpoint(
        "/content/drive/MyDrive/Anomaly_Detection/Anomaly__Detection/trained_models/AnomalyDetector_UCSD_Ped1_tds.keras",
        monitor="val_loss", save_best_only=True, save_weights_only=False
    )
    callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    metrics_logger = MetricsLogger()

    # Start training
    print("Starting training...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=args.n_epochs,
              callbacks=[callback_save, callback_early_stopping, callback_reduce_lr, metrics_logger])
    print("Training finished")

    # Save the best model as .h5
    model.load_weights("/content/drive/MyDrive/Anomaly_Detection/Anomaly__Detection/trained_models"
                       "/AnomalyDetector_UCSD_Ped1_tds.keras")
    model.save("/content/drive/MyDrive/Anomaly_Detection/Anomaly__Detection/trained_models"
               "/AnomalyDetector_UCSD_Ped1_tds.h5")
    print("Model saved as .h5 file")

    # Save training and validation loss history
    np.save('loss_history.npy', metrics_logger.history['loss'])
    np.save('val_loss_history.npy', metrics_logger.history['val_loss'])
    print("Training and validation loss history saved.")

    # Predict on the validation set and save the results
    print("Predicting on validation set...")
    y_pred = model.predict(val_dataset)
    y_true = np.concatenate([y for _, y in val_dataset], axis=0)

    np.save('y_true.npy', y_true)
    np.save('y_pred.npy', y_pred)
    print("Validation predictions and ground truth saved.")
