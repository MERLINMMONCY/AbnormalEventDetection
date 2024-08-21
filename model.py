import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ConvLSTM2D, Conv2DTranspose, LayerNormalization, TimeDistributed, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

def load_model(time_length=10):
    inputs = Input(shape=(time_length, 256, 256, 1))

    # First convolutional block
    x = TimeDistributed(Conv2D(128, 11, strides=4, padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('elu'))(x)

    # Second convolutional block
    x = TimeDistributed(Conv2D(64, 5, strides=2, padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('elu'))(x)

    # ConvLSTM blocks
    x = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(x)
    x = LayerNormalization()(x)
    x = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(x)
    x = LayerNormalization()(x)
    x = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(x)
    x = LayerNormalization()(x)

    # Deconvolutional blocks
    x = TimeDistributed(Conv2DTranspose(128, (5, 5), padding='same', strides=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('elu'))(x)

    outputs = TimeDistributed(Conv2DTranspose(1, (11, 11), padding='same', strides=4))(x)

    model = Model(inputs, outputs)

    optimizer = Adam(learning_rate=1e-4, epsilon=1e-6)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    return model

def active_learning_loop(model, X_train, Y_train, X_unlabeled, iterations=10, query_size=5):
    """
    Implements the active learning loop.
    """
    for iteration in range(iterations):
        print(f"Active Learning Iteration {iteration + 1}/{iterations}")

        # Train the model on the current training data
        model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1)

        # Predict on the unlabeled data
        predictions = model.predict(X_unlabeled)

        # Compute uncertainty and select the most uncertain samples
        uncertainties = np.var(predictions, axis=-1).mean(axis=(1, 2, 3, 4))
        query_indices = uncertainties.argsort()[-query_size:]

        # Simulate labeling and update the training set
        queried_X = X_unlabeled[query_indices]
        queried_Y = obtain_labels(queried_X)
        X_train = np.concatenate([X_train, queried_X])
        Y_train = np.concatenate([Y_train, queried_Y])

        # Remove the queried samples from the unlabeled pool
        X_unlabeled = np.delete(X_unlabeled, query_indices, axis=0)

    return model

def obtain_labels(queried_X):
    """
    Simulates the labeling process by returning random labels.
    """
    return np.random.rand(*queried_X.shape)
