import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def get_nn(K: int, hidden: np.ndarray) -> keras.Model:
    """
    defines and compiles model according to number K of neighbors and model configuration hidden
    Args:
        K (int): number of nearest neighbors in the model's input
        hidden (np.ndarray): model configuration

    Returns: model compiled with Adam and Mean Squared Error

    """
    model = keras.Sequential(name="ann")
    model.add(layers.Input(shape=(2 * K + 1,), name="input"))
    for _, hidden_size in enumerate(hidden):
        model.add(layers.Dense(hidden_size, activation="sigmoid"))
    model.add(layers.Dense(1, activation="linear", name="output"))
    model.compile(
        optimizer="adam",
        loss="mse",
    )
    return model
