import tensorflow as tf


def create_model():
    """
    Creates and returns the Placement Prediction model.
    """

    model = tf.keras.Sequential([
        # Input Layer (10 features)
        tf.keras.layers.Input(shape=(10,)),

        # Hidden Layer 1
        tf.keras.layers.Dense(16, activation="relu"),

        # Hidden Layer 2
        tf.keras.layers.Dense(8, activation="relu"),

        # Output Layer
        # 2 Classes:
        # 0 -> Not Placed
        # 1 -> Placed
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    return model