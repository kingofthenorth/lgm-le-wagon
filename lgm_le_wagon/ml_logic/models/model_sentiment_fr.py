from colorama import Fore, Style

import time
print(Fore.BLUE + "\nLoading tensorflow..." + Style.RESET_ALL)
start = time.perf_counter()
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from transformers import TFCamembertModel
from tensorflow.keras.optimizers import Adam
end = time.perf_counter()
print(f"\n✅ tensorflow loaded ({round(end - start, 2)} secs)")

from typing import Tuple
import numpy as np



def initialize_model() -> Model:
    """
    Create model with pre-trained layers
    """
    # Load pre-trained model
    transformer_model = TFCamembertModel.from_pretrained('jplu/tf-camembert-base')

    # Define inputs
    entrees_ids = Input(shape=(128,), name='input_token', dtype='int32')
    entrees_masks = Input(shape=(128,), name='masked_token', dtype='int32')

    # Define outputs from pre-trained
    sortie_camemBERT = transformer_model([entrees_ids, entrees_masks])[0]
    l1 = Lambda(lambda seq: seq[:,0,:])(sortie_camemBERT)

    # Add custom layers to tune pre-trained model
    x = Dense(32, activation='relu')(l1)
    x = Dropout(.2)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)

    # Define custom model
    model = Model(inputs=[entrees_ids, entrees_masks], outputs = outputs)

    # Freeze pre-trained layers
    model.layers[2].trainable = False

    model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

    print("\n✅ model trained and compiled")

    return model



def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=32,
                validation_split=0.2) -> Tuple[Model, dict]:
    """
    Fit model and return a tuple (fitted_model, history)
    """

#     print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    es = EarlyStopping(monitor="val_loss",
                       patience=3,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X,
                        y,
                        validation_split=validation_split,
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=1)

    print(f"\n✅ model trained ({len(X)} rows)")

    return model, history


def evaluate_model(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=32) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    """

    print(Fore.BLUE + f"\nEvaluate model on {len(X)} rows..." + Style.RESET_ALL)

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=1,
        return_dict=True)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(accuracy, 2)}")

    return metrics








########################### TAXIFARE #####################################


# def initialize_model(X: np.ndarray) -> Model:
#     """
#     Initialize the Neural Network with random weights
#     """
#     print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

#     reg = regularizers.l1_l2(l2=0.005)

#     model = Sequential()
#     model.add(layers.BatchNormalization(input_shape=X.shape[1:]))
#     model.add(layers.Dense(100, activation="relu", kernel_regularizer=reg))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dropout(rate=0.1))
#     model.add(layers.Dense(50, activation="relu", kernel_regularizer=reg))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dropout(rate=0.1))
#     model.add(layers.Dense(10, activation="relu"))
#     model.add(layers.BatchNormalization(momentum=0.99))  # use momentum=0 for to only use statistic of the last seen minibatch in inference mode ("short memory"). Use 1 to average statistics of all seen batch during training histories.
#     model.add(layers.Dropout(rate=0.1))
#     model.add(layers.Dense(1, activation="linear"))

#     print("\n✅ model initialized")

#     return model


# def compile_model(model: Model, learning_rate: float) -> Model:
#     """
#     Compile the Neural Network
#     """
#     optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#     model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

#     print("\n✅ model compiled")
#     return model


# def train_model(model: Model,
#                 X: np.ndarray,
#                 y: np.ndarray,
#                 batch_size=256,
#                 validation_split=0.3,
#                 validation_data=None) -> Tuple[Model, dict]:
#     """
#     Fit model and return a the tuple (fitted_model, history)
#     """

#     print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

#     es = EarlyStopping(monitor="val_loss",
#                        patience=2,
#                        restore_best_weights=True,
#                        verbose=0)

#     history = model.fit(X,
#                         y,
#                         validation_split=validation_split,
#                         validation_data=validation_data,
#                         epochs=100,
#                         batch_size=batch_size,
#                         callbacks=[es],
#                         verbose=0)

#     print(f"\n✅ model trained ({len(X)} rows)")

#     return model, history


# def evaluate_model(model: Model,
#                    X: np.ndarray,
#                    y: np.ndarray,
#                    batch_size=256) -> Tuple[Model, dict]:
#     """
#     Evaluate trained model performance on dataset
#     """

#     print(Fore.BLUE + f"\nEvaluate model on {len(X)} rows..." + Style.RESET_ALL)

#     metrics = model.evaluate(
#         x=X,
#         y=y,
#         batch_size=batch_size,
#         verbose=1,
#         # callbacks=None,
#         return_dict=True)

#     loss = metrics["loss"]
#     mae = metrics["mae"]

#     print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(mae, 2)}")

#     return metrics
