from typing import Tuple
import numpy as np
from colorama import Fore, Style

print(Fore.BLUE + "\nLoading tensorflow..." + Style.RESET_ALL)
from tensorflow.keras import Model,Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping




def initialize_model_ooo() -> Model:
    """
    Initialize the Neural Network with random weights
    """
    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

    model = Sequential()
    model.add(layers.Masking())
    model.add(layers.LSTM(20, activation='tanh'))
    model.add(layers.Dense(15, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    print("\n✅ model initialized")

    return model

def compile_model(model:Model) -> Model:
    """
    Compile the Neural Network
    """

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print("\n✅ model compiled")

    return model

def train_model(model:Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=32,
                validation_split=0.3) -> Tuple[Model, dict]:

    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    es = EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(X, y,
          batch_size = batch_size,
          epochs=1,
          validation_split=validation_split,
          callbacks=[es]
         )


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
        X=X,
        y=y,
        batch_size=batch_size,
        verbose=1,
        # callbacks=None,
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
