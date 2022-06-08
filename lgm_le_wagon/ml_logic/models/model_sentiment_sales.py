
from pyexpat import XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE
from pyexpat.errors import XML_ERROR_BAD_CHAR_REF
from colorama import Fore, Style

import time
print(Fore.BLUE + "\nLoading tensorflow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from tensorflow.keras import Model, Sequential, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import convert_to_tensor
from tensorflow.keras.utils import to_categorical
from transformers import DistilBertTokenizer, BertTokenizer
from tensorflow.keras import layers, Model
from transformers import TFDistilBertModel, DistilBertConfig, TFBertModel, BertConfig, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ tensorflow loaded ({round(end - start, 2)} secs)")

from typing import Tuple

import numpy as np

def initialize_model():
    '''
    Initialize the model
    '''
    distil_bert = 'distilbert-base-uncased'
    # Load pre-trained and add layers
    config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False

    transformer_model = TFDistilBertModel.from_pretrained(distil_bert, config=config)

    input_ids_in = layers.Input(shape=(128,), name='input_token', dtype='int32')
    input_masks_in = layers.Input(shape=(128,), name='masked_token', dtype='int32')

    # Extract embedding
    embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    cls_token = embedding_layer[:, 0, :]

    # Add layers to be trained
    X = layers.BatchNormalization()(cls_token)

    X = layers.Dense(80, activation='relu')(X)


    X = layers.Dropout(config.dropout)(X)

    # 3 classes
    outputs = layers.Dense(3, activation='softmax')(X)

    model = Model(inputs=[input_ids_in, input_masks_in], outputs=outputs)

    # Set first 3 layers as non-trainable
    for layer in model.layers[:3]:
        layer.trainable = False

    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model





def train_model_sales(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=256,
                validation_split=0.3,
                ):
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    es = EarlyStopping(patience=2,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X,
                        y,
                        validation_split=validation_split,
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=0)

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
