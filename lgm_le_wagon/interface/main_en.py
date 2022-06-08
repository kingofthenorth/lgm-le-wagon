from cgi import test
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from colorama import Fore, Style


#from lgm_le_wagon.ml_logic.data import get_data

from lgm_le_wagon.data_sources.local_disk import get_local_data
from lgm_le_wagon.ml_logic.data import get_data
from lgm_le_wagon.ml_logic.preprocessor import create_tokenizer_en, tokenize
from lgm_le_wagon.ml_logic.models.model_sentiment_en import initialize_model, train_model, evaluate_model

#from ml_logic.params import (VALIDATION_DATASET_SIZE)

from lgm_le_wagon.ml_logic.preprocessor import (clean_text_mail,
                                   clean_text_linkedin,
                                   preproccess_for_ooo)

from lgm_le_wagon.ml_logic.models import (model_ooo,
                             model_sales_or_hr,
                             model_sentiment_hr,
                             model_sentiment_sales)

from lgm_le_wagon.ml_logic.registry import (save_model,
                                load_model)



def preprocess_and_train_SAEN():
    """
    Load data in memory, clean and preprocess it, train a Keras model on it,
    save the model, and finally compute & save a performance metric
    on a validation set holdout at the `model.fit()` level
    """

    df = get_data("Sentiment_EN")
    X_EN = df["reply"]
    y_EN = df[["negative, neutral, positive"]]
    X_EN_train, X_EN_test, y_EN_train, y_EN_test = train_test_split(X_EN, y_EN, test_size=0.3)

    #Import tokenizer and tokenize our X_TRAIN
    tokenizer = create_tokenizer_en()
    inputs_ids, input_masks = tokenize(X_EN_train, tokenizer)

    #Load Model
    model = initialize_model()

    #Train Model
    model, history = train_model(model, [inputs_ids, input_masks], y_EN_train, batch_size=32, validation_split=0.2)

    # compute val_metrics
    val_accuracy = np.min(history.history['val_accuracy'])
    metrics = dict(val_accuracy=val_accuracy)
    # save model
    params = dict(
        # hyper parameters
        #learning_rate=learning_rate,
        batch_size=32,
        # package behavior
        context="preprocess and train")

    save_model(model=model, params=params, metrics=metrics)

    #print(f"\n✅ trained on {row_count} rows ({cleaned_row_count} cleaned) with accuracy {round(val_accuracy, 2)}")

    return val_accuracy




if __name__ == '__main__':
    preprocess_and_train_SAEN()
    #preprocess()
    #train()
    #pred()
    #evaluate(first_row=9000)
