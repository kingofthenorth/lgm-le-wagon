import numpy as np
import pandas as pd

from colorama import Fore, Style
import gcld3
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api
from transformers import CamembertTokenizer
from tensorflow import convert_to_tensor

word2vec_transfer = api.load("glove-wiki-gigaword-300")



def clean_text(string):
    '''
    Take a string and keep if string >20 and <1450
    '''
    if string.len() < 1450 and string.len() >20:
        return string
    else:
        "Text is >1500 .../ NA"

def add_language(string):
    '''
    predict the language of a string
    '''
    detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,
                                        max_num_bytes=1000)
    language = detector.FindLanguage(string).language
    if language == "fr" or language == "en":
        return language
    else:
        return "Language is not detected"


def create_tokenizer_fr(max_length=128):
    """
    Create a tokenizer that pads and truncates according to max_length
    Returns input_token_ids and input_masks
    """
    tokenizer_fr = CamembertTokenizer \
        .from_pretrained('jplu/tf-camembert-base',
                         do_lower_case=True,
                         add_special_tokens=True,
                         max_length=128,
                         pad_to_max_length=True)
    return tokenizer_fr

def create_tokenizer_en(max_length=128):
    """
    Create a tokenizer that pads and truncates according to max_length
    Returns input_token_ids and input_masks
    """
    tokenizer_en = CamembertTokenizer \
        .from_pretrained('distilbert-base-uncased',
                         do_lower_case=True,
                         add_special_tokens=True,
                         max_length=128,
                         pad_to_max_length=True)
    return tokenizer_en

def tokenize(sentences, tokenizer):

    input_ids, input_masks = list(), list()

    for sentence in sentences:
        inputs = tokenizer.encode_plus(sentence,
                                       add_special_tokens=True,
                                       max_length=128,
                                       padding='max_length',
                                       truncation=True,
                                       return_attention_mask=True)

        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])

    return np.array(input_ids, dtype='int32'), \
            np.array(input_masks, dtype='int32')



def to_categorical(y,number_of_classes):
    '''
    Take a dataframe column and create as many columns as classes
    '''
    y_cat = to_categorical(y,num_classes=number_of_classes)

    return y_cat

def words_to_vectors():
    pass

def embed_sentence_with_TF(word2vec, sentence):
    '''
    Function to convert a sentence (list of words) into a matrix representing the words in the embedding space
    '''
    embedded_sentence = []
    for word in sentence:
        if word in word2vec:
            embedded_sentence.append(word2vec[word])

    return np.array(embedded_sentence)

def texts_to_embedding_with_transfer(word2vec:word2vec_transfer, sentences):
    '''
    Function that converts a list of sentences into a list of matrices
    '''

    embed = []

    for sentence in sentences:
        embedded_sentence = embed_sentence_with_TF(word2vec, sentence)
        embed.append(embedded_sentence)

    return embed

def padding_embed(embed):
    '''
    Pad the embedded matrices
    '''
    padding = pad_sequences(embed, dtype='float32', padding='post', maxlen=200)

    return padding

def preproccess_for_ooo(X):
    '''
    Play ALL functions to preprocess for ooo prediction
    '''
    X_embed = texts_to_embedding_with_transfer(word2vec_transfer, X)
    X_pad = padding_embed(X_embed)
    return X_pad



########################### TAXIFARE #####################################


# def preprocess_features(X: pd.DataFrame) -> np.ndarray:

#     def create_sklearn_preprocessor() -> ColumnTransformer:
#         """
#         Create a stateless scikit-learn "Column Transformer" object
#         that transforms a cleaned dataset of shape (_, 7)
#         into a preprocessed one of different fixed shape (_, 65)
#         ready for machine-learning models
#         """


#         # PASSENGER PIPE
#         p_min = 1
#         p_max = 8
#         passenger_pipe = FunctionTransformer(lambda p: (p - p_min) /
#                                              (p_max - p_min))

#         # DISTANCE PIPE
#         dist_min = 0
#         dist_max = 100
#         distance_pipe = make_pipeline(
#             FunctionTransformer(transform_lonlat_features),
#             FunctionTransformer(lambda dist: (dist - dist_min)/(dist_max - dist_min))
#         )

#         # TIME PIPE
#         year_min = 2009
#         year_max = 2019
#         time_categories = {
#             0: np.arange(0, 7, 1),  # days of the week
#             1: np.arange(1, 13, 1)  # months of the year
#         }
#         time_pipe = make_pipeline(
#             FunctionTransformer(transform_time_features),
#             make_column_transformer(
#                 (OneHotEncoder(
#                     categories=time_categories,
#                     sparse=False,
#                     handle_unknown="ignore"), [2,3]), # correspond to columns ["day of week", "month"], not the others columns
#                 (FunctionTransformer(lambda year: (year-year_min)/(year_max-year_min)), [4]), # min-max scale the columns 4 ["year"]
#                 remainder="passthrough" # keep hour_sin and hour_cos
#                 )
#             )

#         # GEOHASH PIPE
#         lonlat_features = [
#             "pickup_latitude", "pickup_longitude", "dropoff_latitude",
#             "dropoff_longitude"
#         ]

#         # Below are the 20 most frequent district geohash of precision 5,
#         # covering about 99% of all dropoff/pickup location,
#         # according to prior analysis in a separate notebook
#         most_important_geohash_districts = [
#             "dr5ru", "dr5rs", "dr5rv", "dr72h", "dr72j", "dr5re", "dr5rk",
#             "dr5rz", "dr5ry", "dr5rt", "dr5rg", "dr5x1", "dr5x0", "dr72m",
#             "dr5rm", "dr5rx", "dr5x2", "dr5rw", "dr5rh", "dr5x8"
#         ]

#         geohash_categories = {
#             0: most_important_geohash_districts,  # pickup district list
#             1: most_important_geohash_districts  # dropoff district list
#         }

#         geohash_pipe = make_pipeline(
#             FunctionTransformer(compute_geohash),
#             OneHotEncoder(categories=geohash_categories,
#                           handle_unknown="ignore",
#                           sparse=False))

#         # COMBINED PREPROCESSOR
#         final_preprocessor = ColumnTransformer(
#             [
#                 ("passenger_scaler", passenger_pipe, ["passenger_count"]),
#                 ("time_preproc", time_pipe, ["pickup_datetime"]),
#                 ("dist_preproc", distance_pipe, lonlat_features),
#                 ("geohash", geohash_pipe, lonlat_features),
#             ],
#             n_jobs=-1,
#         )

#         return final_preprocessor


#     print(Fore.BLUE + "\nPreprocess features..." + Style.RESET_ALL)

#     preprocessor = create_sklearn_preprocessor()

#     X_processed = preprocessor.fit_transform(X)

#     print("\n✅ X_processed, with shape", X_processed.shape)

#     return X_processed
