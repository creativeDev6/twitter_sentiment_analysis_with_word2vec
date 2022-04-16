import logging
import multiprocessing
import os
import warnings
from enum import Enum
from time import perf_counter

import gensim.downloader as api
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from pandas import DataFrame

from helper import show_used_time

# region variables

cwd = os.getcwd()

# models
models_path = f"{cwd}/models"
# trained models' extension
extension = ".model"
pretrained_model_name = "word2vec-google-news-300"

# model parameters
# when changing parameters make sure force_retrain_w2v_models is set to True in run.py otherwise model will be loaded
# from disk without applying model parameters.
min_count = 1
window = 2
vector_size = 300
sample = 6e-5
alpha = 0.03
min_alpha = 0.0007
negative = 20
sg = 1  # word2vec training algorithm: CBoW (0) (default), skip gram (1)
dm = 1  # doc2vec training algorithm: distributed memory (1) (default), distributed bag of words (0 or anything else)
workers = multiprocessing.cpu_count() - 1

# when building vocabulary table show progress
progress_per = 10000

# train parameters
epochs = 30

# endregion


class Method(Enum):
    WORD2VEC = "word2vec"
    DOC2VEC = "doc2vec"


# region models

# todo move to init()
def create_w2v_model():
    # https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
    return Word2Vec(min_count=min_count,
                    window=window,
                    vector_size=vector_size,
                    sample=sample,
                    alpha=alpha,
                    min_alpha=min_alpha,
                    negative=negative,
                    sg=sg,
                    workers=workers)


def create_d2v_model():
    return Doc2Vec(min_count=min_count,
                   window=window,
                   vector_size=vector_size,
                   sample=sample,
                   alpha=alpha,
                   min_alpha=min_alpha,
                   negative=negative,
                   dm=dm,
                   workers=workers)


def read_corpus(df, tokens_only=False):
    for i in df.index:
        tokens = df[i]
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield TaggedDocument(tokens, [i])


def load_pretrained_model():
    return api.load(pretrained_model_name)


def train_model(method: Method, df: DataFrame):
    if method == Method.WORD2VEC:
        model = create_w2v_model()
        corpus = df.copy()
    elif method == Method.DOC2VEC:
        model = create_d2v_model()
        corpus = list(read_corpus(df))
        corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(df)]
    else:
        raise ValueError(f"'{method}' is not defined. Defined values: {[e.name for e in Method]}")

    logging.debug(f"CORPUS:\n{corpus}")

    # building the vocabulary table
    t = perf_counter()
    model.build_vocab(corpus, progress_per=progress_per)
    show_used_time(t, "Time for building vocabulary")
    print("*" * 30)
    print(f"len(df): {len(df.index)}, type(model): {type(model)}")
    print("*" * 30)

    # training the model
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs,
                report_delay=1)

    return model


def load_or_create_model(method: Method, df, tweet_count, force_retrain=False):
    """
    Try to load model from disk. If it cannot be loaded from disk it will be trained and stored on disk.
    """
    if method == Method.WORD2VEC:
        method_object = Word2Vec
        method_description_short = "w2v"
    elif method == Method.DOC2VEC:
        method_object = Doc2Vec
        method_description_short = "d2v"
    else:
        raise ValueError(f"'{method}' is not defined. Defined values: {[e.name for e in Method]}")

    folder = f"{models_path}/{method.value}"
    filename = f"{tweet_count}_tweets_{method_description_short}"
    model_path = f"{folder}/{filename}{extension}"

    if not force_retrain and os.path.exists(model_path):
        model = method_object.load(model_path)
    else:
        if os.path.exists(model_path):
            warnings.warn(f"Model: '{model_path}' exists. Model will be overwritten.")

        t = perf_counter()
        model = train_model(method, df.iloc[:int(tweet_count)])
        # only needed when models are deleted manually without deleting folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        print(f"model: {model}")
        model.save(model_path)
        show_used_time(t, f"Time for training {method.value} model with {tweet_count} tweets")

    logging.info(f"MODEL:  {model}")
    logging.info(f"MODEL PATH: {model_path}")

    return model

# endregion
