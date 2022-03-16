import multiprocessing
import os
import warnings
from enum import Enum, auto
from time import time

import gensim
from gensim.models.doc2vec import TaggedDocument
from pandas import DataFrame
from gensim.models import Word2Vec, Doc2Vec
import gensim.downloader as api

# todo rename in utility
from helper import show_used_time

# region variables

cwd = os.getcwd()
# make sure you get this repository as your cwd
print(f"Current Working Directory: {cwd}")

# data
data_path = f"{cwd}/data"

# models
models_path = f"{cwd}/models"
# trained models' extension
extension = ".model"
pretrained_model_name = "word2vec-google-news-300"

# todo why these parameters?
# todo adjust parameters
# model parameters
# todo should be the same as min_count during preprocessing (remove_less_frequent_words -> min_word_frequency)?
min_count = 1  # 10
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
epochs = 30  # 30


# endregion


class TweetPool:
    initialized = False
    tweets = []

    def __init__(self, tweets: list):
        if not self.initialized:
            self.tweets = tweets


class WordEmbeddingMethod(Enum):
    MEAN = 0
    TFID = 1


class Method(Enum):
    WORD2VEC = auto()
    DOC2VEC = auto()


class TweetModel:
    """Word2Vec model with corresponding number of tweets used for training (tweet_count)"""

    """
    method = None
    all_tweets = []
    tweet_pool = None
    initialized = False
    """

    # todo can Word2Vec also be used for Doc2Vec models?
    def __init__(self, method: Method, model: Word2Vec, tweet_pool: DataFrame, tweet_count=0):
        """
        :type model: Word2Vec
        """
        self.method = method
        self.tweet_pool = tweet_pool
        self.model = self.__load_model(self)
        # todo use tweets used for training?
        self.tweet_count = tweet_count
        """Number of tweets used for training model"""
        # todo implement
        self.trained_model_tweets = tweet_count
        self.trained_classifier_tweets = tweet_count
        self.tweets = tweet_count
        """Tweets as tokens (List) used for training model"""

        self.initialized = True
        if self.initialized:
            self.set_tweet_pool(tweet_pool)
            raise UserWarning("Tweet pool was already initialized. Current tweet pool will be overwritten.")

    def __load_model(self, force_retrain=False):

        """Try to load model from disk. If it cannot be loaded from disk it will be trained and stored on disk."""
        if self.method == Method.WORD2VEC:
            method_object = Word2Vec
            method_description = "word2vec"
            method_description_short = "w2v"
        elif self.method == Method.DOC2VEC:
            method_object = Doc2Vec
            method_description = "doc2vec"
            method_description_short = "d2v"
        else:
            raise ValueError(f"'{self.method}' is not defined. Defined values: {[e.name for e in Method]}")

        folder = f"{models_path}/{method_description}"
        filename = f"{self.tweet_count}_tweets_{method_description_short}"
        model_path = f"{folder}/{filename}{extension}"

        if not force_retrain and os.path.exists(model_path):
            model = method_object.load(model_path)
        else:
            if os.path.exists(model_path):
                print(f"Model: '{model_path}' exists. Model will be overwritten.")

            t = time()
            model = train_model(self.method, self.tweet_pool.iloc[:int(self.tweet_count)])
            # only needed when models are deleted manually without deleting folder
            if not os.path.exists(folder):
                os.makedirs(folder)
            print(f"model: {model}")
            model.save(model_path)
            show_used_time(t, f"Time for training {method_description} model with {self.tweet_count} tweets")

        print("MODEL: ", model)
        print(f"model path: {model_path}")
        return model

    def __repr__(self):
        return "TweetModel"

    def __str__(self):
        return f"tweet_count: {self.tweet_count}, model: {self.model}"

    def set_tweet_pool(self, all_tweets: DataFrame):
        # todo test it
        if self.initialized:
            raise UserWarning("Tweet pool was already initialized. Current tweet pool will be overwritten.")

        self.tweet_pool = all_tweets
        self.initialized = True


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
        # tokens = gensim.utils.simple_preprocess(clean_tweet)
        # clean_tweet = preprocess(df["tweet"][i])
        # tokens = clean_tweet["tidy_tweet"][i]
        # todo use already preprocessed df
        tokens = df[i]
        print(tokens)
        print("-" * 30)
        # try:
        #     tokens = df[i]
        #     # tokens = df.iloc[i]
        # except KeyError as err:
        #     print(f"i: {i}")
        #     # print(f"i: {i}, df[i]: {df[i]}")
        # # tokens = df["tidy_tweet"][i]
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield TaggedDocument(tokens, [i])
            # yield TaggedDocument(words=tokens, tags=[i])


def load_pretrained_model() -> Word2Vec:
    return api.load(pretrained_model_name)


def train_model(method: Method, df: DataFrame):
    if method == Method.WORD2VEC:
        model = create_w2v_model()
        # todo do not use copy?
        corpus = df.copy()
        print("corpus: ", corpus)
    elif method == Method.DOC2VEC:
        model = create_d2v_model()
        # todo what is better as tags, unique integer or label (0/1)?
        #  problem: passed df contains only tidy_tweets
        corpus = list(read_corpus(df))
        corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(df)]
        # corpus = [TaggedDocument(doc, 'tag') for doc in df]
        print("corpus: ", corpus)

    # building the vocabulary table
    t = time()
    model.build_vocab(corpus, progress_per=progress_per)
    show_used_time(t, "Time for building vocabulary")
    print("*" * 30)
    print(f"len(df): {len(df.index)}, type(model): {type(model)}")
    print("*" * 30)

    # training the model
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs,
                report_delay=1)

    return model


# todo how to move to TweetModel (method creates a TweetModel)?
def train_specific_models(method: Method, df: DataFrame, tweet_counts: list) -> [TweetModel]:
    """Create [TweetModel] from tweets and respective tweet_counts"""
    # models = OrderedDict()
    models = []
    for tweet_count in tweet_counts:
        # models[str(tweet_count)] = load_w2v_model(df, tweet_count)
        # models.append(load_w2v_model(df, tweet_count))
        models.append(
            TweetModel(method=method, model=load_or_create_model(method, df, tweet_count),
                       tweet_pool=df, tweet_count=tweet_count))

    return models


def load_or_create_model(method: Method, df, tweet_count, force_retrain=False):
    """Try to load model from disk. If it cannot be loaded from disk it will be trained and stored on disk."""
    if method == Method.WORD2VEC:
        method_object = Word2Vec
        method_description = "word2vec"
        method_description_short = "w2v"
    elif method == Method.DOC2VEC:
        method_object = Doc2Vec
        method_description = "doc2vec"
        method_description_short = "d2v"
    else:
        raise ValueError(f"'{method}' is not defined. Defined values: {[e.name for e in Method]}")

    folder = f"{models_path}/{method_description}"
    filename = f"{tweet_count}_tweets_{method_description_short}"
    model_path = f"{folder}/{filename}{extension}"

    if not force_retrain and os.path.exists(model_path):
        model = method_object.load(model_path)
    else:
        if os.path.exists(model_path):
            warnings.warn(f"Model: '{model_path}' exists. Model will be overwritten.")

        t = time()
        model = train_model(method, df.iloc[:int(tweet_count)])
        # only needed when models are deleted manually without deleting folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        print(f"model: {model}")
        model.save(model_path)
        show_used_time(t, f"Time for training {method_description} model with {tweet_count} tweets")

    print("MODEL: ", model)
    print(f"model path: {model_path}")
    return model

# endregion
