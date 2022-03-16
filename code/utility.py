from enum import Enum, auto
import gensim
from pandas import DataFrame
from gensim.models import Word2Vec, Doc2Vec


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
    method = None
    all_tweets = []
    tweet_pool = None
    initialized = False

    # todo can Word2Vec also be used for Doc2Vec models?
    def __init__(self, method: Method, model: Word2Vec, tweet_pool: DataFrame, tweet_count=0):
        """
        :type model: Word2Vec
        """
        self.method = method
        self.model = model
        # todo use tweets used for training?
        self.tweet_count = tweet_count
        """Number of tweets used for training model"""
        # todo implement
        self.trained_model_tweets = tweet_count
        self.trained_classifier_tweets = tweet_count
        self.tweets = tweet_count
        """Tweets as tokens (List) used for training model"""

        if self.initialized:
            self.set_tweet_pool(tweet_pool)
            raise UserWarning("Tweet pool was already initialized. Current tweet pool will be overwritten.")

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
