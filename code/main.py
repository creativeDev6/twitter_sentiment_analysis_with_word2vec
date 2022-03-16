# own packages
from typing import List

from code.utility import TweetModel, Method

from pprint import pp, pprint
import enum
import os.path
import random
import re
from enum import Enum

import gensim
import nltk
import pandas
import pandas as pd
from pandas import DataFrame
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.parsing.preprocessing import preprocess_string
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import models
import gensim.downloader as api
from gensim import corpora
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import tempfile
import multiprocessing

# temp fix not using dot: https://stackoverflow.com/a/57461593/964551
from twitter_corpus import TwitterCorpus

from time import time
from collections import Counter, OrderedDict, namedtuple, defaultdict

# setting up the loggings to monitor gensim
import logging
import warnings
import sys

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# pandas.set_option("display.max_colwidth", 200)

# region variables

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


# endregion

# region measurements

def show_used_time(start_time, text="Time"):
    print(f"{text}: {format(round((time() - start_time) / 60, 2))} min")


# endregion

# region preprocessing

def remove_pattern(text, regex):
    patterns = re.findall(regex, text)
    for pattern in patterns:
        text = re.sub(pattern, "", text)

    return text


# function to collect hashtags
def hashtag_extract(text):
    return re.findall(r"#(\w+)", text)


# source: https://towardsdatascience.com/a-guide-to-cleaning-text-in-python-943356ac86ca
def remove_unicode(text):
    # encoding the text to ASCII format
    encoded_text = text.encode(encoding="ascii", errors="ignore")

    # decoding the text
    decoded_text = encoded_text.decode()

    # cleaning the text to remove extra whitespace
    return " ".join([word for word in decoded_text.split()])


def remove_html_entities(text):
    return remove_pattern(text, r"&[a-zA-Z0-9#]+;")


# todo remove and use gensim.parsing.preprocessing.utils.keep_vocab_item()
def remove_less_frequent_words(texts: list, word_frequency: defaultdict, min_word_frequency: int = 1):
    # https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#core-concepts-document
    # Only keep words that appear more than once
    processed_corpus = [[token for token in text if word_frequency[token] >= min_word_frequency] for text in texts]

    return processed_corpus


def preprocess_doc(doc: str):
    # remove hashtags and users
    # todo also remove hashtags?
    doc = remove_pattern(doc, r"[#@]\w*")
    doc = doc.replace("[^a-zA-Z#")
    # doc = gensim.utils.simple_preprocess(doc)
    from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, stem_text
    # filters = [remove_stopwords, stem_text]
    # doc = gensim.parsing.preprocess_string(doc, filters=filters)
    doc = gensim.parsing.preprocess_string(doc)

    return doc


def preprocess(df: DataFrame):
    # the preprocessed tweet will be stored in the column "tidy_tweet"
    # remove twitter handles (@user)
    df.loc[:, "tidy_tweet"] = df["tweet"].apply(remove_pattern, args=(r"@\w*",))

    # todo remove &amp; Unicode, e.g. 
    df.loc[:, "tidy_tweet"] = df["tidy_tweet"].apply(remove_html_entities)

    # remove special characters, numbers, punctuations (everything except letters and #)
    # todo what about ', like in don't, haven't?
    df.loc[:, "tidy_tweet"] = df["tidy_tweet"].str.replace("[^a-zA-Z#]", " ")
    # df.loc[:, "tidy_tweet"] = df["tidy_tweet"].apply(remove_unicode)

    # extract hashtags into separate column
    df.loc[:, "hashtags"] = df["tidy_tweet"].apply(hashtag_extract)

    # remove hashtags
    # todo IMPORTANT recheck below comment after wrangling data (does it still worsen the result?)
    # Info: the F1-score was worse than leaving them (0.38 to 0.48) when compared with all the training tweets. Also
    # this lead to more rows having zero word tokens (94 to 4) and therefore removing them in the last step of this
    # function (remove words with no word tokens)
    df.loc[:, "tidy_tweet"] = df["tidy_tweet"].str.replace(r"#(\w+)", " ")

    # todo use other gensim preprocessing functions
    # gensim.parsing.preprocessing.utils.keep_vocab_item()
    # gensim.parsing.preprocessing.
    # todo remove short, stopwords, stem
    custom_filters = [gensim.parsing.preprocessing.strip_tags,
                      gensim.parsing.preprocessing.strip_short,
                      gensim.parsing.preprocessing.remove_stopwords,
                      gensim.parsing.preprocessing.stem_text]
    # df.loc[:, "tidy_tweet"] = df["tidy_tweet"].apply(gensim.parsing.preprocessing.preprocess_string, filters=gensim.parsing.preprocessing.DEFAULT_FILTERS)
    df.loc[:, "tidy_tweet"] = df["tidy_tweet"].apply(lambda x: preprocess_string(x, filters=custom_filters))

    # for checking if data was wrangled
    df.to_csv(f"{data_path}/clean/train.csv")
    # sys.exit()

    # use gensim for preprocessing
    # simple preprocess without stemming
    # df.loc[:, "tidy_tweet"] = df["tidy_tweet"].apply(gensim.utils.simple_preprocess)
    # preprocess with stemming
    # df.loc[:, "tidy_tweet"] = df["tidy_tweet"].apply(preprocess_string)

    # display rows (tweets) with no words
    counter = 0
    for row in df.itertuples():
        if len(row.tidy_tweet) == 0:
            print(f"No words in row. Row will be removed: {row}")
            counter += 1
    print(f"Removed rows: {counter}")
    # remove rows with no word tokens after preprocessing
    # Info: prevents having to calculate the document_vector for no words in tweet or rather to define a value for
    # empty tweets
    df = df[df["tidy_tweet"].map(len) > 0]
    df = df.reset_index(drop=True)

    return df


# todo use custom class with iterator for memory efficiency (only one document at a time is read in RAM)
class TwitterCorpus:
    """A constructor that receives a DataFrame."""

    def __init__(self, df):
        self.df = df

    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        # todo path to csv?
        # corpus_path = data_path('')
        for i in self.df.index:
            clean_tweet = remove_pattern(self.df['tweet'][i], r"@\w*")
            clean_tweet = clean_tweet.replace("[^a-zA-Z#]", " ")
            clean_tweet = preprocess(i)
            yield clean_tweet
        # for line in open(corpus_path):
        #     # assume there's one document per line, tokens separated by whitespace
        #     yield utils.simple_preprocess(line)
        # todo remove
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            # todo do preprocessing here
            # yield preprocess(line)
            yield gensim.utils.simple_preprocess(line)


# endregion

# region models

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


def train_model(method: Method, df: DataFrame):
    if method == Method.WORD2VEC:
        model = create_w2v_model()
        # todo do not use copy?
        corpus = df.copy()
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

    # training the model
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs,
                report_delay=1)

    return model


def train_specific_models(method: Method, df: DataFrame, tweet_counts: list) -> [TweetModel]:
    """Create [TweetModel] from tweets and respective tweet_counts"""
    # models = OrderedDict()
    models = []
    for tweet_count in tweet_counts:
        # models[str(tweet_count)] = load_w2v_model(df, tweet_count)
        # models.append(load_w2v_model(df, tweet_count))
        models.append(
            TweetModel(method=method, model=load_model(method, df, tweet_count),
                       tweet_pool=df, tweet_count=tweet_count))

    return models


def load_model(method: Method, df, tweet_count, force_retrain=False):
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
            print(f"Model: '{model_path}' exists. Model will be overwritten.")

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

def document_vector(doc, word2vec_model: Word2Vec):
    # remove out-of-vocabulary words
    try:
        doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    except KeyError:
        print("KeyError")
    # print("-" * 30)
    # print(f"document_vector()")
    # print("-" * 30)
    # print(f"{len(doc)}")
    # print(f"{doc}")

    # print(f"# of tokens: {len(doc)}")
    # todo how to handle the cases when no word is found in model? What should be then returned?
    #  e.g. tweet for train data with id=4800
    #  @user @ user @ user @ user @ user @ user @ user @ user @ user @ user @ user
    #  --> ignore word (use exception handling)
    # Info: make sure len(doc) > 0 otherwise no mean can be calculated.
    # return np.mean(word2vec_model[doc], axis=0)
    # gensim 4
    return np.mean(word2vec_model.wv[doc], axis=0)


# todo rename doc_vector
def mean_doc_vector(tokens, model_w2v: Word2Vec, is_pretrained=False):
    # vec = np.zeros(vector_size).reshape((vector_size, ))
    vec = np.zeros(vector_size)
    count = 0
    words_not_in_vocab = []

    if is_pretrained:
        w2v_model = model_w2v
    else:
        w2v_model = model_w2v.wv

    for word in tokens:
        try:
            # vec += model_w2v[word].reshape((1, vector_size))
            # vec += model_w2v.wv[word].reshape((vector_size, ))
            # vec += model_w2v.wv[word]
            vec += w2v_model[word]
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            words_not_in_vocab.append(word)
            continue
    if count != 0:
        vec /= count
    # print(f"Words not in vocabulary (# {len(words_not_in_vocab)}): {words_not_in_vocab}")
    return vec


# Word2Vec.load_word2vec_format()
# region word embedding vectorizers

# from http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors

        # commented old code that no longer works in python 3 see https://stackoverflow.com/a/30418498/964551
        # self.dim = len(word2vec.itervalues().next())
        self.dim = len(word2vec.values())
        # self.dim = len(word2vec)

        # if len(word2vec) > 0:
        #     self.dim = len(word2vec[next(iter(glove_small))])
        # else:
        #     self.dim = 0

    def fit(self, tokens: list, labels: list):
        return self

    def transform(self, tokens: list):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in tokens
        ])


# from http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        # self.dim = len(word2vec.itervalues().next())
        self.dim = len(word2vec.values())

    def fit(self, tokens: list, labels: list):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(tokens)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, tokens: list):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in tokens
        ])


def word_embedding(w2v: dict, docs, labels, method=1):
    # tfidf word embedding
    if method == 1:
        vectorizer = TfidfEmbeddingVectorizer(w2v)
    # mean word embedding
    elif method == 0:
        vectorizer = MeanEmbeddingVectorizer(w2v)
    length = len(docs)
    vectorizer.fit(docs, labels)

    return vectorizer.transform(docs)


# endregion

def test_w2v_model(model: Word2Vec):
    print(model.wv.most_similar(positive=["trump"]))
    print(model.wv.most_similar(positive=["love"]))

    # how similar
    print(model.wv.similarity("trump", "love"))

    # odd word
    print(model.wv.doesnt_match(["trump", "donald", "love"]))

    # relation man to trump and woman to ? (man + trump - woman = ?)
    print(model.wv.most_similar(positive=["man", "trump"], negative=["man"], topn=3))

    # woman + king - man = ?
    print(model.wv.most_similar(positive=["woman", "king"], negative=["man"]))


# todo test
def extract_feature_vectors(models: OrderedDict, tweets: DataFrame, is_test_data: bool = True) -> OrderedDict:
    result = OrderedDict()
    for key, value in models.items():
        # length = len(value.docvecs)
        # gensim 4
        length = len(value.dv)
        if is_test_data:
            result[str(key)] = [value.infer_vector(tweets[i]) for i in range(length)]
        else:
            # result[str(key)] = [value.docvecs[i] for i in range(length)]
            # gensim 4
            result[str(key)] = [value.dv[i] for i in range(length)]

    return result


# region classifiers

# todo how to extract features from w2v models and pass to classifier?
def train_classifier(vectors: list, df_labels: DataFrame):
    # todo retrieve vector for doc by tag
    # model.docvecs["0"]
    # gensim 4
    # model.dv["0"]
    # INFO: ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    # therefore increased max_iter
    # https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
    # For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
    logreg = LogisticRegression(solver='liblinear', max_iter=1000)
    # logreg.fit(train_clean["tidy_tweet"], train_clean["label"])
    print(f"train_classifier -> len(vectors): {len(vectors)}, len(labels): {len(df_labels)}")
    print(f"shape: {vectors[0].shape}")
    logreg = logreg.fit(vectors, df_labels)

    return logreg


def train_d2v_sentiment_classifiers(models: OrderedDict, tweets: DataFrame, labels: DataFrame,
                                    is_test_data: bool = True) -> namedtuple("mdc", "model docvecs classifier"):
    # OrderedDict with same keys as models, value is a namedtuple mdc (model, docvecs, classifier)
    result = OrderedDict()
    mdc = namedtuple("mdc", "model docvecs classifier")
    for key, model in models.items():
        # save doc vectors as a list of floats for training classifier (docvecs of type 'Doc2VecKeyedVectors' cannot
        # be used)
        if is_test_data:
            # todo unknown words are discarded
            # if tweets[key] not in model.build_vocab:
            #     pass
            # infer new vector for every document
            try:
                length = len(tweets.index)
                docvecs = [model.infer_vector(tweets[i]) for i in range(length)]
            except:
                print(f"EXCEPTION: infer_vector")
        else:
            # use already trained vectors
            # length = len(model.docvecs)
            # gensim 4
            length = len(model.dv)
            # docvecs = [model.docvecs[i] for i in range(length)]
            # gensim 4
            docvecs = [model.dv[i] for i in range(length)]

        # make sure vectors are all finite numbers
        if (not np.all(np.isfinite(docvecs))) or (np.any(np.isnan(docvecs))):
            print(f"Any NAN: {np.any(np.isnan(docvecs))}")
            print(f"All FINITE: {np.all(np.isfinite(docvecs))}")
            sys.exit("Make sure vectors are all finite number")

        classifier = train_classifier(docvecs, labels[:length])

        # train classifier with feature vectors and labels
        # result[str(key)] = train_classifier(docvecs, labels[:length])
        result[str(key)] = mdc(model, docvecs, classifier)

    return result


# endregion

# region evaluation


def evaluate_f1_scores(dict_mdc: OrderedDict, labels: DataFrame):
    for key, value in dict_mdc.items():
        # length = len(value.docvecs)
        # gensim 4
        length = len(value.dv)
        # output model parameters and F1-score
        print("-" * 30)
        print(f"model['{key}']: {value.model}")
        # print(f"{evaluate_f1_score(value.classifier, value.docvecs, labels[:length])}")
        # gensim 4
        print(f"{evaluate_f1_score(value.classifier, value.dv, labels[:length])}")


# todo maybe remove
def evaluate_d2v_specific_models(models: OrderedDict, tweets: DataFrame, labels: DataFrame,
                                 is_test_data: bool = True):
    for key, model in models.items():
        # save doc vectors as a list of floats for training classifier (docvecs of type 'Doc2VecKeyedVectors' cannot
        # be used)
        if is_test_data:
            # infer new vector for every document
            length = len(tweets.index)
            train_vectors = [model.infer_vector(tweets[i]) for i in range(length)]
        else:
            # use already trained vectors
            # length = len(model.docvecs)
            train_vectors = [model.docvecs[i] for i in range(length)]
            # gensim 4
            length = len(model.dv)
            train_vectors = [model.dv[i] for i in range(length)]

        # make sure vectors are all finite numbers
        if (not np.all(np.isfinite(train_vectors))) or (np.any(np.isnan(train_vectors))):
            print(f"Any NAN: {np.any(np.isnan(train_vectors))}")
            print(f"All FINITE: {np.all(np.isfinite(train_vectors))}")
            sys.exit("Make sure vectors are all finite number")

        # FIXME Wrong: You have to pass a log_reg from an already trained classifier
        #   and use this with the given unknown data to evaluate the f1-score!
        # train classifier with feature vectors and labels
        log_reg = train_classifier(train_vectors, labels[:length])
        # output model parameters and F1-score
        print("-" * 30)
        print(f"model['{key}']: {model}")
        print(f"{evaluate_f1_score(log_reg, train_vectors, labels[:length])}")


# todo remove or maybe use?
def benchmark(model, X, y, n):
    test_size = 1 - (n / float(len(y)))
    scores = []
    for train, test in StratifiedShuffleSplit(y, n_iter=5, test_size=test_size):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
    return np.mean(scores)


# endregion

# region statistics

def show_pos_neg_ratio(df: DataFrame, column: str = None):
    # count positive and negative labels for train data
    pos_count = len(df[df.label == 0])
    neg_count = len(df[df.label == 1])

    print(f"positive: {pos_count}")
    print(f"negative: {neg_count}")
    print(f"positive ratio: {pos_count / len(df)}")
    print(f"negative ratio: {neg_count / len(df)}")
    print(f"{pos_count + neg_count} == {len(df)}")


# todo add other statistics functions (tf-idf)
def count_word_frequencies(texts: list):
    # https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#core-concepts-document
    # Count word frequencies
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    return frequency


# endregion

def pipeline_w2v(models: OrderedDict, train_df: DataFrame, test_df: DataFrame):
    # trained_log_reg = train_classifier(models["20000"].wv.vectors_norm, train_df["label"][:20000])
    # gensim 4
    trained_log_reg = train_classifier(models["20000"].wv.get_normed_vectors(), train_df["label"][:20000])
    prediction = trained_log_reg.predict_proba(test_df["tweet"])
    # if prediction is greater than or equal to 0.3 than 1 else 0
    prediction_int = prediction[:, 1] >= 0.3
    prediction_int = prediction_int.astype(np.int)
    f1_score(test_df["label"], prediction_int, pos_label=0)
    sys.exit()


def pipeline(models: OrderedDict, train_df: DataFrame, test_df: DataFrame):
    # [str, model_doc2vec]
    print(f'MODEL: {models["20000"]}')
    # fixme
    # trained_log_reg = train_classifier(models["20000"].docvecs, train_df["label"][:20000])
    # gensim 4
    trained_log_reg = train_classifier(models["20000"].dv, train_df["label"][:20000])
    prediction = trained_log_reg.predict_proba(test_df["tweet"])
    # if prediction is greater than or equal to 0.3 than 1 else 0
    prediction_int = prediction[:, 1] >= 0.3
    prediction_int = prediction_int.astype(np.int)
    f1_score(test_df["label"], prediction_int, pos_label=0)
    sys.exit()
    dict_mdc_train = train_d2v_sentiment_classifiers(specific_d2v_models, train["tidy_tweet"], train["label"], False)
    evaluate_f1_scores(dict_mdc_train, train["label"])


# Todo w2v test
def w2v_test(model: Word2Vec, train: DataFrame, test: DataFrame):
    # todo classify w2v
    w2v = {w: vec for w, vec in zip(model.wv.index_to_key, model.wv.vectors)}
    # X tokenized sentences -> tweets["tidy_tweet]
    # n train size -> key in specific_w2v_models
    # y labels -> tweet["label"]
    # X, y = np.array(X), np.array(y)
    print(f"index_to_key: {type(model.wv.index_to_key).__name__}")
    print(f"w2v: {type(w2v).__name__}")
    # print(f"w2v['when']: {w2v['when']}")

    all_val = len(train)
    key = all_val
    # tv = document_vector(train["tidy_tweet"][0], model)
    # print(tv)
    # train_vectors = train["tidy_tweet"].apply(document_vector, args=(model,))
    train_vectors = []

    # is working fine
    # for doc in train["tidy_tweet"].iloc[:int(key)]:
    #     # print(f"doc: {doc}")
    #     train_vectors.append(document_vector(doc, model))
    train_vectors_doc_vector = [document_vector(doc, model) for doc in train["tidy_tweet"]]
    mv = MeanEmbeddingVectorizer(w2v)
    # mv = TfidfEmbeddingVectorizer(w2v)
    mv.fit(train["tidy_tweet"][:key], train["label"][:key])
    train_array = np.array(train["tidy_tweet"][:key])
    train_vectors = mv.transform(train["tidy_tweet"][:key])

    log_reg = train_classifier(train_vectors_doc_vector, train["label"][:key])
    print(f"document_vector: {evaluate_f1_score(log_reg, train_vectors, train['label'][:key])}")

    log_reg = train_classifier(train_vectors, train["label"][:key])
    print(f"MeanEmbeddingVectorizer: {evaluate_f1_score(log_reg, train_vectors, train['label'][:key])}")


def w2v_evaluate(model: Word2Vec, train: DataFrame, test: DataFrame, is_pretrained=False):
    # w2v = {w: vec for w, vec in zip(model.wv.index_to_key, model.wv.vectors)}
    if is_pretrained:
        w2v = {w: vec for w, vec in zip(model.index_to_key, model.vectors)}
    else:
        w2v = {w: vec for w, vec in zip(model.wv.index_to_key, model.wv.vectors)}
    # print(f"w2v['when']: {w2v['when']}")

    train_doc_vectors = [mean_doc_vector(doc, model, is_pretrained) for doc in train["tidy_tweet"]]
    test_doc_vectors = [mean_doc_vector(doc, model, is_pretrained) for doc in test["tidy_tweet"]]
    print(f"shape train: {train_doc_vectors[0].shape}")
    print(f"shape test: {test_doc_vectors[0].shape}")
    print(f"len(test_vectors) == len(test): {len(test_doc_vectors) == len(test)}")
    mv = MeanEmbeddingVectorizer(w2v)
    # mv = TfidfEmbeddingVectorizer(w2v)
    mv.fit(train["tidy_tweet"], train["label"])
    train_vectors = mv.transform(train["tidy_tweet"])

    print("*-" * 30)
    print(f"Tweets used: {len(train)}")
    log_reg = train_classifier(train_doc_vectors, train["label"])
    print("-" * 30)
    print(f"insanity check (train):")
    print("-" * 30)
    print(f"{evaluate_f1_score(log_reg, train_doc_vectors, train['label'])}")

    print("-" * 30)
    print(f"evaluate (test):")
    print("-" * 30)
    print(f"{evaluate_f1_score(log_reg, test_doc_vectors, test['label'])}")
    print("*-" * 30)


def evaluate_f1_score(logreg: LogisticRegression, test_vectors: list, test_labels: DataFrame):
    def scores(test_labels_param, prediction_int_param, average_param, pos_label_param):
        # prevent warning: ignoring pos_label when average != "binary"
        if average_param != "binary":
            f1_res = f1_score(test_labels_param, prediction_int_param, average=average_param)
            precision_res = precision_score(test_labels_param, prediction_int_param, average=average_param)
            recall_res = recall_score(test_labels_param, prediction_int_param, average=average_param)
        else:
            f1_res = f1_score(test_labels_param, prediction_int_param, average=average_param, pos_label=pos_label_param)
            precision_res = precision_score(test_labels_param, prediction_int_param, average=average_param,
                                            pos_label=pos_label_param)
            recall_res = recall_score(test_labels_param, prediction_int_param, average=average_param,
                                      pos_label=pos_label_param)

        return f1_res, precision_res, recall_res

    # todo difference to predict_log_proba()?
    # prediction = logreg.predict_proba(test["tidy_tweet"])
    print(f"evaluate_f1_score")
    print(f"len(vectors): {len(test_vectors)}, len(labels): {len(test_labels)}")
    # probability estimates
    prediction = logreg.predict_proba(test_vectors)
    # non-thresholded decision values
    # prediction = logreg.decision_function(test_vectors)
    # todo why 0.3? because there are more positive labeled tweets? (0.3 also gets the best result)
    # if prediction is greater than or equal to 0.3 than 1 else 0
    prediction_int = prediction[:, 1] >= 0.3
    prediction_int = prediction_int.astype(np.int)

    # todo remove
    # from sklearn.metrics import roc_auc_score
    # print("ROC AUC: ", roc_auc_score(test_labels, prediction[:, 1]))
    # print("with decision function: ", roc_auc_score(test_labels, logreg.decision_function(test_vectors)))

    # print(f1_score(test["label"], prediction_int))
    # f1, precision and recall
    result = ""
    # (1, 'weighted') is not needed because result will be calculated from both labels
    for i in [(0, 'binary'), (1, 'binary'), (0, 'weighted')]:
        pos_label = i[0]
        # average = 'binary' # Only report results for the class specified by pos_label
        average = i[1]
        # todo get score for individual labels
        # f1 = f1_score(test_labels, prediction_int, average=average, pos_label=pos_label)
        # precision = precision_score(test_labels, prediction_int, average=average, pos_label=pos_label)
        # recall = recall_score(test_labels, prediction_int, average=average, pos_label=pos_label)
        f1, precision, recall = scores(test_labels, prediction_int, average, pos_label)
        result += f"(label: {pos_label}, average: {average}) F1-Score: {f1}, Precision: {precision}, Recall: {recall}\n"

    # return f"F1-score: {f1_score(test_labels, prediction_int, pos_label=0)}"
    return result


def evaluate_scores(method: Method, model: Word2Vec, train: DataFrame, test: DataFrame, is_pretrained=False):
    def scores(test_labels_param, prediction_int_param, average_param, pos_label_param):
        # prevent warning: ignoring pos_label when average != "binary"
        if average_param != "binary":
            f1_res = f1_score(test_labels_param, prediction_int_param, average=average_param)
            precision_res = precision_score(test_labels_param, prediction_int_param, average=average_param)
            recall_res = recall_score(test_labels_param, prediction_int_param, average=average_param)
        else:
            f1_res = f1_score(test_labels_param, prediction_int_param, average=average_param, pos_label=pos_label_param)
            precision_res = precision_score(test_labels_param, prediction_int_param, average=average_param,
                                            pos_label=pos_label_param)
            recall_res = recall_score(test_labels_param, prediction_int_param, average=average_param,
                                      pos_label=pos_label_param)

        return f1_res, precision_res, recall_res

    if method == Method.WORD2VEC:
        train_doc_vectors = [mean_doc_vector(doc, model, is_pretrained) for doc in train["tidy_tweet"]]
        test_doc_vectors = [mean_doc_vector(doc, model, is_pretrained) for doc in test["tidy_tweet"]]

        log_reg = train_classifier(train_doc_vectors, train["label"])

        prediction = log_reg.predict_proba(test_doc_vectors)
        # non-thresholded decision values
        # prediction = logreg.decision_function(test_vectors)
        # todo why 0.3? because there are more positive labeled tweets? (0.3 also gets the best result)
        # if prediction is greater than or equal to 0.3 than 1 else 0
        prediction_int = prediction[:, 1] >= 0.3
        prediction_int = prediction_int.astype(np.int)

    elif method == Method.DOC2VEC:
        # creating taggedDocuments with tweet and label
        # train_tagged = train[:limit].apply(
        # todo the doc2vec models were trained with unique integers as tags, now by labels does it matter?
        train_tagged = train.apply(
            lambda r: TaggedDocument(words=r.tidy_tweet, tags=[r.label]), axis=1)
        test_tagged = test.apply(
            lambda r: TaggedDocument(words=r.tidy_tweet, tags=[r.label]), axis=1)

        labels_train, train_doc_vectors = vec_for_learning(model, train_tagged)
        labels_test, test_doc_vectors = vec_for_learning(model, test_tagged)

        # INFO: ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
        # therefore increased max_iter
        # https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
        # todo look at first 2 options before trying to increase max_iter to 1000
        log_reg = LogisticRegression(n_jobs=1, C=1e5, max_iter=3000)
        log_reg.fit(train_doc_vectors, labels_train)
        # todo old as backup, remove later
        # predicted_labels_test = log_reg.predict(test_doc_vectors)
        # predicted_labels_train = log_reg.predict(train_doc_vectors)
        prediction_int = log_reg.predict(test_doc_vectors)
        predicted_labels_train = log_reg.predict(train_doc_vectors)

    else:
        raise ValueError(f"'{method}' is not defined. Defined values: {[e.name for e in Method]}")

    """
    if is_pretrained:
        w2v = {w: vec for w, vec in zip(model.index_to_key, model.vectors)}
    else:
        w2v = {w: vec for w, vec in zip(model.wv.index_to_key, model.wv.vectors)}
    mv = MeanEmbeddingVectorizer(w2v)
    mv = TfidfEmbeddingVectorizer(w2v)
    mv.fit(train["tidy_tweet"], train["label"])
    train_vectors = mv.transform(train["tidy_tweet"])
    """

    # for grouping data (hue)
    tweet_count = len(train)

    # f1, precision and recall
    result = []
    # todo is it possible to pass this as a parameter to make it more flexible?
    score_data = namedtuple("scoreData", "tweet_count label f1 precision recall")
    # get score for individual labels (binary)
    # (1, 'weighted') is not needed because result will be calculated from both labels
    for i in [(0, 'binary'), (1, 'binary'), (0, 'weighted')]:
        average_label = ""
        pos_label = i[0]
        # average = 'binary' # Only report results for the class specified by pos_label
        average = i[1]
        # translate labels
        if i[1] == "binary":
            average_label = "0: Positive" if i[0] == 0 else "1: Negative"
        else:
            average_label = "0 + 1"
        # f1 = f1_score(test_labels, prediction_int, average=average, pos_label=pos_label)
        # precision = precision_score(test_labels, prediction_int, average=average, pos_label=pos_label)
        # recall = recall_score(test_labels, prediction_int, average=average, pos_label=pos_label)
        f1, precision, recall = scores(test["label"], prediction_int, average, pos_label)
        # result.append(zip(average, pos_label, f1, precision, recall))
        result.append(score_data(tweet_count=tweet_count, label=average_label, f1=f1, precision=precision,
                                 recall=recall))

    # return f"F1-score: {f1_score(test_labels, prediction_int, pos_label=0)}"
    # return DataFrame(result)
    return result


# todo testing from tutorial doc2vec (multiclass):
#  https://www.kdnuggets.com/2018/11/multi-class-text-classification-doc2vec-logistic-regression.html
def vec_for_learning(d2v_model: Doc2Vec, tagged_docs: TaggedDocument):
    # fixme
    # sents = tagged_docs.values
    sents = tagged_docs
    # print("-" * 30)
    # print(sents)
    # print("-" * 30)
    # sents = [x for x in tagged_docs]
    # todo what happens if word is not in model, how to handle this case?
    # * unpacks list into zip call
    targets, regressors = zip(*[(doc.tags[0], d2v_model.infer_vector(doc.words, epochs=20)) for doc in sents])
    # print(f"labels: {targets[:1000]}, docvecs: {regressors[:1]}")

    return targets, regressors


#  https://www.kdnuggets.com/2018/11/multi-class-text-classification-doc2vec-logistic-regression.html
def evaluate_d2v_model(model_param: Doc2Vec, train_param: DataFrame, test_param: DataFrame):
    print("evaluate_d2v_model")
    # todo fixme how to make it work with generator?
    # train_tagged = read_corpus(train[:limit])
    # test_tagged = read_corpus(test[:limit])

    # creating taggedDocuments with tweet and label
    # train_tagged = train[:limit].apply(
    train_tagged = train_param.apply(
        lambda r: TaggedDocument(words=r['tidy_tweet'], tags=[r.label]), axis=1)
    test_tagged = test_param.apply(
        lambda r: TaggedDocument(words=r['tidy_tweet'], tags=[r.label]), axis=1)

    labels_train, doc_vectors_train = vec_for_learning(model_param, train_tagged)
    labels_test, doc_vectors_test = vec_for_learning(model_param, test_tagged)

    # INFO: ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    # therefore increased max_iter
    # https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
    # todo look at first 2 options before trying to increase max_iter to 1000
    logreg = LogisticRegression(n_jobs=1, C=1e5, max_iter=3000)
    logreg.fit(doc_vectors_train, labels_train)
    predicted_labels_test = logreg.predict(doc_vectors_test)
    predicted_labels_train = logreg.predict(doc_vectors_train)

    print(f"Evaluation")
    # todo evaluate also labels separately (see w2v_evaluate)
    # todo use proper model description
    print(f"Model: {len(train_param)} tweets")
    print(f"Accuracy: {accuracy_score(labels_test, predicted_labels_test)}")
    print(f"Testing F1-Score: {f1_score(labels_test, predicted_labels_test, average='weighted')}")
    print(f"Testing F1-Score (insanity check): {f1_score(labels_train, predicted_labels_train, average='weighted')}")


def evaluate_d2v_models(d2v_models: OrderedDict, train_param: DataFrame, test_param: DataFrame):
    for key, model in d2v_models.items():
        print(f"-" * 30)
        print(f"{key} tweets model")
        if key == "all":
            temp_train = train_param.copy()
        else:
            temp_train = train_param.iloc[:int(key)].copy()
        evaluate_d2v_model(model, temp_train, test_param)
        print(f"-" * 30)


def testing(train_param):
    for key in [100, 1000, 10000, 20000, "all"]:
        if key == "all":
            print(f"Key = {key} tweets : len = {len(train_param)}")
            print(f"len(train_param.iloc[:{int(key)}]: ", len(train_param.iloc[:int(key)]))
            break
        # fixme why only 2 elements in train (100, 1.000, 10.000)
        print(f"Key = {key} tweets : len = {len(train_param.iloc[:int(key)])}")


def main():
    # always show all columns and rows on a panda's DataFrame
    pandas.options.display.max_columns = None
    # pandas.options.display.max_rows = None

    labeled = pandas.read_csv(f"{data_path}/train_E6oV3lV.csv")
    # preprocess both data separately (adds column "tidy_tweet")
    labeled = preprocess(labeled)

    # fixme shuffle=True not working (should work tested on 2021-09-22)
    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)

    # todo evaluation in each fold + store values and calculate mean
    fold = 0
    # todo divide data into 3 sections:
    #  training, validation (used for tweaking parameters) and test set (testing performance)?
    for train_index, test_index in skf.split(labeled["tweet"], labeled["label"]):
        fold += 1
        print(f"Fold - {fold} -")
        print(f"TRAIN (len: {len(train_index)}: {train_index} TEST (len: {len(test_index)}: {test_index}")
        print("tweet ", labeled["tweet"].iloc[train_index], labeled["tweet"].iloc[test_index])
        print("tidy_tweet ", labeled["tidy_tweet"].iloc[train_index], labeled["tidy_tweet"].iloc[test_index])
        print("hashtags ", labeled["hashtags"].iloc[train_index], labeled["hashtags"].iloc[test_index])
        print("labels ", labeled["label"].iloc[train_index], labeled["label"].iloc[test_index])
        train_tweets, test_tweets = labeled["tweet"].iloc[train_index], labeled["tweet"].iloc[test_index]
        train_tidy_tweets, test_tidy_tweets = labeled["tidy_tweet"].iloc[train_index], labeled["tidy_tweet"].iloc[
            test_index]
        train_hashtags, test_hashtags = labeled["hashtags"].iloc[train_index], labeled["hashtags"].iloc[test_index]
        train_labels, test_labels = labeled["label"].iloc[train_index], labeled["label"].iloc[test_index]
        train_ids, test_ids = labeled["id"].iloc[train_index], labeled["id"].iloc[test_index]
        print(f"Current TRAIN (len: {len(train_tweets)}) TEST (len: {len(test_tweets)})")

    # save stratified data as DataFrame
    train = DataFrame(list(zip(train_ids, train_labels, train_tweets, train_tidy_tweets, train_hashtags)),
                      columns=["id", "label", "tweet", "tidy_tweet", "hashtags"])
    test = DataFrame(list(zip(test_ids, test_labels, test_tweets, test_tidy_tweets, test_hashtags)),
                     columns=["id", "label", "tweet", "tidy_tweet", "hashtags"])

    # todo X decision_function()
    # todo what does 'greater label' mean? https://scikit-learn.org/stable/modules/model_evaluation.html#binary-case
    #  -> The “greater label” corresponds to classifier.classes_[1] and thus classifier.predict_proba(X)[:, 1]
    # switching labels 0 -> 1 and 1 -> 0
    # train.label.replace([0, 1], [1, 0], inplace=True)
    # test.label.replace([0, 1], [1, 0], inplace=True)

    t = time()
    # todo INFO: this was moved above
    # train = preprocess(train)
    # test = preprocess(test)
    show_used_time(t, "Time for preprocessing")
    print(f"{test[test['tidy_tweet'].str.len() < 2]}")

    # t = time()
    # todo uncomment visualization
    # visualize_data(train, "Train")
    # visualize_data(test, "Test")
    # show_used_time(t, "Time for visualization")

    # print(train.head(15))
    # print(test.head(15))

    # todo filter out stopwords + short words (<4)

    print("-" * 30)
    print("Specific model")
    print("-" * 30)

    # todo Do I also need to limit the model to current tweet_count (e.g. google-news model cannot be limited)?
    #  -> solution: set specific models to tweet_count, but when comparing with google-news model use model "all"

    # todo adjust models' keys (either 100, 1000, ... or equally sized partitions 1/5 1, 2, 3, 4, 5)
    tweet_counts = [100, 1000, 10000, 20000, len(train)]

    # todo add test_tweets (17k)?
    # todo only return len(tweet_counts) number of models (do not add all)!
    specific_w2v_tweet_models = train_specific_models(Method.WORD2VEC, train["tidy_tweet"], tweet_counts)

    # todo refactor train_specific_models() + get_scores_dataframe()?
    """
    scores_data = []
    for tweet_count in tweet_counts:
        train_specific_models(Method.WORD2VEC, train["tidy_tweet"], tweet_count)
        scores_data.append(get_scores_dataframe(Method.DOC2VEC, specific_w2v_tweet_models, train, test))

    for tweet_model in specific_w2v_tweet_models:
        print(tweet_model)
    """
    # todo uncomment visualization
    scores_data = get_scores_dataframe(Method.WORD2VEC, specific_w2v_tweet_models, train, test)
    visualize_score(scores_data, title_prefix="W2V Specific Model")
    print(scores_data.head(15))

    print("-" * 30)
    print("Unspecific model")
    print("-" * 30)
    # load a saved model:
    # todo maybe put this in above specific model loop (showing specific and unspecific model results)
    # TODO use 30k tweets model for faster testing, uncomment
    pretrained_model = api.load(pretrained_model_name)
    unspecific_w2v_tweet_models = []
    tweet_counts = [100, 1000, 10000, 20000, len(train)]
    # todo uncomment
    # """
    for tweet_count in tweet_counts:
        # tweet_count is only used for training classifiers with corresponding number of tweets
        # TODO use 30k tweets model for faster testing, uncomment
        unspecific_w2v_tweet_models.append(TweetModel(Method.WORD2VEC, pretrained_model, train["tidy_tweet"], tweet_count))
        # unspecific_w2v_tweet_models.append(
        #     TweetModel(Method.WORD2VEC, specific_w2v_tweet_models[len(specific_w2v_tweet_models) - 1].model, train["tidy_tweet"],
        #                tweet_count))
        # tweet_counts = [len(train)]
        print(f"{tweet_count} tweets : len = {len(train.iloc[:int(tweet_count)])}")
    # """
    # w2v_evaluate(pretrained_model, train.iloc[:int(tweet_count)], test, True)
    # TODO use 30k tweets model for faster testing, use is_pretrained=True

    # todo uncomment visualization
    scores_data = get_scores_dataframe(Method.WORD2VEC, unspecific_w2v_tweet_models, train, test, is_pretrained=True)
    visualize_score(scores_data, title_prefix="W2V Unspecific Model")

    # all tweets
    print(f"{len(train)} tweets : len = {len(train)}")
    # w2v_evaluate(pretrained_model, train, test, True)
    # return

    print("-" * 30)
    print("d2v - Specific model")
    print("-" * 30)

    # todo change keys of models (only 100, 1000, 10.000, 20.000, all (25566))

    # specific_d2v_tweet_models = train_specific_models(Method.DOC2VEC, train["tidy_tweet"], tweet_counts)
    # evaluate_d2v_models(specific_d2v_tweet_models, train, test)
    # todo uncomment visualization
    # scores_data = get_scores_dataframe(Method.DOC2VEC, specific_d2v_tweet_models, train, test)
    # visualize_score(scores_data, title_prefix="D2V Specific Model")
    # print(scores_data.head(15))


def get_scores_dataframe(method: Method, tweet_models: [TweetModel], train: DataFrame, test: DataFrame,
                         is_pretrained=False) -> DataFrame:
    data_list = []

    if method == Method.WORD2VEC:
        method = Method.WORD2VEC
    elif method == Method.DOC2VEC:
        method = Method.DOC2VEC
    else:
        raise ValueError(f"'{method}' is not defined. Defined values: {[e.name for e in Method]}")

    for tweet_model in tweet_models:
        scores = evaluate_scores(method, tweet_model.model,
                                 train.iloc[:int(tweet_model.tweet_count)],
                                 test,
                                 is_pretrained)
        data_list.extend(scores)

    return DataFrame(data=data_list)


def evaluation():
    visualize_score()


def visualize_data(data: DataFrame, title_prefix=None):
    def label_ratio_pie_chart(count_list, *, title, labels):
        # source: https://stackoverflow.com/a/6170354
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))

                return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

            return my_autopct

        plt.title(f"{title_prefix}: {title}")
        plt.pie(count_list,
                labels=labels,
                autopct=make_autopct(count_list),
                wedgeprops=
                {
                    "edgecolor": "1",
                    "linewidth": 1,
                    # "linestyle": "dashed",
                    "antialiased": True,
                })
        plt.show()

    def word_freq_bar_plot(words: DataFrame, *, title="", num_words_to_plot=10, total_count=1, y_suffix=""):
        # Format the text with number or decimal.
        text_format = ".3f" if total_count > 1 else "n"
        if y_suffix != "":
            y_suffix = f" {y_suffix}"
        # unnesting list
        words = sum(words, [])

        freq_dist = nltk.FreqDist(words)
        d = DataFrame({"Word": list(freq_dist.keys()),
                       "Frequency": list(freq_dist.values())})

        # selecting top n most frequent hashtags
        d = d.nlargest(columns="Frequency", n=num_words_to_plot)
        plt.figure(figsize=(16, 5))
        plt.title(f"{title_prefix}: {title}")
        ax = sns.barplot(data=d, x="Word", y="Frequency")
        # ax = sns.barplot(data=d, y="Word", x="Frequency", orient="h")
        ax.set(ylabel="Frequency" + y_suffix)
        ax.set(xlabel="Words")
        ax.yaxis.grid(True, color="#EEEEEE")

        # adding values on top of each bar
        # source: https://www.pythoncharts.com/matplotlib/grouped-bar-charts-matplotlib/
        for bar in ax.patches:
            # The text annotation for each bar should be its height.
            bar_value = bar.get_height()
            text = f"{bar_value / total_count:{text_format}}"
            # This will give the middle of each bar on the x-axis.
            text_x = bar.get_x() + bar.get_width() / 2
            # get_y() is where the bar starts so we add the height to it.
            text_y = bar.get_y() + bar_value
            # If we want the text to be the same color as the bar, we can
            # get the color like so:
            bar_color = bar.get_facecolor()
            bar_color = "#222222"
            # If you want a consistent color, you can just set it as a constant, e.g. #222222
            ax.text(text_x, text_y, text, ha="center", va="bottom", color=bar_color,
                    size=12)

        plt.show()

    pos_words = data["tidy_tweet"][data["label"] == 0]
    neg_words = data["tidy_tweet"][data["label"] == 1]

    print(neg_words.head(15))
    print(pos_words.head(15))

    # todo rethink: pos/neg words will be in both sets so there are no true pos/neg words
    label_ratio_pie_chart([len(pos_words), len(neg_words)],
                          title="Label Ratio",
                          labels=["Positive", "Negative"])

    word_freq_bar_plot(neg_words, title="Negative Labeled Tweets", num_words_to_plot=20)
    word_freq_bar_plot(pos_words, title="Positive Labeled Tweets", num_words_to_plot=20)

    # fixme is this correct, per total words and per tweet??? plots are identical, is len(pos_word)==len(pos_words)???
    # word frequency percentage for total words (to better compare negative/positive word frequencies)
    word_freq_bar_plot(pos_words, title="Positive Labeled Tweets", num_words_to_plot=20, total_count=len(pos_words),
                       y_suffix="(Percentage Per Total Words)")
    word_freq_bar_plot(neg_words, title="Negative Labeled Tweets", num_words_to_plot=20, total_count=len(neg_words),
                       y_suffix="(Percentage Per Total Words)")

    pos_tweets = data["tidy_tweet"][data["label"] == 0]
    neg_tweets = data["tidy_tweet"][data["label"] == 1]

    word_freq_bar_plot(pos_words, title="Positive Labeled Tweets", num_words_to_plot=20,
                       total_count=len(pos_tweets), y_suffix="(Percentage Per Tweet)")
    word_freq_bar_plot(neg_words, title="Negative Labeled Tweets", num_words_to_plot=20,
                       total_count=len(neg_tweets), y_suffix="(Percentage Per Tweet)")


def visualize_score(data: OrderedDict, title_prefix=None):
    def raw_score(d):
        for key, d in d.items():
            print("-" * 30)
            print(f"{key} tweets")
            print("-" * 30)
            for i in d:
                print(i)

    # todo F1-scores for (un)specific models + label 0/1 average='binary' + average='weighted'
    def grouped_bar_chart(d, title):
        # grouping on average_label
        ax = sns.barplot(x="tweet_count", y="f1", hue="label", data=d)
        plt.title(f"{title_prefix}: {title}")
        plt.xlabel("Tweet Count")
        plt.ylabel("F1-Score")
        # todo add description label
        plt.legend(loc='center left', prop={'size': 10}, title="Labels")
        # todo rename hue values

        # ax.set(xlabel="Tweet Count")
        # ax.set(ylabel='F1-Score')
        ax.yaxis.grid(True, color='#EEEEEE')
        # adding values on top of each bar
        # source: https://www.pythoncharts.com/matplotlib/grouped-bar-charts-matplotlib/
        for bar in ax.patches:
            # The text annotation for each bar should be its height.
            bar_value = bar.get_height()
            # Format the text with two decimals.
            text = f'{bar_value:.2f}'
            # This will give the middle of each bar on the x-axis.
            text_x = bar.get_x() + bar.get_width() / 2
            # get_y() is where the bar starts so we add the height to it.
            text_y = bar.get_y() + bar_value
            # If we want the text to be the same color as the bar, we can
            # get the color like so:
            bar_color = bar.get_facecolor()
            bar_color = "#222222"
            # If you want a consistent color, you can just set it as a constant, e.g. #222222
            ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color,
                    size=12)
        # todo save figure
        # plt.savefig("barplot_Seaborn_barplot_Python.png")
        plt.show()

    # raw_score(data)
    grouped_bar_chart(data, "Score Depending on Tweet Count")


def main2():
    # test_tweets data is not annotated with labels for positive/negative tweets (unlabeled). Therefore it will
    # only be used for training the word2vec models. Train data (labeled) will also be used for training the word2vec
    # models as well as training the classifiers according to training/test sets in order to evaluate the model's
    # performance (test tweets will not be used for training the word2vec models nor classifiers)

    # always show all columns and rows on a panda's DataFrame
    pandas.options.display.max_columns = None
    # pandas.options.display.max_rows = None

    labeled = pandas.read_csv(f"{data_path}/train_E6oV3lV.csv")
    unlabeled = pandas.read_csv(f"{data_path}/test_tweets_anuFYb8.csv")

    # todo set shuffle=True and save/load csv todo what ratio to split (80 to 20, 70 to 30, test!) todo which approach (
    #  holdout, k fold cross validation, stratified k fold cross validation)? -> since we are using annotated data (
    #  semi supervised learning) holdout enough, but make sure ratio of positive and negative tweets are equally
    #  spread within training and test data
    # TODO after cross validation still F1-score of 0.89 (same limit=100)? (Yes, 24.08.21)
    # TODO delete models and recreate them (Done 24.08.21)

    # cross validation
    # split data into train and test
    # train, test = train_test_split(labeled, test_size=0.2, shuffle=False)
    # train, test = train_test_split(labeled, test_size=0.2, shuffle=False)
    # fixme shuffle=True not working
    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    # a = skf.get_n_splits(train["tweet"], train["label"])
    # print(a)
    # todo evaluate models for every fold
    # todo calculate for model skill score mean + standard deviation or standard error
    fold = 0
    for train_index, test_index in skf.split(labeled["tweet"], labeled["label"]):
        fold += 1
        print(f"Fold - {fold} -")
        print(f"TRAIN (len: {len(train_index)}: {train_index} TEST (len: {len(test_index)}: {test_index}")
        train_tweets, test_tweets = labeled["tweet"].iloc[train_index], labeled["tweet"].iloc[test_index]
        train_labels, test_labels = labeled["label"].iloc[train_index], labeled["label"].iloc[test_index]
        train_ids, test_ids = labeled["id"].iloc[train_index], labeled["id"].iloc[test_index]
        print(f"Current TRAIN (len: {len(train_tweets)}) TEST (len: {len(test_tweets)})")

    # save stratified data as DataFrame
    train = DataFrame(list(zip(train_ids, train_labels, train_tweets)), columns=["id", "label", "tweet"])
    # train = DataFrame()
    # train["id"] = train_ids
    # train["label"] = train_labels
    # train["tweet"] = train_tweets
    test = DataFrame(list(zip(test_ids, test_labels, test_tweets)), columns=["id", "label", "tweet"])
    # test = DataFrame()
    # test["id"] = test_ids
    # test["label"] = test_labels
    # test["tweet"] = test_tweets

    # reset indices
    # train = train.reset_index(drop=True)
    # test = test.reset_index(drop=True)
    print(f"types: {type(train['tweet']), type(train)}")

    print(
        f"len(train_tweets, labels, test_tweets, labels): {len(train_tweets), len(train_labels), len(test_tweets), len(test_labels)}")
    # simple test for checking if columns were not shifted
    for i in range(50):
        index = random.randint(0, len(train_tweets) - 1)
        index = i
        print(f"id: {train_ids[index]}, label (0,1): {train_labels[index]}, tweet: {train_tweets[index]}")
        print(f"id: {train['id'][index]}, label (0,1): {int(train['label'][index])}, tweet: {train['tweet'][index]}")
        if not train_tweets[index] == train["tweet"][index]:
            print(f"FAIL")
    # check ratios of positive/negative labeled tweets after cross validation
    print("*" * 30)
    print("train")
    show_pos_neg_ratio(train)
    print("*" * 30)
    print("test")
    show_pos_neg_ratio(test)
    # sys.exit()

    print(f"len(labeled): {len(labeled)}")
    print(f"len(unlabeled): {len(unlabeled)}")
    print("---------------------------------")
    print(f"len(train): {len(train)}")
    print(f"len(test): {len(test)}")
    print("---------------------------------")
    print(f"train: {train.head()}")
    print(f"test: {test.head()}")

    t = time()
    # preprocess both data separately (add column "tidy_tweet")
    train = preprocess(train)
    test = preprocess(test)
    show_used_time(t, "Time for preprocessing")

    # use only labeled data because of training classifiers otherwise there will be no labels to use
    tweets = train
    # tweets = tweets.append(test, ignore_index=False)
    # tweets = tweets.reset_index(drop=True)
    # print(f"----> len(tweets): {len(tweets)}")

    # appending unlabeled data
    # tweets = train.append(unlabeled, ignore_index=False)
    # tweets = tweets.reset_index(drop=True)

    # todo use dictionary?
    tweets_dictionary = corpora.Dictionary(tweets['tidy_tweet'])
    # tweets_dictionary.save('')
    print(tweets_dictionary)
    # print(tweets_dictionary.token2id)

    # todo use corpus streaming (one document at a time)?
    # https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#sphx-glr-auto-examples-core-run-corpora-and-vector-spaces-py

    print(f"len(tweets): {len(tweets.index)}")
    print(f"tweets: {tweets['tweet'].head()}")
    print(f"tweets: {tweets['tweet'].iloc[len(unlabeled):len(unlabeled) + 5]}")

    print(f"len(train): {len(train)}")
    print(f"len(test): {len(test)}")

    specific_w2v_models = train_specific_models(Method.WORD2VEC, tweets["tidy_tweet"], [100, 1000, 10000, 20000, 30000])
    specific_d2v_models = train_specific_models(Method.DOC2VEC, tweets, "tidy_tweet",
                                                [100, 1000, 10000, 20000, 30000])
    model = Word2Vec()
    d2v = Doc2Vec()
    # d2v.dv.vectors

    # todo w2v test
    w2v_test(specific_w2v_models["all"], train, test)
    sys.exit()

    # todo TESTING remove afterwards
    # pipeline_w2v(specific_w2v_models, train, test)
    # pipeline(specific_d2v_models, train, test)

    # todo remove until exit (inclusive)
    # specific_d2v_models.pop("all")
    print(f"train tweets:")
    dict_mdc_train = train_d2v_sentiment_classifiers(specific_d2v_models, train["tidy_tweet"], train["label"], False)
    # evaluate_f1_scores(dict_mdc_train, train["label"])

    print("-" * 30)
    print(f"Test Tweets:")
    print("-" * 30)
    # todo use trained classifier and test on test data
    # dict_mdc_test = train_d2v_sentiment_classifiers(specific_d2v_models, test["tidy_tweet"], test["label"])
    # evaluate_f1_scores(dict_mdc_test, test["label"])
    # todo why no error?
    # evaluate_d2v_specific_models(specific_d2v_models, test["tidy_tweet"], test["label"])

    # print(f"END")
    # sys.exit()

    """""
    for key, mdc in dict_mdc_train.items():
        # train classifier, docvecs from test, test labels
        # infer new vector for every document
        length = len(test.index)
        docvecs = [mdc.model.infer_vector(test["tidy_tweet"][i]) for i in range(length)]
        print("-" * 30)
        print(f"model['{key}']: {mdc.model}")
        print(f"{evaluate_f1_score(mdc.classifier, docvecs, test['label'])}")
    """

    print("*" * 30)
    print(train["tidy_tweet"].iloc[0])
    print("*" * 30)
    # train the Logistic Regression Classifier
    # todo setting this to 100 still gets and F1-Score of 0.89, is there a mistake? Also after stratifiedkshuffle 0.89 (without shuffle)
    limit = 100
    current_model = specific_d2v_models[str(limit)]

    print("*" * 30)
    print(f"len(train): {len(train['tidy_tweet'])}")
    print("*" * 30)

    limit = 100
    print(f"train labels: {train.label.head(limit)}")
    print(f"test labels: {test.label.head(limit)}")
    # todo pandas condition label = 1
    #  find real position index -> df.index.get_loc(index)
    print("*-" * 30)
    print(f"{train.label.head(15)}")
    print("*-" * 30)
    print(f"{train.loc[train['label'] == 1].head(limit)}")

    # todo how to kFoldSplit so that within 100 tweets also label=1? more splits?
    # todo how exactly does stratifiedKFold work?
    # sys.exit()
    # this is working
    # evaluate_d2v_model(specific_d2v_models[str(limit)], test, train[:limit])

    # todo change keys of models (only 100, 1000, 10.000, 20.000, all (25566))
    # fixme here train all elements but when iterating over specific_d2v_models.items() train only 2 elements???
    def testing(train_param):
        for key in [100, 1000, 10000, 20000, 30000, "all"]:
            if key == "all":
                print(f"Key = {key} tweets : len = {len(train_param)}")
                break
            # fixme why only 2 elements in train (100, 1.000, 10.000)
            print(f"Key = {key} tweets : len = {len(train_param.iloc[:int(key)])}")
        # sys.exit()

    testing(train)

    for key, model in specific_d2v_models.items():
        if key == "all":
            print(f"Key = {key} tweets : len = {len(train)}")
            break
        # fixme why only 2 elements in train (100, 1.000, 10.000)
        print(f"Key = {key} tweets : len = {len(train.iloc[:int(key)])}")
        # todo visualize_score unique words in model
        # print(f"model with {len(model.dv)} tweets, {len(model.wv)} words, len(train.iloc[:{int(key)}] = {len(train.iloc[:int(key)])}")
        # count of word
        # Doc2Vec().wv.get_vecattr("trump", "count")
    testing(train)
    # sys.exit()
    evaluate_d2v_models(specific_d2v_models, train, test)
    sys.exit()

    # todo do not use evalute_f1_score def, write it out here!
    # todo remove
    mv = MeanEmbeddingVectorizer(w2v)
    # mv = TfidfEmbeddingVectorizer(w2v)
    mv.fit(train["tidy_tweet"][:key], train["label"][:key])
    test_vectors = mv.transform(test["tidy_tweet"][:key])
    print(f"Test MeanEmbeddingVectorizer: {evaluate_f1_score(log_reg, test_vectors, test['label'][:key])}")

    # fixme should not score be close to 1 because training tweets are used (insanity check)?
    # todo try model.wv and not model.wv.syn0, compare (not working even with model.wv.syn0norm -> not iterable)
    for key, model in specific_w2v_models.items():
        print(f"{key} tweets, {model}")
        w2v = {w: vec for w, vec in zip(model.wv.index_to_key, model.wv.syn0)}
        if "all" in key:
            length = len(tweets)
        else:
            length = int(key)

        # mean word embedding
        tweets_vectors = word_embedding(w2v, train["tidy_tweet"][:length], train["label"][:length], 0)
        print(f"-" * 20)
        print(f"len(w2v): {length}")
        print(f"len(vectors): {len(tweets_vectors)}, len(labels): {len(train[:length])}")
        # print(f"{tweets_vectors[0]}")
        print(f"-" * 20)
        log_reg = train_classifier(tweets_vectors, train["label"][:length])
        # insanity check (evaluate by classifying already trained tweets)
        print(f"Training (insanity check) mean: {evaluate_f1_score(log_reg, tweets_vectors, train['label'][:length])}")
        # evaluate by classifying all tweets from training data even if they were not trained, e.g. model
        # with 100 tweets will be evaluated on all available training data)
        # fixme
        # tweets_vectors = word_embedding(w2v, train["tidy_tweet"], train["label"], 0)
        # print(f"Training mean: {evaluate_f1_score(log_reg, tweets_vectors, train['label'])}")

        # todo also do for test data
        # test_length = len(test)
        # tweets_vectors = word_embedding(w2v, test["tidy_tweet"][:test_length], test["label"][:test_length], 0)
        # # fixme
        # # do not train classifier with test data, use previous classifier trained with train data
        # # log_reg = train_classifier(tweets_vectors, test["label"][:test_length])
        # print(f"Test mean: {evaluate_f1_score(log_reg, tweets_vectors, test['label'][:test_length])}")

        # tfidf word embedding
        tweets_vectors = word_embedding(w2v, train["tidy_tweet"][:length], train["label"][:length])
        log_reg = train_classifier(tweets_vectors, train["label"][:length])
        print(f"Training (insanity check) tfidf: {evaluate_f1_score(log_reg, tweets_vectors, train['label'][:length])}")

        # evaluate by classifying all tweets from training data even if they were not trained, e.g. model
        # with 100 tweets will be evaluated on all available training data)
        # fixme
        # tweets_vectors = word_embedding(w2v, train["tidy_tweet"], train["label"])
        # print(f"Training mean: {evaluate_f1_score(log_reg, tweets_vectors, train['label'])}")

        # todo also do for test data
        # tweets_vectors = word_embedding(w2v, test["tidy_tweet"], test["label"][:length])
        # log_reg = train_classifier(tweets_vectors, test["label"][:length])
        # print(f"Test tfidf: {evaluate_f1_score(log_reg, tweets_vectors, test['label'][:length])}")
    # todo for every w2v model get classifier (maybe return mdc)

    # todo for every mdc model evaluate f1-score
    """
    current_model = specific_d2v_models["all"]
    print(f'TWEETS: {tweets["tidy_tweet"][0]}')
    print(f'TYPE: {type(current_model).__name__}')
    vector = current_model.infer_vector(tweets["tidy_tweet"][0])
    print(f"Vector: {vector}")
    print(specific_w2v_models["all"].wv.vectors.shape)
    words = list(specific_w2v_models["all"].wv.vocab)
    print(f"All tweets vocabulary size: {len(words)}")
    """

    # load pretrained model
    # google_news_w2v_model = api.load("word2vec-google-news-300")
    # todo how to use pretrained model with Doc2Vec? (test with type(model).__name__)
    # print(f'TYPE: {type(google_news_w2v_model).__name__}')

    # testing models
    # not working with models with less than 10000 tweets (tested words must be in model)
    # print("Test with 100 tweets")
    # test_w2v_model(specific_w2v_models["100"])
    # print("Test with 1000 tweets")
    # test_w2v_model(specific_w2v_models["1000"])
    # print("Test with 10000 tweets")
    # test_w2v_model(specific_w2v_models["10000"])

    """
    print("Test with 20000 tweets")
    test_w2v_model(specific_w2v_models["20000"])

    print("Test with 30000 tweets")
    test_w2v_model(specific_w2v_models["30000"])

    print("Test with all tweets")
    test_w2v_model(specific_w2v_models["all"])
    print("-----------")
    """

    # test_w2v_model(google_news_w2v_model)

    # print(tweets["tidy_tweet"].head())
    # print(tweets["tidy_tweet"].tail())

    # print(tweets["tidy_tweet"][0])
    # print(current_model.docvecs[0])

    # inferred_vector = current_model.infer_vector(tweets["tidy_tweet"][0])
    # print(current_model.docvecs.most_similar([inferred_vector], topn=len(current_model.docvecs)))

    # todo compare with tutorial sanity check
    # Assessing model
    current_model = specific_d2v_models["100"]
    ranks = []
    second_ranks = []
    count = 100  # len(tweets)
    t = time()
    for doc_id in range(count):
        # show progress
        if doc_id % 10 == 0:
            print(f"{doc_id} / {count}")
        inferred_vector = current_model.infer_vector(tweets["tidy_tweet"].iloc[doc_id], steps=300)
        # sims = current_model.docvecs.most_similar([inferred_vector], topn=len(current_model.docvecs))
        # gensim 4
        sims = current_model.wv.most_similar([inferred_vector], topn=len(current_model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        second_ranks.append(sims[1])
    show_used_time(t)

    counter = Counter(ranks)
    print(f"{counter}")

    print("-" * 30)
    print(tweets["tidy_tweet"])

    doc_id = 0
    print('Document ({}): «{}»\n'.format(doc_id, ' '.join(tweets["tidy_tweet"][doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % current_model)
    for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(tweets["tidy_tweet"][sims[index][0]])))

    sys.exit()
    # todo compare similarity with docvec[0] and infeer_vector()
    d = current_model.docvecs.doctag_syn0norm[doc_id]
    i = current_model.infer_vector(tweets["tidy_tweet"].iloc[doc_id])
    d = np.array(d).reshape(1, -1)
    i = np.array(i).reshape(1, -1)
    print(f"cos: {1 - sklearn.metrics.pairwise.cosine_similarity(i, d)}")
    # from scipy.spatial import distance
    # print(f"cos: {1 - distance.distance.cosine(i, d)}")

    print(f"tweet: {tweets['tidy_tweet'][doc_id]}")
    # todo fixme why is not trained doc returned as the most similar doc? Also similarity should be close to 1
    ivec = current_model.infer_vector(tweets["tidy_tweet"][doc_id], steps=20, alpha=0.025)
    # print(current_model.docvecs.most_similar(positive=[ivec], topn=100))
    # gensim 4
    print(current_model.wv.most_similar(positive=[ivec], topn=100))

    # print(f"Similarity: {current_model.docvecs.similarity(0, 2)}")
    # gensim 4
    print(f"Similarity: {current_model.wv.similarity(0, 2)}")
    # print(f"len(current_model.docvecs): {len(current_model.docvecs)}")
    # print(f"type(model.docvecs): {type(current_model.docvecs).__name__}")

    # classification

    # for i in range(len(current_model.docvecs)):
    #     train_vectors.append(current_model.docvecs[i])
    #     if train_vectors[i].dtype != np.float32:
    #         print(f"dtype: {train_vectors[0].dtype}")
    #         print(f"dtype: {train_vectors[0]}")

    # df.apply does not work
    # test_vectors = test["tidy_tweet"].apply(current_model.infer_vector)

    print(f"train tweets:")
    dict_mdc_train = train_d2v_sentiment_classifiers(specific_d2v_models, train["tidy_tweet"], train["label"], False)
    evaluate_f1_scores(dict_mdc_train, train["label"])

    print(f"test tweets:")
    dict_mdc_test = train_d2v_sentiment_classifiers(specific_d2v_models, test["tidy_tweet"], test["label"])
    evaluate_f1_scores(dict_mdc_test, test["label"])

    """
    print(f"-" * 30)
    print(f"train tweets:")
    print(f"-" * 30)
    evaluate_d2v_specific_models(specific_d2v_models, train["tidy_tweet"], train["label"], False)

    print(f"-" * 30)
    print(f"test tweets:")
    print(f"-" * 30)
    evaluate_d2v_specific_models(specific_d2v_models, test["tidy_tweet"], test["label"])
    """


if __name__ == '__main__':
    t = time()
    main()
    show_used_time(t, "Time for main()")
    # print(api.load('word2vec-google-news-300', return_path=True))

# todo use sklearn.pipeline?
"""
How to call

# TSA -> TweetSentimentAnalyzer
tsa = TSA(csv)
# tsa.readData(csv)
tsa.preprocess()
tsa.create_training_test(test_size) / tsa.cross_validation(k_fold)

tsa.visualize_train_data() / tsa.visualize_data(tsa.train)
tsa.visualize_test_data() / tsa.visualize_data(tsa.test)

scores = []
for tweet_count in [100, 1000, 10000, 20000, len(train)]:
    tsa.train_specific_model(tweet_count=len(self.train))

    tsa.train_classifier()
    
    # evaluation (print F1-Score and result dataframe?)
    scores.append(tsa.test_classifier())

tsa.visualize_score(scores)

Notes structure of program

readData(csv)
    as streamed corpus (custom class with __iter__)
preprocessData
createTrainingData
createTestData
visualize_score
    train
    test
createModels
    w2v serializeW2v (https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#corpus-formats)
    
        specific
            100
            1000
            10000
            20000
            25000
        unspecific
            google-news
    d2v
        specific
        unspecific
createClassifier
    train
    test
evaluate (benchmark)
    create DataFrame
    visualize_score
        plotA
        plotB
        ...
.gitignore models?
"""
