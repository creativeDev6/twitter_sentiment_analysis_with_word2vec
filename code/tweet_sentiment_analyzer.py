import multiprocessing
import os
from time import time

import gensim
import pandas
from gensim.models.doc2vec import TaggedDocument, Word2Vec
from gensim.parsing import preprocess_string
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold, train_test_split

from helper import show_used_time
from code.data import ColumnNames, oversample, distribute_equally
from preprocessing import remove_pattern, remove_html_entities, hashtag_extract
from model import Method, load_or_create_model, load_pretrained_model, create_w2v_model, create_d2v_model, read_corpus
from classification import Classifier
from visualization import ratio_pie_chart, word_freq_bar_plot, grouped_bar_chart

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

# always show all columns and rows on a panda's DataFrame
pandas.options.display.max_columns = None
# pandas.options.display.max_rows = None


class TweetSentimentAnalyzer:
    """
    csv = None

    raw_data = None
    data = None

    train = None
    test = None

    method = Method.WORD2VEC
    # depending on k_fold
    fold_size = None
    """

    def __init__(self, csv, column_names: ColumnNames):
        self.csv = csv
        self.column = column_names
        self.raw_data = None
        self.data = None

        self.train = None
        self.test = None

        self.method: Method = Method.WORD2VEC
        self.model: Word2Vec = None
        self.classifier: Classifier = None

        # depending on k_fold
        self.fold_size = None

        self.__read_data(csv)

    def __read_data(self, csv):
        self.raw_data = pandas.read_csv(csv)
        self.preprocess(self.raw_data)
        # todo remove
        print("*" * 30)
        print(f"len(self.raw_data): {len(self.raw_data)}")
        print(f"len(self.data): {len(self.data)}")
        print("*" * 30)

    def preprocess(self, df: DataFrame):
        """
        The preprocessed tweet will be stored in the column "tidy_tweet".

        Removed hashtags will be stored in the column "hashtags".
        """
        # drop all other except essential columns (tweet and label)
        df = df[[self.column.tweet, self.column.label]]

        # remove twitter handles (@user)
        df.loc[:, self.column.tidy_tweet] = df[self.column.tweet].apply(remove_pattern, args=(r"@\w*",))

        # todo remove &amp; Unicode, e.g. 
        df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].apply(remove_html_entities)

        # remove special characters, numbers, punctuations (everything except letters and #)
        # todo what about ', like in don't, haven't?
        df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].str.replace("[^a-zA-Z#]", " ")
        # df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].apply(remove_unicode)

        # extract hashtags into separate column
        df.loc[:, self.column.hashtags] = df[self.column.tidy_tweet].apply(hashtag_extract)

        # remove hashtags todo IMPORTANT recheck below comment after wrangling data (does it still worsen the
        #  result?) Info: the F1-score was worse than leaving them (0.38 to 0.48) when compared with all the training
        #  tweets. Also this lead to more rows having zero word tokens (94 to 4) and therefore removing them in the
        #  last step of this function (remove words with no word tokens) df.loc[:, self.column.tidy_tweet] = df[
        #  self.column.tidy_tweet].str.replace(r"#(\w+)", " ")
        #  remove only hashtags '#' (otherwise 833 rows will be removed and
        #  hashtags might contain important words for deciding the tweet's sentiment)
        df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].str.replace("#", "")

        # todo use other gensim preprocessing functions
        # gensim.parsing.preprocessing.utils.keep_vocab_item()
        # gensim.parsing.preprocessing.
        # todo remove short, stopwords, stem
        custom_filters = [gensim.parsing.preprocessing.strip_tags,
                          gensim.parsing.preprocessing.strip_short,
                          gensim.parsing.preprocessing.remove_stopwords,
                          gensim.parsing.preprocessing.stem_text]
        # df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].apply(
        # gensim.parsing.preprocessing.preprocess_string, filters=gensim.parsing.preprocessing.DEFAULT_FILTERS)
        df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].apply(
            lambda x: preprocess_string(x, filters=custom_filters))

        # use gensim for preprocessing
        # simple preprocess without stemming
        # df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].apply(gensim.utils.simple_preprocess)
        # preprocess with stemming
        # df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].apply(preprocess_string)

        # display rows (tweets) with no words
        counter = 0
        for row in df.itertuples():
            # fixme tweets with a few words are also removed (833 rows removed)
            #  Pandas(Index=28324, id=28325, label=0, tweet='#cute   how are you? see you? -  ', tidy_tweet=[], hashtags=['cute'])
            if len(row.tidy_tweet) == 0:
                print(f"No words in row. Row will be removed: {row}")
                counter += 1
        print(f"Removed rows: {counter}")
        # remove rows with no word tokens after preprocessing
        # Info: prevents having to calculate the document_vector for no words in tweet or rather to define a value for
        # empty tweets
        df = df[df[self.column.tidy_tweet].map(len) > 0]
        df = df.reset_index(drop=True)

        self.data = df

        return df

    def split_train_test(self, test_size=0.2):
        self.train, self.test = train_test_split(self.data, test_size=test_size, stratify=self.data[self.column.label])

    def cross_validation(self, k_fold=5):
        # fixme shuffle=True not working (should work tested on 2021-09-22)
        skf = StratifiedKFold(n_splits=k_fold, shuffle=False, random_state=None)

        # fixme
        #  evaluation in each fold + store values and calculate mean
        #  otherwise (holdout validation) last fold = test and {1-4} folds = train
        fold = 0
        # todo divide data into 3 sections:
        #  training, validation (used for tweaking parameters) and test set (testing performance)?
        for train_index, test_index in skf.split(self.data[self.column.tweet], self.data[self.column.label]):
            fold += 1

            print(f"Fold - {fold} -")
            print(f"TRAIN (len: {len(train_index)}: {train_index} TEST (len: {len(test_index)}: {test_index}")
            print("tweet ", self.data[self.column.tweet].iloc[train_index],
                  self.data[self.column.tweet].iloc[test_index])
            print("tidy_tweet ", self.data[self.column.tidy_tweet].iloc[train_index],
                  self.data[self.column.tidy_tweet].iloc[test_index])
            print("hashtags ", self.data[self.column.hashtags].iloc[train_index],
                  self.data[self.column.hashtags].iloc[test_index])
            print("labels ", self.data[self.column.label].iloc[train_index],
                  self.data[self.column.label].iloc[test_index])

            train_tweets, test_tweets = self.data[self.column.tweet].iloc[train_index], \
                                        self.data[self.column.tweet].iloc[test_index]
            train_tidy_tweets, test_tidy_tweets = self.data[self.column.tidy_tweet].iloc[train_index], \
                                                  self.data[self.column.tidy_tweet].iloc[test_index]
            train_hashtags, test_hashtags = self.data[self.column.hashtags].iloc[train_index], \
                                            self.data[self.column.hashtags].iloc[test_index]
            train_labels, test_labels = self.data[self.column.label].iloc[train_index], \
                                        self.data[self.column.label].iloc[test_index]
            print(f"Current TRAIN (len: {len(train_tweets)}) TEST (len: {len(test_tweets)})")

        # save stratified data as DataFrame
        self.train = DataFrame(list(zip(train_labels, train_tweets, train_tidy_tweets, train_hashtags)),
                               columns=[self.column.label, self.column.tweet, self.column.tidy_tweet,
                                        self.column.hashtags])
        self.test = DataFrame(list(zip(test_labels, test_tweets, test_tidy_tweets, test_hashtags)),
                              columns=[self.column.label, self.column.tweet, self.column.tidy_tweet,
                                       self.column.hashtags])

        self.fold_size = len(self.test)

    def oversample(self, ratio=1):
        self.train = oversample(self.train, ratio=ratio)

    def distribute_labels_equally_in_train(self):
        self.train = distribute_equally(self.train, self.column.label)

    def __show_class_distribution(self, df: DataFrame, tweet_counts, title_prefix=""):
        from collections import Counter
        print("*" * 30)
        print(f"{title_prefix} Class distribution")
        print("*" * 30)
        for tweet_count in tweet_counts:
            dist = Counter(df.iloc[:tweet_count][self.column.label])
            ratio = dist[0] // dist[1]
            print(f"tweet_count: {tweet_count}")
            print(f"Distribution: {Counter(df.iloc[:tweet_count][self.column.label])}")
            print(f"Ratio: 1:{ratio}")
            print("-" * 30)

    def show_train_class_distribution(self, tweet_counts: [int]):
        self.__show_class_distribution(self.train, tweet_counts, title_prefix="Train")

    def show_test_class_distribution(self):
        self.__show_class_distribution(self.test, tweet_counts=[len(self.test)], title_prefix="Test")

    def __visualize_data(self, data: DataFrame, title_prefix: str):
        pos_words = data[self.column.tidy_tweet][data[self.column.label] == 0]
        neg_words = data[self.column.tidy_tweet][data[self.column.label] == 1]

        print(neg_words.head(15))
        print(pos_words.head(15))

        # todo rethink: pos/neg words will be in both sets so there are no true pos/neg words
        ratio_pie_chart([len(pos_words), len(neg_words)],
                        title_prefix=title_prefix,
                        title="Label Ratio",
                        labels=["Positive", "Negative"])

        word_freq_bar_plot(neg_words, title_prefix=title_prefix, title="Negative Labeled Tweets", num_words_to_plot=20)
        word_freq_bar_plot(pos_words, title_prefix=title_prefix, title="Positive Labeled Tweets", num_words_to_plot=20)

        # fixme is this correct, per total words and per tweet??? plots are identical, is len(pos_word)==len(pos_words)???
        # word frequency percentage for total words (to better compare negative/positive word frequencies)
        word_freq_bar_plot(pos_words, title_prefix=title_prefix, title="Positive Labeled Tweets", num_words_to_plot=20,
                           total_count=len(pos_words),
                           y_suffix="(Percentage Per Total Words)")
        word_freq_bar_plot(neg_words, title_prefix=title_prefix, title="Negative Labeled Tweets", num_words_to_plot=20,
                           total_count=len(neg_words),
                           y_suffix="(Percentage Per Total Words)")

        pos_tweets = data[self.column.tidy_tweet][data[self.column.label] == 0]
        neg_tweets = data[self.column.tidy_tweet][data[self.column.label] == 1]

        word_freq_bar_plot(pos_words, title_prefix=title_prefix, title="Positive Labeled Tweets", num_words_to_plot=20,
                           total_count=len(pos_tweets), y_suffix="(Percentage Per Tweet)")
        word_freq_bar_plot(neg_words, title_prefix=title_prefix, title="Negative Labeled Tweets", num_words_to_plot=20,
                           total_count=len(neg_tweets), y_suffix="(Percentage Per Tweet)")

    def visualize_train_data(self):
        self.__visualize_data(self.train, "Train")

    def visualize_test_data(self):
        self.__visualize_data(self.test, "Test")

    def train_model(self, pretrained_model: Word2Vec = None, tweet_count=None, force_retrain=False):
        """ If tweet_count=None then entire train set will be used. """

        if pretrained_model is None:
            print(f"New model: will be trained.")
            # todo remove force_retrain
            self.model = load_or_create_model(self.method, self.train[self.column.tidy_tweet], tweet_count,
                                              force_retrain=True)
            # todo remove
            print(f"****** len(train): {len(self.train)}")
            # print(f"****** vocabulary: {self.model.wv.index_to_key}")
        else:
            print(f"Pretrained model: '{pretrained_model}' will be used.")
            # self.model = pretrained_model
            # set other fields
            # todo which fold_size?
            # self.fold_size = len(pretrained_model.wv)
            self.fold_size = len(self.train)
            self.method = Method.WORD2VEC

            self.model = pretrained_model

    def load_pretrained_model(self):
        return load_pretrained_model()

    def train_classifier(self, tweet_count: int = None):
        if tweet_count is None:
            tweet_count = len(self.train)

        self.classifier = Classifier(self.method, self.model, vector_size,
                                     self.train.iloc[:tweet_count],
                                     self.test, self.column)
        self.classifier.train_classifier()

    def test_classifier(self):
        return self.classifier.test_classifier()

    def visualize_score(self, scores: [], title_prefix=None):
        # score_data = namedtuple("scoreData", "tweet_count label f1 precision recall")
        scores_data = DataFrame(data=scores)
        grouped_bar_chart(scores_data, title="Score Depending on Tweet Count", title_prefix=title_prefix)


"""
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
"""