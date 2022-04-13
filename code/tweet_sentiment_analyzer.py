import logging
import os
from collections import Counter
from enum import Enum

import gensim
import numpy as np
import pandas
from gensim.models.doc2vec import Word2Vec
from gensim.parsing import preprocess_string
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from classification import Classifier
from code.data import ColumnNames, oversample, distribute_equally, get_duplicates, count_duplicates
from model import Method, load_or_create_model, load_pretrained_model
from preprocessing import remove_pattern, remove_html_entities, hashtag_extract
from visualization import ratio_pie_chart, word_freq_bar_plot, grouped_bar_chart

# region variables

cwd = os.getcwd()

# data
data_path = f"{cwd}/data"
cleaned_data_path = f"{data_path}/cleaned"

# plots
plot_path = f"{cwd}/plots"

# endregion

# always show all columns and rows on a panda's DataFrame
# pandas.options.display.max_rows = None
pandas.options.display.max_columns = None


class TestSet(Enum):
    VALIDATION = "Validation"
    TEST = "Test"


class TweetSentimentAnalyzer:
    """
    csv = None

    raw_data = None
    data = None

    train = None
    test = None

    method = Method.WORD2VEC
    """

    def __init__(self, csv, column_names: ColumnNames):
        self.csv = csv
        self.column = column_names
        self.essential_columns = [self.column.id, self.column.label, self.column.tweet]
        self.raw_data = None
        self.data = None

        self.train = None
        self.validation = None
        self.test = None

        self.method: Method = Method.WORD2VEC
        self.model: Word2Vec = None
        self.is_pretrained_model = False
        self.classifier: Classifier = None

        self.__read_data(csv)

    def __read_data(self, csv):
        self.raw_data = pandas.read_csv(csv)

    def __remove_duplicates(self, df: DataFrame):
        """
        Removes all duplicates in place except the first occurrence.
        :param df:
        :return:
        """
        duplicates = get_duplicates(df)
        duplicates_count = len(duplicates.index)
        print("*" * 30)
        print("Duplicates, following rows will be removed:")
        print("*" * 30)
        for row in duplicates.itertuples():
            print(f"Duplicate: {row}")
        print(f"Removed rows: {duplicates_count}")

        df.drop_duplicates(subset=self.essential_columns, inplace=True)

    def __preprocess(self, df: DataFrame):
        """
        The preprocessed tweet will be stored in the column "tidy_tweet".

        Hashtags (their content) will be stored in the column "hashtags".
        """
        self.__remove_duplicates(df)

        # drop all other except essential columns (tweet and label)
        df = df[self.essential_columns]

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
        logging.info(f"Column headers: {self.data.columns.values}")

        return df

    def preprocess(self):
        self.__preprocess(self.raw_data)

    def save_preprocessed_data(self):
        if not os.path.exists(cleaned_data_path):
            os.makedirs(cleaned_data_path)

        write_index = False

        self.data.to_csv(f"{cleaned_data_path}/data.csv", index=write_index)
        self.train.to_csv(f"{cleaned_data_path}/train.csv", index=write_index)
        self.validation.to_csv(f"{cleaned_data_path}/validation.csv", index=write_index)
        self.test.to_csv(f"{cleaned_data_path}/test.csv", index=write_index)

    def load_preprocessed_data(self):
        try:
            self.data = pandas.read_csv(f"{cleaned_data_path}/data.csv")
            self.train = pandas.read_csv(f"{cleaned_data_path}/train.csv")
            self.validation = pandas.read_csv(f"{cleaned_data_path}/validation.csv")
            self.test = pandas.read_csv(f"{cleaned_data_path}/test.csv")
        except FileNotFoundError:
            print(f"Make sure '{cleaned_data_path}' and cleaned csv files exist")
        logging.info(f"Column headers: {self.data.columns.values}")

    def train_validation_test_split(self, test_size=0.1, validation_size=0.2, train_size=0.7, shuffle: bool = True,
                                    random_state: int = None):
        # needs round otherwise raises error, e.g. with default size values sum -> 0.999
        sum_sizes = round(train_size + validation_size + test_size, 2)
        if sum_sizes != 1.0:
            raise ValueError(
                f"Sizes: {train_size} (train), {validation_size} (validation) and {test_size} (test) should add up to "
                f"1.0")

        # train_size = 1 - (validation_size + test_size)
        relative_test_size = test_size / (validation_size + test_size)

        self.train, temp_validation = train_test_split(self.data, train_size=train_size, shuffle=shuffle,
                                                       random_state=random_state,
                                                       stratify=self.data[self.column.label])
        self.validation, self.test = train_test_split(temp_validation, test_size=relative_test_size, shuffle=shuffle,
                                                      random_state=random_state,
                                                      stratify=temp_validation[self.column.label])

    def __oversample(self, df: DataFrame, ratio=1, random_state=None):
        print(f"Before oversampling: {Counter(df[self.column.label])}")
        df_oversampled, labels_oversampled = oversample(df, ratio=ratio, random_state=random_state)
        print(f"After oversampling: {Counter(labels_oversampled)}")
        return df_oversampled

    def oversample_validation(self, ratio=1, random_state=None):
        self.validation = self.__oversample(self.validation, ratio=ratio, random_state=random_state)

    def oversample_train(self, ratio=1, random_state=None):
        self.train = self.__oversample(self.train, ratio=ratio, random_state=random_state)

    def show_train_duplicates_distribution(self, tweet_counts: [int]):
        print("*" * 30)
        print("Duplicate Distribution")
        print("*" * 30)
        for tweet_count in tweet_counts:
            current_train = self.train.iloc[:tweet_count]
            without_duplicates = current_train.drop_duplicates(subset=[self.column.tweet])
            without_duplicates_minority = without_duplicates[without_duplicates[self.column.label] == 1]
            without_duplicates_minority_count = len(without_duplicates_minority)
            duplicates_count = count_duplicates(current_train, self.essential_columns)
            try:
                # ratio = tweet_count / duplicates_count
                ratio = duplicates_count / tweet_count
            except ZeroDivisionError:
                ratio = 0
            print(f"Total tweets: {tweet_count}")
            print(f"Distribution: {duplicates_count} : {tweet_count} (duplicates : tweet_count)")
            print(f"Ratio: 1 : {ratio:.2f}")
            print(f"No duplicates: {without_duplicates_minority_count} (in minority class 1)")
            print("-" * 30)

    def distribute_labels_equally_in_train(self):
        self.train = distribute_equally(self.train, self.column.label)

    def __show_class_distribution(self, df: DataFrame, tweet_counts, title_prefix=""):
        print("*" * 30)
        print(f"{title_prefix} Class Labels Distribution")
        print("*" * 30)
        for tweet_count in tweet_counts:
            dist = Counter(df.iloc[:tweet_count][self.column.label])
            ratio = dist[0] / dist[1]
            print(f"Total tweets: {tweet_count}")
            print(f"Distribution: {Counter(df.iloc[:tweet_count][self.column.label])}")
            print(f"Negative tweets: {dist[1]}")
            print(f"Positive tweets: {dist[0]}")
            print(f"Ratio: 1 : {ratio:.2f}")
            print("-" * 30)

    def show_data_class_distribution(self):
        self.__show_class_distribution(self.data, [len(self.data.index)], title_prefix="Data")

    def show_train_class_distribution(self, tweet_counts: [int]):
        self.__show_class_distribution(self.train, tweet_counts, title_prefix="Train")

    def show_validation_class_distribution(self):
        self.__show_class_distribution(self.validation, tweet_counts=[len(self.validation)], title_prefix="Validation")

    def show_test_class_distribution(self):
        self.__show_class_distribution(self.test, tweet_counts=[len(self.test)], title_prefix="Test")

    def __visualize_data(self, data: DataFrame, title_prefix: str):
        num_words_to_plot = 20
        current_plot_path = f"{plot_path}/{title_prefix.lower()}"

        def merge_lists(df: DataFrame):
            concatenated = []
            for tweet in df:
                concatenated.extend(tweet)
            return concatenated

        pos_tweets = data[self.column.tidy_tweet][data[self.column.label] == 0]
        neg_tweets = data[self.column.tidy_tweet][data[self.column.label] == 1]

        # words from positive/negative tweets
        pos_words = merge_lists(pos_tweets)
        neg_words = merge_lists(neg_tweets)

        # unique words from positive/negative tweets
        pos_words_unique = np.unique(pos_words)
        neg_words_unique = np.unique(neg_words)

        # region debugging

        logging.info(f"\n{neg_tweets.head(num_words_to_plot)}")
        logging.info(f"\n{pos_tweets.head(num_words_to_plot)}")

        logging.info(f"len(pos_words): {len(pos_words)}")
        logging.info(f"len(neg_words): {len(neg_words)}")

        logging.info(f"len(pos_words_unique): {len(pos_words_unique)}")
        logging.info(f"len(neg_words_unique): {len(neg_words_unique)}")

        logging.debug(f"pos_words\n{pos_words}")
        logging.debug(f"neg_words\n{neg_words}")

        # endregion

        ratio_pie_chart([len(pos_tweets), len(neg_tweets)],
                        title_prefix=title_prefix,
                        title="Label Ratio",
                        labels=["Positive", "Negative"],
                        save_path=f"{current_plot_path}/1-class_distribution_pie_chart")

        word_freq_bar_plot(neg_tweets, title_prefix=title_prefix, title="Negative Labeled Tweets",
                           num_words_to_plot=num_words_to_plot,
                           save_path=f"{current_plot_path}/2-word_freq-neg_tweets")
        word_freq_bar_plot(pos_tweets, title_prefix=title_prefix, title="Positive Labeled Tweets",
                           num_words_to_plot=num_words_to_plot,
                           save_path=f"{current_plot_path}/3-word_freq-pos_tweets")

        # region for better comparison of word frequencies due to imbalanced class distribution

        # word frequency percentage (occurrences / total (positive|negative) tweets)
        word_freq_bar_plot(pos_tweets, title_prefix=title_prefix, title="Positive Labeled Tweets",
                           num_words_to_plot=num_words_to_plot,
                           total_count=len(pos_tweets.index),
                           multiplier=100,
                           y_label="Word Frequency ÷ Tweets (in %)",
                           save_path=f"{current_plot_path}/4-word_freq_with_tweets-pos_tweets")
        word_freq_bar_plot(neg_tweets, title_prefix=title_prefix, title="Negative Labeled Tweets",
                           num_words_to_plot=num_words_to_plot,
                           total_count=len(neg_tweets.index),
                           multiplier=100,
                           y_label="Word Frequency ÷ Tweets (in %)",
                           save_path=f"{current_plot_path}/5-word_freq_with_tweets-neg_tweets")

        # word frequency percentage (occurrences / total {positive, negative} words)
        word_freq_bar_plot(pos_tweets, title_prefix=title_prefix, title="Total Words From Positive Labeled Tweets",
                           num_words_to_plot=num_words_to_plot,
                           total_count=len(pos_words),
                           multiplier=100,
                           y_label="Word Frequency ÷ Total Words (in %)",
                           save_path=f"{current_plot_path}/6-word_freq_with_words-pos_tweets")
        word_freq_bar_plot(neg_tweets, title_prefix=title_prefix, title="Total Words From Negative Labeled Tweets",
                           num_words_to_plot=num_words_to_plot,
                           total_count=len(neg_words),
                           multiplier=100,
                           y_label="Word Frequency ÷ Total Words (in %)",
                           save_path=f"{current_plot_path}/7-word_freq_with_words-neg_tweets")

        """
        # word frequency percentage (occurrences ÷ total unique (positive|negative) words)
        word_freq_bar_plot(pos_tweets, title_prefix=title_prefix, title="Total Words From Positive Labeled Tweets",
                           num_words_to_plot=num_words_to_plot,
                           total_count=len(pos_words_unique),
                           multiplier=100,
                           y_label="Word Frequency ÷ Total Words (in %)")
        word_freq_bar_plot(neg_tweets, title_prefix=title_prefix, title="Total Words From Negative Labeled Tweets",
                           num_words_to_plot=num_words_to_plot,
                           total_count=len(neg_words_unique),
                           multiplier=100,
                           y_label="Word Frequency ÷ Total Words (in %)")
        """

        # endregion

    def visualize_train_data(self):
        self.__visualize_data(self.train, "Train")

    def visualize_validation_data(self):
        self.__visualize_data(self.validation, "Validation")

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
            self.method = Method.WORD2VEC

            self.model = pretrained_model

    def load_pretrained_model(self):
        self.is_pretrained_model = True
        return load_pretrained_model()

    def train_classifier(self, tweet_count: int = None):
        if tweet_count is None:
            tweet_count = len(self.train)

        self.classifier = Classifier(self.method, self.model, self.is_pretrained_model, self.model.vector_size,
                                     self.train.iloc[:tweet_count],
                                     self.validation,
                                     self.test,
                                     self.column)
        self.classifier.fit()

    def validate_classifier(self):
        return self.classifier.validate_classifier()

    def test_classifier(self):
        return self.classifier.test_classifier()

    @staticmethod
    def visualize_score(scores: [], title=None):
        # score_data = namedtuple("scoreData", "tweet_count label f1 precision recall mcc")
        scores_data = DataFrame(data=scores)
        # y="mcc" can also be used
        # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
        grouped_bar_chart(scores_data, y="f1", title=title)

    # region encapsulate multiple operations

    def __test_model(self, use_set: TestSet = TestSet.VALIDATION, pretrained_model=None, tweet_counts=None):
        test_func = None
        scores = []

        if use_set == TestSet.VALIDATION:
            test_func = self.validate_classifier
        elif use_set == TestSet.TEST:
            test_func = self.test_classifier
        else:
            logging.error(f"Invalid use set '{use_set}'. Expected one of {[e.name for e in TestSet]}.")

        if pretrained_model:
            from model import pretrained_model_name
            for tweet_count in tweet_counts:
                self.train_model(pretrained_model=pretrained_model)
                self.train_classifier(tweet_count)
                scores.extend(test_func())
            self.visualize_score(scores, title=f"{use_set.value}: {pretrained_model_name}")
        else:
            for tweet_count in tweet_counts:
                self.train_model(tweet_count=tweet_count)
                self.train_classifier(tweet_count)
                scores.extend(test_func())
            self.visualize_score(scores, title=f"{use_set.value}: word2vec Specific Models")

    def validate_specific_models_by(self, tweet_counts: [int]):
        self.__test_model(TestSet.VALIDATION, tweet_counts=tweet_counts)

    def validate_unspecific_pretrained_model(self, tweet_counts: [int]):
        self.__test_model(TestSet.VALIDATION, pretrained_model=self.load_pretrained_model(), tweet_counts=tweet_counts)

    def test_specific_models_by(self, tweet_counts: [int]):
        self.__test_model(TestSet.TEST, tweet_counts=tweet_counts)

    def test_unspecific_pretrained_model(self, tweet_counts: [int]):
        self.__test_model(TestSet.TEST, pretrained_model=self.load_pretrained_model(), tweet_counts=tweet_counts)

    # endregion
