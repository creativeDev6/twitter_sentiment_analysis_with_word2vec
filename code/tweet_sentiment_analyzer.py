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

from code.classification import Classifier
from code.data import ColumnNames, oversample, distribute_equally, get_duplicates, count_duplicates
from code.model import Method, load_or_create_model, load_pretrained_model
from code.preprocessing import remove_usernames, remove_html_entities, remove_tags, hashtag_extract
from code.visualization import ratio_pie_chart, word_freq_bar_plot, grouped_bar_chart, bar_chart

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
# show entire column width
pandas.options.display.max_colwidth = None


class TestSet(Enum):
    VALIDATION = "Validation"
    TEST = "Test"


class TweetSentimentAnalyzer:
    """
    Class to compare different word2vec models for sentiment classification on tweets.
    """

    def __init__(self, csv, column_names: ColumnNames, random_state=None):
        self.csv = csv
        self.column = column_names
        self.random_state = random_state

        self.essential_columns = [self.column.id, self.column.label, self.column.tweet]
        self.raw_data = None
        self.data = None

        self.train = None
        self.validation = None
        self.test = None

        self.method: Method = Method.WORD2VEC
        self.model: Word2Vec = None
        self.pretrained_model = None
        self.classifier: Classifier = None

        self.partition_size = None

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

        :param df: DataFrame to be preprocessed.
        :return:
        """

        self.__remove_duplicates(df)

        # drop all other except essential columns (id, tweet and label)
        df = df[self.essential_columns]

        df.loc[:, self.column.tidy_tweet] = df[self.column.tweet].apply(remove_usernames)
        # this is needed because otherwise, e.g. '&amp;' will become 'amp' in the str.replace step (same for tags)
        df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].apply(remove_html_entities)
        df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].apply(remove_tags)

        # remove special characters, numbers, punctuations (everything except letters and #)
        # contradictions (e.g. I'm, didn't, don't or haven't) will be removed because they often contain only stopwords
        # and will later on be removed anyway
        df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].str.replace(r"[^a-zA-Z#]+", " ")

        # extract hashtags into separate column
        df.loc[:, self.column.hashtags] = df[self.column.tidy_tweet].apply(hashtag_extract)

        # remove hashtags
        #  remove only hashtags '#' (otherwise 833 rows will be removed and
        #  hashtags might contain important words for deciding the tweet's sentiment)
        df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].str.replace("#", "")

        # use gensim for preprocessing
        custom_filters = [
            gensim.parsing.preprocessing.strip_short,
            gensim.parsing.preprocessing.remove_stopwords,
            gensim.parsing.preprocessing.stem_text,
        ]
        df.loc[:, self.column.tidy_tweet] = df[self.column.tidy_tweet].apply(
            lambda x: preprocess_string(x, filters=custom_filters))

        # display rows (tweets) with no words
        counter = 0
        for row in df.itertuples():
            if len(getattr(row, self.column.tidy_tweet)) == 0:
                print(f"No words in row. Row will be removed: {row}")
                counter += 1
        print(f"Total removed rows: {counter}")
        # remove rows with no word tokens after preprocessing
        # Info: prevents having to calculate the document_vector for no words in tweet or rather to define a value for
        # empty tweets
        df = df[df[self.column.tidy_tweet].map(len) > 0]
        df = df.reset_index(drop=True)

        self.data = df
        logging.info(f"Column headers after preprocessing: {self.data.columns.values}")

        return df

    def preprocess(self):
        self.__preprocess(self.raw_data)

    def save_all_preprocessed_data(self):
        if not os.path.exists(cleaned_data_path):
            os.makedirs(cleaned_data_path)

        # by default (index = True), the index of a df is written as the first column
        write_row_names = False

        self.data.to_csv(f"{cleaned_data_path}/data.csv", index=write_row_names)
        self.train.to_csv(f"{cleaned_data_path}/train.csv", index=write_row_names)
        self.validation.to_csv(f"{cleaned_data_path}/validation.csv", index=write_row_names)
        self.test.to_csv(f"{cleaned_data_path}/test.csv", index=write_row_names)

    def load_preprocessed_data(self):
        try:
            self.data = pandas.read_csv(f"{cleaned_data_path}/data.csv")
            self.train = pandas.read_csv(f"{cleaned_data_path}/train.csv")
            self.validation = pandas.read_csv(f"{cleaned_data_path}/validation.csv")
            self.test = pandas.read_csv(f"{cleaned_data_path}/test.csv")
        except FileNotFoundError:
            print(f"Make sure '{cleaned_data_path}' and cleaned csv files exist")
        logging.info(f"Column headers: {self.data.columns.values}")

    def train_validation_test_split(self, test_size=0.1, validation_size=0.2, train_size=0.7):
        """
        Divides data into train, validation and test set.
        Always stratifying and shuffling on label.

        :param test_size:
        :param validation_size:
        :param train_size:
        :param shuffle:
        :return:
        """
        # needs round otherwise raises error, e.g. with default size values sum -> 0.999
        sum_sizes = round(train_size + validation_size + test_size, 2)
        if sum_sizes != 1.0:
            raise ValueError(
                f"Sizes: {train_size} (train), {validation_size} (validation) and {test_size} (test) should add up to "
                f"1.0")

        relative_test_size = test_size / (validation_size + test_size)

        self.train, temp_validation = train_test_split(self.data, train_size=train_size, shuffle=True,
                                                       random_state=self.random_state,
                                                       stratify=self.data[self.column.label])
        self.validation, self.test = train_test_split(temp_validation, test_size=relative_test_size, shuffle=True,
                                                      random_state=self.random_state,
                                                      stratify=temp_validation[self.column.label])

    def get_partition_list_for_train(self, n: int):
        """
        Gets a list of partition sizes for training models.

        :param n: Number of partitions to create.
        :return: Partition list with the total size for each partition.
        """

        self.partition_size = len(self.train.index) // n
        print(f"partition_size: {self.partition_size}")
        print(f"len(train): {len(self.train.index)}")
        print(f"number of values not used: {len(self.train.index) - (n * self.partition_size)}")

        partition_list = [x * self.partition_size for x in range(1, n + 1)]
        print(f"partition_list: {partition_list}")

        return partition_list

    def __oversample(self, df: DataFrame, ratio=1):
        print(f"Before oversampling: {Counter(df[self.column.label])}")
        df_oversampled, labels_oversampled = oversample(df, ratio=ratio, random_state=self.random_state)
        print(f"After oversampling: {Counter(labels_oversampled)}")
        return df_oversampled

    def oversample_validation(self, ratio=1):
        self.validation = self.__oversample(self.validation, ratio=ratio)

    def oversample_train(self, ratio=1):
        self.train = self.__oversample(self.train, ratio=ratio)

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
                ratio = duplicates_count / tweet_count
            except ZeroDivisionError:
                ratio = 0
            print(f"Total tweets: {tweet_count}")
            print(f"Distribution: {duplicates_count} : {tweet_count} (duplicates : tweet_count)")
            print(f"Ratio: 1 : {ratio:.2f}")
            print(f"Total unique tweets (no duplicates): {without_duplicates_minority_count} (in minority class 1)")
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

        word_freq_bar_plot(pos_tweets, title_prefix=title_prefix, title="Positive Labeled Tweets",
                           num_words_to_plot=num_words_to_plot,
                           save_path=f"{current_plot_path}/2-word_freq-pos_tweets")
        word_freq_bar_plot(neg_tweets, title_prefix=title_prefix, title="Negative Labeled Tweets",
                           num_words_to_plot=num_words_to_plot,
                           save_path=f"{current_plot_path}/3-word_freq-neg_tweets")

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

        # endregion

    def visualize_data(self):
        self.__visualize_data(self.data, "Data")

    def visualize_train_data(self):
        self.__visualize_data(self.train, "Train")

    def visualize_validation_data(self):
        self.__visualize_data(self.validation, "Validation")

    def visualize_test_data(self):
        self.__visualize_data(self.test, "Test")

    def train_model(self, pretrained_model: Word2Vec = None, tweet_count=None, force_retrain=False):
        """ If tweet_count=None then entire train set will be used. """

        if pretrained_model:
            logging.info(f"Pretrained model: '{pretrained_model}' will be used.")
            # set other fields
            self.method = Method.WORD2VEC
            self.model = pretrained_model
        else:
            logging.info(f"New model: will be trained.")
            self.model = load_or_create_model(self.method, self.train[self.column.tidy_tweet], tweet_count,
                                              force_retrain=force_retrain)

    def load_pretrained_model(self):
        if self.pretrained_model:
            return self.pretrained_model
        self.pretrained_model = load_pretrained_model()
        return self.pretrained_model

    def train_classifier(self, tweet_count: int = None, pretrained_model=None):
        if tweet_count is None:
            tweet_count = len(self.train)

        self.classifier = Classifier(self.method, self.model, self.model.vector_size,
                                     self.train.iloc[:tweet_count],
                                     self.validation,
                                     self.test,
                                     pretrained_model,
                                     self.column,
                                     random_state=self.random_state)
        self.classifier.fit()

    def validate_classifier(self):
        return self.classifier.validate_classifier()

    def test_classifier(self):
        return self.classifier.test_classifier()

    @staticmethod
    def visualize_score(scores: [], title_prefix=None, title=None, is_pretrained_model: bool = False):
        x_label = "Tweet_Count"

        def get_mcc_scores(df: DataFrame):
            mcc_columns = ["tweet_count", "mcc"]
            df = df[mcc_columns]
            return df.drop_duplicates(subset=mcc_columns)

        if is_pretrained_model:
            f1_filename = "10-evaluation_f1-unspecific_w2v_models"
            mcc_filename = "11-evaluation_mcc-unspecific_w2v_models"
        else:
            f1_filename = "8-evaluation_f1-specific_w2v_models"
            mcc_filename = "9-evaluation_mcc-specific_w2v_models"

        f1_plot_path = f"{plot_path}/{title_prefix.lower()}/{f1_filename}"
        mcc_plot_path = f"{plot_path}/{title_prefix.lower()}/{mcc_filename}"
        # score_data = namedtuple("scoreData", "tweet_count label f1 precision recall mcc")
        scores_data = DataFrame(data=scores)
        grouped_bar_chart(scores_data, y="f1", title=title, x_label=x_label, y_label="F1-Score",
                          title_prefix=title_prefix, save_path=f1_plot_path)
        mcc_scores = get_mcc_scores(scores_data)
        bar_chart(mcc_scores, y="mcc", title=title, x_label=x_label, y_label="MCC",
                  title_prefix=title_prefix, save_path=mcc_plot_path)

    # region encapsulate multiple operations

    def __test_model(self, use_set: TestSet = TestSet.VALIDATION, pretrained_model=None, tweet_counts: [int] = None,
                     force_retrain: bool = False):
        """
        Test the specified Word2vec model on the test set. Train Word2vec model according to given tweet_counts (not
        if pretrained_model = True), train classifiers for given tweet_counts and visualizes scores.

        :param use_set: TestSet to be used for evaluation.
        :param pretrained_model: Use pretrained model.
        :param tweet_counts: Train different models according to given partitions by tweet_counts.
        :param force_retrain: Will not work if pretrained model is used. Classifiers will always be retrained.
        :return:
        """

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
                self.train_classifier(tweet_count, pretrained_model=pretrained_model)
                scores.extend(test_func())
            self.visualize_score(scores, title_prefix=use_set.value, title=f"{pretrained_model_name}",
                                 is_pretrained_model=True)
        else:
            for tweet_count in tweet_counts:
                self.train_model(tweet_count=tweet_count, force_retrain=force_retrain)
                self.train_classifier(tweet_count)
                scores.extend(test_func())
            self.visualize_score(scores, title_prefix=use_set.value, title=f"word2vec Specific Models")

    def validate_specific_models_by(self, tweet_counts: [int], force_retrain: bool = False):
        self.__test_model(TestSet.VALIDATION, tweet_counts=tweet_counts, force_retrain=force_retrain)

    def validate_unspecific_pretrained_model(self, tweet_counts: [int]):
        self.__test_model(TestSet.VALIDATION, pretrained_model=self.load_pretrained_model(), tweet_counts=tweet_counts)

    def test_specific_models_by(self, tweet_counts: [int], force_retrain: bool = False):
        self.__test_model(TestSet.TEST, tweet_counts=tweet_counts, force_retrain=force_retrain)

    def test_unspecific_pretrained_model(self, tweet_counts: [int]):
        self.__test_model(TestSet.TEST, pretrained_model=self.load_pretrained_model(), tweet_counts=tweet_counts)

    # endregion
