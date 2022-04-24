import logging
from collections import namedtuple

import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

from code.data import ColumnNames
from code.model import Method


# region doc2vec functions

def create_tagged_docs(train, test, column: ColumnNames):
    # todo the doc2vec models were trained with unique integers as tags, now by labels does it matter?
    train_tagged = train.apply(
        lambda r: TaggedDocument(words=r[column.tidy_tweet], tags=[r[column.label]]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=r[column.tidy_tweet], tags=[r[column.label]]), axis=1)

    return train_tagged, test_tagged


# todo testing from tutorial doc2vec (multiclass):
#  https://www.kdnuggets.com/2018/11/multi-class-text-classification-doc2vec-logistic-regression.html
def vec_for_learning(d2v_model: Doc2Vec, tagged_docs: TaggedDocument, epochs=20):
    # fixme
    # sents = tagged_docs.values
    sents = tagged_docs
    # print("-" * 30)
    # print(sents)
    # print("-" * 30)
    # sents = [x for x in tagged_docs]
    # todo what happens if word is not in model, how to handle this case?
    # * unpacks list into zip call
    targets, regressors = zip(*[(doc.tags[0], d2v_model.infer_vector(doc.words, epochs=epochs)) for doc in sents])
    # print(f"labels: {targets[:1000]}, docvecs: {regressors[:1]}")

    return targets, regressors


# endregion

class Classifier:
    def __init__(self, method: Method, model: Word2Vec, vector_size: int,
                 train: DataFrame, validation: DataFrame, test: DataFrame,
                 pretrained_model: Word2Vec = None,
                 column_names: ColumnNames = ColumnNames(), random_state=None):
        self.method = method
        self.model = model
        self.pretrained_model = pretrained_model
        self.vector_size = vector_size
        self.train = train
        self.validation = validation
        self.test = test
        self.column = column_names
        self.random_state = random_state

        self.classifier = None
        self.vectors = None
        self.labels = None

        logging.info(f"Use pretrained model = {pretrained_model}")

    def mean_doc_vector(self, tokens):
        doc_vector = np.zeros(self.vector_size)
        count = 0
        words_not_in_vocab = []

        if self.pretrained_model:
            w2v_model = self.pretrained_model
        else:
            w2v_model = self.model.wv

        for word in tokens:
            try:
                doc_vector += w2v_model[word]
                count += 1.
            # handling the case where the token is not in vocabulary
            except KeyError:
                words_not_in_vocab.append(word)
                continue

        if count != 0:
            doc_vector /= count

        if words_not_in_vocab:
            logging.debug(f"Words not in vocabulary (# {len(words_not_in_vocab)}): {words_not_in_vocab}")

        return doc_vector

    def fit(self):
        """
        Fit train data to classifier (training classifier).
        Train data is converted to document vectors used for training classifier.
        :return:
        """
        train_vectors = [self.mean_doc_vector(doc) for doc in self.train[self.column.tidy_tweet]]
        log_reg = LogisticRegression(solver="liblinear", max_iter=100, random_state=self.random_state)

        logging.debug(f"fit() -> len(vectors): {len(train_vectors)}, "
                      f"len(labels): {len(self.train[self.column.label])}, "
                      f"shape: {train_vectors[0].shape}")

        self.classifier = log_reg.fit(train_vectors, self.train[self.column.label])

        return self.classifier

    def predict_labels(self, test: DataFrame):
        if self.classifier is None:
            raise ValueError("Make sure fit() is called before using predict_labels().")

        if self.method == Method.WORD2VEC:
            test_doc_vectors = [self.mean_doc_vector(doc) for doc in test[self.column.tidy_tweet]]
            prediction = self.classifier.predict_proba(test_doc_vectors)
            # manual threshold: if prediction is greater than or equal to 0.5 than 1 else 0
            predicted_labels = prediction[:, 1] >= 0.5
            predicted_labels = predicted_labels.astype(np.int)
            # this can be used instead when no manual threshold will be defined
            # predicted_labels = self.classifier.predict(test_doc_vectors)

        elif self.method == Method.DOC2VEC:
            # creating taggedDocuments with tweet and label
            # todo the doc2vec models were trained with unique integers as tags, now by labels does it matter?
            test_tagged = test.apply(
                lambda r: TaggedDocument(words=r[self.column.tidy_tweet], tags=[r[self.column.label]]), axis=1)
            labels_test, test_doc_vectors = vec_for_learning(self.model, test_tagged)
            predicted_labels = self.classifier.predict(test_doc_vectors)
        else:
            raise ValueError(f"'{self.method}' is not defined. Defined values: {[e.name for e in Method]}")

        return predicted_labels

    def __test_classifier(self, df_test: DataFrame):
        def print_score_data(data):
            print("-" * 100)
            print(f"Tweet count: {len(self.train)}")
            print("-" * 100)
            for score in data:
                print(
                    f"Label: {score.label} F1-Score: {score.f1} (Precision: {score.precision}, Recall: {score.recall})"
                    f", MCC: {score.mcc}")
            print("-" * 100)

        score_data = self.evaluate_score_data(df_test)
        print_score_data(score_data)

        return score_data

    def validate_classifier(self):
        return self.__test_classifier(self.validation)

    def test_classifier(self):
        return self.__test_classifier(self.test)

    def evaluate_score_data(self, test: DataFrame):
        # prevents warning: ignoring pos_label when average != "binary"
        def scores(true_labels_param, prediction_param, average_param, pos_label_param):
            if average_param != "binary":
                f1_res = f1_score(true_labels_param, prediction_param, average=average_param)
                precision_res = precision_score(true_labels_param, prediction_param, average=average_param)
                recall_res = recall_score(true_labels_param, prediction_param, average=average_param)
            else:
                f1_res = f1_score(true_labels_param, prediction_param,
                                  average=average_param, pos_label=pos_label_param)
                precision_res = precision_score(true_labels_param, prediction_param,
                                                average=average_param, pos_label=pos_label_param)
                recall_res = recall_score(true_labels_param, prediction_param,
                                          average=average_param, pos_label=pos_label_param)

            mcc_res = matthews_corrcoef(true_labels_param, prediction_param)

            return f1_res, precision_res, recall_res, mcc_res

        predicted_labels = self.predict_labels(test)
        # for grouping data (hue)
        tweet_count = len(self.train.index)
        # todo is it possible to pass this as a parameter to make it more flexible?
        # measurements according to namedtuple
        score_data = namedtuple("scoreData", "tweet_count label f1 precision recall mcc")
        result = []
        # get score for individual (average='binary') and weighted (average='weighted') labels
        #   see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        # (1, 'weighted') is not needed because result will be calculated from both labels when using 'weighted',
        # so (0. 'weighted') is enough
        for i in [(0, 'binary'), (1, 'binary'), (0, 'weighted')]:
            pos_label = i[0]
            # average = 'binary' # Only report results for the class specified by pos_label
            average = i[1]
            # translate labels
            if i[1] == "binary":
                average_label = "0: Positive" if i[0] == 0 else "1: Negative"
            # weighted is used
            else:
                average_label = "average(0 & 1)"
            f1, precision, recall, mcc = scores(test[self.column.label], predicted_labels, average, pos_label)
            result.append(score_data(tweet_count=tweet_count, label=average_label,
                                     f1=f1, precision=precision, recall=recall,
                                     mcc=mcc))

        return result
