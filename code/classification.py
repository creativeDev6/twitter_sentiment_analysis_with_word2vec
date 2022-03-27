import sys
from collections import OrderedDict, namedtuple

import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from code.data import ColumnNames
from model import Method


def create_tagged_docs(train, test, column: ColumnNames):
    # todo the doc2vec models were trained with unique integers as tags, now by labels does it matter?
    train_tagged = train.apply(
        lambda r: TaggedDocument(words=r[column.tidy_tweet], tags=[r[column.label]]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=r[column.tidy_tweet], tags=[r[column.label]]), axis=1)

    return train_tagged, test_tagged


class Classifier:
    def __init__(self, method: Method, model: Word2Vec, vector_size: int,
                 train: DataFrame, test: DataFrame, column_names: ColumnNames = ColumnNames()):
        self.method = method
        self.model = model
        self.vector_size = vector_size
        self.train = train
        self.test = test
        self.column = column_names

        print("-" * 30)
        print(f"model: Type: {type(self.model)}, len: {len(self.model.wv)}")
        print("-" * 30)

        self.classifier = None
        self.vectors = None
        self.labels = None

    # todo rename doc_vector
    def mean_doc_vector(self, tokens, is_pretrained=False):
        # vec = np.zeros(vector_size).reshape((vector_size, ))
        vec = np.zeros(self.vector_size)
        count = 0
        words_not_in_vocab = []

        if is_pretrained:
            w2v_model = self.model
        else:
            w2v_model = self.model.wv

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

    # todo testing from tutorial doc2vec (multiclass):
    #  https://www.kdnuggets.com/2018/11/multi-class-text-classification-doc2vec-logistic-regression.html
    def vec_for_learning(self, d2v_model: Doc2Vec, tagged_docs: TaggedDocument, epochs=20):
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

    # todo how to extract features from w2v models and pass to classifier?
    def train_classifier(self):
        # train = self.train.iloc[:len(self.train)]

        # todo retrieve vector for doc by tag
        # model.docvecs["0"]
        # gensim 4
        # model.dv["0"]
        # INFO: ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
        # therefore increased max_iter
        # https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
        # For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.

        self.vectors = [self.mean_doc_vector(doc) for doc in self.train[self.column.tidy_tweet]]
        log_reg = LogisticRegression(solver='liblinear', max_iter=1000)
        # logreg.fit(train_clean[self.column.tidy_tweet], train_clean[self.column.label])
        print(f"train_classifier -> len(vectors): {len(self.vectors)}, len(labels): {len(self.train[self.column.label])}")
        print(f"shape: {self.vectors[0].shape}")
        log_reg = log_reg.fit(self.vectors, self.train[self.column.label])

        self.classifier = log_reg
        return log_reg

    def get_prediction_int(self, train, test, is_pretrained=False):
        if self.method == Method.WORD2VEC:
            train_doc_vectors = [self.mean_doc_vector(doc, is_pretrained) for doc in train[self.column.tidy_tweet]]
            test_doc_vectors = [self.mean_doc_vector(doc, is_pretrained) for doc in test[self.column.tidy_tweet]]

            log_reg = self.train_classifier()
            log_reg = self.classifier

            prediction = log_reg.predict_proba(test_doc_vectors)
            print("|" * 30)
            print("len, type, prediction", len(prediction), type(prediction), prediction)
            print("|" * 30)
            # non-thresholded decision values
            # prediction = logreg.decision_function(test_vectors)
            # if prediction is greater than or equal to 0.5 than 1 else 0
            prediction_int = prediction[:, 1] >= 0.5
            prediction_int = prediction_int.astype(np.int)
            print("|" * 30)
            print("len, type, prediction_int", len(prediction_int), type(prediction_int), prediction_int)
            print("|" * 30)

        elif self.method == Method.DOC2VEC:
            # creating taggedDocuments with tweet and label
            # train_tagged = train[:limit].apply(
            # todo the doc2vec models were trained with unique integers as tags, now by labels does it matter?
            train_tagged = train.apply(
                lambda r: TaggedDocument(words=r[self.column.tidy_tweet], tags=[r[self.column.label]]), axis=1)
            test_tagged = test.apply(
                lambda r: TaggedDocument(words=r[self.column.tidy_tweet], tags=[r[self.column.label]]), axis=1)

            labels_train, train_doc_vectors = self.vec_for_learning(self.model, train_tagged)
            labels_test, test_doc_vectors = self.vec_for_learning(self.model, test_tagged)

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
            raise ValueError(f"'{self.method}' is not defined. Defined values: {[e.name for e in Method]}")

        return prediction_int

    def test_classifier(self):
        def print_score_data(data):
            print("-" * 100)
            print(f"Tweet count: {len(self.train)} ({self.train.iloc[0].index} - {self.train.iloc[-1].index})")
            print("-" * 100)
            for score in data:
                print(f"Label: {score.label} F1-Score: {score.f1} (Precision: {score.precision}, Recall {score.recall})")
            print("-" * 100)

        score_data = self.evaluate_score_data(self.train, self.test)
        print_score_data(score_data)

        return score_data

    def evaluate_score_data(self, train: DataFrame, test: DataFrame, is_pretrained=False):
        def scores(test_labels_param, prediction_int_param, average_param, pos_label_param):
            # prevent warning: ignoring pos_label when average != "binary"
            if average_param != "binary":
                f1_res = f1_score(test_labels_param, prediction_int_param, average=average_param)
                precision_res = precision_score(test_labels_param, prediction_int_param, average=average_param)
                recall_res = recall_score(test_labels_param, prediction_int_param, average=average_param)
            else:
                f1_res = f1_score(test_labels_param, prediction_int_param, average=average_param,
                                  pos_label=pos_label_param)
                precision_res = precision_score(test_labels_param, prediction_int_param, average=average_param,
                                                pos_label=pos_label_param)
                recall_res = recall_score(test_labels_param, prediction_int_param, average=average_param,
                                          pos_label=pos_label_param)

            return f1_res, precision_res, recall_res

        prediction_int = self.get_prediction_int(train, test, is_pretrained)

        """
        if is_pretrained:
            w2v = {w: vec for w, vec in zip(self.model.index_to_key, self.model.vectors)}
        else:
            w2v = {w: vec for w, vec in zip(self.model.wv.index_to_key, self.model.wv.vectors)}
        mv = MeanEmbeddingVectorizer(w2v)
        mv = TfidfEmbeddingVectorizer(w2v)
        mv.fit(train[self.column.tidy_tweet], train[self.column.label])
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
                average_label = "0 & 1"
            # f1 = f1_score(test_labels, prediction_int, average=average, pos_label=pos_label)
            # precision = precision_score(test_labels, prediction_int, average=average, pos_label=pos_label)
            # recall = recall_score(test_labels, prediction_int, average=average, pos_label=pos_label)
            f1, precision, recall = scores(test[self.column.label], prediction_int, average, pos_label)
            # result.append(zip(average, pos_label, f1, precision, recall))
            result.append(score_data(tweet_count=tweet_count, label=average_label, f1=f1, precision=precision,
                                     recall=recall))

        # return f"F1-score: {f1_score(test_labels, prediction_int, pos_label=0)}"
        # return DataFrame(result)
        return result
