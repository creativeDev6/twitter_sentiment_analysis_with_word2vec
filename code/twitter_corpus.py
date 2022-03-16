import re
import pandas as pd
import numpy as np

from gensim.test.utils import datapath
from gensim import utils
from gensim.parsing.preprocessing import preprocess_documents
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.porter import PorterStemmer


def remove_pattern(input_txt, regex):
    patterns = re.findall(regex, input_txt)
    for pattern in patterns:
        input_txt = re.sub(pattern, '', input_txt)

    return input_txt


class TwitterCorpus(object):
    """A constructor that receives a DataFrame."""

    def __init__(self, df):
        self.df = df

    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        # todo open twitter data
        for i in self.df.index:
            clean_tweet = remove_pattern(self.df['tweet'][i], r"@\w*")
            clean_tweet = clean_tweet.replace("[^a-zA-Z#]", " ")
            # assume there's one document per line, tokens separated by whitespace
            yield preprocess_string(clean_tweet)
        # for line in open(corpus_path):
        #     # assume there's one document per line, tokens separated by whitespace
        #     yield utils.simple_preprocess(line)
