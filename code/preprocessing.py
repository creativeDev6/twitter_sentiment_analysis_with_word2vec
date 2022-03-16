import re
from collections import defaultdict
from pandas import DataFrame

import gensim
from gensim.parsing import preprocess_string


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
