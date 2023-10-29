# Twitter Sentiment Analysis With Word2Vec

A twitter sentiment analysis (TSA) project to show the effect of different training sizes on word2vec models.

This repository is based on the following two experiments:

1. Comparison between different sizes (number of tweets trained) of a self-trained word2vec model:
The word2vec models are trained on a predefined number of tweets and then compared to each other.

2. Comparison between a specific word2vec model trained on the tweets themselves and a pretrained word2vec model ([word2vec-google-news-300](https://code.google.com/archive/p/word2vec/)):
The specific word2vec model trained with the largest number of tweets (from experiment 1) is compared with the general pretrained word2vec model.

## Data

The data is from https://github.com/prateekjoshi565/twitter_sentiment_analysis and is originally from a contest regarding TSA (https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/).

## Getting Started

### Installing

Use 

```
pip -r requirements.txt
```

If it does not work, you will need the following dependencies:
- Python 3.8 (You can use [Simple Python version management (pyenv)](https://github.com/pyenv/pyenv) to easily manage different versions)
- [Python Development Workflow for Humans (Pipenv)](https://pipenv.pypa.io)

and use

```
pyenv install 3.8.10
pyenv local 3.8.10
pipenv install
```

### Executing program

- `python code/run.py` (feel free to comment out parts before executing).

---

Program structure:

![program structure](program_structure.png)

The program will perform the following operations (in yellow or blue) on the data (in grey):

![program flow](program_flow.png)

Visualizations are also created to summarize the results. Generation of all models and visualizations will take around 5 min (depending on your hardware).
Afterwards the following additional file structure is generated:

```
./
├── data
│   └── cleaned
│       ├── data.csv
│       ├── test.csv
│       ├── train.csv
│       └── validation.csv
├── models
│   └── word2vec
│       ├── 100_tweets_w2v.model
│       ├── 1000_tweets_w2v.model
│       ├── 10000_tweets_w2v.model
│       ├── 20000_tweets_w2v.model
│       └── 26971_tweets_w2v.model
└── plots
    ├── data
    │   ├── 1-class_distribution_pie_chart.pdf
    │   ├── 2-word_freq-pos_tweets.pdf
    │   ├── 3-word_freq-neg_tweets.pdf
    │   ├── 4-word_freq_with_tweets-pos_tweets.pdf
    │   ├── 5-word_freq_with_tweets-neg_tweets.pdf
    │   ├── 6-word_freq_with_words-pos_tweets.pdf
    │   └── 7-word_freq_with_words-neg_tweets.pdf
    ├── test
    │   ├── 1-class_distribution_pie_chart.pdf
    │   ├── 2-word_freq-pos_tweets.pdf
    │   ├── 3-word_freq-neg_tweets.pdf
    │   ├── 4-word_freq_with_tweets-pos_tweets.pdf
    │   ├── 5-word_freq_with_tweets-neg_tweets.pdf
    │   ├── 6-word_freq_with_words-pos_tweets.pdf
    │   ├── 7-word_freq_with_words-neg_tweets.pdf
    │   ├── 8-evaluation_f1-specific_w2v_models.pdf
    │   ├── 9-evaluation_mcc-specific_w2v_models.pdf
    │   ├── 10-evaluation_f1-unspecific_w2v_models.pdf
    │   └── 11-evaluation_mcc-unspecific_w2v_models.pdf
    ├── train
    │   ├── 1-class_distribution_pie_chart.pdf
    │   ├── 2-word_freq-pos_tweets.pdf
    │   ├── 3-word_freq-neg_tweets.pdf
    │   ├── 4-word_freq_with_tweets-pos_tweets.pdf
    │   ├── 5-word_freq_with_tweets-neg_tweets.pdf
    │   ├── 6-word_freq_with_words-pos_tweets.pdf
    │   └── 7-word_freq_with_words-neg_tweets.pdf
    └── validation
        ├── 1-class_distribution_pie_chart.pdf
        ├── 2-word_freq-pos_tweets.pdf
        ├── 3-word_freq-neg_tweets.pdf
        ├── 4-word_freq_with_tweets-pos_tweets.pdf
        ├── 5-word_freq_with_tweets-neg_tweets.pdf
        ├── 6-word_freq_with_words-pos_tweets.pdf
        ├── 7-word_freq_with_words-neg_tweets.pdf
        ├── 8-evaluation_f1-specific_w2v_models.pdf
        ├── 9-evaluation_mcc-specific_w2v_models.pdf
        ├── 10-evaluation_f1-unspecific_w2v_models.pdf
        └── 11-evaluation_mcc-unspecific_w2v_models.pdf
```
