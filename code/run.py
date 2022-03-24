import os
from tweet_sentiment_analyzer import TweetSentimentAnalyzer, ColumnNames

cwd = os.getcwd()
data_path = f"{cwd}/data"

# make sure you get this repository's root folder as your cwd
print(f"Current Working Directory: {cwd}")


def run():
    tsa = TweetSentimentAnalyzer(f"{data_path}/train_E6oV3lV.csv", ColumnNames())
    print(f"column headers: {tsa.data.columns.values}")
    print(f'# positive tweets: {len(tsa.raw_data[tsa.raw_data[tsa.column.label] == 0].index)}')
    print(f'# negative tweets: {len(tsa.raw_data[tsa.raw_data[tsa.column.label] == 1].index)}')

    # todo remove later
    # fixme why clean/train.csv same amount of data as train_E6oV3lV.csv?
    # for checking if data was wrangled
    tsa.data.to_csv(f"{data_path}/clean/train.csv")

    # todo move to loop and use how it is meant to be used
    # tsa.cross_validation(k_fold=5)
    # tsa.fold_size = len(tsa.train) / 5
    # print(f"tsa.fold_size: {tsa.fold_size}")
    tsa.split_train_test(test_size=0.2)

    # tsa.visualize_train_data()
    # tsa.visualize_test_data()

    tsa.oversample(ratio=1)
    # tweet_counts = [tsa.fold_size * x for x in range(1, 6)]
    # todo remove later
    # tweet_counts = [tsa.fold_size]
    tweet_counts = [100, 1_000, 10_000, 20_000, 30_000, 40_000, len(tsa.train)]
    specific_scores = []
    # for tweet_count in [100, 1_000, 10_000, 20_000, len(tsa.train)]:
    # specific model
    # todo how to select incrementally more tweets (100, 1000, 10000, ...)?
    for tweet_count in tweet_counts:
        tsa.train_model(tweet_count=tweet_count)
        tsa.train_classifier(tweet_count=tweet_count)
        specific_scores.extend(
            tsa.test_classifier())

    # fixme not showing with different tweet_counts (fold_size is used)
    tsa.visualize_score(specific_scores, "W2V Specific Model")

    """
    # unspecifc (pretrained) model
    unspecific_scores = []
    pretrained_model = tsa.load_pretrained_model()
    tsa.train_model(pretrained_model=pretrained_model)
    tsa.train_classifier()
    unspecific_scores.append(
        tsa.test_classifier())

    unspecific_scores.clear()
    for tweet_count in tweet_counts:
        tsa.train_model(pretrained_model=pretrained_model, tweet_count=tweet_count)
        tsa.train_classifier()
        unspecific_scores.extend(
            tsa.test_classifier())
    
    tsa.visualize_score(unspecific_scores, "W2V Unspecific Model")
    """


if __name__ == '__main__':
    run()
