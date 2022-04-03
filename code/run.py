import logging
import os
from tweet_sentiment_analyzer import TweetSentimentAnalyzer, ColumnNames
from time import perf_counter
from helper import show_used_time

cwd = os.getcwd()
data_path = f"{cwd}/data"
cleaned_data_path = f"{cwd}/data/cleaned"

# make sure you get this repository's root folder as your cwd
print(f"Current Working Directory: {cwd}")


def run():
    data_was_cleaned = False
    random_state = 1

    if data_was_cleaned:
        tsa = TweetSentimentAnalyzer(f"{cleaned_data_path}/data.csv", ColumnNames())
        tsa.load_preprocessed_data()
    else:
        tsa = TweetSentimentAnalyzer(f"{data_path}/train_E6oV3lV.csv", ColumnNames())
        tsa.preprocess()
    print(f"column headers: {tsa.data.columns.values}")
    print(f'# positive tweets: {len(tsa.raw_data[tsa.raw_data[tsa.column.label] == 0].index)}')
    print(f'# negative tweets: {len(tsa.raw_data[tsa.raw_data[tsa.column.label] == 1].index)}')

    # todo move to loop and use how it is meant to be used
    # tsa.cross_validation(k_fold=5)
    # tsa.fold_size = len(tsa.train) / 5
    # print(f"tsa.fold_size: {tsa.fold_size}")
    tsa.train_validation_test_split(test_size=0.1, validation_size=0.2, train_size=0.7, shuffle=True,
                                    random_state=random_state)

    # tsa.visualize_train_data()
    # tsa.visualize_validation_data()
    ## tsa.visualize_test_data()

    tsa.oversample_train(ratio=1, random_state=random_state)
    # tweet_counts = [tsa.fold_size * x for x in range(1, 6)]
    # todo remove later
    # tweet_counts = [tsa.fold_size]
    tweet_counts = [100, 1_000, 10_000, 20_000, 30_000, len(tsa.train)]
    specific_scores = []
    # for tweet_count in [100, 1_000, 10_000, 20_000, len(tsa.train)]:

    # Test run label distribution
    print("*" * 50)
    print("Before distribute_labels_equally_in_train()")
    print("*" * 50)
    tsa.show_train_class_distribution(tweet_counts)
    tsa.show_validation_class_distribution()
    tsa.show_test_class_distribution()

    tsa.distribute_labels_equally_in_train()
    tsa.save_preprocessed_data()

    print("*" * 50)
    print("After distribute_labels_equally_in_train()")
    print("*" * 50)
    tsa.show_train_class_distribution(tweet_counts)
    tsa.show_test_class_distribution()

    # specific model
    for tweet_count in tweet_counts:
        tsa.train_model(tweet_count=tweet_count)
        tsa.train_classifier(tweet_count=tweet_count)
        specific_scores.extend(
            tsa.validate_classifier())

    # todo save best performing model chosen by validation
    # todo add final tsa.test_classifier at the end

    tsa.visualize_score(specific_scores, "Validation: W2V Specific Model")

    # unspecifc (pretrained) model
    unspecific_scores = []
    pretrained_model = tsa.load_pretrained_model()
    tsa.train_model(pretrained_model=pretrained_model)
    tsa.train_classifier()
    unspecific_scores.extend(
        tsa.validate_classifier())

    tsa.visualize_score(unspecific_scores, "Validation: W2V Unspecific Model")


if __name__ == '__main__':
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    t = perf_counter()
    run()
    show_used_time(t, "run()")
