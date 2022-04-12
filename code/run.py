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

    tsa.show_data_class_distribution()

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
    tweet_counts = [100, 1_000, 10_000, 20_000, 30_000, len(tsa.train)]
    tsa.show_train_duplicates_distribution(tweet_counts)

    # Test run label distribution
    print("*" * 50)
    print("Before distribute_labels_equally_in_train()")
    print("*" * 50)
    tsa.show_train_class_distribution(tweet_counts)
    tsa.show_validation_class_distribution()
    # tsa.show_test_class_distribution()

    tsa.distribute_labels_equally_in_train()
    tsa.save_preprocessed_data()

    print("*" * 50)
    print("After distribute_labels_equally_in_train()")
    print("*" * 50)
    tsa.show_train_class_distribution(tweet_counts)
    tsa.show_validation_class_distribution()
    # tsa.show_test_class_distribution()

    tsa.show_train_duplicates_distribution(tweet_counts)

    tsa.validate_specific_models_by(tweet_counts)
    # todo save best performing model chosen by validation
    tsa.validate_unspecific_pretrained_model(tweet_counts)

    # tsa.test_specific_models_by(tweet_counts=tweet_counts)
    # tsa.test_unspecific_pretrained_model()


if __name__ == '__main__':
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    t = perf_counter()
    run()
    show_used_time(t, "run()")
