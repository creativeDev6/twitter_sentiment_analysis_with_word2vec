import logging
import os
from time import perf_counter

from helper import show_used_time
from tweet_sentiment_analyzer import TweetSentimentAnalyzer, ColumnNames

cwd = os.getcwd()
data_path = f"{cwd}/data"
cleaned_data_path = f"{cwd}/data/cleaned"

# make sure you get this repository's root folder as your cwd
print(f"Current Working Directory: {cwd}")

# region variables

data_was_cleaned = False
random_state = 1

force_retrain_w2v_models = True

logging_level = logging.INFO
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt="%H:%M:%S", level=logging_level)


# endregion

def run():
    tsa = TweetSentimentAnalyzer(f"{data_path}/train_E6oV3lV.csv", ColumnNames(), random_state=random_state)
    if data_was_cleaned:
        tsa.load_preprocessed_data()
    else:
        tsa.preprocess()

    tsa.show_data_class_distribution()

    tsa.train_validation_test_split(test_size=0.1, validation_size=0.2, train_size=0.7)

    # tsa.visualize_data()
    # tsa.visualize_train_data()
    # tsa.visualize_validation_data()
    ## tsa.visualize_test_data()

    tsa.oversample_train(ratio=1)
    tweet_counts = [100, 1_000, 10_000, 20_000, 30_000, len(tsa.train)]
    # tweet_counts = tsa.get_partition_list_for_train(5)
    tsa.show_train_duplicates_distribution(tweet_counts)

    # Test run label distribution
    print("*" * 50)
    print("Before distribute_labels_equally_in_train()")
    print("*" * 50)
    tsa.show_train_class_distribution(tweet_counts)
    tsa.show_validation_class_distribution()
    # tsa.show_test_class_distribution()

    tsa.distribute_labels_equally_in_train()
    tsa.save_all_preprocessed_data()

    print("*" * 50)
    print("After distribute_labels_equally_in_train()")
    print("*" * 50)
    tsa.show_train_class_distribution(tweet_counts)
    tsa.show_validation_class_distribution()
    # tsa.show_test_class_distribution()

    tsa.show_train_duplicates_distribution(tweet_counts)

    tsa.validate_specific_models_by(tweet_counts, force_retrain=force_retrain_w2v_models)
    tsa.validate_unspecific_pretrained_model(tweet_counts)

    # tsa.test_specific_models_by(tweet_counts=tweet_counts, force_retrain=force_retrain_w2v_models)
    # tsa.test_unspecific_pretrained_model()


if __name__ == '__main__':
    t = perf_counter()
    run()
    show_used_time(t, "run()")
