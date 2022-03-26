from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from pandas import DataFrame


class ColumnNames:
    def __init__(self, tweet: str = 'tweet', tidy_tweet: str = 'tidy_tweet', label: str = 'label',
                 hashtags: str = 'hashtags'):
        # essential
        self.tweet = tweet
        self.label = label

        # columns will be added
        self.tidy_tweet = tidy_tweet
        self.hashtags = hashtags


column = ColumnNames()


# todo test different ratios
def oversample(df: DataFrame, ratio=1):
    print(f"Before oversampling: {Counter(df[column.label])}")
    # todo try different oversampling methods (e.g. SMOTE)
    oversample_method = RandomOverSampler(sampling_strategy=ratio)
    labels = df[column.label]
    data_oversampled, labels_oversampled = oversample_method.fit_resample(df, labels)
    print(f"After oversampling: {Counter(labels_oversampled)}")
    return data_oversampled


def get_duplicates(df: DataFrame, subset: [str] = None):
    if subset is None:
        return df[df.duplicated()]
    return df[df.duplicated(subset=subset)]


def count_duplicates(df: DataFrame, subset: [str] = None):
    return len(get_duplicates(df, subset).index)


def distribute_equally(df: DataFrame, target_column: str):
    # source https://stackoverflow.com/a/69079436
    return df.assign(rank=df.groupby(target_column).cumcount()).sort_values('rank').drop(columns='rank')
