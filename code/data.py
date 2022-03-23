class ColumnNames:
    def __init__(self, tweet: str = 'tweet', tidy_tweet: str = 'tidy_tweet', label: str = 'label',
                 hashtags: str = 'hashtags'):
        # essential
        self.tweet = tweet
        self.label = label

        # columns will be added
        self.tidy_tweet = tidy_tweet
        self.hashtags = hashtags
