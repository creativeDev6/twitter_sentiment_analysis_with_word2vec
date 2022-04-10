import nltk
from pandas import DataFrame

import matplotlib.pyplot as plt
import seaborn as sns


def ratio_pie_chart(count_list, *, title_prefix, title, labels):
    # source: https://stackoverflow.com/a/6170354
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))

            return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

        return my_autopct

    plt.title(f"{title_prefix}: {title}")
    plt.pie(count_list,
            labels=labels,
            autopct=make_autopct(count_list),
            wedgeprops=
            {
                "edgecolor": "1",
                "linewidth": 1,
                # "linestyle": "dashed",
                "antialiased": True,
            })
    plt.show()


def word_freq_bar_plot(words: DataFrame, *, title_prefix="", title="", num_words_to_plot=10, total_count=1,
                       y_suffix=""):
    # Format the text with number or decimal.
    text_format = ".3f" if total_count > 1 else "n"
    if y_suffix != "":
        y_suffix = f" {y_suffix}"
    # unnesting list
    words = sum(words, [])

    freq_dist = nltk.FreqDist(words)
    d = DataFrame({"Word": list(freq_dist.keys()),
                   "Frequency": list(freq_dist.values())})

    # selecting top n most frequent hashtags
    d = d.nlargest(columns="Frequency", n=num_words_to_plot)
    plt.figure(figsize=(16, 5))
    plt.title(f"{title_prefix}: {title}")
    ax = sns.barplot(data=d, x="Word", y="Frequency")
    # ax = sns.barplot(data=d, y="Word", x="Frequency", orient="h")
    ax.set(ylabel="Frequency" + y_suffix)
    ax.set(xlabel="Words")
    ax.yaxis.grid(True, color="#EEEEEE")

    # adding values on top of each bar
    # source: https://www.pythoncharts.com/matplotlib/grouped-bar-charts-matplotlib/
    for bar in ax.patches:
        # The text annotation for each bar should be its height.
        bar_value = bar.get_height()
        text = f"{bar_value / total_count:{text_format}}"
        # This will give the middle of each bar on the x-axis.
        text_x = bar.get_x() + bar.get_width() / 2
        # get_y() is where the bar starts so we add the height to it.
        text_y = bar.get_y() + bar_value
        # If we want the text to be the same color as the bar, we can
        # get the color like so:
        bar_color = bar.get_facecolor()
        bar_color = "#222222"
        # If you want a consistent color, you can just set it as a constant, e.g. #222222
        ax.text(text_x, text_y, text, ha="center", va="bottom", color=bar_color,
                size=12)

    plt.show()


# show F1-scores for (un)specific models + label 0/1 average='binary' + average='weighted'
def grouped_bar_chart(d, y, title):
    # grouping on average_label
    ax = sns.barplot(x="tweet_count", y=y, hue="label", data=d)
    plt.title(f"{title}")
    plt.xlabel("Tweet Count")
    plt.ylabel(f"{y.upper()}")
    plt.legend(loc='lower right', prop={'size': 10}, title="Labels")

    # ax.set(xlabel="Tweet Count")
    # ax.set(ylabel='F1-Score')
    ax.yaxis.grid(True, color='#EEEEEE')
    # adding values on top of each bar
    # source: https://www.pythoncharts.com/matplotlib/grouped-bar-charts-matplotlib/
    for bar in ax.patches:
        # The text annotation for each bar should be its height.
        bar_value = bar.get_height()
        # Format the text with two decimals.
        text = f'{bar_value:.2f}'
        # This will give the middle of each bar on the x-axis.
        text_x = bar.get_x() + bar.get_width() / 2
        # get_y() is where the bar starts so we add the height to it.
        text_y = bar.get_y() + bar_value
        # If we want the text to be the same color as the bar, we can
        # get the color like so:
        bar_color = bar.get_facecolor()
        bar_color = "#222222"
        # If you want a consistent color, you can just set it as a constant, e.g. #222222
        ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color,
                size=12)
    # plt.savefig("barplot_Seaborn_barplot_Python.png")
    plt.show()


def raw_score(d):
    for key, d in d.items():
        print("-" * 30)
        print(f"{key} tweets")
        print("-" * 30)
        for i in d:
            print(i)
