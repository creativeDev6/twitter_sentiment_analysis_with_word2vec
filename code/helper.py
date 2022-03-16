from time import time


def show_used_time(start_time, text="Time"):
    print(f"{text}: {format(round((time() - start_time) / 60, 2))} min")