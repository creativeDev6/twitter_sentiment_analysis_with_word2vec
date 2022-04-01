from time import perf_counter


def show_used_time(start_time, text="Time"):
    print(f"{text}: {format(round((perf_counter() - start_time) / 60, 2))} min")
