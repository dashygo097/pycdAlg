import functools
import time

from termcolor import colored


def timer(func):
    "decorator function that counts the time of a function"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tik = time.time()
        result = func(*args, **kwargs)
        tok = time.time()
        print(
            "Time elapsed of "
            + colored(f"{func.__name__}", color="yellow", attrs=["bold"])
            + ": "
            + colored(f"{tok - tik:.6f} s \n", color="red", attrs=["bold"])
        )

        return result

    return wrapper
