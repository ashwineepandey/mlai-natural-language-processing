import os
import log
import time
import functools
from datetime import datetime

from pyhocon import ConfigFactory


logger = log.get_logger(__name__)

def timer(func):
    """ Print the runtime of the decorated function """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        logger.info(f"Starting {func.__name__!r}.")
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs.")
        return value
    return wrapper_timer


def debug(func):
    """ Print the function signature and return value """
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.debug(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        logger.debug(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug


@timer
def load_config(filename: str):
    """
    Load the configuration file from a hocon file object and returns it (https://github.com/chimpler/pyhocon).
    """
    return ConfigFactory.parse_file(f"../conf/{filename}.conf")


@timer
def load_text_data(file_path: str) -> str:
    """
    Function to load text data from a file.

    Args:
    file_path (str): Path to the text file.

    Returns:
    str: Text from the file.
    """
    with open(file_path, 'r') as file:
        data = file.read()
    return data


def write_text_data(file_path: str, data: str):
    """
    Function to write text data to a file.

    Args:
    file_path (str): Path to the text file.
    data (str): Text data to be written to the file.
    """
    with open(file_path, 'w') as file:
        file.write(data)


def _get_current_dt() -> str:
    """
    Returns the current date and time as a string in the format "DDMMYYYY_HHMMSS".

    Returns:
        A string representing the current date and time, in the format "DDMMYYYY_HHMMSS".
    """
    now = datetime.now()
    return now.strftime("%d%m%Y_%H%M%S")


def save_plot(plot_filepath, fig, filename):
    """
    Saves the plot to file.
    """
    # generate file name
    prefix = f"{filename}-{_get_current_dt()}.png"
    filepath = os.path.join(plot_filepath, prefix)
    # save figure to file
    fig.write_image(filepath)
    logger.info(f"Plot saved: {prefix}")