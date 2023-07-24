import os
import log
import time
import functools
import json
import pickle
from collections import defaultdict, Counter
from datetime import datetime
from ast import literal_eval
from typing import List, Tuple, Dict, Union
import torch

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


def write_text_data(file_path: str, sentences: str):
    """
    Function to write text data to a file.

    Args:
    file_path (str): Path to the text file.
    sentences (str): Corpus.
    """
    with open(file_path, 'w') as file:
        file.write(sentences)


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


def save_model(model: Dict[Tuple[str, str], Counter], file_path: str) -> None:
    """
    Function to save the language model to a JSON file.

    Args:
    model (Dict[Tuple[str, str], Counter]): Language model.
    file_path (str): Path to the JSON file where the model will be saved.
    """
    # Convert the defaultdict and Counter objects to regular dictionaries
    model_dict = {k: dict(v) for k, v in model.items()}

    # Convert the tuples to strings (JSON keys need to be strings)
    model_dict = {str(k): v for k, v in model_dict.items()}

    with open(file_path, 'w') as file:
        json.dump(model_dict, file, ensure_ascii=False)


def load_model(file_path: str) -> Dict[Tuple[str, str], Counter]:
    """
    Function to load the language model from a JSON file.

    Args:
    file_path (str): Path to the JSON file where the model is saved.

    Returns:
    Dict[Tuple[str, str], Counter]: Language model.
    """
    with open(file_path, 'r') as file:
        model_dict = json.load(file)

    # Convert the strings back to tuples and the dictionaries back to Counters
    model = {literal_eval(k): Counter(v) for k, v in model_dict.items()}

    # Convert the dictionary back to a defaultdict
    model = defaultdict(Counter, model)

    return model


def save_pickle(filepath: str, obj_name: str, obj):
    """
    Saves an object to a pickle file.

    Args:
    filepath: path to the pickle file
    obj_name: name of the object to be saved
    obj: the object to be saved
    """
    with open(f'{filepath}{obj_name}.pkl', 'wb') as f:
        pickle.dump(obj, f)
        logger.info(f"Object {obj_name} saved to pickle file.")


def load_pickle(filepath: str, obj_name: str):
    """
    Loads a pickle file.

    Args:
    filepath: path to the pickle file
    obj_name: name of the object to be loaded

    Returns:
    obj: the loaded object
    """
    with open(f'{filepath}{obj_name}.pkl', 'rb') as f:
        obj = pickle.load(f)
        logger.info(f"Object {obj_name} loaded from pickle file.")
        return obj


def save_pytorch_model(model, file_path: str) -> None:
    """
    Function to save a PyTorch model to a file.

    Args:
    model (torch.nn.Module): PyTorch model.
    file_path (str): Path to the file where the model will be saved.
    """
    torch.save(model.state_dict(), file_path)


def load_pytorch_model(model, file_path: str) -> torch.nn.Module:
    """
    Function to load a PyTorch model from a file.

    Args:
    model (torch.nn.Module): PyTorch model.
    file_path (str): Path to the file where the model is saved.

    Returns:
    torch.nn.Module: PyTorch model.
    """
    model.load_state_dict(torch.load(file_path))
    return model