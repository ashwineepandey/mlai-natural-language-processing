import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
import plotly.express as px
from collections import defaultdict, Counter
from unidecode import unidecode
from typing import List, Tuple, Dict, Union
import log
import mynlputils as nu

logger = log.get_logger(__name__)

@nu.timer
def generate_trigrams(text: str) -> List[Tuple[str, str, str]]:
    """
    Function to generate character-level trigrams from text.

    Args:
    text (str): Text data.

    Returns:
    List[Tuple[str, str, str]]: List of character-level trigrams.
    """
    return list(ngrams(text, 3, pad_left=True, pad_right=True, left_pad_symbol='<s><s>', right_pad_symbol='</s>')) # pad_left=True, pad_right=True, left_pad_symbol='<s><s>', right_pad_symbol='</s>'

@nu.timer
def build_language_model(trigrams: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str], Counter]:
    """
    Function to build a character-level trigram language model.

    Args:
    trigrams (List[Tuple[str, str, str]]): List of character-level trigrams.

    Returns:
    Dict[Tuple[str, str], Counter]: Language model.
    """
    model = defaultdict(Counter)

    for t1, t2, t3 in trigrams:
        model[(t1, t2)][t3] += 1

    return model

@nu.timer
def generate_text(model: Dict[Tuple[str, str], Counter], max_length: int = 200) -> str:
    """
    Function to generate text from a character-level trigram language model.

    Args:
    model (Dict[Tuple[str, str], Counter]): Language model.
    max_length (int, optional): Maximum length of the generated text. Defaults to 200.

    Returns:
    str: Generated text.
    """
    text = ['<s>', '<s>']
    while len(text) < max_length:
        t1, t2 = text[-2], text[-1]
        next_char = model[(t1, t2)].most_common(1)[0][0]
        text.append(next_char)
    return ''.join(text)


@nu.timer
def calculate_perplexity(model: Dict[Tuple[str, str], Counter], text: str) -> float:
    """
    Function to calculate the perplexity of a language model on a given text.

    Args:
    model (Dict[Tuple[str, str], Counter]): Language model.
    text (str): Text data.

    Returns:
    float: Perplexity of the language model on the text.
    """
    trigrams = generate_trigrams(text)
    N = len(trigrams)
    log_prob = 0
    for t1, t2, t3 in trigrams:
        prob = model[(t1, t2)][t3] / sum(model[(t1, t2)].values())
        log_prob += np.log2(prob) if prob > 0 else 0
    return np.power(2, -log_prob/N)

@nu.timer
def plot_perplexity(df: pd.DataFrame):
    """
    Function to plot a DataFrame of perplexity scores.

    Args:
    df (pd.DataFrame): DataFrame with perplexity scores, with validation set languages as rows
        and model languages as columns.
    """
    # Convert DataFrame from wide format to long format
    df_long = df.reset_index().melt(id_vars='index', var_name='Model Language', value_name='Perplexity')

    # Create bar chart
    fig = px.bar(df_long, x='index', y='Perplexity', color='Model Language', 
                 labels={'index': 'Validation Set Language'}, barmode='group')

    # Show plot
    fig.show()

@nu.timer
def main():
    conf = nu.load_config("a1") # Load config
    for lang in conf.langs:
        norm_train = nu.load_text_data(f"{conf.paths.normalized_txt}norm_train.{lang}.txt") # load data
        trigrams = generate_trigrams(norm_train) # Generate trigrams        
        model = build_language_model(trigrams) # Build language model
        nu.save_model(model, f"{conf.paths.models}model_{lang}.json") # save model


if __name__ == "__main__":
    main()