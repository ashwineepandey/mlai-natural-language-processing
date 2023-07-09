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


def initial_eda(data: str) -> None:
    """
    Function to perform initial exploratory data analysis.

    Args:
    data (str): Text data.
    """
    num_chars = len(data)
    num_words = len(data.split())
    num_sentences = data.count('.')

    logger.info(f'Number of characters: {num_chars}')
    logger.info(f'Number of words: {num_words}')
    logger.info(f'Number of sentences: {num_sentences}')


def plot_character_frequency(data: str, plot_filepath, title='Character Frequency') -> None:
    """
    Function to plot character frequency.

    Args:
    data (str): Corpus.
    """
    character_counts = Counter(data) # Count the frequency of each character
    df = pd.DataFrame.from_dict(character_counts, orient='index').reset_index() # Convert the counter to a DataFrame for plotting
    fig = px.bar(df, x='index', y=0, labels={'index': 'Characters', '0': 'Count'}, title=title,
                 log_y=True) # Plot character frequency
    nu.save_plot(plot_filepath, fig, title)


def plot_word_length(data: str, plot_filepath, title='Word Length Distribution') -> None:
    """
    Function to plot word length.

    Args:
    data (str): Corpus.
    """
    word_lengths = [len(word) for word in data.split()] # Get the length of each word
    df = pd.DataFrame(word_lengths, columns=['Word Length']) # Convert the list to a DataFrame for plotting
    fig = px.histogram(df, x='Word Length', nbins=50, title=title, log_y=True) # Plot word length
    nu.save_plot(plot_filepath, fig, title)


def plot_zipfs_law(data: str, plot_filepath, title="Zipf's Law") -> None:
    """
    Function to plot Zipf's law.

    Args:
    data (str): Corpus.
    """
    word_counts = Counter(data.split()) # Count the frequency of each word
    words, counts = zip(*word_counts.most_common()) # Sort words by frequency
    df = pd.DataFrame({'Word': words, 'Frequency': counts}) # Convert the lists to a DataFrame for plotting
    df['Rank'] = df['Frequency'].rank(method='min', ascending=False) # Add a column for rank
    fig = px.scatter(df, x='Rank', y='Frequency', title=title, log_x=True, log_y=True) # Plot Zipf's law
    # Add annotations for top 5 words
    for i in range(5):
        fig.add_annotation(
            x=np.log10(df.loc[i, 'Rank']),  # Apply log transformation
            y=np.log10(df.loc[i, 'Frequency']),  # Apply log transformation
            text=df.loc[i, 'Word'],
            # showarrow=False
        )
    nu.save_plot(plot_filepath, fig, title)

@nu.timer
def normalize_text(text: str) -> str:
    """
    Function to normalize the text data.

    Args:
    text (str): Text data.

    Returns:
    str: Normalized text data.
    """
    text = text.lower() # Lowercase the text
    text = re.sub(r'\d', '0', text) # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    sentences = sent_tokenize(text) # Tokenize into sentences
    normalized_sentences = []
    for sentence in sentences:
        sentence = re.sub(r"[^\w\s]", '', sentence) # Remove punctuation
        normalized_sentences.append(sentence)
    # Remove extra spaces
    normalized_sentences = normalized_sentences.strip()
    return normalized_sentences


def main():
    conf = nu.load_config("a1")

    for lang in conf.langs:
        logger.info(f"Loading {lang} data.")
        train_data = nu.load_text_data(f"{conf.paths.raw_txt}train.{lang}.txt")
        initial_eda(data=train_data)
        norm_train = normalize_text(train_data)
        nu.write_text_data(f"{conf.paths.normalized_txt}norm_train_{lang}.txt", norm_train)

        plot_character_frequency(norm_train, f"{conf.paths.reporting_plots}", title=f"Character Frequency for {lang}")
        plot_word_length(norm_train, f"{conf.paths.reporting_plots}", title=f"Word Length Distribution for {lang}")
        plot_zipfs_law(norm_train, f"{conf.paths.reporting_plots}", title=f"Zipf's Law for {lang}")


if __name__ == '__main__':
    main()