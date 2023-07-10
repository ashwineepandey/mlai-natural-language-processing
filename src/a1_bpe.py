import re
import os
import plotly.express as px
from sklearn.metrics import confusion_matrix
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import log
import mynlputils as nu

logger = log.get_logger(__name__)

@nu.timer
def load_datasets(base_path_train: str, base_path_val: str, langs: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Function to load the training and validation datasets.

    Args:
    base_path_train (str): Base path to the training files.
    base_path_val (str): Base path to the validation files.
    langs (List[str]): List of keys for the dictionaries.

    Returns:
    Tuple[Dict[str, str], Dict[str, str]]: Tuple of two dictionaries containing the training and validation data.
    """
    training_data = {}
    validation_data = {}
    for lang in langs:
        training_file = os.path.join(base_path_train, f"norm_train.{lang}.txt")
        validation_file = os.path.join(base_path_val, f"val.{lang}.txt")
        training_data[lang] = nu.load_text_data(training_file)
        validation_data[lang] = nu.load_text_data(validation_file)
    return training_data, validation_data


def get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """
    Function to compute the frequency of each pair of symbols in the vocabulary.

    Args:
    vocab (Dict[str, int]): Vocabulary with frequencies.

    Returns:
    Dict[Tuple[str, str], int]: A dictionary with pair of symbols as keys and their frequency as values.
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs


def merge_vocab(pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
    """
    Function to replace the most frequent pair in the vocabulary with a new symbol.

    Args:
    pair (Tuple[str, str]): Most frequent pair of symbols.
    v_in (Dict[str, int]): Initial vocabulary.

    Returns:
    Dict[str, int]: Updated vocabulary after the merge.
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def get_vocab(text: str) -> Dict[str, int]:
    """
    Function to initialize the vocabulary so that each word is a sequence of separate characters.

    Args:
    text (str): Text data.

    Returns:
    Dict[str, int]: Vocabulary with frequencies.
    """
    words = text.split()
    vocab = Counter(words)
    return {' '.join(word): freq for word, freq in vocab.items()}


def byte_pair_encoding(text: str, num_iterations: int) -> Dict[str, int]:
    """
    Function to learn byte pair encoding (BPE) vocabulary.

    Args:
    text (str): Text data.
    num_iterations (int): Number of iterations.

    Returns:
    Dict[str, int]: BPE vocabulary.
    """
    vocab = get_vocab(text)
    for i in range(num_iterations):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return vocab


def get_merged_chars(vocab: Dict[str, int], num_iterations: int) -> List[Tuple[str, str]]:
    """
    Function to get the merged characters for the first 'num_iterations' iterations.

    Args:
    vocab (Dict[str, int]): Vocabulary with frequencies.
    num_iterations (int): Number of iterations.

    Returns:
    List[Tuple[str, str]]: List of merged characters.
    """
    merged_chars = []
    for i in range(num_iterations):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merged_chars.append(best)
        vocab = merge_vocab(best, vocab)
    return merged_chars


def get_vocab_overlap(vocabs: Dict[str, Dict[str, int]]) -> Dict[Tuple[str, str], int]:
    """
    Function to get the overlap of the BPE subword vocabulary between each of the languages.

    Args:
    vocabs (Dict[str, Dict[str, int]]): Dictionary of BPE vocabularies for each language.

    Returns:
    Dict[Tuple[str, str], int]: Overlap of the BPE subword vocabulary between each of the languages.
    """
    overlaps = {}
    languages = list(vocabs.keys())
    for i in range(len(languages)):
        for j in range(i+1, len(languages)):
            lang1, lang2 = languages[i], languages[j]
            overlap = len(set(vocabs[lang1].keys()) & set(vocabs[lang2].keys()))
            overlaps[(lang1, lang2)] = overlap
    return overlaps


def main():
    conf = nu.load_config("a1") # Load config
    train_data, valid_data = load_datasets(conf.paths.normalized_txt, conf.paths.raw_txt, conf.langs) # load data

    bpe_vocabs = {}
    for lang in conf.langs:
        bpe_vocab = byte_pair_encoding(train_data[lang], 100)
        bpe_vocabs[lang] = bpe_vocab
        logger.info(f'{lang} | BPE vocab size: {len(bpe_vocab)}')
        logger.info(f'{lang} | BPE vocab: {bpe_vocab}')

        # Report chars merged in first 10 iters
        merged_chars = get_merged_chars(bpe_vocab, 10)
        logger.info(f'{lang} | Merged characters in the first 10 iters: {merged_chars}')

    # Report overlap of BPE subword vocab between langs
    vocab_overlap = get_vocab_overlap(bpe_vocabs)
    logger.info(f'BPE subword vocabulary overlap between each of the languages: {vocab_overlap}')


if __name__ == '__main__':
    main()