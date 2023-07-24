from typing import List, Tuple, Dict
from collections import Counter
import pandas as pd
import log
import mynlputils as nu
from sklearn.model_selection import train_test_split


logger = log.get_logger(__name__)


@nu.timer
def load_data(raw_txt_train_path: str, raw_txt_test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads training and testing data from specified paths.

    Parameters:
    raw_txt_train_path (str): Path to the training data file.
    raw_txt_test_path (str): Path to the testing data file.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas DataFrames: the training and testing data.
    """
    df_train = pd.read_csv(raw_txt_train_path, header=None, names=["label", "title", "description"])
    df_test = pd.read_csv(raw_txt_test_path, header=None, names=["label", "title", "description"])
    return df_train[["label", "description"]], df_test[["label", "description"]]


@nu.timer
def create_validation_set(corpus: pd.DataFrame, valid_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the given corpus into training and validation sets.

    Parameters:
    corpus (pd.DataFrame): The corpus to be split.
    valid_size (float): The proportion of the corpus to include in the validation set.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas DataFrames: the training and validation sets.
    """
    train_corpus, valid_corpus = train_test_split(corpus, test_size=valid_size, random_state=1)
    return train_corpus.reset_index(drop=True), valid_corpus.reset_index(drop=True)


@nu.timer
def clean_text(docs: pd.DataFrame) -> pd.DataFrame:
    """
    Performs text cleaning on the given document DataFrame.

    Parameters:
    docs (pd.DataFrame): DataFrame containing the documents to clean.

    Returns:
    pd.DataFrame: A DataFrame containing the cleaned documents.
    """
    clean_docs = docs['description']
    # add your cleaning steps here
    return clean_docs.to_frame()


@nu.timer
def split_docs(docs: pd.DataFrame) -> List[List[str]]:
    """
    Splits the documents into words (tokens).

    Parameters:
    docs (pd.DataFrame): DataFrame containing the documents to split.

    Returns:
    List[List[str]]: A list of lists, where each inner list contains the tokens for a single document.
    """
    return docs['description'].str.split().to_list()


@nu.timer
def tokenize(tokens: List[List[str]], min_freq: int = 5) -> Tuple[List[str], List[List[int]], Dict[str, int]]:
    """
    Tokenizes the given tokens, generating a vocabulary, a list of tokenized documents, and a word-to-index mapping.

    Parameters:
    tokens (List[List[str]]): A list of lists, where each inner list contains the tokens for a single document.
    min_freq (int, optional): Minimum frequency for a word to be included in the vocabulary. Defaults to 5.

    Returns:
    Tuple[List[str], List[List[int]], Dict[str, int]]: The vocabulary (a list of unique words), the tokenized documents, 
    and a word-to-index mapping.
    """
    word_freq = Counter([word for sentence in tokens for word in sentence])
    vocab = [word for word, freq in word_freq.items() if freq >= min_freq]
    vocab = ['<PAD>', '<UNK>'] + vocab
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx_tokens = [[word2idx.get(word, 1) for word in sentence] for sentence in tokens]
    return vocab, idx_tokens, word2idx


@nu.timer
def create_skipgrams(corpus: List[List[int]], window_size: int, pad_idx: int) -> List[Tuple[List[int], int]]:
    """
    Creates skip-gram pairs from the given corpus.

    Parameters:
    corpus (List[List[int]]): The corpus, which is a list of lists, where each inner list contains the token IDs for a single document.
    window_size (int): The size of the context window for skip-gram generation.
    pad_idx (int): The index to use for padding.

    Returns:
    List[Tuple[List[int], int]]: A list of tuples, where each tuple contains the context and target word IDs for a skip-gram pair.
    """
    data = []
    for sentence in corpus:
        padded_sentence = [pad_idx] * window_size + sentence + [pad_idx] * window_size
        for word_index in range(window_size, len(padded_sentence) - window_size):
            contexts = padded_sentence[word_index - window_size : word_index] + padded_sentence[word_index + 1 : word_index + window_size + 1]
            data.append((contexts, padded_sentence[word_index]))
    logger.info(f"Created {len(data)} skip-gram pairs.")
    return data


def main():
    conf = nu.load_config("a3")
    df_train, df_test = load_data(conf.paths.raw_txt_train, conf.paths.raw_txt_test)
    df_train, df_valid = create_validation_set(df_train, conf.preprocess.valid_size)
    df_train_clean = clean_text(df_train)
    df_valid_clean = clean_text(df_valid)
    df_test_clean = clean_text(df_test)

    train_tokens = split_docs(df_train_clean)
    logger.info(f"Number of training documents: {len(train_tokens)}")
    valid_tokens = split_docs(df_valid_clean)
    logger.info(f"Number of validation documents: {len(valid_tokens)}")
    test_tokens = split_docs(df_test_clean)
    logger.info(f"Number of testing documents: {len(test_tokens)}")

    vocab, idx_train_tokens, word2idx = tokenize(train_tokens)
    logger.info(f"Vocabulary size: {len(vocab)}")
    _, idx_valid_tokens, _ = tokenize(valid_tokens)
    _, idx_test_tokens, _ = tokenize(test_tokens)

    pad_idx = word2idx[conf.preprocess.pad_token]
    skipgrams_train = create_skipgrams(idx_train_tokens, window_size=conf.preprocess.window_size, pad_idx=pad_idx)
    skipgrams_valid = create_skipgrams(idx_valid_tokens, window_size=conf.preprocess.window_size, pad_idx=pad_idx)
    skipgrams_test = create_skipgrams(idx_test_tokens, window_size=conf.preprocess.window_size, pad_idx=pad_idx)
    
    # nu.save_pickle(conf.paths.skipgrams, "skipgrams_train", skipgrams_train)
    # nu.save_pickle(conf.paths.skipgrams, "skipgrams_valid", skipgrams_valid)
    # nu.save_pickle(conf.paths.skipgrams, "skipgrams_test", skipgrams_test)
    # nu.save_pickle(conf.paths.vocab, "vocab", vocab)
    # nu.save_pickle(conf.paths.vocab, "word2idx", word2idx)
    df_train_clean['class'] = df_train['label']
    df_valid_clean['class'] = df_valid['label']
    df_test_clean['class'] = df_test['label']
    df_train_clean.to_csv(conf.paths.model_input, index=False)
    df_valid_clean.to_csv(conf.paths.model_input, index=False)
    df_test_clean.to_csv(conf.paths.model_input, index=False)
    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    main()