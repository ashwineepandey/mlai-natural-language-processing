import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import plotly.express as px
from typing import List, Tuple, Dict, Union
import log
import mynlputils as nu

logger = log.get_logger(__name__)


def load_data(raw_txt_train_path: str, raw_txt_test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_excel(raw_txt_train_path)
    df_test = pd.read_excel(raw_txt_test_path)
    logger.info(f"df_train.shape: {df_train.shape}")
    logger.info(f"df_train unique tokens: {df_train['Token'].nunique()}")
    logger.info(f"df_train unique POS: {df_train['POS'].nunique()}")
    logger.info(f"df_test.shape: {df_test.shape}")
    logger.info(f"df_test unique tokens: {df_test['Token'].nunique()}")
    logger.info(f"df_test unique POS: {df_test['POS'].nunique()}")
    return df_train, df_test


def remove_punctuation(data: pd.DataFrame):
    """
    Removes rows with 'PUNCT' in the 'POS' column from the dataset.

    Args:
    data (pd.DataFrame): DataFrame containing the tokenized isiZulu data with 'Token' and 'POS' columns.

    Returns:
    pd.DataFrame: DataFrame with rows containing 'PUNCT' removed.
    """
    return data[data['POS'] != 'PUNCT']


def split_into_sentences(df: pd.DataFrame, start_token: List[str], stop_token: List[str]) -> List[pd.DataFrame]:
    """
    Splits a DataFrame into a list of DataFrames each representing a sentence.
    Adds start and stop tokens to each sentence.

    Args:
    df (pd.DataFrame): DataFrame containing the tokenized isiZulu data with 'Token' and 'POS' columns.

    Returns:
    List[pd.DataFrame]: List of DataFrames each representing a sentence.
    """
    df['Sentence'] = (df['Token'].isna().cumsum())
    df = df.dropna()
    sentences = [group for _, group in df.groupby('Sentence')]
    for i in range(len(sentences)):
        _start_token_df = pd.DataFrame([start_token], columns=['Token', 'POS'])
        _stop_token_df = pd.DataFrame([stop_token], columns=['Token', 'POS'])
        sentences[i] = pd.concat([_start_token_df, sentences[i], _stop_token_df], ignore_index=True)
    return sentences


def create_validation_set(sentences: List[pd.DataFrame], valid_size: float, seed: int) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Creates a validation set from a list of DataFrames each representing a sentence.

    Args:
    sentences (List[pd.DataFrame]): List of DataFrames each representing a sentence.
    valid_size (float): Proportion of sentences to include in the validation set.

    Returns:
    Tuple[List[pd.DataFrame], List[pd.DataFrame]]: Training and validation sets.
    """
    train_sentences, valid_sentences = train_test_split(sentences, test_size=valid_size, random_state=seed)
    return train_sentences, valid_sentences


def main():
    conf = nu.load_config("a2")
    df_train, df_test = load_data(conf.paths.raw_txt_train, conf.paths.raw_txt_test)
    df_train = remove_punctuation(df_train)
    df_test = remove_punctuation(df_test)
    sentences_train = split_into_sentences(df_train, conf.preprocess.start_token, conf.preprocess.stop_token)
    sentences_test = split_into_sentences(df_test, conf.preprocess.start_token, conf.preprocess.stop_token)
    sentences_train, sentences_valid = create_validation_set(sentences_train, conf.preprocess.valid_size, conf.preprocess.seed)
    nu.save_pickle(conf.paths.processed_txt, "sentences_test", sentences_test)
    nu.save_pickle(conf.paths.processed_txt, "sentences_train", sentences_train)
    nu.save_pickle(conf.paths.processed_txt, "sentences_valid", sentences_valid)


if __name__ == "__main__":
    main()