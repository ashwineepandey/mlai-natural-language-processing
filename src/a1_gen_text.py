import numpy as np
import pandas as pd
import random
import string
import os
from nltk.util import ngrams
import plotly.express as px
from sklearn.metrics import confusion_matrix
from collections import Counter
from typing import List, Tuple, Dict
import log
import mynlputils as nu
logger = log.get_logger(__name__)


def generate_trigrams(text: str) -> List[Tuple[str, str, str]]:
    """
    Function to generate character-level trigrams from text.

    Args:
    text (str): Text data.

    Returns:
    List[Tuple[str, str, str]]: List of character-level trigrams.
    """
    return list(ngrams(text, 3, pad_left=True, pad_right=True, left_pad_symbol='<s><s>', right_pad_symbol='</s>'))

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

@nu.timer
def load_models(base_path: str, langs: List[str]) -> Dict[str, Dict[Tuple[str, str], Counter]]:
    """
    Function to load the language models.

    Args:
    base_path (str): Base path to the model files.
    langs (List[str]): List of keys for the dictionary.

    Returns:
    Dict[str, Dict[Tuple[str, str], Counter]]: Dictionary containing the language models.
    """
    models = {}
    for lang in langs:
        model_file = os.path.join(base_path, f"model_{lang}.json")
        models[lang] = nu.load_model(model_file)
    return models


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
        if (t1, t2) in model:
            # next_char = model[(t1, t2)].most_common(1)[0][0]
            possible_chars = list(model[(t1, t2)].keys())
            probabilities = [count / sum(model[(t1, t2)].values()) for count in model[(t1, t2)].values()]
            next_char = random.choices(possible_chars, probabilities)[0]
        else:
            # If the current pair of characters is not in the model,
            # append a random character or implement another strategy
            next_char = random.choice(string.ascii_lowercase + ' ')
        # t1, t2 = text[-2], text[-1]
        # next_chars = model[(t1, t2)].most_common(1)
        # if next_chars:
        #     next_char = next_chars[0][0]
        # else:
        #     next_char = '<s>'  # Or some other strategy
        text.append(next_char)
    return ''.join(text)

@nu.timer
def save_generated_text(generated_text: Dict[str, str], dir_path: str):
    """
    Function to save the generated text to files.

    Args:
    generated_text (Dict[str, str]): Dictionary containing the generated text, with language names as keys.
    dir_path (str): Directory path where the files will be saved.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path) # Create directory if it does not exist

    for lang, text in generated_text.items():
        file_path = os.path.join(dir_path, f'generated_text_{lang}.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)


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
        total = sum(model[(t1, t2)].values())
        if total == 0:
            # Choose a small nonzero value if the bigram does not exist in the model
            prob = 1e-10
        else:
            prob = model[(t1, t2)][t3] / total
        log_prob += np.log2(prob) if prob > 0 else 0
    return np.power(2, -log_prob/N)

@nu.timer
def generate_perplexity_table(models: Dict[str, Dict[Tuple[str, str], Counter]], 
                              validation_sets: Dict[str, str]) -> pd.DataFrame:
    """
    Function to generate a table of perplexity scores for multiple language models
    and multiple validation sets.

    Args:
    models (Dict[str, Dict[Tuple[str, str], Counter]]): Dictionary of language models,
        with language names as keys.
    validation_sets (Dict[str, str]): Dictionary of validation sets, with language
        names as keys.

    Returns:
    pd.DataFrame: DataFrame with perplexity scores, with validation set languages as rows
        and model languages as columns.
    """
    perplexity_scores = {}
    for model_lang, model in models.items():
        scores = {}
        for val_lang, val_text in validation_sets.items():
            score = calculate_perplexity(model, val_text)
            scores[val_lang] = score
        perplexity_scores[model_lang] = scores

    df = pd.DataFrame(perplexity_scores)
    return df

@nu.timer
def plot_perplexity(df: pd.DataFrame, plot_filepath: str, title: str):
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
                 labels={'index': 'Validation Set Language'}, barmode='group', title=title)
    # Save plot
    nu.save_plot(plot_filepath, fig, title)


def calculate_accuracy(predicted: List[str], actual: List[str]) -> float:
    """
    Function to calculate the accuracy of predicted language labels.

    Args:
    predicted (List[str]): List of predicted language labels.
    actual (List[str]): List of actual language labels.

    Returns:
    float: Accuracy score.
    """
    correct_predictions = sum(p == a for p, a in zip(predicted, actual))
    total_predictions = len(predicted)
    return correct_predictions / total_predictions

@nu.timer
def classify_text(filepath: str, models: Dict[str, Dict[Tuple[str, str], Counter]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Function to classify texts based on language models.

    Args:
    filepath (str): Filepath to the text file to be classified.
    models (Dict[str, Dict[Tuple[str, str], Counter]]): Dictionary of language models.

    Returns:
    Tuple[List[str], List[str], List[str]]: Tuple of three lists. 
    The first list contains the predicted language labels.
    The second list contains the actual language labels.
    The third list contains the texts that were classified.
    """
    predicted_texts = []
    actual_texts = []
    texts = []
    with open(filepath, 'r') as file:
        for line in file:
            actual_lang = line[:2]
            text = line[3:].strip()
            perplexities = {lang: calculate_perplexity(model, text) for lang, model in models.items()}
            predicted_lang = min(perplexities, key=perplexities.get)
            predicted_texts.append(predicted_lang)
            actual_texts.append(actual_lang)
            texts.append(text)
    return predicted_texts, actual_texts, texts



def get_confusion_matrix(actual, predicted, labels):
    """
    Function to compute the confusion matrix.

    Args:
    actual (list): List of actual labels.
    predicted (list): List of predicted labels.
    labels (list): List of unique labels.

    Returns:
    np.array: The confusion matrix.
    """
    return confusion_matrix(actual, predicted, labels=labels)

@nu.timer
def plot_confusion_matrix(cm, labels, filepath):
    """
    Function to plot the confusion matrix.

    Args:
    cm (np.array): The confusion matrix.
    labels (list): List of unique labels.
    filepath (str): Filepath to save the plot.

    Returns:
    None
    """
    df_cm = pd.DataFrame(cm, index=labels, columns=labels).T
    fig = px.imshow(df_cm, labels=dict(x="True Label", y="Predicted Label", color="Count"), 
                     x=labels, y=labels, title="Confusion Matrix")
    fig.write_html(filepath)



def main():
    conf = nu.load_config("a1") # Load config
    train_data, valid_data = load_datasets(conf.paths.normalized_txt, conf.paths.raw_txt, conf.langs) # load data
    models = load_models(conf.paths.models, conf.langs) # load models
    # Generate text
    generated_text = {}
    for lang in conf.langs:
        generated_text[lang] = generate_text(models[lang])

    # Save generated text
    save_generated_text(generated_text, conf.paths.gen_text)
    # Generate perplexity table
    perplexity_table_train = generate_perplexity_table(models, train_data)
    perplexity_table_valid = generate_perplexity_table(models, valid_data)
    # Plot perplexity
    plot_perplexity(perplexity_table_train, conf.paths.reporting_plots, 'Perplexity of trigram language models on training sets')
    plot_perplexity(perplexity_table_valid, conf.paths.reporting_plots, 'Perplexity of trigram language models on validation sets')

    predicted_texts, actual_texts, texts = classify_text(f"{conf.paths.raw_txt}test.lid.txt", models)
    accuracy = calculate_accuracy(predicted_texts, actual_texts)
    # Calculate confusion matrix
    cm = get_confusion_matrix(actual_texts, predicted_texts, list(models.keys()))
    plot_confusion_matrix(cm, list(models.keys()), f'{conf.paths.reporting_plots}confusion_matrix.html') # Plot confusion matrix
    logger.info(f'Accuracy: {accuracy}')


    # # Calculate perplexity
    # perplexity = {}
    # for lang in conf.langs:
    #     perplexity[lang] = {}
    #     for model_lang in conf.langs:
    #         perplexity[lang][model_lang] = calculate_perplexity(models[model_lang], validation_data[lang])


if __name__ == '__main__':
    main()