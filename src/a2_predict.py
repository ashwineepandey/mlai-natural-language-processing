import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
import log
import mynlputils as nu
from a2_train_hmm import HMM, train_hmm

logger = log.get_logger(__name__)

def viterbi_algorithm(model: HMM, sentence: pd.DataFrame, unk_word_prob: float) -> pd.DataFrame:
    """
    Uses the Viterbi algorithm to find the most probable sequence of hidden states (POS tags).
    Handles unknown words by assigning a small constant probability for every state.

    Args:
    model (HMM): Trained HMM model.
    sentence (pd.DataFrame): DataFrame representing a sentence.

    Returns:
    pd.DataFrame: DataFrame containing the tokens, actual tags, and predicted tags for each sentence.
    """
    tokens = sentence['Token'].tolist()
    actual_tags = sentence['POS'].tolist()
    states = list(set([state for state, _ in model.emission_probs.keys()]))
    n_states = len(states)
    n_tokens = len(tokens)

    dp = np.zeros((n_states, n_tokens))
    ptr = np.zeros((n_states, n_tokens), dtype=int)

    # # Initialization
    # for i, state in enumerate(states):
    #     dp[i, 0] = model.start_probs.get(state, 0) * model.emission_probs.get((state, tokens[0]), unk_word_prob)

    # # Recursion
    # for t in range(1, n_tokens):
    #     for j, state in enumerate(states):
    #         max_prob = 0
    #         max_state = 0
    #         for i, prev_state in enumerate(states):
    #             prob = dp[i, t-1] * model.transition_probs.get((prev_state, state), 0) * model.emission_probs.get((state, tokens[t]), unk_word_prob)
    #             if prob > max_prob:
    #                 max_prob = prob
    #                 max_state = i
    #         dp[j, t] = max_prob
    #         ptr[j, t] = max_state

    # Vectorized Initialization
    dp[:, 0] = [model.start_probs.get(state, 0) * model.emission_probs.get((state, tokens[0]), unk_word_prob) for state in states]

    # Optimized Recursion
    for t in range(1, n_tokens):
        for j, state in enumerate(states):
            probabilities = dp[:, t-1] * np.array([model.transition_probs.get((prev_state, state), 0) for prev_state in states]) * model.emission_probs.get((state, tokens[t]), unk_word_prob)
            max_state = np.argmax(probabilities)
            dp[j, t] = probabilities[max_state]
            ptr[j, t] = max_state

    # Traceback
    best_path = [np.argmax(dp[:, -1])]
    for t in range(n_tokens-1, 0, -1):
        best_path.append(ptr[best_path[-1], t])
    best_path = [states[i] for i in reversed(best_path)]

    result = pd.DataFrame({
        'Token': tokens,
        'Actual_POS': actual_tags,
        'Predicted_POS': best_path
    })
    return result

@nu.timer
def evaluate_hmm(model: HMM, sentences: List[pd.DataFrame], unk_word_prob) -> float:
    """
    Evaluates the performance of the HMM model by calculating the accuracy of POS tagging.

    Args:
    model (HMM): Trained HMM model.
    sentences (List[pd.DataFrame]): List of DataFrames, each representing a sentence.

    Returns:
    float: Accuracy of POS tagging.
    df_results (pd.DataFrame): DataFrame containing the 'Token', 'Actual_POS', and 'Predicted_POS' for each sentence.

    """
    results = []
    for sentence in sentences:
        result = viterbi_algorithm(model, sentence, unk_word_prob)
        results.append(result)
    df_results = pd.concat(results, ignore_index=True)
    correct_tags = np.sum(df_results['Actual_POS'] == df_results['Predicted_POS'])
    total_tags = len(df_results)
    accuracy = correct_tags / total_tags
    return accuracy, df_results


@nu.timer
def cross_validation(sentences: List[pd.DataFrame], smooth_values: List[float], k: int) -> float:
    """
    Performs k-fold cross-validation to choose the best smoothing value.

    Args:
    sentences (List[pd.DataFrame]): List of DataFrames, each representing a sentence.
    smooth_values (List[float]): The values to try for Laplace smoothing.
    k (int): The number of folds for cross-validation.

    Returns:
    float: The smoothing value that resulted in the highest average accuracy.
    """
    n = len(sentences)
    fold_size = n // k

    best_smooth_value = None
    best_accuracy = 0

    for smooth_value in smooth_values:
        accuracies = []
        for i in range(k):
            validation_sentences = sentences[i*fold_size:(i+1)*fold_size]
            training_sentences = sentences[:i*fold_size] + sentences[(i+1)*fold_size:]
            model = train_hmm(training_sentences, smooth_value)
            accuracy, _ = evaluate_hmm(model, validation_sentences)
            accuracies.append(accuracy)
        average_accuracy = sum(accuracies) / len(accuracies)
        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            best_smooth_value = smooth_value
    return best_smooth_value


def main():
    conf = nu.load_config("a2")
    sentences_test = nu.load_pickle(conf.paths.processed_txt, "sentences_test")
    sentences_valid = nu.load_pickle(conf.paths.processed_txt, "sentences_valid")
    hmm_model = nu.load_pickle(conf.paths.models, "hmm_model")
    valid_accuracy, valid_results = evaluate_hmm(hmm_model, sentences_valid, conf.model.unk_word_prob)
    test_accuracy, test_results = evaluate_hmm(hmm_model, sentences_test, conf.model.unk_word_prob)
    logger.info(f'Validation Set Accuracy: {valid_accuracy * 100:.2f}%')
    logger.info(f'Test Set Accuracy: {test_accuracy * 100:.2f}%')
    # Save results
    valid_results.to_csv(f"{conf.paths.model_output}valid_results.csv", index=False)
    test_results.to_csv(f"{conf.paths.model_output}test_results.csv", index=False)


if __name__ == "__main__":
    main()