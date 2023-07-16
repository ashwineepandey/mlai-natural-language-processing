import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
import log
import mynlputils as nu

logger = log.get_logger(__name__)

def viterbi_algorithm(model: HMM, sentence: pd.DataFrame) -> pd.DataFrame:
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
    unknown_word_prob = 1e-6  # Small constant probability for unknown words

    # Initialization
    for i, state in enumerate(states):
        dp[i, 0] = model.start_probs.get(state, 0) * model.emission_probs.get((state, tokens[0]), unknown_word_prob)

    # Recursion
    for t in range(1, n_tokens):
        for j, state in enumerate(states):
            max_prob = 0
            max_state = 0
            for i, prev_state in enumerate(states):
                prob = dp[i, t-1] * model.transition_probs.get((prev_state, state), 0) * model.emission_probs.get((state, tokens[t]), unknown_word_prob)
                if prob > max_prob:
                    max_prob = prob
                    max_state = i
            dp[j, t] = max_prob
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
def evaluate_hmm(model: HMM, sentences: List[pd.DataFrame]) -> float:
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
        result = viterbi_algorithm(model, sentence)
        results.append(result)
    df_results = pd.concat(results, ignore_index=True)
    correct_tags = np.sum(df_results['Actual_POS'] == df_results['Predicted_POS'])
    total_tags = len(df_results)
    accuracy = correct_tags / total_tags
    return accuracy, df_results


def main():
    conf = nu.load_config("a2")
    sentences_test = nu.load_pickle(conf.paths.processed_txt, "sentences_test")
    sentences_valid = nu.load_pickle(conf.paths.processed_txt, "sentences_valid")
    hmm_model = nu.load_pickle(conf.paths.models, "hmm_model")
    valid_accuracy, valid_results = evaluate_hmm(hmm_model, sentences_valid)
    test_accuracy, test_results = evaluate_hmm(hmm_model, sentences_test)
    logger.info(f'Validation Set Accuracy: {valid_accuracy * 100:.2f}%')
    logger.info(f'Test Set Accuracy: {test_accuracy * 100:.2f}%')
    # Save results
    valid_results.to_csv(conf.paths.model_output / "valid_results.csv", index=False)
    test_results.to_csv(conf.paths.model_output / "test_results.csv", index=False)


if __name__ == "__main__":
    main()