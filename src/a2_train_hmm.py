import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Union
import log
import mynlputils as nu

logger = log.get_logger(__name__)

class HMM:
    def __init__(self):
        self.transition_probs = dict()
        self.emission_probs = dict()
        self.start_probs = dict()

@nu.timer
def train_hmm(sentences: List[pd.DataFrame], smooth_value: float = 1.0) -> HMM:
    """
    Trains a Hidden Markov Model (HMM) given a list of sentences. 
    Applies Laplace smoothing when calculating probabilities.

    Args:
    sentences (List[pd.DataFrame]): List of DataFrames, each representing a sentence.

    Returns:
    HMM: Trained HMM model.
    """
    model = HMM()

    transition_counts = defaultdict(int)
    emission_counts = defaultdict(int)
    start_counts = defaultdict(int)

    for sentence in sentences:
        prev_tag = None
        for _, row in sentence.iterrows():
            token, tag = row['Token'], row['POS']
            if prev_tag is None:
                start_counts[tag] += 1
            else:
                transition_counts[(prev_tag, tag)] += 1
            emission_counts[(tag, token)] += 1
            prev_tag = tag

    unique_transitions = len(transition_counts)
    unique_emissions = len(emission_counts)

    total_transitions = sum(transition_counts.values())
    total_emissions = sum(emission_counts.values())
    total_starts = sum(start_counts.values())

    model.transition_probs = {k: (v + smooth_value) / (total_transitions + smooth_value*unique_transitions) for k, v in transition_counts.items()}
    model.emission_probs = {k: (v + smooth_value) / (total_emissions + smooth_value*unique_emissions) for k, v in emission_counts.items()}
    model.start_probs = {k: v / total_starts for k, v in start_counts.items()}
    return model


def main():
    conf = nu.load_config("a2")
    sentences_train = nu.load_pickle(conf.paths.processed_txt, "sentences_train")
    hmm_model = train_hmm(sentences_train, conf.model.smooth_value)
    nu.save_pickle(conf.paths.models, "hmm_model", hmm_model)
    logger.info(f"Trained HMM model saved to {conf.paths.models}")


if __name__ == "__main__":
    main()

