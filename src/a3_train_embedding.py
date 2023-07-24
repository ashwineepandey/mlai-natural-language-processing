import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import plotly.graph_objects as go
import random
import log
import mynlputils as nu
from typing import List, Tuple, Dict

logger = log.get_logger(__name__)

@nu.timer
def create_batches(skipgrams: List[Tuple[List[int], int]], vocab_size: int, pad_idx: int, batch_size: int, num_neg_samples: int) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Creates batches of skip-gram pairs for model training.

    Parameters:
    skipgrams (List[Tuple[List[int], int]]): List of skipgram pairs.
    vocab_size int: Length of vocab.
    pad_idx (int): The index to use for padding.
    batch_size (int): The size of each batch.
    num_neg_samples (int): The number of negative samples to use.

    Returns:
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: List of batches.
    """
    n = len(skipgrams)

    # Shuffle skipgrams
    random.shuffle(skipgrams)

    batches = []

    for batch_start in range(0, n, batch_size):
        context_batch = []
        target_batch = []
        negative_batch = []

        # Create batches
        for contexts, target in skipgrams[batch_start:batch_start + batch_size]:
            negatives = [random.choice(range(vocab_size)) for _ in range(num_neg_samples)]
            context_batch.append(torch.LongTensor(contexts))
            target_batch.append(torch.LongTensor([target]))
            negative_batch.append(torch.LongTensor(negatives))

        # If this is the last batch and it's not full, skip it
        if len(context_batch) < batch_size:
            continue

        # Pad context sequences in batch
        context_batch = pad_sequence(context_batch, batch_first=True, padding_value=pad_idx)

        # Convert target and negative batches to tensors
        target_batch = torch.stack(target_batch)
        negative_batch = torch.stack(negative_batch)

        batches.append((context_batch, target_batch, negative_batch))
        
    return batches


class CBOW_NS(nn.Module):
    """
    Continuous Bag of Words model with Negative Sampling.

    Methods:
    forward(context_words, target_word, negative_words): Forward pass through the model.
    """
    def __init__(self, vocab_size, embed_size):
        super(CBOW_NS, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)

    def forward(self, context_words, target_word, negative_words):
        """
        Forward pass through the model.

        Parameters:
        context_words (torch.Tensor): Tensor of context words.
        target_word (torch.Tensor): Tensor of target words.
        negative_words (torch.Tensor): Tensor of negative words.

        Returns:
        torch.Tensor: The loss for the forward pass.
        """
        # Get embeddings for context words, target word and negative words
        context_embeds = self.embeddings(context_words)  # (batch_size, window_size*2, embed_size)
        target_embeds = self.embeddings(target_word)    # (batch_size, 1, embed_size)
        negative_embeds = self.embeddings(negative_words)# (batch_size, num_neg_samples, embed_size)

        # Sum the context word embeddings
        context_embeds_sum = torch.sum(context_embeds, dim=1, keepdim=True)  # (batch_size, 1, embed_size)

        # Compute positive score
        pos_score = torch.bmm(context_embeds_sum, target_embeds.transpose(1,2)) # (batch_size, 1, 1)
        pos_score = F.logsigmoid(pos_score)

        # Compute negative score
        neg_score = torch.bmm(context_embeds_sum, negative_embeds.transpose(1,2)) # (batch_size, 1, num_neg_samples)
        neg_score = F.logsigmoid(-neg_score)

        # Return negative of total score
        return -(torch.sum(pos_score) + torch.sum(neg_score))
    
@nu.timer
def train(model: nn.Module, epochs: int, train_batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], val_batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], lr: float) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Trains the model for a specified number of epochs.

    Parameters:
    model (nn.Module): The model to train.
    epochs (int): The number of epochs to train for.
    train_batches (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): The training data batches.
    val_batches (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): The validation data batches.
    lr (float): The learning rate for the Adam optimizer.

    Returns:
    Tuple[nn.Module, List[float], List[float]]: The trained model and lists of training and validation losses over epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        for context_batch, target_batch, negative_batch in train_batches:
            model.zero_grad()
            loss = model(context_batch, target_batch, negative_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        train_loss = total_loss / batch_count
        train_losses.append(train_loss)

        model.eval()  # set model to eval mode
        val_loss = evaluate(model, val_batches)
        val_losses.append(val_loss)

        logger.info(f'Epoch {epoch}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')
    return model, train_losses, val_losses


def evaluate(model: nn.Module, batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
    """
    Evaluates the model on given data batches.

    Parameters:
    model (nn.Module): The model to evaluate.
    batches (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): The data batches.

    Returns:
    float: The average loss over all batches.
    """
    total_loss = 0
    batch_count = 0
    with torch.no_grad():  # disable gradient computation to save memory
        for context_batch, target_batch, negative_batch in batches:
            loss = model(context_batch, target_batch, negative_batch)
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count


def plot_losses(train_losses: List[float], val_losses: List[float], epochs: int, plot_filepath: str, filename: str):
    """
    Plots the training and validation losses over epochs.

    Parameters:
    train_losses (List[float]): The training losses over epochs.
    val_losses (List[float]): The validation losses over epochs.
    epochs (int): The number of epochs.
    plot_filepath (str): The filepath to save the plot.
    filename (str): The filename for the plot.
    """
    epochs_range = list(range(1, epochs + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs_range, y=train_losses, mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(x=epochs_range, y=val_losses, mode='lines', name='Validation Loss'))
    fig.update_layout(title='Loss over Epochs', xaxis=dict(title='Epoch'), yaxis=dict(title='Loss'))
    nu.save_plot(plot_filepath, fig, filename)


def main():
    conf = nu.load_config("a3")
    skipgrams_train = nu.load_pickle(conf.paths.skipgrams, "skipgrams_train")
    skipgrams_valid = nu.load_pickle(conf.paths.skipgrams, "skipgrams_valid")
    vocab = nu.load_pickle(conf.paths.vocab, "vocab")
    word2idx = nu.load_pickle(conf.paths.vocab, "word2idx")
    train_batches = create_batches(skipgrams_train, len(vocab), word2idx[conf.preprocess.pad_token], conf.model.batch_size, conf.model.num_neg_samples)
    valid_batches = create_batches(skipgrams_valid, len(vocab), word2idx[conf.preprocess.pad_token], conf.model.batch_size, conf.model.num_neg_samples)
    model = CBOW_NS(len(vocab), conf.model.embed_size)
    trained_model, train_losses, val_losses = train(model, conf.model.epochs, train_batches, valid_batches, conf.model.lr)
    nu.save_pytorch_model(trained_model, file_path=f"{conf.paths.models}cbow_ns_{nu._get_current_dt()}.pt")
    plot_losses(train_losses, val_losses, conf.model.epochs, conf.paths.reporting_plots, "training_loss")


if __name__ == "__main__":
    main()