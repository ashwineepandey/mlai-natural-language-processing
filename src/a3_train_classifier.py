import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import random
from typing import List, Tuple
import plotly.graph_objects as go
from a3_train_embedding import CBOW_NS
import log
import mynlputils as nu

logger = log.get_logger(__name__)


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[List[float]], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class TextClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super(TextClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


def load_embeddings(model_path: str, vocab: List[str], embed_size) -> dict:
    """
    Load embeddings from a trained PyTorch model.

    Parameters:
    model_path (str): The path of the trained model file.

    Returns:
    dict: A dictionary where keys are the words and values are the word embeddings.
    """
    model = CBOW_NS(len(vocab), embed_size)
    trained_model = nu.load_pytorch_model(model, model_path)
    embeddings = trained_model.embeddings.weight.detach().numpy()
    embedding_dict = {word: embeddings[i] for i, word in enumerate(vocab)} # Create a dictionary {word: embedding}
    return embedding_dict


def average_embeddings(text: str, embeddings: dict) -> np.array:
    """
    Calculates the average embeddings of a sentence.
    
    Args:
        text (str): The sentence to calculate embeddings for.
        embeddings (dict): The word embeddings.
    
    Returns:
        np.array: The average embedding of the sentence.
    """
    words = text.split()
    return np.mean([embeddings.get(word, np.zeros((300,))) for word in words], axis=0)


def create_batches_classifier(data: List[Tuple[List[np.ndarray], int]], batch_size: int, pad_value: np.ndarray) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Creates batches of text and label pairs for model training.

    Parameters:
    data (List[Tuple[List[np.ndarray], int]]): List of (text, label) pairs where text is a list of word embeddings and label is an integer.
    batch_size (int): The size of each batch.
    pad_value (np.ndarray): The value to use for padding.

    Returns:
    List[Tuple[torch.Tensor, torch.Tensor]]: List of batches.
    """
    n = len(data)

    # Shuffle data
    random.shuffle(data)

    batches = []

    for batch_start in range(0, n, batch_size):
        text_batch = []
        label_batch = []

        # Create batches
        for text, label in data[batch_start:batch_start + batch_size]:
            text_batch.append(torch.FloatTensor(text))
            label_batch.append(torch.LongTensor([label]))

        # If this is the last batch and it's not full, skip it
        if len(text_batch) < batch_size:
            continue

        # Pad text sequences in batch
        text_batch = pad_sequence(text_batch, batch_first=True, padding_value=pad_value)

        # Convert label batch to tensor
        label_batch = torch.cat(label_batch)

        batches.append((text_batch, label_batch))
        
    return batches


@nu.timer
def train(model: nn.Module, epochs: int, train_loader: DataLoader, val_loader: DataLoader, lr: float, device: torch.device) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Trains the model for a specified number of epochs.

    Parameters:
    model (nn.Module): The model to train.
    epochs (int): The number of epochs to train for.
    train_loader (DataLoader): DataLoader for the training data.
    val_loader (DataLoader): DataLoader for the validation data.
    lr (float): The learning rate for the Adam optimizer.
    device (torch.device): The device to train on.

    Returns:
    Tuple[nn.Module, List[float], List[float]]: The trained model and lists of training and validation losses over epochs.
    """
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        # Validation
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        val_losses.append(total_loss / len(val_loader))

        logger.info(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')

    return model, train_losses, val_losses


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
    vocab = nu.load_pickle(conf.paths.vocab, "vocab")
    embedding_dict = load_embeddings(f"{conf.paths.models}cbow_ns_24072023_012312.pt", vocab, conf.model.embed_size)
    # df_train = pd.read_csv(conf.paths.raw_txt_train, header=None, names=["label", "title", "description"])
    df_train_clean = pd.read_csv(conf.paths.model_input)
    df_valid_clean = pd.read_csv(conf.paths.model_input)
    
    df_train_clean['embeddings'] = df_train_clean['description'].apply(lambda x: average_embeddings(x, embedding_dict))
    df_valid_clean['embeddings'] = df_valid_clean['description'].apply(lambda x: average_embeddings(x, embedding_dict))

    encoded_train_data = list(zip(df_train_clean['embeddings'], df_train_clean['class']))
    encoded_valid_data = list(zip(df_valid_clean['embeddings'], df_valid_clean['class']))

    train_batches = create_batches_classifier(encoded_train_data, conf.model.batch_size, 0)
    valid_batches = create_batches_classifier(encoded_valid_data, conf.model.batch_size, 0)

    device = torch.device("cpu")
    model = TextClassifier(conf.model.embed_size, conf.model.num_classes)
    trained_model, train_losses, val_losses = train(model, conf.model.classifier.epochs, train_batches, valid_batches, conf.model.classifier.lr, device)
    nu.save_pytorch_model(trained_model, file_path=f"{conf.paths.models}text_classifier_{nu._get_current_dt()}.pt")
    plot_losses(train_losses, val_losses, conf.model.classifier.epochs, conf.paths.plots, "text_classifier_losses.png")

if __name__ == "__main__":
    main()