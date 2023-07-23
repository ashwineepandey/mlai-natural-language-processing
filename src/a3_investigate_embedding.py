import mynlputils as nu
from src.a3_train_embedding import CBOW_NS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import plotly.express as px
from typing import List
import log

logger = log.get_logger(__name__)

@nu.timer
def create_tsne_embeddings(embeddings: np.ndarray, n_components: int, vocab: List[str], filepath: str, obj_name: str):
    # Create TSNE model
    tsne = TSNE(n_components=n_components, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings[:len(vocab), :])
    nu.save_pickle(filepath, obj_name, embeddings_2d)
    return embeddings_2d

@nu.timer
def create_pca_embeddings(embeddings: np.ndarray, n_components: int, vocab: List[str], filepath: str, obj_name: str):
    # Create PCA model
    pca = PCA(n_components=n_components)
    embeddings_2d = pca.fit_transform(embeddings[:len(vocab), :])
    nu.save_pickle(filepath, obj_name, embeddings_2d)
    return embeddings_2d

@nu.timer
def plot_embeddings(embeddings_2d, vocab_subset: List[str], path: str, filename: str, embedding_type: str):
    """
    Creates an embedding plot of the given word embeddings.
    """
    fig = px.scatter(
        embeddings_2d, x=0, y=1, text=vocab_subset, labels={"0": "Dimension 1", "1": "Dimension 2"}, title=f"Word Embeddings {embedding_type}"
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(showlegend=False)
    nu.save_plot(path, fig, filename)


def find_closest_embeddings(embedding, word, word2idx, idx2word, n=10):
    word_embedding = embedding[word2idx[word]]
    similarities = cosine_similarity([word_embedding], embedding)[0] # Compute cosine similarities between word_embedding and all embeddings
    closest_idxs = np.argsort(similarities)[-n:] # Get the indices of the top n similar embeddings
    return [(idx2word[idx], similarities[idx]) for idx in reversed(closest_idxs)] # Convert these indices back to words and return

@nu.timer
def process_word_list(word_list, embedding, word2idx, idx2word, n, output_filepath):
    data = []
    for word in word_list:
        try:
            closest_words_with_distances = find_closest_embeddings(embedding, word, word2idx, idx2word, n)
            for close_word, distance in closest_words_with_distances:
                data.append({"input_word": word, "closest_word": close_word, "cosine_similarity": distance})
        except KeyError:
            logger.error(f"Word '{word}' not found in vocabulary.")
    df = pd.DataFrame(data)
    df.to_csv(output_filepath, index=False)
    logger.info(f"Closest words saved: {output_filepath}")


def main():
    conf = nu.load_config("a3")
    vocab_subset = conf.embedding_viz.vocab_subset
    vocab = nu.load_pickle(conf.paths.vocab, "vocab")
    word2idx = nu.load_pickle(conf.paths.vocab, "word2idx")
    idx2word = {v: k for k, v in word2idx.items()}
    model = CBOW_NS(len(vocab), conf.model.embed_size)
    model = nu.load_pytorch_model(model, f"{conf.paths.model}")
    embeddings = model.embeddings.weight.detach().numpy()
    embeddings_2d_tsne = create_tsne_embeddings(embeddings, 2, vocab_subset, conf.paths.embeddings, "embeddings_2d_tsne")
    embeddings_2d_pca = create_pca_embeddings(embeddings, 2, vocab_subset, conf.paths.embeddings, "embeddings_2d_pca")
    plot_embeddings(embeddings_2d_tsne, vocab_subset, conf.paths.reporting_plots, "word-embeddings-tsne", "TSNE")
    plot_embeddings(embeddings_2d_pca, vocab_subset, conf.paths.reporting_plots, "word-embeddings-pca", "PCA")
    process_word_list(vocab_subset, embeddings, word2idx, idx2word, conf.embedding_viz.num_closest_words, f"{conf.paths.model_output}closest_words.csv")


if __name__ == "__main__":
    main()