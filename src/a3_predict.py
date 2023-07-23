import mynlputils as nu


def main():
    conf = nu.load_config("a3")
    skipgrams_test = nu.load_pickle(conf.paths.skipgrams, "skipgrams_test")
    vocab = nu.load_pickle(conf.paths.vocab, "vocab")

    model = nu.load_model(conf.paths.model, "model")

if __name__ == "__main__":
    main()