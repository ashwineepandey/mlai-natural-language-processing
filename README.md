# mlai-natural-language-processing

## Table of Contents

1. [A1: Language modelling and byte-pair Encoding](#assignment1)
2. [A2: Test](#assignment2)

## Language modelling and byte-pair Encoding <a name="assignment1"></a>

This project aims to create language models for multiple languages including Afrikaans, English, Dutch, Xhosa and Zulu. The project is divided into various steps from data exploration, preprocessing, training the model, to generating text and predicting language models.

### Directory Structure

The project has the following directory structure:

```
.
├── README.md
├── conf
│   └── a1.conf
├── data
│   ├── 01_raw
│   │   └── wiki_5lang
│   │       ├── test.lid.txt
│   │       ├── train.af.txt
│   │       ├── train.en.txt
│   │       ├── train.nl.txt
│   │       ├── train.xh.txt
│   │       ├── train.zu.txt
│   │       ├── val.af.txt
│   │       ├── val.en.txt
│   │       ├── val.nl.txt
│   │       ├── val.xh.txt
│   │       └── val.zu.txt
│   ├── 02_intermediate
│   │   └── wiki_5lang
│   ├── 03_primary
│   ├── 04_feature
│   ├── 05_model_input
│   ├── 06_models
│   │   └── wiki_5lang
│   ├── 07_model_output
│   │   └── wiki_5lang
│   │       └── gen_text
│   ├── 08_reporting
│   │   └── wiki_5lang
│   │       └── plots
│   └── logs
├── notebooks
├── src
```

### Setup and Run

Before running the scripts, please ensure the data is available in the `01_raw` directory. Each language should have its own train, test, and validation txt files.

To run the entire project, execute the scripts in the following order:

1. `src/a1_preprocess.py`
2. `src/a1_train_lang_model.py`
3. `src/a1_predict_lang_model.py`
4. `src/a1_bpe.py`

Please ensure that the `a1.conf` configuration file is set up appropriately before running the scripts.

To run the scripts, in terminal:

```
cd src
python a1_preprocess.py
```

### Detailed Description

- `src/a1_preprocess.py`: Preprocesses the raw text files by normalizing and tokenizing the text.
- `src/a1_bpe.py`: Applies Byte Pair Encoding (BPE) to the preprocessed text.
- `src/a1_train_lang_model.py`: Trains the language model using the preprocessed and encoded text.
- `src/a1_predict_lang_model.py`: Predicts the output text using the trained language model.

You can find the data exploration notebooks and helper scripts in the `notebooks` directory. The `log.py`, and `mynlputils.py` are helper scripts containing utility functions and logging configuration.

### Outputs

The outputs of each step of the project are stored in the corresponding `data` subdirectory:

- `02_intermediate`: Contains the normalized text files.
- `06_models`: Contains the trained language model files.
- `07_model_output`: Contains the generated text files.
- `08_reporting`: Contains the various plots generated during the project, including character frequency, perplexity, word length distribution, and Zipf's law plots.

In case of errors or other events, check the `logs` subdirectory.

## Test <a name="assignment2"></a>