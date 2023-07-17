# mlai-natural-language-processing

## Table of Contents

1. [A1: Language modelling and byte-pair Encoding](#assignment1)
2. [A2: IsiZulu Part-of-Speech Tagger using Hidden Markov Models](#assignment2)

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

## IsiZulu Part-of-Speech Tagger using Hidden Markov Models <a name="assignment2"></a>

This project consists of three python scripts, `a2_preprocess.py`, `a2_train_hmm.py`, and `a2_predict.py`, which are responsible for pre-processing the data, training a Hidden Markov Model (HMM) for isiZulu Part-of-Speech (POS) tagging, and evaluating the model respectively.

### Pre-Requisites

The code relies on the following Python libraries:

- numpy
- pandas
- scikit-learn
- plotly
- logging

Moreover, it uses a custom-made `mynlputils` module for utilities and logging.


### Data
The data for this project is obtained from the South African Centre for Digital Language Resources (SADiLaR). You can download the necessary data as follows:

1. Visit the [SADiLaR](https://repo.sadilar.org) Resource Catalogue: 

2. Search for "NCHLT isiZulu Annotated Text Corpora" and download the data.

3. Extract the downloaded zip file. You will find Excel files for training and test data in the directory 2.POS Annotated/.

Ensure that the paths to these files are correctly set in the `a2.conf` configuration file used by the scripts.

### Code Overview

- a2_preprocess.py: The script is responsible for data preprocessing tasks, such as loading the data, removing punctuation, splitting the data into sentences, and creating a validation set.
- a2_train_hmm.py: This script trains an HMM model for isiZulu POS tagging. The HMM model's transition, emission, and start probabilities are computed with Laplace smoothing.
- a2_predict.py: The script uses the trained HMM model to predict the POS tags for each token in a sentence using the Viterbi algorithm. It also includes a cross-validation function to select the best Laplace smoothing value.

To run the scripts, in terminal:

```
cd src
python a2_preprocess.py
```

### Directory Structure

The project has the following directory structure:

```
├── README.md
├── conf
│   └── a2.conf
├── data
│   ├── 01_raw
│   │   ├── pos_tag_hmm
│   │   │   └── zu
│   │   │       ├── 1.Lemmatized
│   │   │       │   ├── GOV-ZA.50000ParallelleEnWoorde.zu.lemma.full.lara2
│   │   │       │   ├── GOV-ZA.50000ParallelleEnWoorde.zu.lemma.full.xls
│   │   │       │   ├── GOV-ZA.Toetsteks.5000ParallelleEnWoorde.zu.lemma.full.lara2
│   │   │       │   ├── GOV-ZA.Toetsteks.5000ParallelleEnWoorde.zu.lemma.full.xls
│   │   │       │   ├── Protocol.NCHLT.LemmatizationIsiZulu.1.0.1.NLO.2013-03-31.doc
│   │   │       │   └── Protocol.NCHLT.TokenisationIsiZulu.1.0.1.MJP.2013-03-31.doc
│   │   │       ├── 2.POS Annotated
│   │   │       │   ├── GOV-ZA.50000ParallelleEnWoorde.zu.pos.full.lara2
│   │   │       │   ├── GOV-ZA.50000ParallelleEnWoorde.zu.pos.full.xls
│   │   │       │   ├── GOV-ZA.Toetsteks.5000ParallelleEnWoorde.zu.pos.full.lara2
│   │   │       │   ├── GOV-ZA.Toetsteks.5000ParallelleEnWoorde.zu.pos.full.xls
│   │   │       │   ├── Protocol.NCHLT.PartOfSpeechTaggingIsiZulu.1.0.1.NLO.2013-03-31.doc
│   │   │       │   └── Protocol.NCHLT.TokenisationIsiZulu.1.0.1.MJP.2013-03-31.doc
│   │   │       ├── 3.Morphologically Analyzed
│   │   │       │   ├── GOV-ZA.50000ParallelleEnWoorde.zu.morph.full.lara2
│   │   │       │   ├── GOV-ZA.50000ParallelleEnWoorde.zu.morph.full.xls
│   │   │       │   ├── GOV-ZA.Toetsteks.5000ParallelleEnWoorde.zu.morph.full.lara2
│   │   │       │   ├── GOV-ZA.Toetsteks.5000ParallelleEnWoorde.zu.morph.full.xls
│   │   │       │   ├── Protocol.NCHLT.MorphologicalAnalysisIsiZulu.1.0.1.NLO.2013-03-31.doc
│   │   │       │   └── Protocol.NCHLT.TokenisationIsiZulu.1.0.1.MJP.2013-03-31.doc
│   │   │       └── License.NCHLT.AnnotatedCorpora.txt
│   ├── 02_intermediate
│   ├── 03_primary
│   ├── 04_feature
│   ├── 05_model_input
│   │   └── pos_tag_hmm
│   │       ├── sentences_test.pkl
│   │       ├── sentences_train.pkl
│   │       └── sentences_valid.pkl
│   ├── 06_models
│   │   ├── pos_tag_hmm
│   │   │   └── hmm_model.pkl
│   ├── 07_model_output
│   │   ├── pos_tag_hmm
│   │   │   ├── test_results.csv
│   │   │   └── valid_results.csv
│   ├── 08_reporting
│   │   ├── pos_tag_hmm
│   └── logs
├── notebooks
│   ├── A2-explore-data-001.ipynb
│   ├── A2-explore-data-002.ipynb
│   ├── A2-explore-data-003.ipynb
│   ├── A2-simple-hmm-001.ipynb
│   ├── A2-simple-hmm-002.ipynb
│   ├── A2-simple-hmm-003.ipynb
│   ├── A2-simple-hmm-004.ipynb
│   ├── __utils.py
│   ├── debug.ipynb
│   ├── log.py
│   └── mynlputils.py
├── src
│   ├── a2_predict.py
│   ├── a2_preprocess.py
│   ├── a2_train_hmm.py
│   ├── log.py
│   └── mynlputils.py
```

### Setup and Run

1. Run `a2_preprocess.py` to load and preprocess the isiZulu text data.
2. Run `a2_train_hmm.py` to train the Hidden Markov Model.
3. Run `a2_predict.py` to evaluate the model's performance on the validation and test sets.

Each of these scripts is intended to be run independently and in the order mentioned above. The preprocessing script saves its output to the disk, which is loaded by the training script. The training script similarly saves the model to disk, which is loaded by the prediction script.

Make sure that the required packages are installed, and that you have configured your paths and other settings properly in the `a2.conf` file. Also, check that the data files exist in the specified paths.

### Outputs

The output will include accuracy of POS tagging on the validation and test sets, as well as predicted POS tags for each sentence in these sets.
Accuracy output will be to terminal.

In case of errors or other events, check the `logs` subdirectory.