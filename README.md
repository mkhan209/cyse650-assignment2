# Assignment 2, Stage 1: Sentiment Classification with Neural Language Models

Binary sentiment classifier (0 = negative, 1 = positive) for movie reviews,
trained on a small, imbalanced training set (240 reviews: 180 positive /
60 negative) and evaluated on a balanced public test set (400 reviews:
200/200 positive/negative).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgments)

## Installation

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place `train.csv` and `public_test.csv` in a `data/` folder before running
(not included in this repo — see the assignment's data release).

## Usage

```bash
jupyter notebook stage1_notebook.ipynb
```

Run all cells top to bottom. The notebook is fully self-contained — no
internet access or external model downloads are required, so it runs the
same locally, in Colab, or offline. It will:

1. Load and inspect the data
2. Train **Model A**: TF-IDF (unigrams+bigrams) + Logistic Regression,
   with the regularization strength selected via 5-fold cross-validation
3. Train **Model B**: TF-IDF → SVD(100) → a small neural network, trained
   with a class-weighted loss and early stopping
4. Evaluate both on `public_test.csv` (accuracy + confusion matrix) and
   pick the better one
5. Save the winning model to `model_checkpoint/`
6. Write `public_test_predictions.csv`
7. Reload the saved checkpoint and confirm it reproduces the same
   predictions without retraining

### Running in Google Colab

Upload `stage1_notebook.ipynb` to Colab and run all cells — everything
needed (pandas, scikit-learn, torch) is preinstalled in Colab, so no extra
setup is required. If `train.csv`/`public_test.csv` aren't already
present, the data-loading cell opens an upload widget. A Colab-only cell
at the end downloads `model_checkpoint.zip` and
`public_test_predictions.csv`, since Colab storage doesn't persist between
sessions.

## Project Structure

```
stage1_notebook.ipynb       # full pipeline: data loading, training, evaluation, checkpoint saving
README.md                   # this file
requirements.txt            # Python dependencies
model_checkpoint/           # saved model (created by the notebook)
  ├── tfidf_vectorizer.joblib
  ├── logreg_classifier.joblib
  └── config.json
public_test_predictions.csv # id,predicted_label for public_test.csv (created by the notebook)
data/                        # place train.csv and public_test.csv here (not committed)
```

## Results

Two models are trained and compared, as the assignment allows, on the
public test set (400 reviews, 200/200 balanced):

| Model                                   | Accuracy | Macro F1 |
|------------------------------------------|:--------:|:--------:|
| **Model A — TF-IDF + Logistic Regression** (selected) | **0.715** | 0.70 |
| Model B — TF-IDF + SVD + Neural network  | 0.7025   | 0.68     |

Model A confusion matrix (rows = true, columns = predicted):

|              | pred negative | pred positive |
|--------------|:--------------:|:--------------:|
| **true negative** | 106 | 94 |
| **true positive** | 20  | 180 |

The model favors recall on the positive class (90%) over the negative
class (53%) — expected given the 3:1 imbalance in training data, even
after class weighting. See `stage1_notebook.ipynb` Section 7 for the full
classification report and both models' confusion matrices.


## Configuration

Model choices and hyperparameters, also recorded in
`model_checkpoint/config.json`:

- **Model A (selected):** TF-IDF, `max_features=6000`, `ngram_range=(1,2)`,
  `min_df=2`, `sublinear_tf=True` → Logistic Regression,
  `class_weight="balanced"`, `C=0.0005` (selected via 5-fold stratified
  cross-validation, scored on macro-F1), `random_state=42`
- **Model B (comparison):** same TF-IDF features → TruncatedSVD
  (`n_components=100`) → `Linear(100,32)-ReLU-Dropout(0.4)-Linear(32,1)`,
  Adam optimizer (`lr=1e-3`, `weight_decay=1e-4`), batch size 16,
  class-weighted `BCEWithLogitsLoss`, early stopping (patience 15 epochs)
- **Decision threshold:** 0.5 for both models
- **Seed:** 42 throughout, for reproducibility

## Acknowledgments

Anthropic's Claude was used to help design this pipeline, implement and
debug both models, analyse possible hyperparameters and select hyperparameters 
via cross-validation, run the notebook to create a version of results and write 
this documentation.
