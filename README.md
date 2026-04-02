# Text-Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![GitHub repo size](https://img.shields.io/github/repo-size/MAYANK479/Text-Classification)](https://github.com/MAYANK479/Text-Classification)
[![Build Status](https://github.com/MAYANK479/Text-Classification/actions/workflows/main.yml/badge.svg)](https://github.com/MAYANK479/Text-Classification/actions)
[![Downloads](https://img.shields.io/github/downloads/MAYANK479/Text-Classification/total.svg)](https://github.com/MAYANK479/Text-Classification/releases)

> A complete reproducible implementation of sentiment analysis from the `NLP_Mayank_Final rp.pdf` lab report.
> Includes preprocessing, TF-IDF + Naive Bayes pipeline and comparative classifiers (SVM, Logistic, NBSVM).

Reproduction of NLP sentiment analysis project from lab report (`NLP_Mayank_Final rp.pdf`) using IMDB + Sentiment140 datasets.

## Author

- Mayank Pandey

## Project structure

- `nlp_nb_sentiment.py` - main training/evaluation pipeline
- `generate_doc.py` - auto-generate PDF summary documentation
- `NLP_Mayank_Final rp.pdf` - original report source
- `NLP_Mayank_Final_repro_documentation.pdf` - generated report summary
- `aclImdb/` - IMDB dataset folders (local copy)
- `trainingandtestdata/` - Sentiment140 dataset CSV (local copy)
- `*.png` - evaluation plots (confusion matrix, loss curve, model comparison)

## Requirements

```bash
pip install scikit-learn pandas numpy matplotlib seaborn nltk reportlab
```

## Data download

1. IMDB dataset:
   - http://ai.stanford.edu/~amaas/data/sentiment/ (extract into `aclImdb`)
2. Sentiment140 dataset:
   - http://help.sentiment140.com/for-students (extract into `trainingandtestdata`)

## Run

```bash
python3 nlp_nb_sentiment.py
```

## Output

- `IMDB_(sample_25k_positive_+_25k_negative)_confusion_matrix.png`
- `IMDB_(sample_25k_positive_+_25k_negative)_loss_curve.png`
- `IMDB_(sample_25k_positive_+_25k_negative)_model_comparison.png`
- `Sentiment140_(40k_sample)_confusion_matrix.png`
- `Sentiment140_(40k_sample)_loss_curve.png`
- `Sentiment140_(40k_sample)_model_comparison.png`

## Performance (example)

- IMDB Accuracy: 86.7%
- Sentiment140 Accuracy: 75.26%

## License

MIT License

```
MIT License

Copyright (c) 2026 Mayank Pandey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
