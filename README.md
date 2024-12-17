# Gender Bias Detection in Movie Scripts using Machine Learning and NLP

## Project Overview
This project analyzes **gender bias** in movie scripts using advanced **Natural Language Processing (NLP)** and machine learning techniques. Inspired by the research paper *"Gender Stereotypes in Hollywood Movies and Their Evolution over Time: Insights from Network Analysis"* by Kumar Arjun et al. (2022), the analysis explores the distribution of dialogues, word usage, and character attributes across genders.

The dataset is sourced from [Convokit](https://convokit.cornell.edu), containing **9035 speakers**, **304,713 utterances**, and **83,097 conversations**.

---

## Key Features
1. **Preprocessing Pipeline**:
   - Text cleaning, tokenization, stemming, and lemmatization.
   - Feature engineering: line length, word count, and credit position.

2. **Embeddings & Feature Extraction**:
   - **TF-IDF** and **Count Vectorization** for traditional NLP features.
   - Integration of **BERT** and **RoBERTa embeddings** for semantic understanding.

3. **Models Implemented**:
   - **Naive Bayes (Multinomial and Bernoulli)**
   - **Logistic Regression** (with hyperparameter tuning)
   - **Random Forest** (basic and fine-tuned)
   - Random Forest and Logistic Regression using **BERT** and **RoBERTa** embeddings.

4. **Performance Comparison**:
   - Accuracy scores range from **54%** to **65%** across models.
   - Fine-tuning techniques and cross-validation used to address overfitting.

5. **Visualization & Insights**:
   - Gender-based line length, word count distributions, and role prominence.
   - Confusion matrices and running time comparisons across models.

---

## Results Summary
| Model                        | Accuracy |
|-----------------------------|----------|
| Multinomial Naive Bayes      | 54%      |
| Bernoulli Naive Bayes        | 63%      |
| Logistic Regression          | 60%      |
| Random Forest (Untuned)      | 92%      *(Overfitting)*|
| Random Forest (Fine-Tuned)   | 65%      |
| Random Forest + BERT         | 65%      |
| Logistic Regression + BERT   | 60%      |
| Random Forest + RoBERTa      | 65%      |
| Logistic Regression + RoBERTa| 60%      |

> *Note*: Overfitting was evident in initial Random Forest implementations, but mitigated with hyperparameter tuning.

---

## Installation
### Dependencies
Make sure you have the following libraries installed:
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn transformers torch torchvision torchaudio wordcloud convokit
```

### Clone the Repository
```bash
git clone https://github.com/your-username/gender-bias-detection
cd gender-bias-detection
```

### Download Dataset
The dataset can be directly accessed using Convokit:
```python
from convokit import Corpus, download
corpus = Corpus(filename=download("movie-corpus"))
```

---

## Usage
### Preprocessing and Feature Engineering
Run the preprocessing script to clean the data and extract features:
```bash
python preprocess.py
```

### Train Models
Train and evaluate all models:
```bash
python train_models.py
```

### Visualize Results
Generate visualizations and performance metrics:
```bash
python visualize_results.py
```

---

## Visualizations
- **Line Length Distribution by Gender**:
![Line Length](images/line_length.png)

- **Confusion Matrix for BERT + Random Forest**:
![Confusion Matrix](images/confusion_matrix.png)

- **Cross-Validation Scores**:
![CV Scores](images/cv_scores.png)

---

## Acknowledgments
- Dataset: [Convokit](https://convokit.cornell.edu)
- Inspiration: *Kumar Arjun et al. (2022)*

---

## Future Improvements
- Implement deep learning models (e.g., LSTMs, Transformers).
- Explore additional features such as character sentiment and emotion.
- Fine-tune pre-trained embeddings for improved accuracy.

---

## Contact
For any questions or collaborations, feel free to reach out:
- **LinkedIn**: [Your LinkedIn](https://www.linkedin.com/in/francescasalute/e)

---

*Happy Analyzing!* 🌟
