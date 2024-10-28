# Fake News Detection Using Logistic Regression

This project demonstrates a machine learning pipeline for detecting fake news using [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression). The dataset used in this project is a CSV file containing news articles with labels indicating whether they are real or fake. The pipeline includes data preprocessing, feature extraction, model training, and evaluation.

## Prerequisites

- [Python 3](https://www.python.org/download/releases/3.0/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)

## Code Overview

1. **Import Libraries**: The code begins by importing necessary libraries such as `numpy`, `pandas`, `nltk`, and `scikit-learn`.

2. **Load Dataset**: The dataset is loaded using `pandas.read_csv()` from a file named `train.csv`.

3. **Data Preprocessing**:
   - Missing values are filled with empty strings.
   - A new column `content` is created by concatenating the `author` and `title` columns.
   - The `stemming` function is defined to clean and stem the text data. It removes non-alphabetic characters, converts text to lowercase, removes stopwords, and applies stemming using `PorterStemmer`.

4. **Feature Extraction**:
   - The `TfidfVectorizer` is used to convert the text data into numerical features suitable for machine learning.

5. **Train-Test Split**:
   - The dataset is split into training and testing sets using `train_test_split` with a test size of 20% and stratification based on the label.

6. **Model Training**:
   - A `LogisticRegression` model is instantiated and trained on the training data.

7. **Model Evaluation**:
   - The accuracy of the model is printed for both the training and testing datasets.

8. **Prediction**:
   - A sample from the test set is used to demonstrate prediction. The model predicts whether the news is real or fake, and the result is printed.

## Results

The model achieves an accuracy of approximately 98.72% on the training data and 97.52% on the testing data, indicating a high level of performance in distinguishing between real and fake news.

## Usage

To use this code, ensure your dataset is formatted similarly to the `train.csv` file used here, with columns for `author`, `title`, `text`, and `label`. Adjust the file path as necessary and run the script to train and evaluate the model on your data.

## Conclusion

This project provides a framework for fake news detection using logistic regression. It can be further improved by experimenting with different models, feature extraction techniques, and hyperparameter tuning.
