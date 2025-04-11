# 🐦 Twitter Sentiment Analysis Using CNN

This documentation provides a detailed explanation of the Twitter Sentiment Analysis project implemented using a Convolutional Neural Network (CNN) in Python. The goal of this system is to classify tweets into sentiment categories using deep learning techniques.

## Overview

The project analyzes Twitter data to automatically determine sentiment expressed in tweets. Using a CNN architecture, the model processes text data and classifies it into predefined sentiment categories. The implemented model achieved an accuracy of 94.39%.

## Libraries Used

The following Python libraries were used to implement the model:
- **NumPy:** For numerical operations
- **Pandas:** For data manipulation and analysis
- **Matplotlib & Seaborn:** For data visualization
- **TensorFlow/Keras:** For building and training the CNN model
- **Scikit-learn:** For data preprocessing and evaluation metrics
- **Re:** For regular expression operations in text cleaning
- **SpaCy:** For natural language processing functionality

## Project Structure

**1. Data Loading and Preprocessing:**
- Twitter datasets (training and validation) are loaded from CSV files
- Unnecessary columns are removed
- Text data is cleaned using regular expressions to remove URLs, user references, special characters, and numbers
- Empty or duplicate entries are removed
- Text is converted to lowercase for consistency

**2. Text Processing:**
- Tweets are tokenized, converted to sequences, and padded to ensure uniform length
- Labels are encoded using LabelEncoder

**3. CNN Model Architecture:**
- A Sequential model is created with:
  - Embedding layer for text representation
  - Two Conv1D layers with ReLU activation
  - BatchNormalization and Dropout layers for regularization
  - GlobalMaxPooling1D layer to reduce dimensionality
  - Dense layers with ReLU and Softmax activations

**4. Model Training:**
- The model is trained for 20 epochs with the Adam optimizer
- Sparse Categorical Crossentropy is used as the loss function
- Training progress is monitored with accuracy metrics

**5. Model Evaluation:**
- Classification report showing precision, recall, and F1-score
- Confusion matrix visualization
- Training and validation accuracy/loss plots

**6. Prediction Function:**
- A custom function is implemented to make predictions on new text inputs
- Text is cleaned, tokenized, and processed before prediction

## Evaluation Results

- **Test Accuracy:** 94.39%
- The model demonstrates strong performance in classifying tweet sentiments
- The accuracy curves show good convergence without significant overfitting
- The confusion matrix reveals the distribution of correct predictions across sentiment classes

## Conclusion

The Twitter sentiment analysis system built with CNN demonstrates high accuracy in classifying tweets into sentiment categories. This model can be used for real-time sentiment monitoring on Twitter data, providing valuable insights for brand management, market research, and customer feedback analysis.
