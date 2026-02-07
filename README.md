# Anthropic NLP Roadmap
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Meshojs/Anthropic-NLP-Roadmap)

This repository documents a learning journey through a structured roadmap covering foundational Machine Learning and Natural Language Processing concepts. It features a series of hands-on projects, from building neural networks from scratch with NumPy to leveraging PyTorch for image classification and exploring core NLP techniques.

## Repository Structure

The curriculum is divided into blocks and weeks, with each folder containing Jupyter notebooks, datasets, and project deliverables.

-   **`Block1-M/`**: Focuses on Machine Learning fundamentals.
    -   **`weekone/`**: Introduces Python basics, NumPy operations, and the from-scratch implementation of neural networks for regression tasks.
    -   **`weektwo/`**: Covers classification, demonstrating a binary classifier built with NumPy and a multi-class image classifier with PyTorch.
-   **`Block2-M/`**: Begins the NLP section of the roadmap.
    -   **`main1.ipynb`**: Explores fundamental text preprocessing techniques using NLTK and SpaCy.

## Projects and Deliverables

### 1. House Price Prediction
-   **Location**: `Block1-M/weekone/deliverable-1/`
-   **Description**: A neural network built from scratch using only NumPy to predict house prices based on features like square footage, number of bedrooms, and location signals. The implementation includes data loading, feature scaling, a custom training loop, and the use of ReLU activation functions.
-   **Key Skills**: Regression, Data Preprocessing, NumPy, Neural Network Architecture.

### 2. Student Final Grade Prediction
-   **Location**: `Block1-M/weekone/deliverable-2/`
-   **Description**: A regression model implemented in NumPy to predict final student grades (`G3`) based on demographic, social, and school-related features. The project involves significant data cleaning, label encoding for categorical features, and outlier handling.
-   **Key Skills**: Regression, Feature Engineering, Data Cleaning, NumPy.

### 3. Titanic Survival Classification
-   **Location**: `Block1-M/weektwo/deliverable-3/`
-   **Description**: A binary classification model built with NumPy to predict passenger survival on the Titanic. This project addresses class imbalance using SMOTE (Synthetic Minority Over-sampling TEchnique) and uses a sigmoid activation function for the final output layer.
-   **Key Skills**: Binary Classification, Class Imbalance (SMOTE), Feature Scaling, NumPy.

### 4. MNIST Digit Recognition
-   **Location**: `Block1-M/weektwo/deliverable-4/`
-   **Description**: A multi-class image classifier built with PyTorch to recognize handwritten digits from the MNIST dataset. The model utilizes linear layers, ReLU activations, and the `CrossEntropyLoss` criterion for training.
-   **Key Skills**: Multi-class Classification, PyTorch, `nn.Module`, `DataLoader`, Image Classification.

### 5. NLP Foundations
-   **Location**: `Block2-M/main1.ipynb`
-   **Description**: An exploratory notebook covering the building blocks of Natural Language Processing. It demonstrates techniques such as tokenization, stop-word removal, stemming, lemmatization, Part-of-Speech (POS) tagging, and the use of regular expressions for text pattern matching.
-   **Key Skills**: Tokenization, Lemmatization, Stemming, Regex, NLTK, SpaCy.

## Technologies Used
-   Python
-   NumPy
-   Pandas
-   PyTorch
-   Scikit-learn
-   NLTK
-   SpaCy
-   idx2numpy
