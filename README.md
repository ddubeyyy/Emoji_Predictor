# Emoji-Predictor Model

## Project Overview
This project focuses on building an AI-powered emoji predictor that can predict the most suitable emoji based on a given text input. The process involves data loading, preprocessing, visualization, model building, evaluation, and prediction.

## Approach

### 1.1 Loading the Dataset
The datasets used include:
- **Mapping.csv**: A mapping of emojis.
- **Train.csv**: The training dataset.
- **Test.csv**: The test dataset.
![image](https://github.com/user-attachments/assets/e8a78397-25da-432f-b4de-7f88cf3bbaf0)
![image](https://github.com/user-attachments/assets/68169edb-7422-4672-b3ce-fa5bae09310f)
![image](https://github.com/user-attachments/assets/f2da5fd4-5e7a-4d4d-bb28-1c6b2138b41c)

These datasets were loaded using the pandas library.

### 1.2 Data Preprocessing
Text data was cleaned using the `re` library for regular expressions. Steps include:
- Removing URLs, user references, hashtags, special characters, numbers, and extra spaces.
![image](https://github.com/user-attachments/assets/50364f96-cafe-43ac-b0ba-5ceefcfa9e98)

This step ensures that the input text is clean and consistent for the model.


### 1.3 Data Visualization
Visualization was done using `seaborn` and `matplotlib` to analyze the distribution of emojis in the training set. This helps in understanding the frequency of each emoji and identifying any imbalances in the dataset.
![image](https://github.com/user-attachments/assets/f31131fd-f36e-4641-a6b5-ab008102395b)


### 1.4 Tokenization and Padding
Text data was tokenized and padded using `tensorflow.keras.preprocessing.text.Tokenizer` and `tensorflow.keras.preprocessing.sequence.pad_sequences`. This converts the text into a numerical format suitable for the neural network.

### 1.5 Model Building and Training
A neural network model was built using `tensorflow.keras.models.Sequential` with:
- An **Embedding** layer
- **LSTM** layers
- **Dense** layers for classification
![image](https://github.com/user-attachments/assets/3c4a994d-ce96-4388-94ca-212e3f7961a0)


The model was trained on the processed text data.

### 1.6 Model Evaluation with Additional Metrics
The model's performance was evaluated using metrics such as:
- Precision
- Recall
- F1-score
- Confusion matrices
![image](https://github.com/user-attachments/assets/37fa2340-7ab2-4b7a-9ebf-bd0b8606d1fd)

These metrics were calculated using `sklearn.metrics`.

### 1.7 Predictions to Test Data
The trained model was used to predict emojis for the test dataset. The predictions were added to the test dataframe for further analysis.
![image](https://github.com/user-attachments/assets/295cb765-9f93-4081-ad5e-e58401c3731f)

## Challenges Faced
- **Imbalanced Dataset**: Certain emojis were more frequent than others, leading to a biased model.
- **Text Cleaning and Preprocessing**: Ensuring noise was removed without losing meaningful content.
- **Model Overfitting**: Preventing overfitting using dropout layers and regularization techniques.
- **Hyperparameter Tuning**: Extensive experimentation to fine-tune hyperparameters.

## Potential Improvements
- **Advanced Text Embeddings**: Using embeddings like BERT or GPT for better text understanding.
- **Data Augmentation**: Balancing the dataset using data augmentation techniques.
- **Ensemble Models**: Combining multiple models to enhance performance.
- **Hyperparameter Optimization**: Automating hyperparameter optimization using Grid Search or Bayesian Optimization.
- **Additional Features**: Incorporating features like sentiment analysis or part-of-speech tags for better context.

## Requirements
- Python 3.x
- Pandas
- TensorFlow
- Keras
- Numpy
- Seaborn
- Matplotlib
- Scikit-learn

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/ddubeyyy/Emoji_Predictor.git
