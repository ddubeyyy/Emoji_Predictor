# Emoji Predictor

## Overview
The Emoji Predictor is a machine learning project designed to predict the most likely emoji based on the text of social media posts. This project leverages natural language processing (NLP) and deep learning techniques to analyze text and suggest appropriate emojis.

## Installation
To get started with the project, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/ddubeyyy/Emoji_Predictor.git
    cd Emoji_Predictor
    ```

2. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage
### Data Preprocessing
Ensure your data is in the correct format. The project expects a CSV file with columns for the text and the corresponding emojis.

### Model Training
To train the model, run the following command:
```sh
python train.py
```
This will preprocess the data, train the model, and save the trained model to disk.

### Making Predictions
To make predictions using the trained model, run:
```sh
python predict.py --input "Your text here"
```

## Data Visualization
The project includes functions to visualize the distribution of emojis and the training history.
```

## Evaluation
Evaluate the model performance using precision, recall, F1-score, and a confusion matrix.

### Confusion Matrix
Plot the confusion matrix using:
```python
plot_confusion_matrix(y_test, predicted_labels, classes)
```

### Classification Report
Generate a classification report:
```python
report = classification_report(y_test, predicted_labels, target_names=classes)
print('Classification Report')
print(report)
```

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


