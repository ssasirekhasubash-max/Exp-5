# Experiment 5 : Multilayer Perceptron (MLP) Neural Network

## Aim

To implement a Multilayer Perceptron (MLP) neural network for
handwritten digit classification using Python and evaluate its
performance on the test dataset.

> **Note:** The source page contains an inconsistent aim statement
> mentioning K-Means clustering. The experiment title and algorithm
> steps describe an MLP neural network, so this README follows the MLP
> experiment.

## Overview

A **Multilayer Perceptron (MLP)** is a feed-forward artificial neural
network consisting of an input layer, one or more hidden layers, and an
output layer. It learns patterns in data by adjusting weights and biases
during training using backpropagation.

In this experiment, the MLP model is trained on the **Digits dataset**
available in scikit-learn. Each sample is an 8 × 8 image of a
handwritten digit (0--9). The pixel values are used as input features,
and the model predicts the corresponding digit.

## Algorithm

1.  **Start.**
2.  Load the Digits dataset and separate the input features and target
    labels.
3.  Split the dataset into training and testing sets.
4.  Standardize the input features.
5.  Create an MLP classifier with the specified hidden-layer structure.
6.  Train the model using the training data.
7.  Predict the labels for the test data.
8.  Evaluate the model using test accuracy and a classification report.
9.  Plot the training loss curve.
10. **Stop.**
ry slightly with changes to the train-test split,
    random seed, or model parameters.


## Program
    import matplotlib.pyplot as plt
    
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    # Load Digits dataset
    digits = load_digits()
    
    # Input features and target labels
    X = digits.data
    y = digits.target
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create MLP model
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    
    # Train model
    mlp.fit(X_train, y_train)
    
    # Predict test data
    y_pred = mlp.predict(X_test)
    
    # Evaluate model
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot training loss
    plt.plot(mlp.loss_curve_)
    plt.title("MLP Training Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

## output

     Test Accuracy: 0.975
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        36
               1       0.92      0.92      0.92        36
               2       1.00      1.00      1.00        35
               3       1.00      1.00      1.00        37
               4       0.97      0.97      0.97        36
               5       0.97      1.00      0.99        37
               6       1.00      1.00      1.00        36
               7       1.00      1.00      1.00        36
               8       0.94      0.89      0.91        35
               9       0.95      0.97      0.96        36
    
        accuracy                           0.97       360
       macro avg       0.97      0.97      0.97       360
    weighted avg       0.97      0.97      0.97       360



## Result
Thus, the Multilayer Perceptron (MLP) neural network was successfully implemented using Python and the Scikit-learn Digits dataset. The model was trained to recognize handwritten digits from 0 to 9, and its performance was evaluated using test accuracy, classification report and training loss curve.
