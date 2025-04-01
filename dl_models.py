import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Set random seed for reproducibility
np.random.seed(42)

def preprocess_data(X, y, test_size=0.2):
    """Preprocess data for deep learning models."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_dense_network_1layer(X, y):
    """Simulate a 1-layer dense neural network using scikit-learn's MLPClassifier."""
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Define model architecture - 1 hidden layer with 16 neurons
    model = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_prob, X_test, y_test

def train_dense_network_2layer(X, y):
    """Simulate a 2-layer dense neural network using scikit-learn's MLPClassifier."""
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Define model architecture - 2 hidden layers with 32 and 16 neurons
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_prob, X_test, y_test

def train_dense_network_3layer(X, y):
    """Simulate a 3-layer dense neural network using scikit-learn's MLPClassifier."""
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Define model architecture - 3 hidden layers with 64, 32, and 16 neurons
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_prob, X_test, y_test