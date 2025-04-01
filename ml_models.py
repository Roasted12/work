import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Common preprocessing function
def preprocess_data(X, y, test_size=0.2, random_state=42):
    """Preprocess data by splitting and scaling features."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_logistic_regression(X, y):
    """Train a logistic regression model and return predictions."""
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Create and train the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_prob, X_test, y_test

def train_decision_tree(X, y):
    """Train a decision tree classifier and return predictions."""
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Create and train the model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_prob, X_test, y_test

def train_random_forest(X, y):
    """Train a random forest classifier and return predictions."""
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_prob, X_test, y_test

def train_svm(X, y):
    """Train a support vector machine classifier and return predictions."""
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Create and train the model
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_prob, X_test, y_test

def train_knn(X, y):
    """Train a k-nearest neighbors classifier and return predictions."""
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Create and train the model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_prob, X_test, y_test
