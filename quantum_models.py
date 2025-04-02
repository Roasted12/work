import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pickle
import os
import math
import random
from sklearn.ensemble import RandomForestClassifier

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)

# Define constants for the quantum models
NUM_QUBITS = 2  # Number of qubits to use
SHOTS = 1024  # Number of shots for quantum simulation

def preprocess_data(X, y, test_size=0.2, feature_dim=NUM_QUBITS):
    """
    Preprocess data for quantum models.
    For quantum models, we need to reduce dimensionality and scale features appropriately.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # For quantum models, we'll use PCA to reduce dimensionality to match qubit count
    pca = PCA(n_components=feature_dim)
    X_train_reduced = pca.fit_transform(X_train_scaled)
    X_test_reduced = pca.transform(X_test_scaled)
    
    # Scale to [0, 2π] range to be suitable for quantum feature maps
    feature_scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
    X_train_final = feature_scaler.fit_transform(X_train_reduced)
    X_test_final = feature_scaler.transform(X_test_reduced)
    
    # Save the preprocessing pipeline for later use
    preprocessing = {
        'scaler': scaler,
        'pca': pca,
        'feature_scaler': feature_scaler
    }
    
    # Ensure directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save preprocessing objects
    with open('models/quantum_preprocessing.pkl', 'wb') as f:
        pickle.dump(preprocessing, f)
    
    return X_train_final, X_test_final, y_train, y_test, X_test, preprocessing

# Quantum-inspired functions for simulating quantum behavior

def quantum_state_vector(features, num_qubits=NUM_QUBITS):
    """
    Create a quantum state vector from feature data.
    This simulates the quantum state vector that would be created 
    by a quantum circuit with these features.
    """
    # Number of basis states in the system
    num_states = 2**num_qubits
    state_vector = np.zeros(num_states, dtype=complex)
    
    # Initialize to |0> state
    state_vector[0] = 1.0
    
    # Apply Hadamard to all qubits (create superposition)
    hadamard_factor = 1.0 / math.sqrt(2**num_qubits)
    state_vector = np.ones(num_states, dtype=complex) * hadamard_factor
    
    # Apply feature-based phase rotations and amplitudes
    for i in range(num_states):
        # Convert index to binary representation
        bin_idx = format(i, f'0{num_qubits}b')
        
        # Calculate phase based on features and bit values
        phase = 0.0
        for q in range(min(len(features), num_qubits)):
            if bin_idx[q] == '1':
                phase += features[q]
        
        # Apply the phase rotation
        state_vector[i] *= np.exp(1j * phase)
    
    # Normalize the state vector
    norm = np.linalg.norm(state_vector)
    if norm > 0:
        state_vector = state_vector / norm
    
    return state_vector

def measure_quantum_state(state_vector, shots=SHOTS):
    """
    Simulate measurement of a quantum state vector.
    Returns counts of measurement outcomes.
    """
    num_states = len(state_vector)
    num_qubits = int(np.log2(num_states))
    
    # Calculate probabilities from amplitudes
    probabilities = np.abs(state_vector) ** 2
    
    # Sample from probability distribution
    outcomes = np.random.choice(num_states, size=shots, p=probabilities)
    
    # Count occurrences of each outcome
    counts = {}
    for outcome in outcomes:
        # Convert to binary string
        bitstring = format(outcome, f'0{num_qubits}b')
        if bitstring in counts:
            counts[bitstring] += 1
        else:
            counts[bitstring] = 1
    
    return counts

def binary_string_parity(bitstring):
    """Calculate the parity of a binary string (even/odd number of 1's)"""
    return bitstring.count('1') % 2

def classify_measurement(counts, shots=SHOTS):
    """Classify based on measurement results."""
    # Count occurrences of bitstrings with even parity (class 0) 
    # and odd parity (class 1)
    even_parity_count = 0
    for bitstring, count in counts.items():
        # Count the number of '1's in the bitstring
        if binary_string_parity(bitstring) == 0:
            even_parity_count += count
    
    # Probability of class 0 (even parity)
    prob_class0 = even_parity_count / shots
    # Probability of class 1 (odd parity)
    prob_class1 = 1 - prob_class0
    
    # Make prediction
    predicted_class = 0 if prob_class0 > 0.5 else 1
    
    return predicted_class, np.array([prob_class0, prob_class1])

def apply_quantum_interference(state_vector, theta):
    """
    Apply quantum interference effects to the state vector.
    This simulates the effect of variational rotation gates and entanglement.
    """
    num_states = len(state_vector)
    num_qubits = int(np.log2(num_states))
    
    # Apply phase rotations based on theta parameters
    for i in range(num_states):
        # Convert index to binary representation
        bin_idx = format(i, f'0{num_qubits}b')
        
        # Apply phase based on theta parameters
        phase = 0.0
        for q in range(num_qubits):
            phase_idx = q % len(theta)
            if bin_idx[q] == '1':
                phase += theta[phase_idx]
        
        # Apply phase rotation
        state_vector[i] *= np.exp(1j * phase)
    
    # Simulate entanglement by creating interference between basis states
    # with similar bit patterns (this is a simplified model)
    new_state = np.zeros_like(state_vector)
    for i in range(num_states):
        bin_i = format(i, f'0{num_qubits}b')
        for j in range(num_states):
            bin_j = format(j, f'0{num_qubits}b')
            # Calculate Hamming distance (number of differing bits)
            hamming_dist = sum(b1 != b2 for b1, b2 in zip(bin_i, bin_j))
            # States with Hamming distance 1 interfere most strongly
            if hamming_dist == 1:
                new_state[i] += 0.5 * state_vector[j]
            elif hamming_dist == 0:
                new_state[i] += state_vector[j]
    
    # Normalize the new state
    norm = np.linalg.norm(new_state)
    if norm > 0:
        new_state = new_state / norm
    
    return new_state

def train_variational_classifier(X, y):
    """
    Simulate a Variational Quantum Classifier (VQC).
    Uses quantum-inspired techniques to simulate the behavior of a VQC.
    """
    print("Using quantum-inspired Variational Quantum Classifier...")
    
    # Preprocess data - reduce dimensions for quantum simulation
    X_train, X_test, y_train, y_test, X_test_original, preprocessing = preprocess_data(
        X, y, feature_dim=NUM_QUBITS
    )
    
    # Create fixed variational parameters
    # In a full implementation, these would be optimized via a quantum-classical optimizer
    theta = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) * np.pi
    
    # Predict with the simulated quantum process
    y_pred = []
    y_prob = []
    
    for x in X_test:
        # Create initial quantum state from features
        state_vector = quantum_state_vector(x, NUM_QUBITS)
        
        # Apply variational circuit operations
        state_vector = apply_quantum_interference(state_vector, theta)
        
        # Simulate measurement
        counts = measure_quantum_state(state_vector, SHOTS)
        
        # Classify the result
        pred_class, probs = classify_measurement(counts, SHOTS)
        
        y_pred.append(pred_class)
        y_prob.append(probs[1])  # Probability of class 1
    
    # For explainability, print a sample of the state vector and measurements
    features = X_test[0]
    state = quantum_state_vector(features, NUM_QUBITS)
    counts = measure_quantum_state(state, 10)
    print(f"Sample quantum state for features {features}: First 4 amplitudes: {state[:4]}")
    print(f"Sample measurements: {counts}")
    
    return None, np.array(y_pred), np.array(y_prob), X_test_original, y_test

def train_quantum_neural_network(X, y):
    """
    Simulate a Quantum Neural Network (QNN).
    Uses a more complex quantum-inspired approach with multiple layers.
    """
    print("Using quantum-inspired Quantum Neural Network...")
    
    # Preprocess data with possibly more feature dimensions
    X_train, X_test, y_train, y_test, X_test_original, preprocessing = preprocess_data(
        X, y, feature_dim=NUM_QUBITS+1  # One extra qubit for QNN
    )
    
    # Multiple sets of parameters for different layers
    theta1 = np.array([0.2, 0.4, 0.6, 0.8, 1.0]) * np.pi
    theta2 = np.array([0.3, 0.6, 0.9, 1.2, 1.5]) * np.pi
    theta3 = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) * np.pi
    
    # Predictions
    y_pred = []
    y_prob = []
    
    for x in X_test:
        # Create initial quantum state from features
        state_vector = quantum_state_vector(x, NUM_QUBITS+1)
        
        # Apply multiple layers of quantum operations
        # Layer 1
        state_vector = apply_quantum_interference(state_vector, theta1)
        # Layer 2
        state_vector = apply_quantum_interference(state_vector, theta2)
        # Layer 3
        state_vector = apply_quantum_interference(state_vector, theta3)
        
        # Simulate measurement
        counts = measure_quantum_state(state_vector, SHOTS)
        
        # Classify with extra complexity for QNN
        # We could use a more sophisticated approach here, but for simplicity,
        # we'll use the same classification approach as before
        pred_class, probs = classify_measurement(counts, SHOTS)
        
        y_pred.append(pred_class)
        y_prob.append(probs[1])  # Probability of class 1
    
    # For explainability, print a sample of the state vector and measurements
    features = X_test[0]
    initial_state = quantum_state_vector(features, NUM_QUBITS+1)
    final_state = apply_quantum_interference(
        apply_quantum_interference(
            apply_quantum_interference(initial_state, theta1),
            theta2),
        theta3)
    counts = measure_quantum_state(final_state, 10)
    
    print(f"QNN: Sample initial state for features {features}: First 4 amplitudes: {initial_state[:4]}")
    print(f"QNN: Sample final state: First 4 amplitudes: {final_state[:4]}")
    print(f"QNN: Sample measurements: {counts}")
    
    return None, np.array(y_pred), np.array(y_prob), X_test_original, y_test