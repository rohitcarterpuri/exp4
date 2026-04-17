import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
import numpy as np

def create_standard_model(input_shape):
    """
    Create a standard ANN model without regularization
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), 
                keras.metrics.AUC()]
    )
    
    return model

def create_model_with_dropout(input_shape, dropout_rate=0.3):
    """
    Create an ANN model with dropout layers
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(16, activation='relu'),
        layers.Dropout(dropout_rate/2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(),
                keras.metrics.AUC()]
    )
    
    return model

def create_model_with_early_stopping(input_shape, patience=10):
    """
    Create standard model with early stopping callback
    """
    model = create_standard_model(input_shape)
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    return model, early_stopping

def create_model_with_both(input_shape, dropout_rate=0.3, patience=10):
    """
    Create model with both dropout and early stopping
    """
    model = create_model_with_dropout(input_shape, dropout_rate)
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    return model, early_stopping
