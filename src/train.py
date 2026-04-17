import tensorflow as tf
from tensorflow import keras
import numpy as np
from src.model import (
    create_standard_model,
    create_model_with_dropout,
    create_model_with_early_stopping,
    create_model_with_both
)

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, 
                callbacks=None, model_name="model"):
    """
    Train the model and return history
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data
    """
    results = model.evaluate(X_test, y_test, verbose=0)
    metrics = dict(zip(model.metrics_names, results))
    
    return metrics

def train_all_models(X_train, X_val, X_test, y_train, y_val, y_test, epochs=100):
    """
    Train all model variants and return results
    """
    input_shape = X_train.shape[1]
    models_results = {}
    
    # Model 1: Standard Model
    print("\n" + "="*50)
    print("Training Standard Model")
    print("="*50)
    model_standard = create_standard_model(input_shape)
    history_standard, model_standard = train_model(
        model_standard, X_train, y_train, X_val, y_val, 
        epochs=epochs, model_name="standard"
    )
    results_standard = evaluate_model(model_standard, X_test, y_test)
    models_results['standard'] = {
        'model': model_standard,
        'history': history_standard,
        'metrics': results_standard
    }
    
    # Model 2: Model with Dropout
    print("\n" + "="*50)
    print("Training Model with Dropout")
    print("="*50)
    model_dropout = create_model_with_dropout(input_shape)
    history_dropout, model_dropout = train_model(
        model_dropout, X_train, y_train, X_val, y_val,
        epochs=epochs, model_name="dropout"
    )
    results_dropout = evaluate_model(model_dropout, X_test, y_test)
    models_results['dropout'] = {
        'model': model_dropout,
        'history': history_dropout,
        'metrics': results_dropout
    }
    
    # Model 3: Model with Early Stopping
    print("\n" + "="*50)
    print("Training Model with Early Stopping")
    print("="*50)
    model_es, early_stopping = create_model_with_early_stopping(input_shape)
    history_es, model_es = train_model(
        model_es, X_train, y_train, X_val, y_val,
        epochs=epochs, callbacks=[early_stopping], model_name="early_stopping"
    )
    results_es = evaluate_model(model_es, X_test, y_test)
    models_results['early_stopping'] = {
        'model': model_es,
        'history': history_es,
        'metrics': results_es
    }
    
    # Model 4: Model with Both
    print("\n" + "="*50)
    print("Training Model with Both Dropout and Early Stopping")
    print("="*50)
    model_both, early_stopping_both = create_model_with_both(input_shape)
    history_both, model_both = train_model(
        model_both, X_train, y_train, X_val, y_val,
        epochs=epochs, callbacks=[early_stopping_both], model_name="both"
    )
    results_both = evaluate_model(model_both, X_test, y_test)
    models_results['both'] = {
        'model': model_both,
        'history': history_both,
        'metrics': results_both
    }
    
    return models_results
