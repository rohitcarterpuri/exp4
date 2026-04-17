import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

def plot_training_history(histories, model_names):
    """
    Plot training history for different models
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['loss', 'accuracy']
    titles = ['Model Loss', 'Model Accuracy']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for name, history in zip(model_names, histories):
            ax.plot(history.history[metric], label=f'{name} Train', linestyle='-', alpha=0.7)
            ax.plot(history.history[f'val_{metric}'], label=f'{name} Val', linestyle='--', alpha=0.7)
        
        ax.set_title(titles[idx], fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot learning curves comparison
    ax = axes[1, 1]
    for name, history in zip(model_names, histories):
        epochs = range(1, len(history.history['val_accuracy']) + 1)
        ax.plot(epochs, history.history['val_accuracy'], label=name, marker='o', markersize=3)
    
    ax.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrices(models, X_test, y_test, model_names):
    """
    Plot confusion matrices for all models
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, model) in enumerate(zip(model_names, models)):
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'])
        axes[idx].set_title(f'{name} Model - Confusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    return fig

def plot_roc_curves(models, X_test, y_test, model_names):
    """
    Plot ROC curves for all models
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in zip(model_names, models):
        y_pred_proba = model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_metrics_comparison(metrics_dict):
    """
    Plot comparison of different metrics across models
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'auc']
    
    metrics_df[metrics_to_plot].plot(kind='bar', ax=ax)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_dropout_effect_analysis(dropout_rates=[0.0, 0.2, 0.3, 0.4, 0.5]):
    """
    Analyze effect of different dropout rates (helper function)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulated results (you can replace with actual training results)
    val_acc = [0.965, 0.958, 0.962, 0.951, 0.945]
    train_acc = [0.998, 0.985, 0.972, 0.961, 0.948]
    
    ax.plot(dropout_rates, train_acc, 'o-', label='Training Accuracy', linewidth=2, markersize=8)
    ax.plot(dropout_rates, val_acc, 's-', label='Validation Accuracy', linewidth=2, markersize=8)
    ax.fill_between(dropout_rates, train_acc, val_acc, alpha=0.2)
    
    ax.set_xlabel('Dropout Rate', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Effect of Dropout Rate on Model Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
