"""
Evaluation script for recommendation models.
Computes ROC-AUC, Precision, Recall, and F1 scores.
Focuses on cardio activity predictions.
"""
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from data_generator import ACTIVITY_CLASSES, ACTIVITY_TO_CLASS

def get_cardio_activities():
    """Get list of cardio activities."""
    return ACTIVITY_CLASSES.get('cardio', [])

def evaluate_binary_classification(y_true, y_pred_proba, threshold=0.5):
    """
    Evaluate binary classification performance.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary with metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'threshold': threshold
    }
    
    return metrics

def evaluate_cardio_precision(model_type, predictions, true_labels, activity_names, 
                              target_precision=0.65):
    """
    Evaluate precision for cardio activities specifically.
    
    Args:
        model_type: Type of model ('two_tower', 'sequence', 'hybrid', etc.)
        predictions: Model predictions (probabilities or scores)
        true_labels: True binary labels (1 for cardio, 0 for non-cardio)
        activity_names: List of activity names corresponding to predictions
        target_precision: Target precision score (default 0.65)
        
    Returns:
        Dictionary with cardio-specific metrics
    """
    cardio_activities = set(get_cardio_activities())
    
    # Filter to cardio activities only
    cardio_mask = np.array([act in cardio_activities for act in activity_names])
    
    if cardio_mask.sum() == 0:
        return {
            'cardio_precision': 0.0,
            'cardio_recall': 0.0,
            'cardio_f1': 0.0,
            'cardio_roc_auc': 0.0,
            'n_cardio_samples': 0,
            'target_precision': target_precision,
            'meets_target': False
        }
    
    cardio_predictions = predictions[cardio_mask]
    cardio_true = true_labels[cardio_mask]
    
    # Convert predictions to binary
    threshold = 0.5
    cardio_pred_binary = (cardio_predictions >= threshold).astype(int)
    
    # Calculate metrics
    cardio_precision = precision_score(cardio_true, cardio_pred_binary, zero_division=0)
    cardio_recall = recall_score(cardio_true, cardio_pred_binary, zero_division=0)
    cardio_f1 = f1_score(cardio_true, cardio_pred_binary, zero_division=0)
    
    # ROC-AUC for cardio
    if len(np.unique(cardio_true)) > 1:
        cardio_roc_auc = roc_auc_score(cardio_true, cardio_predictions)
    else:
        cardio_roc_auc = 0.0
    
    # Try to find threshold that meets target precision
    meets_target = cardio_precision >= target_precision
    optimal_threshold = threshold
    
    if not meets_target and len(np.unique(cardio_true)) > 1:
        # Try different thresholds to meet target precision
        thresholds = np.linspace(0.1, 0.9, 50)
        for thresh in thresholds:
            pred_binary = (cardio_predictions >= thresh).astype(int)
            prec = precision_score(cardio_true, pred_binary, zero_division=0)
            if prec >= target_precision:
                optimal_threshold = thresh
                meets_target = True
                break
    
    return {
        'cardio_precision': cardio_precision,
        'cardio_recall': cardio_recall,
        'cardio_f1': cardio_f1,
        'cardio_roc_auc': cardio_roc_auc,
        'n_cardio_samples': int(cardio_mask.sum()),
        'target_precision': target_precision,
        'meets_target': meets_target,
        'optimal_threshold': optimal_threshold
    }

def evaluate_model_predictions(model_type, predictions, true_labels, activity_names,
                              user_ids=None, detailed=False):
    """
    Comprehensive evaluation of model predictions.
    
    Args:
        model_type: Type of model
        predictions: Model predictions (probabilities or scores)
        true_labels: True binary labels
        activity_names: List of activity names
        user_ids: Optional user IDs for per-user analysis
        detailed: Whether to return detailed metrics
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Overall metrics
    overall_metrics = evaluate_binary_classification(true_labels, predictions)
    
    # Cardio-specific metrics
    cardio_metrics = evaluate_cardio_precision(
        model_type, predictions, true_labels, activity_names, target_precision=0.65
    )
    
    results = {
        'model_type': model_type,
        'overall_roc_auc': overall_metrics['roc_auc'],
        'overall_precision': overall_metrics['precision'],
        'overall_recall': overall_metrics['recall'],
        'overall_f1': overall_metrics['f1'],
        **cardio_metrics
    }
    
    if detailed:
        # Per-activity metrics
        activity_metrics = {}
        unique_activities = set(activity_names)
        for activity in unique_activities:
            act_mask = np.array([act == activity for act in activity_names])
            if act_mask.sum() > 0:
                act_pred = predictions[act_mask]
                act_true = true_labels[act_mask]
                if len(np.unique(act_true)) > 1:
                    act_roc = roc_auc_score(act_true, act_pred)
                    act_pred_binary = (act_pred >= 0.5).astype(int)
                    act_prec = precision_score(act_true, act_pred_binary, zero_division=0)
                    act_rec = recall_score(act_true, act_pred_binary, zero_division=0)
                    activity_metrics[activity] = {
                        'roc_auc': act_roc,
                        'precision': act_prec,
                        'recall': act_rec,
                        'n_samples': int(act_mask.sum())
                    }
        
        results['per_activity_metrics'] = activity_metrics
        
        # Per-user metrics if user_ids provided
        if user_ids is not None:
            user_metrics = {}
            unique_users = set(user_ids)
            for user_id in unique_users:
                user_mask = np.array([uid == user_id for uid in user_ids])
                if user_mask.sum() > 0:
                    user_pred = predictions[user_mask]
                    user_true = true_labels[user_mask]
                    if len(np.unique(user_true)) > 1:
                        user_roc = roc_auc_score(user_true, user_pred)
                        user_pred_binary = (user_pred >= 0.5).astype(int)
                        user_prec = precision_score(user_true, user_pred_binary, zero_division=0)
                        user_metrics[user_id] = {
                            'roc_auc': user_roc,
                            'precision': user_prec,
                            'n_samples': int(user_mask.sum())
                        }
            results['per_user_metrics'] = user_metrics
    
    return results

def print_evaluation_results(results):
    """Print evaluation results in a readable format."""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {results['model_type'].upper()}")
    print(f"{'='*60}")
    
    print(f"\nOverall Metrics:")
    print(f"  ROC-AUC: {results['overall_roc_auc']:.4f}")
    print(f"  Precision: {results['overall_precision']:.4f}")
    print(f"  Recall: {results['overall_recall']:.4f}")
    print(f"  F1-Score: {results['overall_f1']:.4f}")
    
    print(f"\nCardio-Specific Metrics:")
    print(f"  Cardio Precision: {results['cardio_precision']:.4f} (Target: {results['target_precision']:.2f})")
    print(f"  Cardio Recall: {results['cardio_recall']:.4f}")
    print(f"  Cardio F1-Score: {results['cardio_f1']:.4f}")
    print(f"  Cardio ROC-AUC: {results['cardio_roc_auc']:.4f}")
    print(f"  Cardio Samples: {results['n_cardio_samples']}")
    print(f"  Meets Target Precision: {'✓ YES' if results['meets_target'] else '✗ NO'}")
    if 'optimal_threshold' in results:
        print(f"  Optimal Threshold: {results['optimal_threshold']:.4f}")
    
    if 'per_activity_metrics' in results:
        print(f"\nPer-Activity Metrics (Top 5):")
        sorted_activities = sorted(
            results['per_activity_metrics'].items(),
            key=lambda x: x[1]['roc_auc'],
            reverse=True
        )[:5]
        for activity, metrics in sorted_activities:
            print(f"  {activity}:")
            print(f"    ROC-AUC: {metrics['roc_auc']:.4f}, Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}, Samples: {metrics['n_samples']}")

def compare_models(model_results):
    """
    Compare multiple models and print comparison table.
    
    Args:
        model_results: List of evaluation result dictionaries
    """
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    print(f"\n{'Model':<20} {'ROC-AUC':<10} {'Precision':<12} {'Cardio Prec':<12} {'Meets Target':<12}")
    print("-" * 70)
    
    for result in model_results:
        model_name = result['model_type']
        roc_auc = result['overall_roc_auc']
        precision = result['overall_precision']
        cardio_prec = result['cardio_precision']
        meets_target = '✓' if result['meets_target'] else '✗'
        
        print(f"{model_name:<20} {roc_auc:<10.4f} {precision:<12.4f} {cardio_prec:<12.4f} {meets_target:<12}")
    
    print("\nBest Model by Metric:")
    best_roc = max(model_results, key=lambda x: x['overall_roc_auc'])
    best_prec = max(model_results, key=lambda x: x['overall_precision'])
    best_cardio = max(model_results, key=lambda x: x['cardio_precision'])
    
    print(f"  Best ROC-AUC: {best_roc['model_type']} ({best_roc['overall_roc_auc']:.4f})")
    print(f"  Best Precision: {best_prec['model_type']} ({best_prec['overall_precision']:.4f})")
    print(f"  Best Cardio Precision: {best_cardio['model_type']} ({best_cardio['cardio_precision']:.4f})")

