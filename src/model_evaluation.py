"""
Model Evaluation Module
Berisi fungsi-fungsi untuk evaluasi model secara detail
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Class untuk evaluasi model machine learning
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
    def confusion_matrix_analysis(self, y_true, y_pred, labels=[0, 1], 
                                  model_name="Model", plot=True):
        """
        Analisis confusion matrix
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        labels : list
            List of label values
        model_name : str
            Nama model
        plot : bool
            Apakah akan plot confusion matrix
            
        Returns:
        --------
        numpy.array
            Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        print(f"\nConfusion Matrix - {model_name}")
        print("="*50)
        print(cm)
        
        # Calculate metrics dari confusion matrix
        if len(labels) == 2:
            tn, fp, fn, tp = cm.ravel()
            
            print(f"\nTrue Negatives:  {tn}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            print(f"True Positives:  {tp}")
            
            # Additional metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"\nSensitivity (Recall): {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
        
        if plot:
            self.plot_confusion_matrix(cm, labels, model_name)
        
        return cm
    
    def plot_confusion_matrix(self, cm, labels, model_name="Model"):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        cm : numpy.array
            Confusion matrix
        labels : list
            List of labels
        model_name : str
            Nama model
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Heart Attack', 'Heart Attack'],
                   yticklabels=['No Heart Attack', 'Heart Attack'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def classification_report_detailed(self, y_true, y_pred, 
                                      target_names=['No Heart Attack', 'Heart Attack'],
                                      model_name="Model"):
        """
        Generate detailed classification report
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        target_names : list
            Names for target classes
        model_name : str
            Nama model
            
        Returns:
        --------
        dict
            Classification report as dictionary
        """
        print(f"\nClassification Report - {model_name}")
        print("="*50)
        
        report = classification_report(y_true, y_pred, 
                                      target_names=target_names,
                                      output_dict=True)
        
        # Print formatted report
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        return report
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name="Model"):
        """
        Plot ROC curve
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        model_name : str
            Nama model
            
        Returns:
        --------
        tuple
            (fpr, tpr, auc_score)
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        return fpr, tpr, roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name="Model"):
        """
        Plot Precision-Recall curve
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        model_name : str
            Nama model
            
        Returns:
        --------
        tuple
            (precision, recall, avg_precision)
        """
        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\nAverage Precision Score: {avg_precision:.4f}")
        
        return precision, recall, avg_precision
    
    def evaluate_model_comprehensive(self, model, X_test, y_test, model_name="Model"):
        """
        Comprehensive evaluation untuk satu model
        
        Parameters:
        -----------
        model : trained sklearn model
            Model yang akan dievaluasi
        X_test : pandas.DataFrame or numpy.array
            Testing features
        y_test : pandas.Series or numpy.array
            Testing target
        model_name : str
            Nama model
            
        Returns:
        --------
        dict
            Dictionary berisi semua hasil evaluasi
        """
        print("\n" + "="*60)
        print(f"Comprehensive Evaluation - {model_name}")
        print("="*60)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Confusion Matrix
        cm = self.confusion_matrix_analysis(y_test, y_pred, model_name=model_name, plot=False)
        
        # Classification Report
        report = self.classification_report_detailed(y_test, y_pred, model_name=model_name)
        
        # ROC-AUC (if model supports probability)
        roc_auc = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"\nROC-AUC Score: {roc_auc:.4f}")
        
        results = {
            'model_name': model_name,
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_auc': roc_auc,
            'predictions': y_pred
        }
        
        self.evaluation_results[model_name] = results
        
        return results
    
    def compare_multiple_models(self, models_dict, X_test, y_test):
        """
        Compare multiple models
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary {model_name: trained_model}
        X_test : pandas.DataFrame or numpy.array
            Testing features
        y_test : pandas.Series or numpy.array
            Testing target
            
        Returns:
        --------
        pandas.DataFrame
            Comparison results
        """
        comparison_results = []
        
        for model_name, model in models_dict.items():
            # Predictions
            y_pred = model.predict(X_test)
            
            # Basic metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC-AUC if available
            roc_auc = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            comparison_results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc if roc_auc else 'N/A'
            })
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("Model Comparison Summary")
        print("="*60)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df, metric='Accuracy'):
        """
        Plot comparison chart untuk multiple models
        
        Parameters:
        -----------
        comparison_df : pandas.DataFrame
            DataFrame hasil comparison
        metric : str
            Metric yang akan di-plot
        """
        plt.figure(figsize=(10, 6))
        
        # Filter out N/A values jika ada
        plot_df = comparison_df[comparison_df[metric] != 'N/A'].copy()
        if metric == 'ROC-AUC':
            plot_df[metric] = plot_df[metric].astype(float)
        
        plt.barh(plot_df['Model'], plot_df[metric], color='steelblue')
        plt.xlabel(metric)
        plt.ylabel('Model')
        plt.title(f'Model Comparison - {metric}')
        plt.xlim([0, 1])
        
        # Add value labels
        for i, v in enumerate(plot_df[metric]):
            plt.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_metrics_comparison(self, comparison_df):
        """
        Plot comparison untuk semua metrics
        
        Parameters:
        -----------
        comparison_df : pandas.DataFrame
            DataFrame hasil comparison
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            plot_df = comparison_df[['Model', metric]].copy()
            
            axes[idx].barh(plot_df['Model'], plot_df[metric], color='steelblue')
            axes[idx].set_xlabel(metric)
            axes[idx].set_ylabel('Model')
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].set_xlim([0, 1])
            
            # Add value labels
            for i, v in enumerate(plot_df[metric]):
                axes[idx].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self, model, feature_names, top_n=15):
        """
        Analyze dan visualisasi feature importance
        
        Parameters:
        -----------
        model : trained sklearn model
            Model yang sudah di-train
        feature_names : list
            List nama features
        top_n : int
            Jumlah top features yang akan ditampilkan
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame berisi feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("Model doesn't have feature importance attribute")
            return None
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop {top_n} Important Features:")
        print(importance_df.head(top_n).to_string(index=False))
        
        # Plot
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def error_analysis(self, y_true, y_pred, X_test, feature_names=None):
        """
        Analyze prediction errors
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        X_test : pandas.DataFrame or numpy.array
            Testing features
        feature_names : list, optional
            List nama features
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame berisi error analysis
        """
        # Identify errors
        errors = y_true != y_pred
        
        print(f"\nError Analysis")
        print("="*50)
        print(f"Total Predictions: {len(y_true)}")
        print(f"Correct Predictions: {(~errors).sum()}")
        print(f"Incorrect Predictions: {errors.sum()}")
        print(f"Error Rate: {errors.sum() / len(y_true):.4f}")
        
        # Analyze error types
        false_positives = (y_true == 0) & (y_pred == 1)
        false_negatives = (y_true == 1) & (y_pred == 0)
        
        print(f"\nFalse Positives: {false_positives.sum()}")
        print(f"False Negatives: {false_negatives.sum()}")
        
        # Create error dataframe
        if isinstance(X_test, pd.DataFrame):
            error_df = X_test[errors].copy()
            error_df['True_Label'] = y_true[errors]
            error_df['Predicted_Label'] = y_pred[errors]
        else:
            if feature_names:
                error_df = pd.DataFrame(X_test[errors], columns=feature_names)
                error_df['True_Label'] = y_true[errors]
                error_df['Predicted_Label'] = y_pred[errors]
            else:
                error_df = None
        
        return error_df


# Helper functions
def calculate_cost_benefit(y_true, y_pred, cost_fp=1, cost_fn=10, benefit_tp=5):
    """
    Calculate cost-benefit analysis
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    cost_fp : float
        Cost of False Positive
    cost_fn : float
        Cost of False Negative
    benefit_tp : float
        Benefit of True Positive
        
    Returns:
    --------
    dict
        Dictionary berisi cost-benefit analysis
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    total_benefit = tp * benefit_tp
    net_value = total_benefit - total_cost
    
    analysis = {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'total_cost': total_cost,
        'total_benefit': total_benefit,
        'net_value': net_value
    }
    
    print("\nCost-Benefit Analysis")
    print("="*50)
    print(f"Total Cost: {total_cost}")
    print(f"Total Benefit: {total_benefit}")
    print(f"Net Value: {net_value}")
    
    return analysis