"""
Model Training Module
Berisi fungsi-fungsi untuk training dan tuning model machine learning
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Class untuk training model machine learning
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.trained_models = {}
        
    def initialize_models(self):
        """
        Initialize berbagai model klasifikasi
        
        Returns:
        --------
        dict
            Dictionary berisi model objects
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=10
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=42,
                n_estimators=100
            )
        }
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_single_model(self, model, model_name, X_train, y_train):
        """
        Train single model
        
        Parameters:
        -----------
        model : sklearn model object
            Model yang akan di-train
        model_name : str
            Nama model
        X_train : pandas.DataFrame or numpy.array
            Training features
        y_train : pandas.Series or numpy.array
            Training target
            
        Returns:
        --------
        trained model
        """
        print(f"\nTraining {model_name}...")
        
        model.fit(X_train, y_train)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        print(f"{model_name} training completed")
        
        return model
    
    def train_all_models(self, X_train, y_train):
        """
        Train semua model yang sudah di-initialize
        
        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.array
            Training features
        y_train : pandas.Series or numpy.array
            Training target
            
        Returns:
        --------
        dict
            Dictionary berisi semua trained models
        """
        if not self.models:
            self.initialize_models()
        
        print("="*60)
        print("Training All Models")
        print("="*60)
        
        for model_name, model in self.models.items():
            self.train_single_model(model, model_name, X_train, y_train)
        
        print("\nAll models trained successfully!")
        
        return self.trained_models
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate model performance
        
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
            Dictionary berisi metrik evaluasi
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"\n{model_name} Evaluation:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Evaluate semua trained models
        
        Parameters:
        -----------
        X_test : pandas.DataFrame or numpy.array
            Testing features
        y_test : pandas.Series or numpy.array
            Testing target
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame berisi hasil evaluasi semua model
        """
        results = []
        
        print("\n" + "="*60)
        print("Evaluating All Models")
        print("="*60)
        
        for model_name, model in self.trained_models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # Store best model
        best_idx = results_df['accuracy'].idxmax()
        self.best_model_name = results_df.loc[best_idx, 'model_name']
        self.best_model = self.trained_models[self.best_model_name]
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Accuracy: {results_df.loc[best_idx, 'accuracy']:.4f}")
        
        return results_df
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='accuracy'):
        """
        Perform cross-validation
        
        Parameters:
        -----------
        model : sklearn model object
            Model yang akan di-cross validate
        X : pandas.DataFrame or numpy.array
            Features
        y : pandas.Series or numpy.array
            Target
        cv : int
            Number of folds
        scoring : str
            Scoring metric
            
        Returns:
        --------
        dict
            Dictionary berisi hasil cross-validation
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        
        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'all_scores': scores
        }
        
        print(f"\nCross-Validation Results ({cv}-fold):")
        print(f"Mean {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        print(f"All scores: {scores}")
        
        return cv_results
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, param_grid, cv=5):
        """
        Perform hyperparameter tuning menggunakan GridSearchCV
        
        Parameters:
        -----------
        model_name : str
            Nama model ('Logistic Regression', 'Decision Tree', 'KNN')
        X_train : pandas.DataFrame or numpy.array
            Training features
        y_train : pandas.Series or numpy.array
            Training target
        param_grid : dict
            Parameter grid untuk tuning
        cv : int
            Number of folds
            
        Returns:
        --------
        tuple
            (best_model, best_params, best_score)
        """
        print(f"\nHyperparameter Tuning for {model_name}")
        print("="*60)
        
        # Get base model
        if model_name not in self.models:
            self.initialize_models()
        
        base_model = self.models[model_name]
        
        # Grid Search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        
        # Update trained model dengan best model
        self.trained_models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def get_default_param_grids(self):
        """
        Get default parameter grids untuk tuning
        
        Returns:
        --------
        dict
            Dictionary berisi parameter grids untuk setiap model
        """
        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            'Decision Tree': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        }
        
        return param_grids
    
    def tune_all_models(self, X_train, y_train, cv=5):
        """
        Tune hyperparameters untuk semua model
        
        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.array
            Training features
        y_train : pandas.Series or numpy.array
            Training target
        cv : int
            Number of folds
            
        Returns:
        --------
        dict
            Dictionary berisi best parameters untuk setiap model
        """
        param_grids = self.get_default_param_grids()
        tuning_results = {}
        
        print("\n" + "="*60)
        print("Hyperparameter Tuning for All Models")
        print("="*60)
        
        for model_name in ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors']:
            if model_name in param_grids:
                best_model, best_params, best_score = self.hyperparameter_tuning(
                    model_name, X_train, y_train, param_grids[model_name], cv
                )
                
                tuning_results[model_name] = {
                    'best_model': best_model,
                    'best_params': best_params,
                    'best_score': best_score
                }
        
        return tuning_results
    
    def save_model(self, model, filepath, model_name="model"):
        """
        Save trained model ke file
        
        Parameters:
        -----------
        model : trained sklearn model
            Model yang akan disimpan
        filepath : str
            Path file untuk menyimpan model
        model_name : str
            Nama model
        """
        try:
            joblib.dump(model, filepath)
            print(f"\n{model_name} saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self, filepath):
        """
        Load trained model dari file
        
        Parameters:
        -----------
        filepath : str
            Path file model
            
        Returns:
        --------
        loaded model
        """
        try:
            model = joblib.load(filepath)
            print(f"Model loaded successfully from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def save_best_model(self, filepath='models/best_model.pkl'):
        """
        Save best model
        
        Parameters:
        -----------
        filepath : str
            Path file untuk menyimpan model
        """
        if self.best_model is not None:
            self.save_model(self.best_model, filepath, self.best_model_name)
        else:
            print("No best model found. Train and evaluate models first.")
    
    def predict(self, model, X):
        """
        Make predictions
        
        Parameters:
        -----------
        model : trained sklearn model
            Model untuk prediksi
        X : pandas.DataFrame or numpy.array
            Features untuk prediksi
            
        Returns:
        --------
        numpy.array
            Predictions
        """
        return model.predict(X)
    
    def predict_proba(self, model, X):
        """
        Make probability predictions
        
        Parameters:
        -----------
        model : trained sklearn model
            Model untuk prediksi
        X : pandas.DataFrame or numpy.array
            Features untuk prediksi
            
        Returns:
        --------
        numpy.array
            Probability predictions
        """
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            print("Model doesn't support probability predictions")
            return None


# Helper functions
def get_feature_importance(model, feature_names, top_n=10):
    """
    Get feature importance dari trained model
    
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
    
    return importance_df


def compare_models_summary(results_df):
    """
    Create summary comparison of models
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame hasil evaluasi models
        
    Returns:
    --------
    dict
        Dictionary berisi summary
    """
    summary = {
        'best_model': results_df.iloc[0]['model_name'],
        'best_accuracy': results_df.iloc[0]['accuracy'],
        'best_f1_score': results_df.iloc[0]['f1_score'],
        'total_models_evaluated': len(results_df)
    }
    
    return summary