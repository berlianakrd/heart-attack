"""
Feature Engineering Module
Berisi fungsi-fungsi untuk feature engineering dan feature selection
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Class untuk feature engineering
    """
    
    def __init__(self):
        self.selected_features = None
        self.pca = None
        
    def create_age_groups(self, df, age_column='age'):
        """
        Create age groups dari kolom age
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset
        age_column : str
            Nama kolom age
            
        Returns:
        --------
        pandas.DataFrame
            Dataset dengan kolom age_group baru
        """
        df_new = df.copy()
        
        bins = [0, 30, 40, 50, 60, 100]
        labels = ['Young', 'Adult', 'Middle_Age', 'Senior', 'Elderly']
        
        df_new['age_group'] = pd.cut(df_new[age_column], bins=bins, labels=labels)
        
        print(f"Age groups created: {df_new['age_group'].value_counts().to_dict()}")
        return df_new
    
    def create_bmi_category(self, df):
        """
        Create BMI category dari obesity flag
        (Simplified version karena dataset sudah punya obesity flag)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset
            
        Returns:
        --------
        pandas.DataFrame
            Dataset dengan kolom bmi_category
        """
        df_new = df.copy()
        
        # Karena dataset sudah ada obesity (1 jika BMI > 30)
        df_new['bmi_category'] = df_new['obesity'].map({
            0: 'Normal',
            1: 'Obese'
        })
        
        return df_new
    
    def create_blood_pressure_category(self, df):
        """
        Create blood pressure category dari systolic dan diastolic
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset dengan kolom blood_pressure_systolic dan blood_pressure_diastolic
            
        Returns:
        --------
        pandas.DataFrame
            Dataset dengan kolom bp_category
        """
        df_new = df.copy()
        
        def categorize_bp(row):
            systolic = row['blood_pressure_systolic']
            diastolic = row['blood_pressure_diastolic']
            
            if systolic < 120 and diastolic < 80:
                return 'Normal'
            elif systolic < 130 and diastolic < 80:
                return 'Elevated'
            elif systolic < 140 or diastolic < 90:
                return 'Hypertension_Stage1'
            elif systolic < 180 or diastolic < 120:
                return 'Hypertension_Stage2'
            else:
                return 'Hypertensive_Crisis'
        
        df_new['bp_category'] = df_new.apply(categorize_bp, axis=1)
        
        print(f"BP categories created: {df_new['bp_category'].value_counts().to_dict()}")
        return df_new
    
    def create_cholesterol_category(self, df):
        """
        Create cholesterol category
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset dengan kolom cholesterol
            
        Returns:
        --------
        pandas.DataFrame
            Dataset dengan kolom cholesterol_category
        """
        df_new = df.copy()
        
        def categorize_cholesterol(value):
            if value < 200:
                return 'Desirable'
            elif value < 240:
                return 'Borderline_High'
            else:
                return 'High'
        
        df_new['cholesterol_category'] = df_new['cholesterol_level'].apply(categorize_cholesterol)
        
        print(f"Cholesterol categories: {df_new['cholesterol_category'].value_counts().to_dict()}")
        return df_new
    
    def create_risk_score(self, df):
        """
        Create composite risk score dari multiple faktor risiko
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset
            
        Returns:
        --------
        pandas.DataFrame
            Dataset dengan kolom risk_score
        """
        df_new = df.copy()
        
        # Calculate risk score berdasarkan faktor risiko utama
        risk_score = 0
        
        # Age risk (>50 tahun = 1 poin)
        risk_score += (df_new['age'] > 50).astype(int)
        
        # Hypertension (1 poin)
        risk_score += df_new['hypertension']
        
        # Diabetes (1 poin)
        risk_score += df_new['diabetes']
        
        # Obesity (1 poin)
        risk_score += df_new['obesity']
        
        # High cholesterol (>240 = 1 poin)
        risk_score += (df_new['cholesterol_level'] > 240).astype(int)
        
        # Smoking (Current smoker = 1 poin)
        if 'smoking_status' in df_new.columns:
            risk_score += (df_new['smoking_status'] == 'Current').astype(int)
        
        # Family history (1 poin)
        risk_score += df_new['family_history']
        
        # Previous heart disease (2 poin - lebih berat)
        risk_score += df_new['previous_heart_disease'] * 2
        
        df_new['risk_score'] = risk_score
        
        print(f"Risk score statistics:")
        print(df_new['risk_score'].describe())
        
        return df_new
    
    def create_interaction_features(self, df):
        """
        Create interaction features antara variabel-variabel penting
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset
            
        Returns:
        --------
        pandas.DataFrame
            Dataset dengan interaction features
        """
        df_new = df.copy()
        
        # Age x Hypertension
        df_new['age_hypertension'] = df_new['age'] * df_new['hypertension']
        
        # Age x Diabetes
        df_new['age_diabetes'] = df_new['age'] * df_new['diabetes']
        
        # Cholesterol x Age
        df_new['cholesterol_age'] = df_new['cholesterol_level'] * df_new['age']
        
        # BP interaction (systolic x diastolic)
        df_new['bp_interaction'] = df_new['blood_pressure_systolic'] * df_new['blood_pressure_diastolic']
        
        # Health conditions sum (total kondisi kesehatan)
        df_new['total_health_conditions'] = (
            df_new['hypertension'] + 
            df_new['diabetes'] + 
            df_new['obesity'] + 
            df_new['previous_heart_disease']
        )
        
        print(f"Interaction features created: {len([col for col in df_new.columns if col not in df.columns])} new features")
        
        return df_new
    
    def select_features_univariate(self, X, y, k=10, score_func='f_classif'):
        """
        Feature selection menggunakan univariate statistical tests
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Features
        y : pandas.Series
            Target variable
        k : int
            Jumlah top features yang akan dipilih
        score_func : str
            Score function ('f_classif', 'chi2', 'mutual_info')
            
        Returns:
        --------
        tuple
            (selected_features_names, selector_object)
        """
        # Pilih score function
        if score_func == 'f_classif':
            func = f_classif
        elif score_func == 'chi2':
            func = chi2
        elif score_func == 'mutual_info':
            func = mutual_info_classif
        else:
            func = f_classif
        
        # Fit selector
        selector = SelectKBest(score_func=func, k=k)
        selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get scores
        scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        print(f"\nTop {k} features selected using {score_func}:")
        print(scores.head(k))
        
        self.selected_features = selected_features
        
        return selected_features, selector
    
    def select_features_correlation(self, df, target_column, threshold=0.1):
        """
        Feature selection berdasarkan correlation dengan target
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset
        target_column : str
            Nama kolom target
        threshold : float
            Threshold minimum correlation
            
        Returns:
        --------
        list
            List features yang dipilih
        """
        # Calculate correlation dengan target
        correlations = df.corr()[target_column].abs().sort_values(ascending=False)
        
        # Filter berdasarkan threshold
        selected = correlations[correlations > threshold].index.tolist()
        selected.remove(target_column)  # Remove target itself
        
        print(f"\nFeatures with correlation > {threshold}:")
        print(correlations[correlations > threshold])
        
        return selected
    
    def apply_pca(self, X, n_components=0.95):
        """
        Apply PCA untuk dimensionality reduction
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.array
            Features
        n_components : int or float
            Jumlah components atau variance yang ingin dipertahankan
            
        Returns:
        --------
        tuple
            (X_transformed, pca_object)
        """
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        print(f"\nPCA applied:")
        print(f"Original dimensions: {X.shape[1]}")
        print(f"Reduced dimensions: {X_pca.shape[1]}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return X_pca, self.pca
    
    def get_feature_importance_from_correlation(self, df, target_column):
        """
        Get feature importance berdasarkan correlation
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset
        target_column : str
            Nama kolom target
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame berisi feature importance
        """
        correlations = df.corr()[target_column].abs().sort_values(ascending=False)
        
        importance_df = pd.DataFrame({
            'Feature': correlations.index,
            'Importance': correlations.values
        })
        
        # Remove target column
        importance_df = importance_df[importance_df['Feature'] != target_column]
        
        return importance_df
    
    def create_all_features(self, df):
        """
        Create all engineered features sekaligus
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset original
            
        Returns:
        --------
        pandas.DataFrame
            Dataset dengan semua engineered features
        """
        print("Creating engineered features...")
        
        df_engineered = df.copy()
        
        # Create categorical features
        df_engineered = self.create_age_groups(df_engineered)
        df_engineered = self.create_bmi_category(df_engineered)
        df_engineered = self.create_blood_pressure_category(df_engineered)
        df_engineered = self.create_cholesterol_category(df_engineered)
        
        # Create numerical features
        df_engineered = self.create_risk_score(df_engineered)
        df_engineered = self.create_interaction_features(df_engineered)
        
        print(f"\nTotal features setelah engineering: {df_engineered.shape[1]}")
        print(f"New features created: {df_engineered.shape[1] - df.shape[1]}")
        
        return df_engineered


# Helper functions
def get_high_correlation_features(df, threshold=0.8):
    """
    Identifikasi features dengan high correlation (multicollinearity)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
    threshold : float
        Threshold correlation
        
    Returns:
    --------
    list
        List pasangan features dengan high correlation
    """
    corr_matrix = df.corr().abs()
    
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    high_corr_pairs = []
    for column in upper.columns:
        high_corr = upper[column][upper[column] > threshold]
        if len(high_corr) > 0:
            for idx in high_corr.index:
                high_corr_pairs.append((column, idx, high_corr[idx]))
    
    return high_corr_pairs


def feature_summary(df_original, df_engineered):
    """
    Summary of feature engineering results
    
    Parameters:
    -----------
    df_original : pandas.DataFrame
        Dataset original
    df_engineered : pandas.DataFrame
        Dataset setelah feature engineering
        
    Returns:
    --------
    dict
        Dictionary berisi summary
    """
    new_features = set(df_engineered.columns) - set(df_original.columns)
    
    summary = {
        'original_features': df_original.shape[1],
        'engineered_features': df_engineered.shape[1],
        'new_features_count': len(new_features),
        'new_features_list': list(new_features)
    }
    
    return summary