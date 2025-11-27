"""
Data Preprocessing Module
Berisi fungsi-fungsi untuk cleaning dan preprocessing data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Class untuk preprocessing data heart attack
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """
        Load data dari CSV file
        
        Parameters:
        -----------
        filepath : str
            Path ke file CSV
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame yang sudah diload
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
            return df
        except Exception as e:
            print(f"Error saat memuat data: {str(e)}")
            return None
    
    def check_missing_values(self, df):
        """
        Cek missing values dalam dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset yang akan dicek
            
        Returns:
        --------
        pandas.Series
            Series berisi jumlah missing values per kolom
        """
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_percent
        })
        
        return missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset dengan missing values
        strategy : str
            Strategi untuk handle missing values ('mean', 'median', 'mode', 'drop')
            
        Returns:
        --------
        pandas.DataFrame
            Dataset yang sudah dihandle missing values-nya
        """
        df_cleaned = df.copy()
        
        # Numerical columns
        numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        # Categorical columns
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        
        if strategy == 'mean':
            for col in numerical_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                    
        elif strategy == 'median':
            for col in numerical_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                    
        elif strategy == 'mode':
            for col in numerical_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                    
        elif strategy == 'drop':
            df_cleaned = df_cleaned.dropna()
        
        # Handle categorical missing values with mode
        for col in categorical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        
        print(f"Missing values handled. Shape: {df_cleaned.shape}")
        return df_cleaned
    
    def remove_duplicates(self, df):
        """
        Remove duplicate rows
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset yang akan dicek duplikatnya
            
        Returns:
        --------
        pandas.DataFrame
            Dataset tanpa duplikat
        """
        initial_shape = df.shape[0]
        df_no_dup = df.drop_duplicates()
        final_shape = df_no_dup.shape[0]
        
        print(f"Duplikat dihapus: {initial_shape - final_shape} baris")
        return df_no_dup
    
    def encode_categorical(self, df, categorical_columns):
        """
        Encode categorical variables menggunakan Label Encoding
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset dengan kolom kategorikal
        categorical_columns : list
            List nama kolom kategorikal yang akan di-encode
            
        Returns:
        --------
        pandas.DataFrame
            Dataset dengan kolom kategorikal yang sudah di-encode
        """
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"Column '{col}' encoded: {list(le.classes_)}")
        
        return df_encoded
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale numerical features menggunakan StandardScaler
        
        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.array
            Training features
        X_test : pandas.DataFrame or numpy.array, optional
            Testing features
            
        Returns:
        --------
        tuple
            (X_train_scaled, X_test_scaled) jika X_test provided
            X_train_scaled jika X_test None
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_basic_stats(self, df):
        """
        Get basic statistics of the dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset
            
        Returns:
        --------
        dict
            Dictionary berisi statistik dasar
        """
        stats = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'numerical_summary': df.describe(),
            'categorical_summary': df.describe(include=['object']),
            'missing_values': self.check_missing_values(df)
        }
        
        return stats
    
    def detect_outliers(self, df, columns, method='iqr'):
        """
        Detect outliers menggunakan IQR method
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset
        columns : list
            List kolom numerical yang akan dicek outliers
        method : str
            Method untuk detect outliers ('iqr' atau 'zscore')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame berisi informasi outliers
        """
        outliers_info = {}
        
        for col in columns:
            if col in df.columns:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outliers_info[col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(df)) * 100,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                    
                elif method == 'zscore':
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(df[col]))
                    outliers = df[z_scores > 3]
                    outliers_info[col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(df)) * 100
                    }
        
        return pd.DataFrame(outliers_info).T
    
    def prepare_data_for_modeling(self, df, target_column, test_size=0.2, random_state=42):
        """
        Prepare data untuk modeling (split train-test)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset yang sudah di-preprocess
        target_column : str
            Nama kolom target
        test_size : float
            Proporsi data test
        random_state : int
            Random state untuk reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Data split completed:")
        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")
        print(f"Target distribution in train: {y_train.value_counts().to_dict()}")
        print(f"Target distribution in test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test


# Helper functions
def get_column_types(df):
    """
    Identifikasi tipe kolom (numerical vs categorical)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
        
    Returns:
    --------
    dict
        Dictionary berisi list kolom numerical dan categorical
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    return {
        'numerical': numerical_cols,
        'categorical': categorical_cols
    }


def data_quality_report(df):
    """
    Generate comprehensive data quality report
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset
        
    Returns:
    --------
    dict
        Dictionary berisi laporan kualitas data
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # in MB
        'column_types': get_column_types(df)
    }
    
    return report