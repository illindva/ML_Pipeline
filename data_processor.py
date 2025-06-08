import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from typing import List, Dict, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handles all data processing operations for the fraud detection application."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
    
    def clean_data(self, df: pd.DataFrame, handle_missing: str = "None", 
                   remove_duplicates: bool = False, outlier_method: str = "None",
                   fix_dtypes: bool = True, drop_empty_columns: bool = False,
                   value_mappings: Dict[str, str] = {}) -> pd.DataFrame:
        """Clean the dataset based on specified parameters."""
        df_cleaned = df.copy()
        
        # Handle missing values
        if handle_missing != "None":
            if handle_missing == "Drop rows":
                df_cleaned = df_cleaned.dropna()
            elif handle_missing == "Fill with mean/mode":
                # Fill numerical columns with mean
                numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                
                # Fill categorical columns with mode
                categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown', inplace=True)
            
            elif handle_missing == "Forward fill":
                df_cleaned = df_cleaned.fillna(method='ffill')
            
            elif handle_missing == "Backward fill":
                df_cleaned = df_cleaned.fillna(method='bfill')
        
        # Remove duplicates
        if remove_duplicates:
            df_cleaned = df_cleaned.drop_duplicates()
        
        # Handle outliers
        if outlier_method != "None":
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            
            if outlier_method == "IQR Method":
                for col in numeric_cols:
                    Q1 = df_cleaned[col].quantile(0.25)
                    Q3 = df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
            
            elif outlier_method == "Z-Score":
                from scipy import stats
                z_scores = np.abs(stats.zscore(df_cleaned[numeric_cols]))
                df_cleaned = df_cleaned[(z_scores < 3).all(axis=1)]
            
            elif outlier_method == "Isolation Forest":
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(df_cleaned[numeric_cols])
                df_cleaned = df_cleaned[outliers == 1]
        
        # Drop columns with all null values
        if drop_empty_columns:
            null_columns = df_cleaned.columns[df_cleaned.isnull().all()].tolist()
            if null_columns:
                df_cleaned = df_cleaned.drop(columns=null_columns)
        
        # Apply value mappings for categorical columns
        if value_mappings:
            for old_value, new_value in value_mappings.items():
                # Find which column contains this value and replace it
                for col in df_cleaned.select_dtypes(include=['object']).columns:
                    df_cleaned[col] = df_cleaned[col].replace(old_value, new_value)
        
        # Fix data types
        if fix_dtypes:
            df_cleaned = self._fix_data_types(df_cleaned)
        
        return df_cleaned
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically fix data types."""
        df_fixed = df.copy()
        
        for col in df_fixed.columns:
            # Try to convert to datetime if column name suggests it's a date
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                try:
                    df_fixed[col] = pd.to_datetime(df_fixed[col])
                    continue
                except:
                    pass
            
            # Try to convert to numeric if it looks numeric
            if df_fixed[col].dtype == 'object':
                try:
                    # Remove common non-numeric characters and try conversion
                    temp_series = df_fixed[col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '')
                    numeric_series = pd.to_numeric(temp_series, errors='coerce')
                    
                    # If most values are successfully converted, use the numeric version
                    if numeric_series.notna().sum() / len(numeric_series) > 0.8:
                        df_fixed[col] = numeric_series
                except:
                    pass
        
        return df_fixed
    
    def create_date_features(self, df: pd.DataFrame, date_columns: List[str], 
                           features: List[str]) -> pd.DataFrame:
        """Create new features from date columns."""
        df_new = df.copy()
        
        for col in date_columns:
            # Ensure column is datetime
            try:
                df_new[col] = pd.to_datetime(df_new[col])
            except:
                continue
            
            if 'year' in features:
                df_new[f'{col}_year'] = df_new[col].dt.year
            
            if 'month' in features:
                df_new[f'{col}_month'] = df_new[col].dt.month
            
            if 'day' in features:
                df_new[f'{col}_day'] = df_new[col].dt.day
            
            if 'dayofweek' in features:
                df_new[f'{col}_dayofweek'] = df_new[col].dt.dayofweek
            
            if 'quarter' in features:
                df_new[f'{col}_quarter'] = df_new[col].dt.quarter
            
            if 'is_weekend' in features:
                df_new[f'{col}_is_weekend'] = (df_new[col].dt.dayofweek >= 5).astype(int)
            
            if 'days_since' in features:
                df_new[f'{col}_days_since'] = (pd.Timestamp.now() - df_new[col]).dt.days
        
        return df_new
    
    def create_polynomial_features(self, df: pd.DataFrame, columns: List[str], 
                                 degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for specified columns."""
        df_new = df.copy()
        
        # Select only the specified columns for polynomial features
        poly_data = df_new[columns]
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(poly_data)
        
        # Get feature names
        feature_names = poly.get_feature_names_out(columns)
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_new.index)
        
        # Remove original columns and add polynomial features
        df_new = df_new.drop(columns=columns)
        df_new = pd.concat([df_new, poly_df], axis=1)
        
        return df_new
    
    def create_interaction_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create interaction features between specified columns."""
        df_new = df.copy()
        
        # Create pairwise interactions
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                interaction_name = f'{col1}_x_{col2}'
                df_new[interaction_name] = df_new[col1] * df_new[col2]
        
        return df_new
    
    def select_features_by_correlation(self, df: pd.DataFrame, target_column: str, 
                                     threshold: float = 0.5) -> List[str]:
        """Select features based on correlation with target variable."""
        # Calculate correlation with target
        correlations = df.corr()[target_column].abs().sort_values(ascending=False)
        
        # Remove target column and select features above threshold
        correlations = correlations.drop(target_column)
        selected_features = correlations[correlations >= threshold].index.tolist()
        
        return selected_features
    
    def select_features_statistical(self, df: pd.DataFrame, feature_columns: List[str],
                                  target_column: str, k: int = 20) -> List[str]:
        """Select features using statistical tests."""
        X = df[feature_columns]
        y = df[target_column]
        
        # Handle mixed data types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        selected_features = []
        
        # For numeric features, use f_classif
        if numeric_features:
            X_numeric = X[numeric_features]
            selector_numeric = SelectKBest(score_func=f_classif, k=min(k, len(numeric_features)))
            selector_numeric.fit(X_numeric, y)
            selected_numeric = [numeric_features[i] for i in selector_numeric.get_support(indices=True)]
            selected_features.extend(selected_numeric)
        
        # For categorical features, use chi2 (after encoding)
        if categorical_features:
            X_categorical = X[categorical_features].copy()
            
            # Label encode categorical features
            for col in categorical_features:
                le = LabelEncoder()
                X_categorical[col] = le.fit_transform(X_categorical[col].astype(str))
            
            remaining_k = max(0, k - len(selected_features))
            if remaining_k > 0:
                selector_categorical = SelectKBest(score_func=chi2, k=min(remaining_k, len(categorical_features)))
                selector_categorical.fit(X_categorical, y)
                selected_categorical = [categorical_features[i] for i in selector_categorical.get_support(indices=True)]
                selected_features.extend(selected_categorical)
        
        return selected_features[:k]  # Ensure we don't exceed k features
    
    def handle_categorical_encoding(self, X: pd.DataFrame, method: str = "One-Hot Encoding",
                                   handle_high_cardinality: bool = False,
                                   cardinality_threshold: int = 20) -> pd.DataFrame:
        """Handle categorical variable encoding."""
        X_encoded = X.copy()
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_cols:
            return X_encoded
        
        if method == "One-Hot Encoding":
            # Handle high cardinality if requested
            if handle_high_cardinality:
                for col in categorical_cols:
                    if X_encoded[col].nunique() > cardinality_threshold:
                        # Keep only top categories, group others as 'Other'
                        top_categories = X_encoded[col].value_counts().head(cardinality_threshold).index
                        X_encoded[col] = X_encoded[col].where(X_encoded[col].isin(top_categories), 'Other')
            
            # One-hot encoding
            X_encoded = pd.get_dummies(X_encoded, columns=categorical_cols, drop_first=True)
        
        elif method == "Label Encoding":
            for col in categorical_cols:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                self.encoders[col] = le
        
        elif method == "Target Encoding":
            # This would require target variable, so we'll use label encoding as fallback
            for col in categorical_cols:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                self.encoders[col] = le
        
        return X_encoded
    
    def scale_features(self, X: pd.DataFrame, method: str = "StandardScaler") -> pd.DataFrame:
        """Scale numerical features."""
        if method == "None":
            return X
        
        X_scaled = X.copy()
        numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return X_scaled
        
        if method == "StandardScaler":
            scaler = StandardScaler()
        elif method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif method == "RobustScaler":
            scaler = RobustScaler()
        else:
            return X_scaled
        
        X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
        self.scalers['feature_scaler'] = scaler
        
        return X_scaled
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, 
                              method: str = "SMOTE") -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using various resampling techniques."""
        if method == "None":
            return X, y
        
        try:
            if method == "SMOTE":
                sampler = SMOTE(random_state=42)
            elif method == "Random Over Sampling":
                sampler = RandomOverSampler(random_state=42)
            elif method == "Random Under Sampling":
                sampler = RandomUnderSampler(random_state=42)
            elif method == "ADASYN":
                sampler = ADASYN(random_state=42)
            else:
                return X, y
            
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        
        except Exception as e:
            print(f"Error in resampling: {e}")
            return X, y
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series, 
                       categorical_encoding: str = None,
                       scaling_method: str = "StandardScaler",
                       resampling_method: str = "None",
                       handle_high_cardinality: bool = False,
                       cardinality_threshold: int = 20) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Complete preprocessing pipeline."""
        preprocessing_info = {
            'original_shape': X.shape,
            'steps_applied': [],
            'feature_changes': {}
        }
        
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Categorical encoding
        if categorical_encoding and categorical_encoding != "None":
            original_cols = len(X_processed.columns)
            X_processed = self.handle_categorical_encoding(
                X_processed, categorical_encoding, handle_high_cardinality, cardinality_threshold
            )
            preprocessing_info['steps_applied'].append(f"Categorical encoding: {categorical_encoding}")
            preprocessing_info['feature_changes']['after_encoding'] = len(X_processed.columns)
            preprocessing_info['feature_changes']['encoding_change'] = len(X_processed.columns) - original_cols
        
        # Feature scaling
        if scaling_method and scaling_method != "None":
            X_processed = self.scale_features(X_processed, scaling_method)
            preprocessing_info['steps_applied'].append(f"Feature scaling: {scaling_method}")
        
        # Handle class imbalance
        if resampling_method and resampling_method != "None":
            original_samples = len(X_processed)
            X_processed, y_processed = self.handle_class_imbalance(X_processed, y_processed, resampling_method)
            preprocessing_info['steps_applied'].append(f"Resampling: {resampling_method}")
            preprocessing_info['feature_changes']['after_resampling'] = len(X_processed)
            preprocessing_info['feature_changes']['sample_change'] = len(X_processed) - original_samples
        
        preprocessing_info['final_shape'] = X_processed.shape
        preprocessing_info['target_distribution'] = y_processed.value_counts().to_dict()
        
        return X_processed, y_processed, preprocessing_info
    
    def transform_new_data(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors."""
        X_transformed = X_new.copy()
        
        # Apply saved encoders
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                try:
                    X_transformed[col] = encoder.transform(X_transformed[col].astype(str))
                except:
                    # Handle unseen categories
                    X_transformed[col] = encoder.transform(['Unknown'] * len(X_transformed))
        
        # Apply saved scalers
        if 'feature_scaler' in self.scalers:
            numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                X_transformed[numeric_cols] = self.scalers['feature_scaler'].transform(X_transformed[numeric_cols])
        
        return X_transformed
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of applied preprocessing steps."""
        return {
            'scalers_fitted': list(self.scalers.keys()),
            'encoders_fitted': list(self.encoders.keys()),
            'feature_selectors_fitted': list(self.feature_selectors.keys())
        }
