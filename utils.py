import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import re
import json
from datetime import datetime, timedelta
import hashlib
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

class Utils:
    """Utility functions for the fraud detection application."""
    
    def __init__(self):
        pass
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = {
            'Total Rows': len(df),
            'Total Columns': len(df.columns),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicate Rows': df.duplicated().sum(),
            'Memory Usage (MB)': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'Numerical Columns': len(df.select_dtypes(include=[np.number]).columns),
            'Categorical Columns': len(df.select_dtypes(include=['object']).columns),
            'Datetime Columns': len(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # Add data quality score
        total_cells = len(df) * len(df.columns)
        missing_ratio = stats['Missing Values'] / total_cells if total_cells > 0 else 0
        duplicate_ratio = stats['Duplicate Rows'] / len(df) if len(df) > 0 else 0
        
        quality_score = max(0, 100 - (missing_ratio * 50) - (duplicate_ratio * 30))
        stats['Data Quality Score'] = round(quality_score, 1)
        
        return stats
    
    def analyze_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze missing data patterns."""
        missing_info = []
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / len(df)) * 100
                missing_info.append({
                    'Column': column,
                    'Missing Count': missing_count,
                    'Missing Percentage': round(missing_percentage, 2),
                    'Data Type': str(df[column].dtype)
                })
        
        if missing_info:
            return pd.DataFrame(missing_info).sort_values('Missing Count', ascending=False)
        else:
            return pd.DataFrame()
    
    def find_high_correlation_pairs(self, correlation_matrix: pd.DataFrame, 
                                   threshold: float = 0.8) -> pd.DataFrame:
        """Find pairs of features with high correlation."""
        high_corr_pairs = []
        
        # Get upper triangle of correlation matrix
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find high correlation pairs
        for column in upper_triangle.columns:
            for index in upper_triangle.index:
                corr_value = upper_triangle.loc[index, column]
                if pd.notna(corr_value) and abs(corr_value) >= threshold:
                    high_corr_pairs.append({
                        'Feature 1': index,
                        'Feature 2': column,
                        'Correlation': round(corr_value, 4),
                        'Absolute Correlation': round(abs(corr_value), 4)
                    })
        
        if high_corr_pairs:
            return pd.DataFrame(high_corr_pairs).sort_values('Absolute Correlation', ascending=False)
        else:
            return pd.DataFrame()
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'IQR') -> Dict[str, Any]:
        """Detect outliers in numerical columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            if method == 'IQR':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
            elif method == 'Z-Score':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df[z_scores > 3]
            
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': round((len(outliers) / len(df)) * 100, 2),
                'lower_bound': lower_bound if method == 'IQR' else None,
                'upper_bound': upper_bound if method == 'IQR' else None
            }
        
        return outlier_info
    
    def suggest_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Suggest optimal data types for columns."""
        suggestions = {}
        
        for col in df.columns:
            current_dtype = str(df[col].dtype)
            
            # Skip if already optimal
            if current_dtype in ['datetime64[ns]', 'category']:
                continue
            
            # Check for date patterns
            if self._is_date_column(df[col]):
                suggestions[col] = 'datetime64[ns]'
                continue
            
            # Check for categorical data
            if current_dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    suggestions[col] = 'category'
                continue
            
            # Check for numeric optimizations
            if current_dtype in ['int64', 'float64']:
                if current_dtype == 'int64':
                    min_val = df[col].min()
                    max_val = df[col].max()
                    
                    if min_val >= 0 and max_val <= 255:
                        suggestions[col] = 'uint8'
                    elif min_val >= -128 and max_val <= 127:
                        suggestions[col] = 'int8'
                    elif min_val >= -32768 and max_val <= 32767:
                        suggestions[col] = 'int16'
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        suggestions[col] = 'int32'
        
        return suggestions
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a series contains date-like values."""
        if series.dtype == 'object':
            # Sample a few non-null values
            sample_values = series.dropna().head(10)
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                r'\d{4}/\d{2}/\d{2}'   # YYYY/MM/DD
            ]
            
            for value in sample_values:
                str_value = str(value)
                for pattern in date_patterns:
                    if re.search(pattern, str_value):
                        return True
        
        return False
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        report = {
            'dataset_overview': self.get_dataset_statistics(df),
            'missing_data': self.analyze_missing_data(df).to_dict('records') if not self.analyze_missing_data(df).empty else [],
            'outliers': self.detect_outliers(df),
            'data_type_suggestions': self.suggest_data_types(df),
            'column_analysis': {}
        }
        
        # Analyze each column
        for col in df.columns:
            col_analysis = {
                'dtype': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'unique_values': df[col].nunique(),
                'memory_usage': df[col].memory_usage(deep=True)
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_analysis.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std()
                })
            elif df[col].dtype == 'object':
                value_counts = df[col].value_counts()
                col_analysis.update({
                    'most_frequent': value_counts.index[0] if not value_counts.empty else None,
                    'most_frequent_count': value_counts.iloc[0] if not value_counts.empty else 0,
                    'average_length': df[col].astype(str).str.len().mean()
                })
            
            report['column_analysis'][col] = col_analysis
        
        return report
    
    def validate_dataset_for_ml(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Validate dataset readiness for machine learning."""
        validation_results = {
            'is_ready': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check minimum requirements
        if len(df) < 100:
            validation_results['issues'].append("Dataset has fewer than 100 rows")
            validation_results['recommendations'].append("Consider collecting more data")
            validation_results['is_ready'] = False
        
        if len(df.columns) < 2:
            validation_results['issues'].append("Dataset has fewer than 2 columns")
            validation_results['is_ready'] = False
        
        # Check missing data
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 30:
            validation_results['issues'].append(f"High missing data percentage: {missing_percentage:.1f}%")
            validation_results['recommendations'].append("Consider data imputation or feature removal")
        
        # Check target column if specified
        if target_column and target_column in df.columns:
            target_missing = df[target_column].isnull().sum()
            if target_missing > 0:
                validation_results['issues'].append(f"Target column has {target_missing} missing values")
                validation_results['is_ready'] = False
            
            # Check class balance for classification
            if df[target_column].dtype in ['object', 'int64'] and df[target_column].nunique() <= 10:
                value_counts = df[target_column].value_counts()
                min_class_ratio = value_counts.min() / value_counts.max()
                if min_class_ratio < 0.1:
                    validation_results['issues'].append("Severe class imbalance detected")
                    validation_results['recommendations'].append("Consider resampling techniques")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            validation_results['issues'].append(f"Constant columns found: {constant_cols}")
            validation_results['recommendations'].append("Remove constant columns")
        
        # Check for high cardinality categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > len(df) * 0.8]
        if high_cardinality_cols:
            validation_results['issues'].append(f"High cardinality categorical columns: {high_cardinality_cols}")
            validation_results['recommendations'].append("Consider feature engineering or removal")
        
        return validation_results
    
    def create_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a summary of all features."""
        feature_summary = []
        
        for col in df.columns:
            summary = {
                'Feature': col,
                'Data Type': str(df[col].dtype),
                'Missing Values': df[col].isnull().sum(),
                'Missing %': round((df[col].isnull().sum() / len(df)) * 100, 2),
                'Unique Values': df[col].nunique(),
                'Unique %': round((df[col].nunique() / len(df)) * 100, 2)
            }
            
            if df[col].dtype in ['int64', 'float64']:
                summary.update({
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Mean': round(df[col].mean(), 2),
                    'Std': round(df[col].std(), 2)
                })
            else:
                summary.update({
                    'Min': 'N/A',
                    'Max': 'N/A',
                    'Mean': 'N/A',
                    'Std': 'N/A'
                })
            
            feature_summary.append(summary)
        
        return pd.DataFrame(feature_summary)
    
    def export_to_csv(self, df: pd.DataFrame, filename: str = None) -> str:
        """Export dataframe to CSV string."""
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return df.to_csv(index=False)
    
    def export_to_excel(self, df: pd.DataFrame, filename: str = None) -> bytes:
        """Export dataframe to Excel bytes."""
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
        
        return output.getvalue()
    
    def calculate_vif(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Variance Inflation Factor for numerical columns."""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # Select only numerical columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return pd.DataFrame()
        
        # Remove columns with zero variance
        numeric_df = numeric_df.loc[:, numeric_df.var() > 0]
        
        if len(numeric_df.columns) < 2:
            return pd.DataFrame()
        
        try:
            vif_data = pd.DataFrame()
            vif_data["Feature"] = numeric_df.columns
            vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) 
                              for i in range(len(numeric_df.columns))]
            
            return vif_data.sort_values('VIF', ascending=False)
        except:
            return pd.DataFrame()
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names for better processing."""
        df_cleaned = df.copy()
        
        # Clean column names
        df_cleaned.columns = (df_cleaned.columns
                             .str.replace(' ', '_')
                             .str.replace('[^\w]', '', regex=True)
                             .str.lower())
        
        return df_cleaned
    
    def get_memory_usage_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed memory usage report."""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        report = {
            'total_memory_mb': round(total_memory / 1024**2, 2),
            'column_memory': {},
            'optimization_potential': {}
        }
        
        for col, memory in memory_usage.items():
            if col != 'Index':
                report['column_memory'][col] = {
                    'memory_mb': round(memory / 1024**2, 4),
                    'percentage': round((memory / total_memory) * 100, 2)
                }
        
        # Check optimization potential
        suggestions = self.suggest_data_types(df)
        for col, suggested_type in suggestions.items():
            current_memory = memory_usage[col]
            
            # Estimate memory savings (simplified)
            if suggested_type in ['int8', 'uint8']:
                potential_savings = current_memory * 0.875  # Rough estimate
            elif suggested_type in ['int16']:
                potential_savings = current_memory * 0.75
            elif suggested_type == 'category':
                potential_savings = current_memory * 0.5
            else:
                potential_savings = 0
            
            if potential_savings > 0:
                report['optimization_potential'][col] = {
                    'suggested_type': suggested_type,
                    'potential_savings_mb': round(potential_savings / 1024**2, 4)
                }
        
        return report
    
    def generate_model_comparison_summary(self, models_results: Dict[str, Any]) -> pd.DataFrame:
        """Generate a summary comparison of model results."""
        summary_data = []
        
        for model_name, results in models_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                summary = {
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0),
                    'ROC-AUC': metrics.get('roc_auc', 0)
                }
                
                # Add training time if available
                if 'training_time' in results:
                    summary['Training Time (s)'] = results['training_time']
                
                # Add cross-validation score if available
                if 'cv_mean' in results:
                    summary['CV Score'] = results['cv_mean']
                
                summary_data.append(summary)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            # Round numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].round(4)
            return df
        else:
            return pd.DataFrame()
    
    def hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate a hash for a dataframe to track changes."""
        # Convert dataframe to string representation
        df_string = df.to_string()
        
        # Create hash
        hash_object = hashlib.md5(df_string.encode())
        return hash_object.hexdigest()
    
    def log_operation(self, operation: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Log an operation with timestamp and details."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details
        }
        
        return log_entry
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = int(np.floor(np.log(size_bytes) / np.log(1024)))
        p = np.power(1024, i)
        s = round(size_bytes / p, 2)
        
        return f"{s} {size_names[i]}"
    
    def validate_file_upload(self, file) -> Dict[str, Any]:
        """Validate uploaded file."""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        if file is None:
            validation['is_valid'] = False
            validation['errors'].append("No file uploaded")
            return validation
        
        # Get file info
        validation['file_info'] = {
            'name': file.name,
            'size': file.size if hasattr(file, 'size') else 0,
            'type': file.type if hasattr(file, 'type') else 'unknown'
        }
        
        # Check file extension
        allowed_extensions = ['.csv', '.xlsx', '.xls']
        file_extension = '.' + file.name.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            validation['is_valid'] = False
            validation['errors'].append(f"Unsupported file type: {file_extension}")
        
        # Check file size (max 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        if hasattr(file, 'size') and file.size > max_size:
            validation['is_valid'] = False
            validation['errors'].append(f"File too large: {self.format_file_size(file.size)} (max: 100MB)")
        
        return validation
