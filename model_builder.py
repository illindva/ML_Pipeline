import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report,
                           roc_curve, precision_recall_curve)
import xgboost as xgb
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelBuilder:
    """Handles all machine learning model operations for the fraud detection application."""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {
            "Logistic Regression": {
                'class': LogisticRegression,
                'default_params': {'random_state': 42, 'max_iter': 1000}
            },
            "Random Forest": {
                'class': RandomForestClassifier,
                'default_params': {'random_state': 42, 'n_estimators': 100}
            },
            "XGBoost": {
                'class': xgb.XGBClassifier,
                'default_params': {'random_state': 42, 'eval_metric': 'logloss'}
            },
            "SVM": {
                'class': SVC,
                'default_params': {'random_state': 42, 'probability': True}
            },
            "Naive Bayes": {
                'class': GaussianNB,
                'default_params': {}
            }
        }
    
    def create_train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                               test_size: float = 0.3, random_state: int = 42,
                               stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create train-test split."""
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_param
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   cross_validation: bool = True, cv_folds: int = 5) -> Dict[str, Any]:
        """Train a specific model."""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not supported")
        
        # Get model configuration
        model_config = self.model_configs[model_name]
        model_class = model_config['class']
        default_params = model_config['default_params']
        
        # Create and train model
        model = model_class(**default_params)
        model.fit(X_train, y_train)
        
        model_info = {
            'model': model,
            'model_name': model_name,
            'training_samples': len(X_train),
            'features': X_train.columns.tolist()
        }
        
        # Cross-validation
        if cross_validation:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            model_info['cv_scores'] = cv_scores
            model_info['cv_mean'] = cv_scores.mean()
            model_info['cv_std'] = cv_scores.std()
        
        return model_info
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance_scores = np.abs(model.coef_[0])
        else:
            # Models without direct feature importance
            return pd.DataFrame({'feature': feature_names, 'importance': [0] * len(feature_names)})
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_hyperparameter_space(self, model_name: str) -> Dict[str, List]:
        """Get hyperparameter space for a specific model."""
        param_spaces = {
            "Logistic Regression": {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            "Random Forest": {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            "XGBoost": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            "SVM": {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            },
            "Naive Bayes": {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        }
        
        return param_spaces.get(model_name, {})
    
    def tune_hyperparameters(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                           param_space: Dict[str, List], method: str = "grid_search",
                           scoring: str = "accuracy", cv: int = 5, n_iter: int = 50) -> Tuple[Any, Dict, float]:
        """Perform hyperparameter tuning."""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not supported")
        
        # Get model configuration
        model_config = self.model_configs[model_name]
        model_class = model_config['class']
        default_params = model_config['default_params']
        
        # Create base model
        base_model = model_class(**default_params)
        
        # Perform hyperparameter search
        if method == "grid_search":
            search = GridSearchCV(
                base_model, param_space, cv=cv, scoring=scoring, 
                n_jobs=-1, verbose=0
            )
        elif method == "random_search":
            search = RandomizedSearchCV(
                base_model, param_space, n_iter=n_iter, cv=cv, 
                scoring=scoring, n_jobs=-1, verbose=0, random_state=42
            )
        else:
            raise ValueError("Method must be 'grid_search' or 'random_search'")
        
        # Fit the search
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Evaluate model performance with various metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC if probabilities are available
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                else:
                    # Multi-class classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def get_classification_report(self, y_true: pd.Series, y_pred: np.ndarray) -> str:
        """Get detailed classification report."""
        return classification_report(y_true, y_pred)
    
    def get_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    def analyze_thresholds(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Analyze different prediction thresholds."""
        # Calculate precision-recall curve
        precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_pred_proba)
        
        # Calculate F1 scores for different thresholds
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        # Use precision-recall thresholds, but we need to match the length
        # precision_recall_curve returns n_thresholds - 1 values
        thresholds = np.append(thresholds_pr, 1.0)
        
        return thresholds, precisions, recalls, f1_scores
    
    def get_roc_curve_data(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve data."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        return fpr, tpr, thresholds
    
    def predict_with_threshold(self, model: Any, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Make predictions with custom threshold."""
        y_pred_proba = model.predict_proba(X)[:, 1]
        return (y_pred_proba >= threshold).astype(int)
    
    def get_model_summary(self, model: Any, model_name: str, X_train: pd.DataFrame, 
                         y_train: pd.Series) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        summary = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'training_samples': len(X_train),
            'features': X_train.columns.tolist(),
            'n_features': len(X_train.columns)
        }
        
        # Model-specific information
        if hasattr(model, 'get_params'):
            summary['parameters'] = model.get_params()
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            importance_df = self.get_feature_importance(model, X_train.columns.tolist())
            summary['top_features'] = importance_df.head(10).to_dict('records')
        
        return summary
    
    def compare_models(self, models_dict: Dict[str, Any], X_test: pd.DataFrame, 
                      y_test: pd.Series) -> pd.DataFrame:
        """Compare multiple models on test data."""
        comparison_results = []
        
        for model_name, model_info in models_dict.items():
            model = model_info.get('tuned_model', model_info['model'])
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = None
            
            # Calculate metrics
            metrics = self.evaluate_model(y_test, y_pred, y_pred_proba)
            
            # Add model name to metrics
            metrics['model_name'] = model_name
            comparison_results.append(metrics)
        
        return pd.DataFrame(comparison_results).set_index('model_name')
    
    def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                           cv_folds: int = 5, scoring: List[str] = None) -> Dict[str, np.ndarray]:
        """Perform cross-validation with multiple scoring metrics."""
        if scoring is None:
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = {}
        
        for score in scoring:
            try:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=score)
                cv_results[score] = scores
            except Exception as e:
                print(f"Error calculating {score}: {e}")
                cv_results[score] = np.array([0] * cv_folds)
        
        return cv_results
    
    def get_learning_curve_data(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, np.ndarray]:
        """Get learning curve data to analyze model performance vs training size."""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=42
        )
        
        return {
            'train_sizes': train_sizes,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }
    
    def explain_prediction(self, model: Any, X_sample: pd.DataFrame, 
                          feature_names: List[str]) -> Dict[str, Any]:
        """Explain a single prediction (basic implementation)."""
        prediction = model.predict(X_sample)[0]
        
        try:
            prediction_proba = model.predict_proba(X_sample)[0]
        except:
            prediction_proba = None
        
        explanation = {
            'prediction': int(prediction),
            'prediction_proba': prediction_proba.tolist() if prediction_proba is not None else None,
            'input_features': dict(zip(feature_names, X_sample.iloc[0].values))
        }
        
        # Add feature importance if available
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            importance_df = self.get_feature_importance(model, feature_names)
            explanation['feature_importance'] = importance_df.to_dict('records')
        
        return explanation
