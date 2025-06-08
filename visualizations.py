import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

class Visualizer:
    """Handles all visualization operations for the fraud detection application."""
    
    def __init__(self):
        # Set default color palette
        self.color_palette = px.colors.qualitative.Set3
        self.primary_color = '#1f77b4'
        self.secondary_color = '#ff7f0e'
    
    def plot_missing_data(self, df: pd.DataFrame) -> go.Figure:
        """Plot missing data heatmap."""
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if missing_data.empty:
            # Create empty plot if no missing data
            fig = go.Figure()
            fig.add_annotation(
                text="No missing data found!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Missing Data Analysis")
            return fig
        
        fig = go.Figure(data=go.Bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            marker_color=self.primary_color
        ))
        
        fig.update_layout(
            title="Missing Data by Column",
            xaxis_title="Number of Missing Values",
            yaxis_title="Columns",
            height=max(400, len(missing_data) * 30)
        )
        
        return fig
    
    def plot_distribution(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Plot distribution of a numerical column."""
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=df[column],
            name="Distribution",
            marker_color=self.primary_color,
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            showlegend=False
        )
        
        return fig
    
    def plot_categorical_distribution(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Plot distribution of a categorical column."""
        value_counts = df[column].value_counts().head(20)  # Show top 20 categories
        
        fig = go.Figure(data=go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            marker_color=self.color_palette[:len(value_counts)]
        ))
        
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Count",
            xaxis_tickangle=-45
        )
        
        return fig
    
    def plot_correlation_matrix(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """Plot correlation matrix heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            width=800,
            height=800
        )
        
        return fig
    
    def plot_target_distribution(self, df: pd.DataFrame, target_column: str) -> go.Figure:
        """Plot target variable distribution."""
        value_counts = df[target_column].value_counts()
        
        # Create pie chart
        fig = go.Figure(data=go.Pie(
            labels=value_counts.index,
            values=value_counts.values,
            hole=0.3,
            marker_colors=[self.primary_color, self.secondary_color]
        ))
        
        fig.update_layout(
            title=f"Distribution of {target_column}",
            annotations=[dict(text=target_column, x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        return fig
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> go.Figure:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Create labels
        labels = ['Legitimate', 'Fraudulent'] if len(cm) == 2 else [str(i) for i in range(len(cm))]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix - {model_name}",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=400,
            height=400
        )
        
        return fig
    
    def plot_metrics_comparison(self, metrics_df: pd.DataFrame) -> go.Figure:
        """Plot comparison of model metrics."""
        fig = go.Figure()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
        
        for metric in available_metrics:
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=metrics_df.index,
                y=metrics_df[metric],
                text=np.round(metrics_df[metric], 3),
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_roc_curves(self, roc_data: List[Dict[str, Any]]) -> go.Figure:
        """Plot ROC curves for multiple models."""
        fig = go.Figure()
        
        for i, data in enumerate(roc_data):
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_scores'])
            auc_score = np.trapz(tpr, fpr)
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f"{data['model_name']} (AUC = {auc_score:.3f})",
                line=dict(color=self.color_palette[i % len(self.color_palette)])
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='black')
        ))
        
        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_precision_recall_curves(self, pr_data: List[Dict[str, Any]]) -> go.Figure:
        """Plot precision-recall curves for multiple models."""
        fig = go.Figure()
        
        for i, data in enumerate(pr_data):
            precision, recall, _ = precision_recall_curve(data['y_true'], data['y_scores'])
            
            fig.add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode='lines',
                name=data['model_name'],
                line=dict(color=self.color_palette[i % len(self.color_palette)])
            ))
        
        fig.update_layout(
            title="Precision-Recall Curves",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_threshold_analysis(self, thresholds: np.ndarray, precisions: np.ndarray, 
                               recalls: np.ndarray, f1_scores: np.ndarray) -> go.Figure:
        """Plot threshold analysis."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=precisions,
            mode='lines',
            name='Precision',
            line=dict(color=self.primary_color)
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=recalls,
            mode='lines',
            name='Recall',
            line=dict(color=self.secondary_color)
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=f1_scores,
            mode='lines',
            name='F1 Score',
            line=dict(color='green')
        ))
        
        # Add optimal threshold line
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        fig.add_vline(
            x=optimal_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Optimal: {optimal_threshold:.3f}"
        )
        
        fig.update_layout(
            title="Threshold Analysis",
            xaxis_title="Threshold",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame) -> go.Figure:
        """Plot feature importance."""
        fig = go.Figure(data=go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color=self.primary_color
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=max(400, len(importance_df) * 25)
        )
        
        return fig
    
    def plot_prediction_probabilities(self, prob_df: pd.DataFrame) -> go.Figure:
        """Plot prediction probabilities."""
        fig = go.Figure(data=go.Bar(
            x=prob_df['Class'],
            y=prob_df['Probability'],
            marker_color=[self.primary_color if x == 'Legitimate' else self.secondary_color for x in prob_df['Class']],
            text=[f"{p:.2%}" for p in prob_df['Probability']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Prediction Probabilities",
            xaxis_title="Class",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_learning_curve(self, learning_data: Dict[str, np.ndarray]) -> go.Figure:
        """Plot learning curves."""
        fig = go.Figure()
        
        train_sizes = learning_data['train_sizes']
        train_mean = learning_data['train_scores_mean']
        train_std = learning_data['train_scores_std']
        val_mean = learning_data['val_scores_mean']
        val_std = learning_data['val_scores_std']
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color=self.primary_color),
            error_y=dict(type='data', array=train_std, visible=True)
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color=self.secondary_color),
            error_y=dict(type='data', array=val_std, visible=True)
        ))
        
        fig.update_layout(
            title="Learning Curve",
            xaxis_title="Training Set Size",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_class_distribution_comparison(self, before_df: pd.DataFrame, after_df: pd.DataFrame, 
                                         target_column: str) -> go.Figure:
        """Plot class distribution before and after resampling."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Before Resampling', 'After Resampling'),
            specs=[[{'type': 'pie'}, {'type': 'pie'}]]
        )
        
        # Before resampling
        before_counts = before_df[target_column].value_counts()
        fig.add_trace(
            go.Pie(
                labels=before_counts.index,
                values=before_counts.values,
                name="Before"
            ),
            row=1, col=1
        )
        
        # After resampling
        after_counts = after_df[target_column].value_counts()
        fig.add_trace(
            go.Pie(
                labels=after_counts.index,
                values=after_counts.values,
                name="After"
            ),
            row=1, col=2
        )
        
        fig.update_layout(title="Class Distribution Comparison")
        
        return fig
    
    def plot_model_comparison_radar(self, metrics_df: pd.DataFrame) -> go.Figure:
        """Plot radar chart for model comparison."""
        fig = go.Figure()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
        
        for i, (model_name, row) in enumerate(metrics_df.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=[row[metric] for metric in available_metrics],
                theta=[metric.replace('_', ' ').title() for metric in available_metrics],
                fill='toself',
                name=model_name,
                line_color=self.color_palette[i % len(self.color_palette)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Model Performance Radar Chart"
        )
        
        return fig
    
    def plot_cross_validation_results(self, cv_results: Dict[str, np.ndarray]) -> go.Figure:
        """Plot cross-validation results."""
        fig = go.Figure()
        
        for metric, scores in cv_results.items():
            fig.add_trace(go.Box(
                y=scores,
                name=metric.replace('_', ' ').title(),
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title="Cross-Validation Results",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_dashboard_summary(self, dataset_info: Dict[str, Any], 
                               model_results: Dict[str, Any]) -> go.Figure:
        """Create a summary dashboard figure."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Dataset Overview', 
                'Model Performance', 
                'Feature Importance', 
                'Class Distribution'
            ),
            specs=[
                [{'type': 'table'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'pie'}]
            ]
        )
        
        # This would be implemented based on specific dashboard requirements
        # For now, return a simple placeholder
        fig.add_annotation(
            text="Dashboard Summary",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        
        fig.update_layout(title="Fraud Detection Dashboard")
        
        return fig
