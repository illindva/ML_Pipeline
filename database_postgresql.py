import os
import json
import pickle
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class Dataset(Base):
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    upload_date = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer)
    columns_info = Column(Text)  # JSON string
    data_blob = Column(LargeBinary)  # Pickled DataFrame

class Analysis(Base):
    __tablename__ = 'analysis'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, nullable=False)
    analysis_type = Column(String(100), nullable=False)
    results = Column(Text)  # JSON string
    created_date = Column(DateTime, default=datetime.utcnow)

class Model(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, nullable=False)
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(100), nullable=False)
    model_data = Column(LargeBinary)  # Pickled model
    metrics = Column(Text)  # JSON string
    parameters = Column(Text)  # JSON string
    created_date = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, nullable=False)
    input_data = Column(Text)  # JSON string
    prediction_result = Column(Text)  # JSON string
    confidence_score = Column(Float)
    created_date = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Manages PostgreSQL database operations for the fraud detection application."""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")
        
        self.engine = create_engine(self.database_url)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Create tables
        Base.metadata.create_all(self.engine)
        logger.info("Database initialized successfully")
    
    def save_dataset(self, name: str, description: str, data: pd.DataFrame) -> int:
        """Save a dataset to the database."""
        try:
            # Prepare dataset metadata
            columns_info = {
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'shape': list(data.shape),
                'memory_usage': int(data.memory_usage(deep=True).sum())
            }
            
            # Pickle the DataFrame
            data_blob = pickle.dumps(data)
            
            dataset = Dataset(
                name=name,
                description=description,
                file_size=len(data_blob),
                columns_info=json.dumps(columns_info),
                data_blob=data_blob
            )
            
            self.session.add(dataset)
            self.session.commit()
            
            logger.info(f"Dataset '{name}' saved with ID: {dataset.id}")
            return dataset.id
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving dataset: {str(e)}")
            raise
    
    def load_dataset(self, dataset_id: int) -> pd.DataFrame:
        """Load a dataset from the database."""
        try:
            dataset = self.session.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset with ID {dataset_id} not found")
            
            # Unpickle the DataFrame
            data = pickle.loads(dataset.data_blob)
            logger.info(f"Dataset loaded: {dataset.name}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def get_all_datasets(self) -> List[Tuple]:
        """Get all datasets metadata."""
        try:
            datasets = self.session.query(Dataset).all()
            result = []
            
            for dataset in datasets:
                columns_info = json.loads(dataset.columns_info)
                result.append((
                    dataset.id,
                    dataset.name,
                    dataset.description,
                    dataset.upload_date,
                    dataset.file_size,
                    columns_info['shape']
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting datasets: {str(e)}")
            raise
    
    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset and all related analysis."""
        try:
            # Delete related records first
            self.session.query(Prediction).filter(
                Prediction.model_id.in_(
                    self.session.query(Model.id).filter(Model.dataset_id == dataset_id)
                )
            ).delete(synchronize_session=False)
            
            self.session.query(Model).filter(Model.dataset_id == dataset_id).delete()
            self.session.query(Analysis).filter(Analysis.dataset_id == dataset_id).delete()
            
            # Delete the dataset
            deleted = self.session.query(Dataset).filter(Dataset.id == dataset_id).delete()
            self.session.commit()
            
            logger.info(f"Dataset {dataset_id} deleted successfully")
            return deleted > 0
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting dataset: {str(e)}")
            raise
    
    def save_analysis(self, dataset_id: int, analysis_type: str, results: Dict[str, Any]) -> int:
        """Save analysis results to the database."""
        try:
            analysis = Analysis(
                dataset_id=dataset_id,
                analysis_type=analysis_type,
                results=json.dumps(results, default=str)
            )
            
            self.session.add(analysis)
            self.session.commit()
            
            logger.info(f"Analysis saved with ID: {analysis.id}")
            return analysis.id
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving analysis: {str(e)}")
            raise
    
    def get_analysis(self, dataset_id: int, analysis_type: str = None) -> List[Dict[str, Any]]:
        """Get analysis results from the database."""
        try:
            query = self.session.query(Analysis).filter(Analysis.dataset_id == dataset_id)
            
            if analysis_type:
                query = query.filter(Analysis.analysis_type == analysis_type)
            
            analyses = query.all()
            
            result = []
            for analysis in analyses:
                result.append({
                    'id': analysis.id,
                    'analysis_type': analysis.analysis_type,
                    'results': json.loads(analysis.results),
                    'created_date': analysis.created_date
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting analysis: {str(e)}")
            raise
    
    def save_model(self, dataset_id: int, model_name: str, model_type: str, 
                   model_data: Any, metrics: Dict[str, Any] = None, 
                   parameters: Dict[str, Any] = None) -> int:
        """Save a trained model to the database."""
        try:
            model_blob = pickle.dumps(model_data)
            
            model = Model(
                dataset_id=dataset_id,
                model_name=model_name,
                model_type=model_type,
                model_data=model_blob,
                metrics=json.dumps(metrics, default=str) if metrics else None,
                parameters=json.dumps(parameters, default=str) if parameters else None
            )
            
            self.session.add(model)
            self.session.commit()
            
            logger.info(f"Model '{model_name}' saved with ID: {model.id}")
            return model.id
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_id: int) -> Dict[str, Any]:
        """Load a model from the database."""
        try:
            model = self.session.query(Model).filter(Model.id == model_id).first()
            if not model:
                raise ValueError(f"Model with ID {model_id} not found")
            
            model_data = pickle.loads(model.model_data)
            
            result = {
                'id': model.id,
                'dataset_id': model.dataset_id,
                'model_name': model.model_name,
                'model_type': model.model_type,
                'model': model_data,
                'metrics': json.loads(model.metrics) if model.metrics else None,
                'parameters': json.loads(model.parameters) if model.parameters else None,
                'created_date': model.created_date
            }
            
            logger.info(f"Model loaded: {model.model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_all_models(self, dataset_id: int = None) -> List[Tuple]:
        """Get all models metadata."""
        try:
            query = self.session.query(Model)
            
            if dataset_id:
                query = query.filter(Model.dataset_id == dataset_id)
            
            models = query.all()
            
            result = []
            for model in models:
                metrics = json.loads(model.metrics) if model.metrics else {}
                result.append((
                    model.id,
                    model.dataset_id,
                    model.model_name,
                    model.model_type,
                    model.created_date,
                    metrics.get('accuracy', 'N/A')
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            raise
    
    def save_prediction(self, model_id: int, input_data: Dict[str, Any], 
                       prediction_result: Dict[str, Any], confidence_score: float = None) -> int:
        """Save a prediction result to the database."""
        try:
            prediction = Prediction(
                model_id=model_id,
                input_data=json.dumps(input_data, default=str),
                prediction_result=json.dumps(prediction_result, default=str),
                confidence_score=confidence_score
            )
            
            self.session.add(prediction)
            self.session.commit()
            
            logger.info(f"Prediction saved with ID: {prediction.id}")
            return prediction.id
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving prediction: {str(e)}")
            raise
    
    def get_predictions(self, model_id: int = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get prediction history."""
        try:
            query = self.session.query(Prediction)
            
            if model_id:
                query = query.filter(Prediction.model_id == model_id)
            
            predictions = query.order_by(Prediction.created_date.desc()).limit(limit).all()
            
            result = []
            for prediction in predictions:
                result.append({
                    'id': prediction.id,
                    'model_id': prediction.model_id,
                    'input_data': json.loads(prediction.input_data),
                    'prediction_result': json.loads(prediction.prediction_result),
                    'confidence_score': prediction.confidence_score,
                    'created_date': prediction.created_date
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            raise
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        try:
            stats = {
                'datasets': self.session.query(Dataset).count(),
                'analyses': self.session.query(Analysis).count(),
                'models': self.session.query(Model).count(),
                'predictions': self.session.query(Prediction).count()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            raise
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old data from the database."""
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Delete old predictions
            old_predictions = self.session.query(Prediction).filter(
                Prediction.created_date < cutoff_date
            ).delete()
            
            # Delete old analyses
            old_analyses = self.session.query(Analysis).filter(
                Analysis.created_date < cutoff_date
            ).delete()
            
            self.session.commit()
            
            logger.info(f"Cleaned up {old_predictions} predictions and {old_analyses} analyses")
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error cleaning up data: {str(e)}")
            raise
    
    def close(self):
        """Close the database session."""
        self.session.close()
    
    def __del__(self):
        """Destructor to ensure session is closed."""
        try:
            self.session.close()
        except:
            pass