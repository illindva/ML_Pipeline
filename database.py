import sqlite3
import pandas as pd
import json
import pickle
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import os

class DatabaseManager:
    """Manages SQLite database operations for the fraud detection application."""
    
    def __init__(self, db_path: str = "fraud_detection.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER,
                    num_rows INTEGER,
                    num_columns INTEGER,
                    data BLOB NOT NULL
                )
            """)
            
            # Create analysis table for storing various analysis results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    analysis_type TEXT NOT NULL,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    results TEXT NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                )
            """)
            
            # Create models table for storing trained models
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_data BLOB NOT NULL,
                    metrics TEXT,
                    parameters TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                )
            """)
            
            # Create predictions table for storing prediction results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    input_data TEXT NOT NULL,
                    prediction_result TEXT NOT NULL,
                    confidence_score REAL,
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            """)
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def save_dataset(self, name: str, description: str, data: pd.DataFrame) -> int:
        """Save a dataset to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Serialize the dataframe
            data_blob = pickle.dumps(data)
            
            # Insert dataset
            cursor.execute("""
                INSERT INTO datasets (name, description, file_size, num_rows, num_columns, data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                name,
                description,
                len(data_blob),
                len(data),
                len(data.columns),
                data_blob
            ))
            
            dataset_id = cursor.lastrowid
            conn.commit()
            
            return dataset_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def load_dataset(self, dataset_id: int) -> pd.DataFrame:
        """Load a dataset from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT data FROM datasets WHERE id = ?", (dataset_id,))
            result = cursor.fetchone()
            
            if result is None:
                raise ValueError(f"Dataset with ID {dataset_id} not found")
            
            # Deserialize the dataframe
            data = pickle.loads(result[0])
            return data
            
        except Exception as e:
            raise e
        finally:
            conn.close()
    
    def get_all_datasets(self) -> List[Tuple]:
        """Get all datasets metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, name, description, upload_date, num_rows, num_columns
                FROM datasets
                ORDER BY upload_date DESC
            """)
            
            return cursor.fetchall()
            
        except Exception as e:
            raise e
        finally:
            conn.close()
    
    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset and all related analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete related analysis
            cursor.execute("DELETE FROM analysis WHERE dataset_id = ?", (dataset_id,))
            
            # Delete related models
            cursor.execute("DELETE FROM models WHERE dataset_id = ?", (dataset_id,))
            
            # Delete dataset
            cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            
            if cursor.rowcount > 0:
                conn.commit()
                return True
            else:
                return False
                
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def save_analysis(self, dataset_id: int, analysis_type: str, results: Dict[str, Any]) -> int:
        """Save analysis results to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Convert results to JSON string
            results_json = json.dumps(results, default=str)
            
            cursor.execute("""
                INSERT INTO analysis (dataset_id, analysis_type, results)
                VALUES (?, ?, ?)
            """, (dataset_id, analysis_type, results_json))
            
            analysis_id = cursor.lastrowid
            conn.commit()
            
            return analysis_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_analysis(self, dataset_id: int, analysis_type: str = None) -> List[Dict[str, Any]]:
        """Get analysis results from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if analysis_type:
                cursor.execute("""
                    SELECT id, analysis_type, analysis_date, results
                    FROM analysis
                    WHERE dataset_id = ? AND analysis_type = ?
                    ORDER BY analysis_date DESC
                """, (dataset_id, analysis_type))
            else:
                cursor.execute("""
                    SELECT id, analysis_type, analysis_date, results
                    FROM analysis
                    WHERE dataset_id = ?
                    ORDER BY analysis_date DESC
                """, (dataset_id,))
            
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            analysis_list = []
            for result in results:
                analysis_list.append({
                    'id': result[0],
                    'analysis_type': result[1],
                    'analysis_date': result[2],
                    'results': json.loads(result[3])
                })
            
            return analysis_list
            
        except Exception as e:
            raise e
        finally:
            conn.close()
    
    def save_model(self, dataset_id: int, model_name: str, model_type: str, 
                   model_data: Any, metrics: Dict[str, Any] = None, 
                   parameters: Dict[str, Any] = None) -> int:
        """Save a trained model to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Serialize the model
            model_blob = pickle.dumps(model_data)
            
            # Convert metrics and parameters to JSON
            metrics_json = json.dumps(metrics, default=str) if metrics else None
            parameters_json = json.dumps(parameters, default=str) if parameters else None
            
            cursor.execute("""
                INSERT INTO models (dataset_id, model_name, model_type, model_data, metrics, parameters)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (dataset_id, model_name, model_type, model_blob, metrics_json, parameters_json))
            
            model_id = cursor.lastrowid
            conn.commit()
            
            return model_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def load_model(self, model_id: int) -> Dict[str, Any]:
        """Load a model from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT model_name, model_type, model_data, metrics, parameters, creation_date
                FROM models
                WHERE id = ?
            """, (model_id,))
            
            result = cursor.fetchone()
            
            if result is None:
                raise ValueError(f"Model with ID {model_id} not found")
            
            # Deserialize the model
            model_data = pickle.loads(result[2])
            
            return {
                'model_name': result[0],
                'model_type': result[1],
                'model_data': model_data,
                'metrics': json.loads(result[3]) if result[3] else None,
                'parameters': json.loads(result[4]) if result[4] else None,
                'creation_date': result[5]
            }
            
        except Exception as e:
            raise e
        finally:
            conn.close()
    
    def get_all_models(self, dataset_id: int = None) -> List[Tuple]:
        """Get all models metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if dataset_id:
                cursor.execute("""
                    SELECT id, model_name, model_type, creation_date, dataset_id
                    FROM models
                    WHERE dataset_id = ?
                    ORDER BY creation_date DESC
                """, (dataset_id,))
            else:
                cursor.execute("""
                    SELECT id, model_name, model_type, creation_date, dataset_id
                    FROM models
                    ORDER BY creation_date DESC
                """)
            
            return cursor.fetchall()
            
        except Exception as e:
            raise e
        finally:
            conn.close()
    
    def save_prediction(self, model_id: int, input_data: Dict[str, Any], 
                       prediction_result: Dict[str, Any], confidence_score: float = None) -> int:
        """Save a prediction result to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Convert to JSON strings
            input_json = json.dumps(input_data, default=str)
            result_json = json.dumps(prediction_result, default=str)
            
            cursor.execute("""
                INSERT INTO predictions (model_id, input_data, prediction_result, confidence_score)
                VALUES (?, ?, ?, ?)
            """, (model_id, input_json, result_json, confidence_score))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            
            return prediction_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_predictions(self, model_id: int = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get prediction history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if model_id:
                cursor.execute("""
                    SELECT id, model_id, prediction_date, input_data, prediction_result, confidence_score
                    FROM predictions
                    WHERE model_id = ?
                    ORDER BY prediction_date DESC
                    LIMIT ?
                """, (model_id, limit))
            else:
                cursor.execute("""
                    SELECT id, model_id, prediction_date, input_data, prediction_result, confidence_score
                    FROM predictions
                    ORDER BY prediction_date DESC
                    LIMIT ?
                """, (limit,))
            
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            predictions_list = []
            for result in results:
                predictions_list.append({
                    'id': result[0],
                    'model_id': result[1],
                    'prediction_date': result[2],
                    'input_data': json.loads(result[3]),
                    'prediction_result': json.loads(result[4]),
                    'confidence_score': result[5]
                })
            
            return predictions_list
            
        except Exception as e:
            raise e
        finally:
            conn.close()
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Count datasets
            cursor.execute("SELECT COUNT(*) FROM datasets")
            stats['total_datasets'] = cursor.fetchone()[0]
            
            # Count analysis
            cursor.execute("SELECT COUNT(*) FROM analysis")
            stats['total_analysis'] = cursor.fetchone()[0]
            
            # Count models
            cursor.execute("SELECT COUNT(*) FROM models")
            stats['total_models'] = cursor.fetchone()[0]
            
            # Count predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            stats['total_predictions'] = cursor.fetchone()[0]
            
            # Database size
            stats['database_size'] = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return stats
            
        except Exception as e:
            raise e
        finally:
            conn.close()
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old data from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete old predictions
            cursor.execute("""
                DELETE FROM predictions
                WHERE prediction_date < datetime('now', '-{} days')
            """.format(days_old))
            
            # Delete old analysis (keep recent ones)
            cursor.execute("""
                DELETE FROM analysis
                WHERE analysis_date < datetime('now', '-{} days')
            """.format(days_old))
            
            conn.commit()
            
            return {
                'predictions_deleted': cursor.rowcount,
                'cleanup_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
