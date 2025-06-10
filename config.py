"""
Configuration file for database selection and application settings
"""

import os

# Database Configuration
# Set DATABASE_TYPE to 'sqlite' or 'postgresql'
DATABASE_TYPE = os.getenv('DATABASE_TYPE', 'sqlite')

# SQLite Configuration
SQLITE_DB_PATH = "fraud_detection.db"

# PostgreSQL Configuration (requires environment variables)
POSTGRESQL_URL = os.getenv('DATABASE_URL', None)

# Application Settings
APP_TITLE = "Fraud Detection ML Platform"
APP_DESCRIPTION = "A comprehensive machine learning platform for fraud detection and analysis"

# Model Configuration
AVAILABLE_MODELS = [
    "Logistic Regression",
    "Random Forest", 
    "XGBoost",
    "SVM",
    "Naive Bayes"
]

# Feature Engineering Settings
MAX_POLYNOMIAL_DEGREE = 3
MAX_INTERACTION_FEATURES = 10
DEFAULT_TEST_SIZE = 0.3
DEFAULT_CV_FOLDS = 5

# UI Configuration
MAX_DISPLAY_ROWS = 1000
CHART_HEIGHT = 400
CHART_WIDTH = 600

def get_database_manager():
    """
    Factory function to return the appropriate database manager
    based on configuration
    """
    if DATABASE_TYPE.lower() == 'postgresql' and POSTGRESQL_URL:
        from database_postgresql import DatabaseManager
        return DatabaseManager()
    else:
        from database import DatabaseManager
        return DatabaseManager()

def get_database_info():
    """
    Get information about the current database configuration
    """
    if DATABASE_TYPE.lower() == 'postgresql' and POSTGRESQL_URL:
        return {
            'type': 'PostgreSQL',
            'status': 'Connected',
            'url': POSTGRESQL_URL,
            'features': ['ACID Compliance', 'Concurrent Access', 'Advanced Queries', 'Scalability']
        }
    else:
        return {
            'type': 'SQLite',
            'status': 'Connected',
            'file': SQLITE_DB_PATH,
            'features': ['Local Storage', 'Zero Configuration', 'Lightweight', 'File-based']
        }

# Environment-specific settings
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Security settings
ALLOW_RAW_SQL = os.getenv('ALLOW_RAW_SQL', 'False').lower() == 'true'
MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', '100'))

# Performance settings
ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'True').lower() == 'true'
CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '3600'))