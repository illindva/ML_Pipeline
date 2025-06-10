#!/usr/bin/env python3
"""
Test script to verify PostgreSQL database connection and functionality
"""

import os
import pandas as pd
import numpy as np
from database_postgresql import DatabaseManager

def test_database_connection():
    """Test basic database operations"""
    print("Testing PostgreSQL database connection...")
    
    try:
        # Initialize database manager
        db = DatabaseManager()
        print("✓ Database connection established")
        
        # Test database stats
        stats = db.get_database_stats()
        print(f"✓ Database stats: {stats}")
        
        # Create a sample dataset for testing
        print("\nCreating sample dataset...")
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Test saving dataset
        dataset_id = db.save_dataset(
            name="Test Dataset",
            description="Sample dataset for testing PostgreSQL integration",
            data=sample_data
        )
        print(f"✓ Dataset saved with ID: {dataset_id}")
        
        # Test loading dataset
        loaded_data = db.load_dataset(dataset_id)
        print(f"✓ Dataset loaded successfully, shape: {loaded_data.shape}")
        
        # Test getting all datasets
        all_datasets = db.get_all_datasets()
        print(f"✓ Found {len(all_datasets)} datasets in database")
        
        # Test saving analysis
        analysis_results = {
            'mean_feature1': float(sample_data['feature1'].mean()),
            'std_feature1': float(sample_data['feature1'].std()),
            'class_distribution': sample_data['target'].value_counts().to_dict()
        }
        
        analysis_id = db.save_analysis(
            dataset_id=dataset_id,
            analysis_type="descriptive_stats",
            results=analysis_results
        )
        print(f"✓ Analysis saved with ID: {analysis_id}")
        
        # Test retrieving analysis
        analyses = db.get_analysis(dataset_id)
        print(f"✓ Retrieved {len(analyses)} analyses for dataset")
        
        # Clean up test data
        db.delete_dataset(dataset_id)
        print("✓ Test dataset cleaned up")
        
        print("\n🎉 All database tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        return False
    finally:
        try:
            db.close()
        except:
            pass

if __name__ == "__main__":
    success = test_database_connection()
    exit(0 if success else 1)