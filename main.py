import sys
import traceback
from data_preparation import DataPreprocessor
from job_automation_analyser import JobAutomationAnalyser
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold

def filter_and_prepare_features(X, variance_threshold=0.01):
    """
    Filtering features based on variance and remove highly correlated columns
    """
    # Print initial information
    print("\n--- Initial Feature Filtering ---")
    print(f"Original features: {X.shape[1]}")
    print("Original columns:", list(X.columns))
    
    # Remove columns with very long names (potential metadata or description columns)
    max_column_name_length = 150  # Increased from 100 to accommodate your long column names
    X = X.loc[:, [col for col in X.columns if len(str(col)) < max_column_name_length]]
    
    # Remove columns with non-numeric content
    X = X.select_dtypes(include=[np.number])
    
    # Print columns after initial filtering
    print("\n--- After Removing Long and Non-Numeric Columns ---")
    print(f"Remaining features: {X.shape[1]}")
    print("Remaining columns:", list(X.columns))
    
    # Variance-based feature selection
    selector = VarianceThreshold(threshold=variance_threshold)
    
    # Fit and transform the data
    X_selected = pd.DataFrame(
        selector.fit_transform(X), 
        columns=X.columns[selector.get_support()],
        index=X.index
    )
    
    # Remove highly correlated features
    correlation_matrix = X_selected.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    
    X_filtered = X_selected.drop(columns=to_drop)
    
    print("\n--- Feature Filtering Summary ---")
    print(f"Original feature count: {X.shape[1]}")
    print(f"Features after variance filtering: {X_selected.shape[1]}")
    print(f"Features after correlation filtering: {X_filtered.shape[1]}")
    print(f"Dropped correlated columns: {to_drop}")
    
    return X_filtered

def main():
    """
    Main execution script for job automation risk analysis
    """
    # Initialize components
    preprocessor = DataPreprocessor()
    analyser = JobAutomationAnalyser()
    
    try:
        # Load and preprocess data
        raw_datasets = preprocessor.load_datasets()
        
        # Debugging: Check if datasets were loaded correctly
        if raw_datasets is None:
            print("Error: Failed to load datasets.")
            return
        
        # Check the number of columns in each dataset
        for i, dataset in enumerate(raw_datasets, 1):
            print(f"\nDataset {i} columns:")
            print(dataset.columns.tolist())
        
        processed_data = preprocessor.preprocess_data(raw_datasets)
        
        # Debugging: Check if processed_data is valid
        if processed_data is None:
            print("Error: Data preprocessing failed.")
            return
        
        X, y = preprocessor.prepare_features_and_target(processed_data)
        
        # Check if feature preparation was successful
        if X is None or y is None:
            print("Error: Feature preparation failed.")
            return
        
        # Filter and prepare features
        X_filtered = filter_and_prepare_features(X)
        
        # Ensure target variable is numeric and matches feature rows
        y = pd.to_numeric(y, errors='coerce')
        y = y.loc[X_filtered.index]
        y = y.fillna(y.median())
        
        # Perform scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_filtered), 
            columns=X_filtered.columns, 
            index=X_filtered.index
        )
        
        # Perform analysis
        rf_results = analyser.train_random_forest(X_scaled, y)
        clustering_results = analyser.perform_clustering(X_scaled)
        reskilling_paths = analyser.collaborative_filtering_reskilling(X_scaled, y)
        
        # Visualization of results
        plt.figure(figsize=(15, 8))
        top_features = rf_results['feature_importance'].head(20)  # Top 20 features
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Top 20 Feature Importance in Automation Risk Prediction')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # Print and save results
        with open('job_automation_analysis_results.txt', 'w') as f:
            f.write("Random Forest Results:\n")
            f.write(f"Mean Absolute Error: {rf_results['mae']}\n")
            f.write(f"Cross-Validation Scores: {rf_results['cv_scores']}\n\n")
            
            f.write("Top 20 Important Features:\n")
            f.write(top_features.to_string())
            f.write("\n\n")
            
            f.write("Clustering Results:\n")
            f.write(f"Optimal Number of Clusters: {clustering_results['optimal_clusters']}\n")
            f.write(f"Silhouette Score: {clustering_results['silhouette_score']}\n\n")
            
            f.write("Reskilling Recommendations:\n")
            for idx, recommendation in reskilling_paths.items():
                f.write(f"High-Risk Occupation: {recommendation['current_occupation']}\n")
                f.write(f"Recommended Occupations: {recommendation['recommended_occupations']}\n")
                f.write(f"Similarity Scores: {recommendation['similarity_scores']}\n\n")
        
        print("Job Automation Risk Analysis completed. Results saved to job_automation_analysis_results.txt")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()