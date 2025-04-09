import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        
    def load_datasets(self):
        """
        Load datasets from file paths
        """
        try:
            labour_market_statistics = pd.read_csv(r'PythonMLA\data\automationreferencetables2.csv')
            ai_awareness = pd.read_csv(r'PythonMLA\data\LabourMarketStatisticsTimeSeries(1).csv')
            automation_probability = pd.read_csv(r'PythonMLA\data\opnai4to14may2023(2).csv')
            
            return labour_market_statistics, ai_awareness, automation_probability
        
        except FileNotFoundError as e:
            print(f"Error loading datasets: {e}")
            return None, None, None
    
    def preprocess_data(self, datasets):
        """
        Preprocessing of datasets with comprehensive cleaning and feature engineering
        """
        if datasets is None or any(dataset is None for dataset in datasets):
            print("Error: Invalid or missing datasets.")
            return None
        
        labour_market_data, ai_awareness_data, automation_probability_data = datasets
        
        print("\n--- DATASET COLUMN INFORMATION ---")
        print("Labour Market Data Columns:", list(labour_market_data.columns))
        print("Labour Market Data Info:")
        labour_market_data.info()
        
        print("\nAI Awareness Data Columns:", list(ai_awareness_data.columns))
        print("AI Awareness Data Info:")
        ai_awareness_data.info()
        
        print("\nAutomation Probability Data Columns:", list(automation_probability_data.columns))
        print("Automation Probability Data Info:")
        automation_probability_data.info()
        
        try:
            # Remove columns with only NaN values
            labour_market_data = labour_market_data.dropna(axis=1, how='all')
            ai_awareness_data = ai_awareness_data.dropna(axis=1, how='all')
            automation_probability_data = automation_probability_data.dropna(axis=1, how='all')
            
            # Remove columns with names starting with 'Unnamed:'
            labour_market_data = labour_market_data.loc[:, ~labour_market_data.columns.str.contains('^Unnamed:')]
            ai_awareness_data = ai_awareness_data.loc[:, ~ai_awareness_data.columns.str.contains('^Unnamed:')]
            automation_probability_data = automation_probability_data.loc[:, ~automation_probability_data.columns.str.contains('^Unnamed:')]
            
            # Convert all columns to numeric where possible
            for df in [labour_market_data, ai_awareness_data, automation_probability_data]:
                for col in df.columns:
                    # Try to convert to numeric, replace non-numeric with NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Merge datasets
            merged_data = pd.concat([labour_market_data, ai_awareness_data, automation_probability_data], axis=1)
            
            # Remove duplicate columns
            merged_data = merged_data.loc[:,~merged_data.columns.duplicated()]
            
            # Fill NaN values with median of each column
            merged_data = merged_data.fillna(merged_data.median())
            
            # Print merged data info
            print("\n--- MERGED DATA INFORMATION ---")
            print("Merged Data Columns:", list(merged_data.columns))
            print("Merged Data Shape:", merged_data.shape)
            merged_data.info()
            
            return merged_data
        
        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_features_and_target(self, processed_data):
        """
        Preparation of features and target variables for machine learning models
        """
        if processed_data is None:
            print("Error: Processed data is None. Check previous preprocessing steps.")
            return None, None

        # Keep only numeric columns
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        
        print("\n--- FEATURE PREPARATION ---")
        print("Numeric Columns Found:", list(numeric_columns))
        
        if len(numeric_columns) == 0:
            print("Error: No numeric columns found for feature preparation.")
            return None, None
        
        # Create features and target
        features = processed_data[numeric_columns]
        
        # Choose a target variable (modify as needed)
        if 'job_stability_score' in features.columns:
            target = features['job_stability_score']
            features = features.drop('job_stability_score', axis=1)
        else:
            # If no specific target, use the first column as target
            target = features.iloc[:, 0]
            features = features.iloc[:, 1:]
        
        print("Features Shape:", features.shape)
        print("Target Shape:", target.shape)
        print("Features Columns:", list(features.columns))
        
        return features, target

# Example usage remains the same
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    try:
        # Load datasets
        raw_datasets = preprocessor.load_datasets()
        
        # Preprocess data
        processed_data = preprocessor.preprocess_data(raw_datasets)
        
        # Prepare features and target
        X, y = preprocessor.prepare_features_and_target(processed_data)
        
        if X is not None:
            print("Data preprocessing completed successfully.")
            print("Features shape:", X.shape)
            print("Target shape:", y.shape)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()