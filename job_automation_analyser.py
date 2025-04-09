import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_preparation import DataPreprocessor

class JobAutomationAnalyser:
    def __init__(self):
        """
        Initialise job automation analysis components
        """
        self.preprocessor = DataPreprocessor()
        self.random_forest_model = None
        self.kmeans_model = None
    
    def train_random_forest(self, X, y):
        """
        Training Random Forest model to predict automation risk with NaN handling
        """
        # Ensure input data is numeric and handle NaNs
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        # Remove rows with NaN in target variable
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a pipeline with imputation, scaling, and random forest
        rf_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Replace NaNs with median
            ('scaler', StandardScaler()),  # Scale features
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Fit the pipeline
        rf_pipeline.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
        
        # Performance metrics
        y_pred = rf_pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Extract feature importances
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_pipeline.named_steps['regressor'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'mae': mae,
            'cv_scores': -cv_scores.mean(),
            'feature_importance': feature_importance,
            'model': rf_pipeline  # Return the entire pipeline
        }
    
    def perform_clustering(self, X):
        """
        Cluster occupations based on automation risk characteristics
        """
        # Ensure input data is numeric and handle NaNs
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Determine optimal number of clusters using silhouette score
        max_clusters = 10
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_imputed)
            score = silhouette_score(X_imputed, cluster_labels)
            silhouette_scores.append(score)
        
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Perform clustering with optimal number of clusters
        self.kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = self.kmeans_model.fit_predict(X_imputed)
        
        return {
            'cluster_labels': cluster_labels,
            'optimal_clusters': optimal_clusters,
            'silhouette_score': max(silhouette_scores)
        }
    
    def collaborative_filtering_reskilling(self, X, y):
        # Ensure input data is numeric and handle NaNs
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        # Remove rows with NaNs
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Compute feature-based similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        similarity_matrix = cosine_similarity(X_scaled)
        
        reskilling_recommendations = {}
        
        # Find similar low-risk jobs for high-risk occupations
        high_risk_indices = np.where(y > y.quantile(0.75))[0]
        
        for idx in high_risk_indices:
            # Find top 3 most similar low-risk jobs
            similar_jobs = np.argsort(similarity_matrix[idx])[::-1][1:4]
            
            reskilling_recommendations[idx] = {
                'current_occupation': X.index[idx],
                'recommended_occupations': X.index[similar_jobs].tolist(),
                'similarity_scores': similarity_matrix[idx][similar_jobs]
            }
        
        return reskilling_recommendations

# Example usage
if __name__ == "__main__":
    analyser = JobAutomationAnalyser()
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    raw_datasets = preprocessor.load_datasets()
    processed_data = preprocessor.preprocess_data(raw_datasets)
    X, y = preprocessor.prepare_features_and_target(processed_data)
    
    # Random Forest Analysis
    rf_results = analyser.train_random_forest(X, y)
    print("Random Forest Results:", rf_results)
    
    # Clustering Analysis
    clustering_results = analyser.perform_clustering(X)
    print("Clustering Results:", clustering_results)
    
    # Reskilling Recommendations
    reskilling_paths = analyser.collaborative_filtering_reskilling(X, y)
    print("Reskilling Recommendations:", reskilling_paths)