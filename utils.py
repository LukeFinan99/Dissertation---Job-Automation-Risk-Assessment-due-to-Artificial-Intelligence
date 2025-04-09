import logging
import os
from datetime import datetime

def setup_logging():
    """
    Configure logging for the job automation analysis project
    """
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    log_filename = os.path.join(log_dir, f'job_automation_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_input_data(data):
    """
    Validate input data for job automation analysis
    
    Args:
        data (pd.DataFrame): Input dataset
    
    Returns:
        bool: True if data is valid, False otherwise
    """
    logger = setup_logging()
    
    # Check for required columns
    required_columns = [
        'industry', 
        'occupation', 
        'salary', 
        'employment_count', 
        'automation_probability'
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for data completeness
    if data.isnull().sum().any():
        logger.warning("Dataset contains missing values")
        return False
    
    # Additional data validation checks can be added here
    
    logger.info("Input data validation successful")
    return True

# Example usage in other scripts
if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Utility functions loaded successfully")