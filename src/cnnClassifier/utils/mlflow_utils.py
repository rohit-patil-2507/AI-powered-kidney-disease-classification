"""
MLflow utility functions for experiment management.
Provides functions to delete experiments (soft delete and hard delete).
"""
import shutil
from pathlib import Path
from urllib.parse import urlparse
import mlflow
from cnnClassifier import logger


def get_mlflow_tracking_uri() -> str:
    """Get the current MLflow tracking URI.
    
    Returns:
        str: The MLflow tracking URI
    """
    return mlflow.get_tracking_uri()


def is_local_file_store() -> bool:
    """Check if using local file store for MLflow tracking.
    
    Returns:
        bool: True if using file:// scheme, False otherwise
    """
    tracking_uri = mlflow.get_tracking_uri()
    scheme = urlparse(tracking_uri).scheme
    return scheme == "file" or tracking_uri.startswith("./") or tracking_uri.startswith("/")


def get_experiment_id_by_name(experiment_name: str) -> str:
    """Get experiment ID by experiment name.
    
    Args:
        experiment_name (str): Name of the experiment
        
    Returns:
        str: Experiment ID
        
    Raises:
        ValueError: If experiment not found
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    return experiment.experiment_id


def get_experiment_name_by_id(experiment_id: str) -> str:
    """Get experiment name by experiment ID.
    
    Args:
        experiment_id (str): ID of the experiment
        
    Returns:
        str: Experiment name
        
    Raises:
        ValueError: If experiment not found
    """
    experiment = mlflow.get_experiment(experiment_id)
    if experiment is None:
        raise ValueError(f"Experiment with ID '{experiment_id}' not found")
    return experiment.name


def list_experiments() -> list:
    """List all experiments in MLflow tracking server.
    
    Returns:
        list: List of experiment objects
    """
    # Use search_experiments for newer MLflow versions
    # Return all experiments (including deleted ones with filter)
    return mlflow.search_experiments()


def soft_delete_experiment(experiment_id: str) -> bool:
    """
    Soft delete an MLflow experiment (marks as deleted).
    
    Args:
        experiment_id (str): ID of the experiment to delete
        
    Returns:
        bool: True if successful
    """
    try:
        mlflow.delete_experiment(experiment_id=experiment_id)
        logger.info(f"Successfully soft deleted experiment with ID: {experiment_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to soft delete experiment: {e}")
        raise e


def soft_delete_experiment_by_name(experiment_name: str) -> bool:
    """
    Soft delete an MLflow experiment by name.
    
    Args:
        experiment_name (str): Name of the experiment to delete
        
    Returns:
        bool: True if successful
    """
    experiment_id = get_experiment_id_by_name(experiment_name)
    return soft_delete_experiment(experiment_id)


def hard_delete_experiment_locally(experiment_id: str) -> bool:
    """
    Hard delete experiment data from local file store.
    Only works if MLflow is using local file storage.
    
    Args:
        experiment_id (str): ID of the experiment
        
    Returns:
        bool: True if successful
    """
    if not is_local_file_store():
        logger.warning("Not using local file store. Hard delete skipped.")
        return False
    
    tracking_uri = mlflow.get_tracking_uri()
    # Parse the path from file:// URI
    parsed = urlparse(tracking_uri)
    mlruns_path = Path(parsed.path) / experiment_id
    
    if mlruns_path.exists():
        try:
            shutil.rmtree(mlruns_path)
            logger.info(f"Successfully hard deleted local experiment data at: {mlruns_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to hard delete local experiment data: {e}")
            raise e
    else:
        logger.warning(f"Local experiment data not found at: {mlruns_path}")
        return False


def delete_experiment(
    experiment_id: str = None,
    experiment_name: str = None,
    hard_delete: bool = False
) -> bool:
    """
    Delete an MLflow experiment.
    
    Soft delete: Marks experiment as deleted in MLflow tracking server.
    Hard delete: Removes local file store data (only for file:// tracking URI).
    
    Args:
        experiment_id (str): ID of the experiment to delete (optional if experiment_name provided)
        experiment_name (str): Name of the experiment to delete (optional if experiment_id provided)
        hard_delete (bool): Whether to also perform hard delete for local file store
        
    Returns:
        bool: True if successful
        
    Raises:
        ValueError: If neither experiment_id nor experiment_name provided
    """
    # Get experiment_id from name if provided
    if experiment_id is None:
        if experiment_name is None:
            raise ValueError("Either experiment_id or experiment_name must be provided")
        experiment_id = get_experiment_id_by_name(experiment_name)
    
    # Soft delete (marks as deleted)
    soft_delete_experiment(experiment_id)
    
    # Hard delete for local file store if requested
    if hard_delete:
        hard_delete_experiment_locally(experiment_id)
    
    return True


# Example usage:
if __name__ == "__main__":
    # Set tracking URI (example: using local file store)
    # mlflow.set_tracking_uri("file:///./mlruns")
    
    # Or using DagsHub remote server
    mlflow.set_tracking_uri("https://dagshub.com/rohit-patil-2507/Kidney-disease-classification-Deep-learning-project.mlflow")
    
    # List all experiments
    print("Listing experiments:")
    experiments = list_experiments()
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
    
    # Example: Delete experiment by name
    # delete_experiment(experiment_name="Kidney-Disease-Classification", hard_delete=False)
    
    # Example: Delete experiment by ID
    # delete_experiment(experiment_id="12345", hard_delete=True)
