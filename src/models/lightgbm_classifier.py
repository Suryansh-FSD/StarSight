import pickle
import lightgbm as lgb
from pathlib import Path
import numpy as np

class LightGBMClassifier:
    """
    A wrapper class for the downstream LightGBM GBDT classifier,
    providing standardized train, predict, save, and load utilities.
    """
    def __init__(self, **kwargs):
        # Default fallback options if none provided
        kwargs.setdefault('n_estimators', 100)
        kwargs.setdefault('learning_rate', 0.05)
        kwargs.setdefault('num_leaves', 31)
        kwargs.setdefault('n_jobs', 1)  # Force single-threaded to prevent macOS deadlocks
        
        self.model = lgb.LGBMClassifier(**kwargs)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'LightGBMClassifier':
        """
        Trains the LightGBM classifier on extracted CNN penultimate feature embeddings.
        """
        self.model.fit(X_train, y_train, **kwargs)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts candidate labels (0 or 1) using GBDT decision boundaries.
        """
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts continuous planet candidate probability in range [0, 1].
        Returns probability of positive class (shape: (N,)).
        """
        proba = self.model.predict_proba(X)
        # Handle cases where LightGBM returns 1-dimensional array or 2-dimensional probabilities
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
        
    def save(self, path: Path) -> None:
        """
        Serializes the LightGBM model to disk using standard pickle protocol.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
            
    def load(self, path: Path) -> 'LightGBMClassifier':
        """
        Deserializes the LightGBM model from disk using pickle.
        """
        path = Path(path)
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self
