import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


class AnomalyDetector:
    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ])

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        images = images.reshape(images.shape[0], -1)

        mean = images.mean(axis=1)
        std = images.std(axis=1)
        max_val = images.max(axis=1)
        min_val = images.min(axis=1)

        features = np.stack([mean, std, max_val, min_val], axis=1)
        return features

    def fit(self, x_clean: np.ndarray, x_adv: np.ndarray):
        x = np.concatenate([x_clean, x_adv], axis=0)
        y = np.concatenate([
            np.zeros(len(x_clean)),
            np.ones(len(x_adv))
        ])

        features = self.extract_features(x)
        self.pipeline.fit(features, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = self.extract_features(x)
        return self.pipeline.predict(features)

    def save(self, path: str):
        joblib.dump(self.pipeline, path)

    def load(self, path: str):
        self.pipeline = joblib.load(path)