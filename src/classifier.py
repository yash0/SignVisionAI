
import joblib
import numpy as np
import pandas as pd


class SignClassifier:

    def __init__(
        self,
        model_path,
        columns_path,
        confidence_threshold=0.40
    ):

        self.model = joblib.load(model_path)
        self.columns = joblib.load(columns_path)
        self.threshold = confidence_threshold

    def predict(self, features):

        data = pd.DataFrame(
            [features],
            columns=self.columns
        )

        probabilities = (
            self.model.predict_proba(data)[0]
        )

        index = np.argmax(probabilities)

        sign = self.model.classes_[index]
        confidence = float(probabilities[index])

        if confidence < self.threshold:
            return "UNCERTAIN", confidence

        return sign, confidence
