
import numpy as np


class FeatureExtractor:

    @staticmethod
    def extract(hand_landmarks):

        wrist = hand_landmarks[0]

        features = []

        for landmark in hand_landmarks:

            features.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])

        features = np.array(
            features,
            dtype=np.float32
        )

        max_value = np.max(
            np.abs(features)
        )

        if max_value > 0:
            features = features / max_value

        return features
