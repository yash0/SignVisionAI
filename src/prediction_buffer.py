
from collections import deque, Counter


class PredictionBuffer:

    def __init__(self, size=8):

        self.buffer = deque(
            maxlen=size
        )

    def add(self, prediction):

        if prediction != "UNCERTAIN":
            self.buffer.append(prediction)

    def get_stable(self):

        if len(self.buffer) < 3:
            return None

        prediction, count = (
            Counter(self.buffer)
            .most_common(1)[0]
        )

        if count >= 3:
            return prediction

        return None

    def clear(self):
        self.buffer.clear()
