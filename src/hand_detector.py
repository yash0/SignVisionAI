import cv2
import mediapipe as mp


class HandDetectionResult:
    """
    Small compatibility wrapper so the rest of
    SignVision AI can continue using:

        result.hand_landmarks
    """

    def __init__(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks


class HandDetector:

    def __init__(
        self,
        model_path=None,
        num_hands=1
    ):

        self.mp_hands = mp.solutions.hands

        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=num_hands,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb_frame.flags.writeable = False

        results = self.detector.process(
            rgb_frame
        )

        landmarks = []

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:
                landmarks.append(
                    hand.landmark
                )

        return HandDetectionResult(
            landmarks
        )
