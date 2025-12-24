import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

class HandTracker:
    def __init__(self, model_path=None):
        if model_path is None:
            # Determine the path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # The model is expected to be in the parent directory (project root)
            model_path = os.path.join(current_dir, '..', 'hand_landmarker.task')
            model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
             # Try checking the current working directory as a fallback
             if os.path.exists('hand_landmarker.task'):
                 model_path = os.path.abspath('hand_landmarker.task')
             else:
                raise FileNotFoundError(f"Hand landmark model file not found at {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=2)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self.results = self.landmarker.detect(mp_image)
        
        if draw and self.results.hand_landmarks:
            pass
        return img

    def get_position(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results and self.results.hand_landmarks:
            if len(self.results.hand_landmarks) > hand_no:
                my_hand = self.results.hand_landmarks[hand_no]
                for id, lm in enumerate(my_hand):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    if draw:
                        pass
        return lm_list