# src/hand_tracker.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

class HandTracker:
    def __init__(self, model_path='hand_landmarker.task'):
        if not os.path.exists(model_path):
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
            for hand_landmarks in self.results.hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_hand_connections_style())
        return img

    def get_position(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results and self.results.hand_landmarks:
            my_hand = self.results.hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return lm_list
