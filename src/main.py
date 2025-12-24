import cv2
from hand_tracker import HandTracker
from mouse_controller import MouseController
from ui import MainWindow
import sys
from PyQt6.QtWidgets import QApplication
import numpy as np
from pynput import keyboard
import time
from one_euro_filter import OneEuroFilter
import torch
from gesture_recognition import GestureRecognition
import os

prev_y_middle = 0

def main():
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        cap = cv2.VideoCapture(0)
        wCam, hCam = 1280, 960
        cap.set(3, wCam)
        cap.set(4, hCam)
        tracker = HandTracker()
        mouse_controller = MouseController(1920, 1080)
        wScr, hScr = mouse_controller.screen_width, mouse_controller.screen_height
        gesture = "None"
        global prev_y_middle
        prev_y_middle = 0

        # Initialize OneEuroFilter for X and Y coordinates
        min_cutoff = 0.1
        beta = 20.0
        one_euro_filter_x = OneEuroFilter(time.time(), 0, min_cutoff=min_cutoff, beta=beta)
        one_euro_filter_y = OneEuroFilter(time.time(), 0, min_cutoff=min_cutoff, beta=beta)

        # Load Gesture Model
        model_path = os.path.join("models", "gesture_model.pth")
        gesture_model = None
        gestures_list = []

        # We need to know the number of classes and input size to initialize the model
        # Assuming we can infer or hardcode for now, or load from a config.
        # For this implementation, let's try to load if available.
        if os.path.exists(model_path) and os.path.exists("data"):
            try:
                gestures_list = [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]
                if gestures_list:
                    num_classes = len(gestures_list)
                    input_size = 42 # 21 landmarks * 2 coords
                    gesture_model = GestureRecognition(input_size, num_classes)
                    gesture_model.load_state_dict(torch.load(model_path))
                    gesture_model.eval()
                    print("Gesture model loaded successfully.")
            except Exception as e:
                print(f"Failed to load gesture model: {e}")

        def on_press(key):
            try:
                if key == keyboard.Key.esc:
                    return False
            except Exception as e:
                print(f"處理按鍵時出錯: {e}")

        with keyboard.Listener(on_press=on_press) as listener:
            while listener.running:
                success, img = cap.read()
                if not success:
                    break
                img = cv2.flip(img, 1)
                img = tracker.find_hands(img)
                lm_list = tracker.get_position(img)

                if len(lm_list) != 0:
                    x1, y1 = lm_list[8][1:]

                    # Map coordinates to screen size
                    x3 = np.interp(x1, (0, wCam), (0, wScr))
                    y3 = np.interp(y1, (0, hCam), (0, hScr))

                    # Update filter parameters based on sensitivity slider
                    sensitivity = window.get_sensitivity()
                    new_min_cutoff = 1.0 / max(sensitivity, 1)

                    one_euro_filter_x.min_cutoff = new_min_cutoff
                    one_euro_filter_y.min_cutoff = new_min_cutoff

                    # Apply OneEuroFilter
                    curr_time = time.time()
                    curr_x = one_euro_filter_x(curr_time, x3)
                    curr_y = one_euro_filter_y(curr_time, y3)

                    mouse_controller.move_mouse(curr_x, curr_y)

                    # Gesture Recognition Logic
                    # Priority: 1. Heuristic (Click/Scroll) 2. ML Model

                    x2, y2 = lm_list[4][1:]
                    x_middle, y_middle = lm_list[12][1:]
                    dist_click = np.hypot(x2 - x1, y2 - y1)
                    dist_scroll = np.hypot(x_middle - x1, y_middle - y1)

                    if dist_click < 30:
                        mouse_controller.left_click()
                        gesture = "Click"
                        cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
                    elif dist_scroll < 30:
                        if prev_y_middle == 0:
                            prev_y_middle = y_middle
                        scroll_amount = (y_middle - prev_y_middle) / 10
                        mouse_controller.scroll(int(scroll_amount))
                        gesture = "Scrolling"
                        cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                        prev_y_middle = y_middle
                    else:
                        prev_y_middle = 0
                        gesture = "Moving"

                        # Try ML model if available and not clicking/scrolling
                        if gesture_model:
                            # Prepare input
                            # lm_list is [[id, x, y], ...]
                            # We need to flatten [x, y] pairs
                            coords = np.array([lm[1:] for lm in lm_list], dtype=np.float32).flatten()
                            # Reshape for LSTM: (1, 1, 42) -> Batch=1, Seq=1, Feature=42
                            input_tensor = torch.tensor(coords).unsqueeze(0).unsqueeze(0)

                            with torch.no_grad():
                                outputs = gesture_model(input_tensor)
                                _, predicted = torch.max(outputs.data, 1)
                                predicted_idx = predicted.item()
                                if predicted_idx < len(gestures_list):
                                    ml_gesture = gestures_list[predicted_idx]
                                    # Only override if confidence is high? Or just display it.
                                    # For now, let's append it to the display
                                    gesture += f" / {ml_gesture}"

                else:
                    gesture = "None"

                window.update_frame(img)
                window.update_gesture(gesture)
                app.processEvents()

        cap.release()
        sys.exit(app.exec())
    except Exception as e:
        print(f"發生錯誤: {e}")
        print("請確保 'hand_landmarker.task' 模型檔案位於專案根目錄下。")
        sys.exit(1)

if __name__ == "__main__":
    main()