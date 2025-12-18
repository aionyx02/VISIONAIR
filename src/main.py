# src/main.py
import cv2
from src.hand_tracker import HandTracker
from src.mouse_controller import MouseController
from src.ui import MainWindow
import sys
from PyQt6.QtWidgets import QApplication
from pynput import keyboard

def main():
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()

        cap = cv2.VideoCapture(0)
        tracker = HandTracker()
        mouse_controller = MouseController(1920, 1080) # Replace with your screen resolution

        # Gesture recognition placeholder
        gesture = "None"

        def on_press(key):
            if key == keyboard.Key.esc:
                return False

        with keyboard.Listener(on_press=on_press) as listener:
            while listener.running:
                success, img = cap.read()
                if not success:
                    break

                img = tracker.find_hands(img)
                lm_list = tracker.get_position(img)

                if len(lm_list) != 0:
                    x1, y1 = lm_list[8][1:]
                    # In a real scenario, you would use the gesture recognition model here
                    # to determine the action. For now, we'll just move the mouse.
                    mouse_controller.move_mouse(x1, y1)
                    # Placeholder for gesture recognition
                    gesture = "Moving"


                window.update_frame(img)
                window.update_gesture(gesture)
                app.processEvents()


        cap.release()
        sys.exit(app.exec())
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please make sure that the 'hand_landmarker.task' file is in the root directory of the project.")
        sys.exit(1)

if __name__ == "__main__":
    main()