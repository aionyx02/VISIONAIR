import cv2
from hand_tracker import HandTracker
from mouse_controller import MouseController
from ui import MainWindow
import sys
from PyQt6.QtWidgets import QApplication
import numpy as np
from pynput import keyboard
smooth = 5
prev_x, prev_y = 0, 0
prev_y_middle = 0
def main():
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        cap = cv2.VideoCapture(0)
        wCam, hCam = 640, 480
        cap.set(3, wCam)
        cap.set(4, hCam)
        tracker = HandTracker()
        mouse_controller = MouseController(1920, 1080)
        wScr, hScr = mouse_controller.screen_width, mouse_controller.screen_height
        gesture = "None"
        global prev_y_middle
        prev_y_middle = 0
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
                    x3 = np.interp(x1, (0, wCam), (0, wScr))
                    y3 = np.interp(y1, (0, hCam), (0, hScr))
                    global prev_x, prev_y
                    curr_x = prev_x + (x3 - prev_x) / smooth
                    curr_y = prev_y + (y3 - prev_y) / smooth
                    mouse_controller.move_mouse(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y
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
                else:
                    gesture = "None"
                window.update_frame(img)
                window.update_gesture(gesture)
                smooth = window.get_sensitivity()
                app.processEvents()
        cap.release()
        sys.exit(app.exec())
    except Exception as e:
        print(f"發生錯誤: {e}")
        print("請確保 'hand_landmarker.task' 模型檔案位於專案根目錄下。")
        sys.exit(1)
if __name__ == "__main__":
    main()