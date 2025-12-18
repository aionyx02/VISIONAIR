# src/data_collector.py
import cv2
import csv
from src.hand_tracker import HandTracker
import os

def collect_data():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    gesture_name = input("Enter gesture name: ")
    data_dir = os.path.join("data", gesture_name)
    os.makedirs(data_dir, exist_ok=True)
    
    sample_num = 0
    while True:
        success, img = cap.read()
        if not success:
            break

        img = tracker.find_hands(img)
        lm_list = tracker.get_position(img)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        
        if key == ord('s') and len(lm_list) != 0:
            with open(os.path.join(data_dir, f"{sample_num}.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(lm_list)
            sample_num += 1
            print(f"Saved sample {sample_num}")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
