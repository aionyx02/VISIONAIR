import cv2
import csv
from hand_tracker import HandTracker
import os
import numpy as np

def collect_data():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    gesture_name = input("Enter gesture name: ")
    data_dir = os.path.join("data", gesture_name)
    os.makedirs(data_dir, exist_ok=True)

    sample_num = len(os.listdir(data_dir))
    print(f"開始收集手勢: {gesture_name}")
    print("操作說明: 按住 'r' 錄製手勢序列，放開 'r' 結束錄製並存檔。按 'q' 退出。")

    is_recording = False
    current_sequence = []

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = tracker.find_hands(img)
        lm_list = tracker.get_position(img)

        # Visual feedback for recording
        if is_recording:
            cv2.putText(img, "RECORDING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)

        if key == ord('r'):
            if not is_recording:
                is_recording = True
                current_sequence = []
                print("Recording started...")

            if len(lm_list) != 0:
                # Flatten the landmark list: [id, x, y] -> [x, y, x, y...]
                # We skip the ID and just take x, y
                # lm_list is [[id, x, y], ...]
                flattened_lm = []
                for lm in lm_list:
                    flattened_lm.extend(lm[1:]) # Append x, y
                current_sequence.append(flattened_lm)

        elif is_recording and key != ord('r'):
            # Key released or another key pressed, stop recording
            is_recording = False
            if len(current_sequence) > 5: # Filter out very short sequences
                file_path = os.path.join(data_dir, f"seq_{sample_num}.csv")
                with open(file_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(current_sequence)
                print(f"Saved sequence {sample_num} with {len(current_sequence)} frames to {file_path}")
                sample_num += 1
            else:
                print("Sequence too short, discarded.")
            current_sequence = []

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()