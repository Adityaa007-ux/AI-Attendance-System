import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace

known_faces_path = "known_faces"

attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Time", "Date"])
    df.to_csv(attendance_file, index=False)

cap = cv2.VideoCapture(0)

print("Starting AI Attendance System...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    try:

        result = DeepFace.find(
            img_path=frame,
            db_path=known_faces_path,
            enforce_detection=False,
            silent=True
        )

        if len(result) > 0 and len(result[0]) > 0:

            identity = result[0].iloc[0]['identity']

            name = os.path.basename(identity).split('.')[0]

            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")
            current_date = now.strftime("%d-%m-%Y")

            df = pd.read_csv(attendance_file)

            if name not in df["Name"].values:

                new_entry = pd.DataFrame({
                    "Name": [name],
                    "Time": [current_time],
                    "Date": [current_date]
                })

                df = pd.concat([df, new_entry])

                df.to_csv(attendance_file, index=False)

            cv2.putText(
                frame,
                f"{name} Present",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    except:
        pass

    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()