"""
Online Cat CV (http.cat)

A openCV program that detects faces and displays Tiktok cats.

"""

import cv2
import mediapipe as mp
import os


face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

cam = cv2.VideoCapture(0)

def cat_tongue(face_landmark_points):
    top_lip = face_landmark_points.landmark[13]
    bottom_lip = face_landmark_points.landmark[14]

    mouth_open = abs(top_lip.y - bottom_lip.y)

    return mouth_open > 0.04

def cat_shock(face_landmark_points):
    l_top = face_landmark_points.landmark[159]
    l_bot = face_landmark_points.landmark[145]
    r_top = face_landmark_points.landmark[386]
    r_bot = face_landmark_points.landmark[374]

    eye_opening = (
        abs(l_top.y - l_bot.y) +
        abs(r_top.y - r_bot.y)
    ) / 2.0

    return (
        eye_opening > 0.03
    )



def main():
    while True:
        ret, image = cam.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        height, width, depth = image.shape

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = face_mesh.process(rgb_image)
        face_landmark_points = processed_image.multi_face_landmarks

        cat_image = "assets/cat-shock.jpeg"
        if face_landmark_points:
            face_landmark_points = face_landmark_points[0]
            if cat_tongue(face_landmark_points):
                cat_image = "assets/cat-tongue.jpeg"
            elif cat_shock(face_landmark_points):
                cat_image = "assets/cat-shock.jpeg"
            elif cat_glare(face_landmark_points):
                cat_image = "assets/cat-glare.jpeg"
            else:
                cat_image = "assets/cat-disgust.jpeg"
        
            height, width = image.shape[:2]
            for lm in face_landmark_points.landmark:
                x = int(lm.x * width)
                y = int(lm.y * height)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            
        
        cv2.imshow('Face Detection', image)

        # Cat Display
        cat = cv2.imread(cat_image)
        if cat is not None:
            cat = cv2.resize(cat, (640, 480))
            cv2.imshow("Cat Image", cat)
        else:
            blank = image * 0
            cv2.putText(blank, f"Missing: {cat_image}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Cat Image", blank)


        key = cv2.waitKey(1)
        if key == 27:
            break


    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()