"""
Online Cat Mediapipe Program - MeowCV


A openCV program that detects faces and displays meme cats.

"""

import cv2
import mediapipe as mp
import os


face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

cam = cv2.VideoCapture(0) #acess the first available webcam

# Thresholds 
eye_opening_threshold = 0.025 
mouth_open_threshold = 0.03
squinting_threshold = 0.018
brow_low_threshold = 0.025
nose_scrunch_threshold = 0.04

def cat_shock(face_landmark_points):
    l_top = face_landmark_points.landmark[159]
    l_bot = face_landmark_points.landmark[145]
    r_top = face_landmark_points.landmark[386]
    r_bot = face_landmark_points.landmark[374]

    eye_opening = (abs(l_top.y - l_bot.y) + abs(r_top.y - r_bot.y)) / 2.0

    return eye_opening > eye_opening_threshold

def cat_tongue(face_landmark_points):
    top_lip = face_landmark_points.landmark[13]
    bottom_lip = face_landmark_points.landmark[14]

    mouth_open = abs(top_lip.y - bottom_lip.y)

    return mouth_open > mouth_open_threshold

def cat_glare(face_landmark_points):
    l_top = face_landmark_points.landmark[159]
    l_bot = face_landmark_points.landmark[145]
    
    r_top = face_landmark_points.landmark[386]
    r_bot = face_landmark_points.landmark[374]

    eye_squint = (
        abs(l_top.y - l_bot.y) +
        abs(r_top.y - r_bot.y)
    ) / 2.0

    return eye_squint < squinting_threshold

def cat_angry(face_landmark_points):
    # Left Brow (65) and Right Brow (295)
    l_brow = face_landmark_points.landmark[65]
    r_brow = face_landmark_points.landmark[295]

    # Top of Left Eye (159) and Right Eye (386)
    l_eye_top = face_landmark_points.landmark[159]
    r_eye_top = face_landmark_points.landmark[386]

    # Calculate distance between eyebrow and eye
    l_dist = abs(l_brow.y - l_eye_top.y)
    r_dist = abs(r_brow.y - r_eye_top.y)

    avg_dist = (l_dist + r_dist) / 2.0

    # If the distance is small, the brows are furrowed/low
    return avg_dist < brow_low_threshold

def cat_scrunch(face_landmark_points):
    # Nose Tip (1) and Top Lip Center (13)
    nose_tip = face_landmark_points.landmark[1]
    top_lip = face_landmark_points.landmark[13]

    # Calculate distance between nose tip and top lip
    nose_lip_dist = abs(nose_tip.y - top_lip.y)

    # If the distance is small, the nose is scrunched
    return nose_lip_dist < nose_scrunch_threshold



def main():
    while True:
        ret, image = cam.read() #grabs frame from webcam
        if not ret:
            break

        image = cv2.flip(image, 1) #acts like mirror
        height, width, depth = image.shape
        print(height, width, depth)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #converts BGR to RGB
        processed_image = face_mesh.process(rgb_image)
        face_landmark_points = processed_image.multi_face_landmarks

        cat_image = "assets/larry.jpeg" #default image
        if face_landmark_points:
            face_landmark_points = face_landmark_points[0] #if face detected... Takes the first detected face
            if cat_tongue(face_landmark_points):
                cat_image = "assets/cat-tongue.jpeg"
            elif cat_shock(face_landmark_points):
                cat_image = "assets/cat-shock.jpeg"
            elif cat_glare(face_landmark_points):
                cat_image = "assets/cat-glare.jpeg"
            elif cat_scrunch(face_landmark_points):
                cat_image = "assets/cat-disgust.jpeg"
            elif cat_angry(face_landmark_points):
                cat_image = "assets/cat_angry.jpeg"
            else:
                cat_image = "assets/larry.jpeg"
        

            height, width = image.shape[:2]
            for lm in face_landmark_points.landmark:
                x = int(lm.x * width)
                y = int(lm.y * height)
                cv2.circle(image, (x, y), 1, (0, 100, 0), -1)
            
        
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
        if key == 'q':
            break


    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
