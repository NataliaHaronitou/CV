import cv2
from random import randrange

# Download file from github.com/opencv 'frontalface' (trained data)
trained_face_data =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# To capture video from webcam
webcam = cv2.VideoCapture('/home/natalia/Desktop/Video.mp4') # <-- if you put (0) it will capture from default webcam

#### Iterate forever over frames 
while True:

    #### Read the current frame
    successful_frame_read, frame = webcam.read()


    # Algorithim can only be trained in greyscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces 
    for (x, y, w, h) in face_coordinates: # <-- allows more than one face with a loop 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128,256), randrange(128,256), randrange(128,256)), 10)


    cv2.imshow('Natalia Face Detector Video', frame)
    key = cv2.waitKey(1)

    #### Stop if Q key is pressed
    if key==81 or key==113:
        break

#### Release the VideoCapture object
webcam.release()

print("Code Completed")
