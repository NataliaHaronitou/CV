import cv2
from random import randrange

# Download file from github.com/opencv 'frontalface' (trained data)
trained_face_data =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose image to detect faces in
img=cv2.imread('/home/natalia/Desktop/Elon.jpg')

# Algorithim can only be trained in greyscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around the faces 
for (x, y, w, h) in face_coordinates: # <-- allows more than one face with a loop 
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128,256), randrange(128,256), randrange(128,256)), 10)


print(face_coordinates)

# Display the image with the faces 
cv2.imshow('Natalia Face Detector', img)
cv2.waitKey()




print("Code Completed")
