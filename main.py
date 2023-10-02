# REQUIRED PACKAGES:
# opencv-python

#imports required packages
import cv2

#routes video capture to variable
cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)

#while loop that loops the frames from the camera input and display them in a window until the escape key is pressed
while True:
    _, frame = cap.read()
    frame_copy = frame.copy()

    cv2.imshow("Camera Window", frame_copy)

    #waits until escape key (0xFF)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()