# REQUIRED PACKAGES:
# opencv-python

#imports required packages
import cv2
import numpy as np

#routes video capture to variable
cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)

#while loop that loops the frames from the camera input and display them in a window until the escape key is pressed
while True:
    _, frame = cap.read()
    frame_copy = frame.copy()

    #converts Blue Green Red color to Hue, Saturation and Value color
    hsv = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
    
    #defines range of color in HSV for mask
    sensitivity = 25

    lower_white = np.array([0, 0, 255-sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    #creates the mask using the HSV threshold range
    mask = cv2.inRange(hsv, lower_white, upper_white)

    result = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

    cv2.imshow("Masked", mask)
    cv2.imshow("Camera Window", frame_copy)

    #waits until escape key (0xFF)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()

# test push