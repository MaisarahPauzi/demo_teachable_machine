import cv2
import numpy as np

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

GREEN = (0,255,0)
WHITE = (255,255,255)



camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)
    # detect faces
    for (x,y,w,h) in faces:
        # draw rectangle to detected face
        cv2.rectangle(frame,(x,y),(x+w,y+h),GREEN,2)
        
        # draw rectange for background of text
        cv2.rectangle(frame,(x,y-40),(x+w,y),GREEN,-1)
        
        # coordinate text
        centre_x = x+int(w/2)
        centre_y = y+int(h/2)
        
        if (centre_x > 0 and centre_x < 300):
            x_direction = 'left'
            
        elif (centre_x >=300 and centre_x < 400):
            x_direction = 'center'
        else:
            x_direction = 'right'
        
            
        coord_text = f'X direction = {x_direction}'
        
        # put dot in the middle of face
        cv2.circle(frame, (centre_x, centre_y), 7, GREEN, -1)
        
        # draw text of detected face (wear mask or not)
        cv2.putText(frame, coord_text, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,WHITE,2)

    # show window frame contain live camera video
    cv2.imshow("frame", frame)

    # wait for key every 1 millisecond
    key = cv2.waitKey(1)

    # close window when click exit button
    if cv2.getWindowProperty("frame",cv2.WND_PROP_VISIBLE) == 0:        
        break

camera.release()
cv2.destroyAllWindows()