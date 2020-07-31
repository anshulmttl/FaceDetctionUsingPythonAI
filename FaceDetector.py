import cv2
print ("FaceDetector.py")

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect the faces in.
# input image using openCV and not normal image read
#img = cv2.imread('RDJ.jpg')
webcam = cv2.VideoCapture(0) #0 => Goes to webcam

#iterate forever over frames
while True:
    successful_frame_read,frame = webcam.read()

    #Must convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #plugin face detection in video
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    #draw rectangles around faces
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
        
        
    cv2.imshow("Grayscaled frame", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break;

webcam.release()
    
# How the algorithm is working
# Haar cascade - haar is person who invented algorithm. It goes down the funnel and
# keeps cascading. Chain of machine learning things that passes through
# HAAR features - Edge features, Line features, Four rectangle feaatures
# 
#
#
#
#
#

#key = cv2.waitKey(1)
#Show the image
#cv2.imshow('Face detector', img)

#wait for key press before closing image.
#cv2.waitKey()

#Convert to grayscale because haar cascade algorithm only takes gray scale images
#grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#plug the grayscale image to the trained algorithm
#face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#print the face coordinates upper left, bottom down, bottom right, top right
#print(face_coordinates)

#(x,y,w,h) = face_coordinates[0]

#draw the green box around the face.
#draw the rectangle using a module in OpenCV
#cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#cv2.imshow('Image with rectangle', img)
#cv2.waitKey()


#Detecting faces in a video
#Run the classifier code into every single frame of video
