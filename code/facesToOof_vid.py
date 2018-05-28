import cv2
import numpy as np

# Functions

def facesToOof(img,oof,x,y,w,h): # ~0.004s
	
	# Datas
	add_h = 13
	add_w = 0

	# Preprocess
	new_h = h+2*add_w
	new_w = w+2*add_h
	resized_oof = cv2.resize(oof,(new_h,new_w),interpolation = cv2.INTER_AREA)
	y1, y2 = y-add_h, y+h+add_h
	x1, x2 = x-add_w, x+w+add_w
	alpha = resized_oof[:, :, 3] / 255
	ctr_alpha = 1.0 - alpha

	# Treatment
	try:
		for c in range(0, 3):
			img[y1:y2, x1:x2, c] = (alpha * resized_oof[:, :, c] + ctr_alpha * img[y1:y2, x1:x2, c])
	except Exception as e:
		pass
	
	return img

#-----------------------------------------------------------------------------------------------------------------------------#

# Main

# Load XML Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load replacing image
oof = cv2.imread("oof.png",-1)

# Configure capture
cap = cv2.VideoCapture(0)

# Stop parameter
cam_quit = 0

# Loop over each frame
while(cam_quit == 0):

	# Take each frame
	_, frame = cap.read()

	# Detect faces and apply function for each
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(25,25))
	for (x,y,w,h) in faces:
		facesToOof(frame,oof,x,y,w,h)
	
	# Display result
	cv2.imshow('frame',frame)
	
	# Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
		cam_quit = 1

cv2.destroyAllWindows()


	





