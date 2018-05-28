import cv2
import numpy as np


# Functions

def addHellEyes(img,begone,ex,ey,ew,eh): # ~0.017s

	# Datas
	add_h = 80
	add_w = 120
	centering_parameter_y = 5
	centering_parameter_x = 0

	# Preprocess
	new_h = eh+2*add_w
	new_w = ew+2*add_h
	resized_b = cv2.resize(begone,(new_h,new_w),interpolation = cv2.INTER_AREA)
	y1, y2 = ey-add_h+centering_parameter_y, ey+eh+add_h+centering_parameter_y
	x1, x2 = ex-add_w+centering_parameter_x, ex+ew+add_w+centering_parameter_x
	alpha = resized_b[:, :, 3] / 255.0
	ctr_alpha = 1.0 - alpha

	# Treatment
	try:
		for c in range(0, 3):
			img[y1:y2, x1:x2, c] = (alpha * resized_b[:, :, c] + ctr_alpha * img[y1:y2, x1:x2, c])
	except Exception as e:
		pass
	
	return img

#-----------------------------------------------------------------------------------------------------------------------------#

# Main

# Load XML Classifiers   
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load replacing image
begone = cv2.imread("begoneThotEyes.png",-1)

# Configure capture
try:
	cap = cv2.VideoCapture(0)
except Exception as e:
	print("Erreur: impossible d'accéder à la webcam")
	raise e

# Loop over each frame
while(1):

	# Take each frame
	try:
		_, frame = cap.read()
	except Exception as e:
		print("Attention: l'image n'a pas été prise")

	# Detect faces
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(25,25))

	for (x,y,w,h) in faces:
		roi_gray = gray[y:y+h, x:x+w]

		# Detect eyes and apply function to each
		eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5, minSize=(10,10))
		for (ex,ey,ew,eh) in eyes:
			addHellEyes(frame,begone,x+ex,y+ey,ew,eh)

		# Display result
		cv2.imshow('frame',frame)
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break

cv2.destroyAllWindows()

	





