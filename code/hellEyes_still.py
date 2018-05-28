import cv2
import numpy as np
import copy
import sys


# Checks

nb_arg = len(sys.argv)

if(nb_arg > 2):
	print("Erreur: nombre d'arguments incorrect")
	print("Commande:",sys.argv[0],"[nom d'un fichier image] 	(utilise la webcam de base)")
	sys.exit(1)

if(nb_arg == 1):
	try:
		cap = cv2.VideoCapture(0)
	except Exception as e:
		print("Erreur: impossible d'accéder à la webcam")
		raise e

	try:
		_, img = cap.read()
	except Exception as e:
		print("Erreur: impossible de prendre une photo depuis la webcam")
		raise e

elif(nb_arg == 2):
	file = sys.argv[1] if nb_arg > 1 else 'ERROR_FILE_ARG'
	try:
		img = cv2.imread(file_or_ip)
		img_dtype = img.dtype
	except Exception as e:
		print("Erreur: nombre d'arguments incorrect")
		print("Commande:",sys.argv[0],"[nom d'un fichier image] 	(utilise la webcam de base)")
		raise e



# Functions

def addHellEyes(img,ex,ey,ew,eh): # ~0.017s

	# Datas
	begone = cv2.imread("begoneThotEyes.png", -1)
	add_h = 80
	add_w = 120
	centering_parameter_y = 0
	centering_parameter_x = 5

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

# Detect faces
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(25,25))

for (x,y,w,h) in faces:
	roi_gray = gray[y:y+h, x:x+w]

	# Detect eyes and apply function to each
	eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5, minSize=(10,10))
	for (ex,ey,ew,eh) in eyes:
		addHellEyes(img,x+ex,y+ey,ew,eh)

# Writing result
print("Ecriture de hell_eyes.jpg")
cv2.imwrite('hell_eyes.jpg',img)

	





