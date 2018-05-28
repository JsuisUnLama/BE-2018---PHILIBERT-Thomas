import cv2
import numpy as np
import copy
import sys
from multiprocessing import Pool



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
		img = cv2.imread(file)
		img_dtype = img.dtype
	except Exception as e:
		print("Erreur: nombre d'arguments incorrect")
		print("Commande:",sys.argv[0],"[nom d'un fichier image] 	(utilise la webcam de base)")
		raise e



# Functions

def facesToOof(img,x,y,w,h): # ~0.008s
	
	# Datas
	oof = cv2.imread("oof.png", -1)
	add_h = 15
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
	for c in range(0, 3):
	    img[y1:y2, x1:x2, c] = (alpha * resized_oof[:, :, c] + ctr_alpha * img[y1:y2, x1:x2, c])

	return img

#-----------------------------------------------------------------------------------------------------------------------------#



# Main

# Load XML Classifier  
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

e3 = cv2.getTickCount()

# Detect faces
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(25,25))
for (x,y,w,h) in faces:
	facesToOof(img,x,y,w,h)
	
# Writing result ~0.01s
print("Ecriture de face2oof.jpg")
cv2.imwrite('face2oof.jpg',img)

e4 = cv2.getTickCount()
time = (e4 - e3)/ cv2.getTickFrequency()
print("The whole process took",time,"seconds to run")



	





