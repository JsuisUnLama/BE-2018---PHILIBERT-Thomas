import cv2
import numpy as np
import copy
import random as rng
import os
import sys
import ipaddress as IP

# Checks

nb_arg = len(sys.argv)

if(nb_arg > 3):
	print("Erreur: nombre d'arguments incorrect")
	print("Commande:",sys.argv[0],"<adresse IP> [-rev]")
	print("Ou:",sys.argv[0],"<nom du fichier>")
	print("Ou:",sys.argv[0]),"	<- Par défaut: utilise la webcam"
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
	file_or_ip = sys.argv[1] if nb_arg > 1 else 'ERROR_FILE_OR_IP_ARG'
	ip_worked = True
	try:
		IP.ip_address(file_or_ip)
	except Exception as e:
		ip_worked = False

	if(ip_worked):
		string = '.\connectPi.sh '+file_or_ip+' 0'
		try:
			os.system(string)
		except Exception as e:
			print("Erreur: la photo n'a pas pu être prise et/ou récupérée")
			raise e

		try:
			img = cv2.imread('hell_eyes.jpeg')
		except Exception as e:
			print("Erreur: l'image n'a pas pu être lue")
			raise e

		try:
			img_dtype = img.dtype
		except Exception as e:
			print("Erreur: l'image récupérée est invalide")
			print("Commande:",sys.argv[0],"<adresse IP> [-rev]")
			print("Ou:",sys.argv[0],"<nom du fichier>")
			print("Ou:",sys.argv[0],"	<- Par défaut: utilise la webcam")
			raise e

	else:
		try:
			img = cv2.imread(file_or_ip)
			img_dtype = img.dtype
		except Exception as e:
			print("Erreur: fichier passé en argument non convenable ou invalide")
			print("Commande:",sys.argv[0],"<adresse IP> [-rev]")
			print("Ou:",sys.argv[0],"<nom du fichier>")
			print("Ou:",sys.argv[0],"	<- Par défaut: utilise la webcam")
			raise e
	
else:
	rev = 0
	ipaddr = sys.argv[1] if nb_arg == 3 else ERROR_IP_ARG
	rev_opt = sys.argv[2] if nb_arg == 3 else ERROR_OPT_ARG_1

	if(rev_opt != "-rev"):
		print("Erreur: argument non reconnu (",rev_opt,")")
		print("Commande:",sys.argv[0],"<adresse IP> [-rev]  OU  ",sys.argv[0],"<nom du fichier>")
		sys.exit(1)
	else:
		rev = 1

	try:
		IP(ipaddr)
	except Exception as e:
		print("Erreur: adresse IP invalide")
		raise e

	try:
		img = os.popen("./connectPi.sh",ipaddr,rev,"","r").read
	except Exception as e:
		print("Erreur: la photo n'a pas pu être prise et/ou récupérée")
		raise e

# Functions
def addHellEyes(img,roi,ex,ey,ew,eh):
	begone_eyes = cv2.imread('begoneThotEyes.png')
	roi_h,roi_w,roi_v = roi.shape
	red_eye = cv2.resize(begone_eyes,(roi_h,roi_w),fx=0,fy=0)
	center = ((ex+ew)/2,(ey+eh)/2)
	rh,rw,rv = red_eye.shape
	mid_rh = int(rh/2)
	mid_rw = int(rw/2)
	c0 = int(center[0])
	c1 = int(center[1])
	roi_begone = img[(c0-mid_rh):(c0+mid_rh), (c1-mid_rw):(c1+mid_rw)]
	img[roi_begone] = red_eye
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
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]

# Detect eyes
eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5, minSize=(10,10))
for (ex,ey,ew,eh) in eyes:
	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	#img = addHellEyes(img,roi_color,ex,ey,ew,eh)
	
# Writing result
print("Ecriture de hell_eyes.jpg")
cv2.imwrite('hell_eyes.jpg',img)

	





