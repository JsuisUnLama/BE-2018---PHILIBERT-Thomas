# Libraries
import cv2
import numpy as np
import copy
import sys

#-----------------------------------------------------------------------------------------------------------------------------#

# Functions

def hellEyes(img,begone):
	# Detect faces
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(25,25))

	for (x,y,w,h) in faces:
		roi_gray = gray[y:y+h, x:x+w]

		# Detect eyes and apply function to each
		eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5, minSize=(10,10))
		for (ex,ey,ew,eh) in eyes:
			addHellEyes(img,begone,x+ex,y+ey,ew,eh)

	return img



def addHellEyes(img,begone,ex,ey,ew,eh): # ~0.017s

	# Datas
	add_w = 80
	add_h = 120
	centering_parameter_y = 0
	centering_parameter_x = 5

	# Preprocess
	new_h = eh+2*add_h
	new_w = ew+2*add_w
	resized_b = cv2.resize(begone,(new_h,new_w),interpolation = cv2.INTER_AREA)
	y1, y2 = ey-add_w+centering_parameter_y, ey+eh+add_w+centering_parameter_y
	x1, x2 = ex-add_h+centering_parameter_x, ex+ew+add_h+centering_parameter_x
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

# Checks

nb_arg = len(sys.argv)
image = "une future image"
method = 'realtime'
init_camera = True

if(nb_arg > 3):
	print("Erreur: nombre d'arguments incorrect")
	print("Commande:",sys.argv[0],"[-s] [image]")
	sys.exit(1)

if(nb_arg == 3):
	meth = sys.argv[1] if nb_arg > 1 else 'ERROR_ARG_OPT'
	img = sys.argv[2] if nb_arg > 2 else 'ERROR_ARG_IMG'
	if(meth != '-s'):
		print("Erreur:",meth,"n'est pas un argument valide")
		print("Commande:",sys.argv[0],"[-s] [image]")
		sys.exit(2)
	
	# Check image eligibility
	image = cv2.imread(img)
	try:
		image_type = image.dtype
	except Exception as e:
		print("L'image fournie est invalide\n.")
		raise e

	# Put method as still
	method = 'still'
	init_camera = False

if(init_camera):
	# Initialize video capture
	try:
		cap = cv2.VideoCapture(0)
	except Exception as e:
		print("Erreur: impossible d'accéder à l'outil de capture principal\n")
		raise e

if(nb_arg == 2):
	meth = sys.argv[1] if nb_arg > 1 else 'ERROR_ARG_OPT'

	if(meth != '-s'):
		print("Erreur:",meth,"n'est pas un argument valide")
		print("Commande:",sys.argv[0],"[-s] [image]")
		sys.exit(2)

	# Put method as still
	method = 'still'

	# Take capture as image to transform
	try:
		_, image = cap.read(0)	
	except Exception as e:
		print("Erreur, la capture n'a pas pu être réalisée")
		raise e

#-----------------------------------------------------------------------------------------------------------------------------#

# Main

# Load XML Classifiers   
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Fetch the eyes from hell
begone = cv2.imread("begoneThotEyes.png", -1)

# Check method
if(method == 'still'):
	# Launch principal function
	image = hellEyes(image,begone)

	# Write result
	print("Ecriture de hell_eyes.jpg")
	cv2.imwrite('hell_eyes.jpg',image)

else:
	# Stop parameter
	cam_quit = 0

	# Initialize result window
	cv2.namedWindow("Demon overlord filter",cv2.WINDOW_NORMAL)

	# Loop over each frame until 'q' is pressed
	while(cam_quit == 0):
		
		# Take a frame
		try:
			_, frame = cap.read(0)
		except Exception as e:
			print("Attention: l'image n'a pas été prise")

		# Launch principal function
		frame = hellEyes(frame,begone)

		# Display result
		cv2.imshow("Demon overlord filter",frame)

		# Poll the keyboard. If 'q' is pressed, exit the main loop.
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			cam_quit = 1

	cv2.destroyAllWindows()

# Worked fine
sys.exit(0)


