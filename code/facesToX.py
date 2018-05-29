# Libraries
import cv2
import sys

#-----------------------------------------------------------------------------------------------------------------------------#

# Function

def facesToOof(img,oof,x,y,w,h): # ~0.008s
	
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

# Checks

nb_arg = len(sys.argv)
image = "une future image"
method = 'realtime'
init_camera = True
custom_image = False

if(nb_arg > 4):
	print("Erreur: nombre d'arguments incorrect")
	print("Commande:",sys.argv[0],"[-v [image]/-s [image] [image]]")
	sys.exit(1)

if(nb_arg == 4):
	meth = sys.argv[1] if nb_arg > 1 else 'ERROR_ARG_OPT'
	img1 = sys.argv[2] if nb_arg > 2 else 'ERROR_ARG_IMG'
	img2 = sys.argv[3] if nb_arg > 3 else 'ERROR_ARG_IMG'

	if(meth != '-s'):
		print("Erreur:",meth,"n'est pas un argument valide")
		print("Commande:",sys.argv[0],"[-v [image]/-s [image] [image]]")
		sys.exit(2)

	# Check first image eligibility
	image = cv2.imread(img1)
	try:
		image_type = image.dtype
	except Exception as e:
		print("La première image fournie est invalide\n.")
		raise e

	# Check second image eligibility
	oof = cv2.imread(img2, -1)
	try:
		oof_type = oof.dtype
	except Exception as e:
		print("La première image fournie est invalide\n.")
		raise e

	# Put method as still
	method = 'still'
	init_camera = False
	custom_image = True

if(nb_arg == 3):
	meth = sys.argv[1] if nb_arg > 1 else 'ERROR_ARG_OPT'
	img = sys.argv[2] if nb_arg > 2 else 'ERROR_ARG_IMG'
	if(meth != '-s' and meth != '-v'):
		print("Erreur:",meth,"n'est pas un argument valide")
		print("Commande:",sys.argv[0],"[-v [image]/-s [image] [image]]")
		sys.exit(2)
	
	img = cv2.imread(img, -1)
	try:
		image_type = img.dtype
	except Exception as e:
		print("L'image fournie est invalide\n.")
		raise e

	if(meth == '-v'):
		oof = img
		custom_image = True
	else:
		image = img
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

	if(meth != '-s' and meth != '-v'):
		print("Erreur:",meth,"n'est pas un argument valide")
		print("Commande:",sys.argv[0],"[-v/-s] [image]")
		sys.exit(2)

	if(meth == '-s'):
		method = 'still'

		# Take capture as image to transform
		try:
			_, image = cap.read(0)	
		except Exception as e:
			print("Erreur, la capture n'a pas pu être réalisée")
			raise e

#-----------------------------------------------------------------------------------------------------------------------------#

# Main

# Load XML Classifier  
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If no custom image, fetch the head of god
if(not custom_image):
	oof = cv2.imread("oof.png", -1)

# Checks
if(method == 'still'):
	# Detect faces
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(25,25))

	# Launch function for each faces detected
	for (x,y,w,h) in faces:
		facesToOof(image,oof,x,y,w,h)
		
	# Writing result ~0.01s
	print("Ecriture de face2x.jpg")
	cv2.imwrite('face2x.jpg',image)

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

		# Detect faces
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(25,25))

		# Launch function for each faces detected
		for (x,y,w,h) in faces:
			facesToOof(frame,oof,x,y,w,h)

		# Display result
		cv2.imshow("Demon overlord filter",frame)

		# Poll the keyboard. If 'q' is pressed, exit the main loop.
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			cam_quit = 1

	cv2.destroyAllWindows()

# Worked fine
sys.exit(0)




	





