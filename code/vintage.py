import cv2
import numpy as np
import copy
import random as rng
import sys



def generateCircleMask(img,center,radius):

	h,w,v = img.shape
	sq_rad = radius**2
	toThreshold = cv2.cvtColor(img.astype('uint8',copy=False),cv2.COLOR_BGR2GRAY)
	for i in range (0,h):
		for j in range (0,w):
			if(((i-center[0])**2 + (j-center[1])**2) <= sq_rad):
				toThreshold[i,j] = 0
			else:
				toThreshold[i,j] = 255
	ret, circleMask = cv2.threshold(toThreshold, 10, 255, cv2.THRESH_BINARY)
	return circleMask


	
def cappedValue(value,least=0,most=255):

	if(value < least):
		value = least
	elif(value > most):
		value = most
	return value



def applyVariationToBGRPixel(pixel,var,withRNG=False,low_rng=0,high_rng=0):

	b,g,r = pixel
	if(withRNG):
		b = cappedValue(b - (var + rng.randint(low_rng,high_rng)))
		g = cappedValue(g - (var + rng.randint(low_rng,high_rng)))
		r = cappedValue(r - (var + rng.randint(low_rng,high_rng)))
	else:
		b = cappedValue(b - var)
		g = cappedValue(g - var)
		r = cappedValue(r - var)
	pixel = [b,g,r]
	return pixel



def getOutOfMaskNeighboursList(img,x,y,mask,expected,size=3):

	neigh = []
	size = int(np.floor(size/2))
	for i in range (x-size,x+size):
		for j in range (y-size,y+size):
			if(mask[i,j] == expected):
				neigh.append(img[i,j])
	return neigh



def maskDifference(mask1,mask2):

	h,w = mask1.shape
	mask_diff = copy.copy(mask1)
	for i in range (0,h):
		for j in range (0,w):
			if(mask1[i,j] == 0 and mask2[i,j] == 255):
				mask_diff[i,j] = 255
			else:
				mask_diff[i,j] = 0
	return mask_diff



def vintage(img):

	# Useful datas 
	height, width, value = img.shape
	
	# Get Canny edges
	grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edgyimg = cv2.Canny(grayimg,50,200,apertureSize=3)

	# Dilate edges
	rect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	thicc_img = cv2.dilate(edgyimg,rect,iterations=1)

	# Thickening edges
	invertimg = cv2.bitwise_not(thicc_img)
	colorgray = cv2.cvtColor(invertimg,cv2.COLOR_GRAY2BGR)	
	ret, mask = cv2.threshold(grayimg, 10, 255, cv2.THRESH_BINARY)
	mask_inv = mask
	roi = grayimg[0:height, 0:width]
	img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
	img2_fg = cv2.bitwise_and(img,img,mask = mask)
	img1_bg = cv2.cvtColor(img1_bg,cv2.COLOR_GRAY2BGR)
	addimg = cv2.add(img1_bg,img2_fg)
	img[0:height, 0:width] = addimg

	# Speckle Noise
	gauss = np.random.randn(height,width,value)
	gauss = gauss.reshape(height,width,value)        
	img = img + (img * gauss)/16

	# Sepia filter I
	sepia = np.matrix([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
	img = cv2.transform(img,sepia)

	# Toneing
	center_h = int(height/2)
	center_w = int(width/2)
	mask = generateCircleMask(img,[center_h,center_w],center_h-5)
	outer_mask = generateCircleMask(img,[center_h,center_w],center_h+5)
	outer_mask = maskDifference(outer_mask,mask)
	var = 5
	unb = 1
	low = -2
	high = 2
	for i in range (0,height):
		for j in range (0,width):
			if(mask[i,j] == 255):
				img[i,j] = applyVariationToBGRPixel(img[i,j],var,True,low,high)
			elif(outer_mask[i,j] == 255):
				img[i,j] = applyVariationToBGRPixel(img[i,j],int(var/2),True,low,high)

	# Sepia filter II
	sepia = np.matrix([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
	img = cv2.transform(img,sepia)

	# Gaussian blur
	img = cv2.GaussianBlur(img,(5,5),0)

	# Border
	bs = 10
	img = cv2.copyMakeBorder(img,bs,bs,bs,bs,cv2.BORDER_CONSTANT,value=(230,245,245))

	# Return result
	return img



def options():
	print("Nom ou chemin de l'image à traiter (avec l'extension)")
	name_i = input("-> ")
	print("")
	image = cv2.imread(name_i)
	try:
		dtype_debug = image.dtype
	except AttributeError as ae:
		print("Erreur,",name_i,"ne peut pas être lu (fichier invalide ou absent)\n")
		raise ae 

	print("Quel nom donner à l'image en sortie (avec l'extension)?")
	name_o = input("-> ")
	print("")

	return image,name_o



def optionTime(time):
	# See Execution time of vintage function
	print("Voulez-vous connaître le temps d'execution de la fonction vintage? [y/n]")
	know = input("-> ")
	if(know == 'y'):
		print("Temps d'execution du programme:",time,"secondes")



# ---------------------------------------------------------------------------------------------- #
# Main

# Checks
nb_arg = len(sys.argv)
image = "emplacement d'une future image"
name_o = "image_out_from_vintage.jpg"

if(nb_arg > 2):
	print("Erreur: nombre d'arguments incorrect")
	print("Commande:",sys.argv[0],"[-o]		(utilise la webcam de base)")
	sys.exit(1)

if(nb_arg == 2):
	opt = sys.argv[1] if nb_arg > 1 else 'ERROR_ARG_OPT'
	if(opt != '-o'):
		print("Erreur:",opt,"n'est pas un argument valide")
		print("Commande:",sys.argv[0],"[-o]		(utilise la webcam de base)")
		sys.exit(2)
	else:
		# Launch some options
		image, name_o = options()

if(nb_arg == 1):
	# Initialize video capture
	try:
		cap = cv2.VideoCapture(0)
	except Exception as e:
		print("Erreur: impossible d'accéder à la webcam")
		raise e
	
	# Capture image
	try:
		_, image = cap.read(0)	
	except Exception as e:
		print("Erreur, la capture n'a pas pu être réalisée")
		raise e

# Start counting time
e1 = cv2.getTickCount()

# Do vintage transformation
image = vintage(image)

# Forced to write it even if I want to simply display it because of the magic of types
cv2.imwrite(name_o,image)
written_image = cv2.imread(name_o)

# Compute elapsed time
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()

# Launch some other options
answer = 'y'
if(nb_arg == 2):
	# Image destiny
	print("Voulez-vous directement voir l'image? [y/n]")
	know1 = input("-> ")
	print("")

if(answer == 'y'):
	# Stop parameter
	quit = 0

	# Display result
	while(quit == 0):
		cv2.imshow('Vintage capture',written_image)

		# Poll the keyboard. If 'q' is pressed, exit the main loop.
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			quit = 1

	cv2.destroyAllWindows()

# Time option
optionTime(time)

