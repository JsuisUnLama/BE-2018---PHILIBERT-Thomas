import cv2
import numpy as np
import copy
import random as rng

def generateWhiteScreen(img):

	h,w,v = img.shape
	white_screen = copy.copy(img)
	for i in range (0,h):
		for j in range (0,w):
			white_screen[i,j] = [255,255,255]
	return white_screen



def generateBlackScreen(img):

	h,w,v = img.shape
	black_screen = copy.copy(img)
	for i in range (0,h):
		for j in range (0,w):
			black_screen[i,j] = [0,0,0]
	return black_screen



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



def changeImageSaturation(img,sat=0,lum=0):

	h,w,v = img.shape
	hsv_img = cv2.cvtColor(img.astype('uint8',copy=False),cv2.COLOR_BGR2HSV)
	for i in range (0,h):
		for j in range (0,w):
			hsv_img[i,j][1] = cappedValue(hsv_img[i,j][1] + sat)
			hsv_img[i,j][2] = cappedValue(hsv_img[i,j][2] + lum)
	hsv_img = cv2.cvtColor(hsv_img,cv2.COLOR_HSV2BGR)
	return hsv_img



def invertSaturation(img):

	hsv_img = cv2.cvtColor(img,COLOR_BGR2HSV)
	hsv_img = cv2.invert(hsv_img)
	hsv_img = cv2.cvtColor(hsv_img,COLOR_HSV2BGR)
	return hsv_img


	
def cappedValue(value,least=0,most=255):

	if(value < least):
		value = least
	elif(value > most):
		value = most
	return value



def generateFold(img):

	h,w,v = img.shape
	lower_w = int(w/12)
	upper_w = w - lower_w
	yA = rng.randint(lower_w,upper_w)
	yB = rng.randint(lower_w,upper_w)
	A = [0,yA]
	B = [h,yB]
	slope = (yB - yA)/h
	start = yA
	if(yB < yA):
		yA = yA + yB
		yB = yA - yB
		yA = yA - yB

	for i in range (0,h):
		for j in range (yA,yB):
			if(i*slope + start == j):
				img[i,j] = pixelFold(img[i,j])
	return img
				


def pixelFold(bgr_array,treatment=0):
	
	lightGray = 160
	max_tone = 1.2
	variation = 20
	test_tone = np.floor(255/max_tone)
	
	if(treatment == 0):
		bgr_array = [lightGray + rng.randint(0,20), lightGray + rng.randint(0,20), lightGray + rng.randint(0,20)]
	elif(treatment == 1):
		bgr_array = [lightGray - rng.randint(0,8), lightGray - rng.randint(0,8), lightGray - rng.randint(0,8)]
	else:
		if(np.amax(bgr_array) < 255 - variation):
			bgr_array = [bgr_array[0] + variation, bgr_array[1] + variation, bgr_array[2] + variation]
	return bgr_array



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



def meanFilter(img,x,y,size=3,withRNG=False,low_rng=0,high_rng=0):

	s = int(np.floor(size/2))
	sum_b = 0
	sum_g = 0
	sum_r = 0
	ite=0
	for i in range (x-s,x+s):
		for j in range (y-s,y+s):
			ite+=1
			sum_b += img[i,j][0]
			sum_g += img[i,j][1]
			sum_r += img[i,j][2]
	mean_b = int(sum_b/ite)
	mean_g = int(sum_g/ite)
	mean_r = int(sum_r/ite)
	if(withRNG):
		mean_b += rng.randint(low_rng,high_rng)
		mean_g += rng.randint(low_rng,high_rng)
		mean_r += rng.randint(low_rng,high_rng)
	mean = [mean_b,mean_g,mean_r]
	return mean



def vintage(img,name):

	# Useful datas 
	height, width, value = img.shape
	white_screen = generateWhiteScreen(img)
	
	# Thickening edges
	grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	lowThreshold = 50
	ratio = 4
	edgyimg = cv2.Canny(grayimg,lowThreshold,lowThreshold*ratio,apertureSize=3)

	rect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	thicc_img = cv2.dilate(edgyimg,rect,iterations=1)
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
	for i in range (0,height):
		for j in range (0,width):
			if(mask[i,j] == 255):
				img[i,j] = applyVariationToBGRPixel(img[i,j],var,True,-2,2)
			elif(outer_mask[i,j] == 255):
				img[i,j] = applyVariationToBGRPixel(img[i,j],int(var/2),True,-2,2)

	"""
	# Median Filter for the outer mask
	for i in range (0,height):
		for j in range (0,width):
			if(outer_mask[i,j] == 255):
				img[i,j] = meanFilter(img,i,j)
	"""

	# Sepia filter II
	sepia = np.matrix([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
	img = cv2.transform(img,sepia)

	# Blur
	#img = cv2.blur(img,(3,3))

	# Gaussian blur
	img = cv2.GaussianBlur(img,(5,5),0)

	# Folds
	#img = generateFold(img)
	
	"""
	# Gaussian Noise
	mean = 0
	var = 0.1
	sigma = var**0.5	
	gauss = np.random.normal(mean,sigma,(height,width,value))
	gauss = gauss.reshape(height,width,value)
	img = img + (gauss * img)/8
	"""

	print("Ecriture de",name)
	cv2.imwrite(name,img)


# Main

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

e1 = cv2.getTickCount()
vintage(image,name_o)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("Temps d'execution du programme:",time,"secondes")

