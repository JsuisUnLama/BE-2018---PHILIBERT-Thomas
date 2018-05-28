import cv2
import numpy as np
import copy

#print version
print("OpenCV Version:",cv2.__version__)

e1 = cv2.getTickCount()
print("Initializing the program's duration measurement")

"""

#bring image of name_img, constant row,col
name_img = 'Lama.jpg'
img = cv2.imread(name_img)

row = 100
col = 100

blue = 0
green = 1
red = 2

### general fonctions ###

#read and print pixel at [row,col]
rdpixel = img[row,col]
print("Blue/Green/Red tab of pixel",row,col,"=",rdpixel)

#read and print only blue pixel
rd_blue = img[row,col,0]
print("Blue color of",row,col,"=",rd_blue)

#change pixel value
old_value = copy.copy(rdpixel)
img[row,col] = [255,255,255]
print("Pixel",row,col,"now is",img[row,col],"(previously ",old_value,")")

#better editing with numpy methods
old_value2 = copy.copy(img.item(row,col,red))
img.itemset((row,col,red),100)
print("Pixel",row,col,"had red value",old_value2,"but has it been set to",img.item(row,col,red))

#accessing image properties
print("Shape of the image:",img.shape) #note: grayscale image only returns row and col
print("Size (total number of pixel) of the image:",img.size)
print("Datatype of the image (useful for debugging):",img.dtype)

### image regioning (ROI) ###

#copy a bit of an image
r1 = 0
r2 = 150
c1 = 110
c2 = 230
lamaface = img[r1:r2, c1:c2]

#writing an image
new_img = 'Lama2.jpg'
print("Writing of",new_img)
cv2.imwrite(new_img,lamaface)

#putting the copied bit into somewhere in the image
r1 = 46
r2 = 54
c1 = 46
c2 = 54
dr = r2-r1
dc = c2-c1
print("Copying a portion of image")
redbit = img[r1:r2, c1:c2]

r3 = 60
c3 = 280
print("Repeating it")
img[r3:r3+dr, c3:c3+dc] = redbit
print("Rewriting",name_img)
cv2.imwrite(name_img,img)

#cv2 image split (time costly)
b,g,r = cv2.split(img)

#cv2 merge
img = cv2.merge((b,g,r))

#numpy indexing
b = img[:,:,0] #retrieve all the blue
img[:,:,2] = 0 #put all red to 0

### padding ###

#create a sort of "photo frame" = cv2.copyMakeBorder()
#it takes following arguments:
#src: input image
#top/bottom/left/right: border width in number of pixels in corresponding directions
#borderType (Flag)
	#cv2.BORDER_CONSTANT: constant colored border, value as extra arg
	#cv2.BORDER_REFLECT: border will be reflexion of the border elements
	#cv2.BORDER_REFLECT_101: "default" reflection
	#cv2.BORDER_REPLICATE: replicate nearest border element
	#cv2.BORDER_WRAP: wrapping border

reflect = cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_REPLICATE)
border_name_img = 'Lama3.jpg'
print("Writing of",border_name_img)
cv2.imwrite(border_name_img,reflect)

### add, sub and blend image together ###

superlama1 = cv2.imread('Superlama.jpg')
superlama2 = cv2.imread('Superlama2.jpg')
addedSuperlama = cv2.add(superlama1,superlama2)  #add
#subbedSuperLama = cv2.sub(superlama2,superlama1) #sub
add_name_img = 'Lama4.jpg'
#sub_name_img = 'Lama5.jpg'
print("Writing of",add_name_img)
cv2.imwrite(add_name_img,addedSuperlama)
#print("Writing of",sub_name_img)
#cv2.imwrite(sub_name_img,subbedSuperLama)

alpha = 0.5
gamma = 0
blendedLama = cv2.addWeighted(superlama1,alpha,superlama2,1-alpha,gamma)
blend_name_img = 'Superlamaaaaa.jpg'
print("Writing of",blend_name_img)
cv2.imwrite(blend_name_img,blendedLama)


### Using a video channel ###

#video flux: webcam
	#https://wiki.labomedia.org/index.php/OpenCV_pour_Python

#timelapse: raspberry + linux
	#https://www.raspberrypi.org/documentation/usage/camera/raspicam/timelapse.md

#Bitwise operation
# Load two images
img1 = cv2.imread('Superlama.jpg')
img2 = cv2.imread('bmw_logo.jpg')
print(img1.dtype)
print(img2.dtype)

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

print("Writing bitwise operation image named as some bullshit or smth")
cv2.imwrite('some_bullshit_or_smth.jpg',img1)

e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("This program took",time,"seconds to run")

#optimization exemple:
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_optimization/py_optimization.html

#color changing
#BGR -> Gray
graylama = cv2.cvtColor(superlama1,cv2.COLOR_BGR2GRAY)

#BGR -> HSV
hsvlama = cv2.cvtColor(superlama1,cv2.COLOR_BGR2HSV)

print("Writing Gray and HSV transformation results")
cv2.imwrite('GrayLama.jpg',graylama)
cv2.imwrite('HSVLama.jpg',hsvlama)

"""

#basic colour tracking with OpenCV with noise
#here tracked: blue 
cap = cv2.VideoCapture(0)

# Stop parameter
cam_quit = 0

# Loop over each frame
while(cam_quit == 0):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Define range of green colour in HSV
    lower_green = np.array([60,100,100])
    upper_green = np.array([70,255,255])

    # Define range of red colour in HSV
    lower_red = np.array([0,140,140]) 
    upper_red = np.array([20,255,255])

    # Threshold the HSV image to get blue, green and red color
    #mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    #mask2 = cv2.inRange(hsv, lower_green, upper_green)
    mask3 = cv2.inRange(hsv, lower_red, upper_red)
    supermask = mask3

    """
    # Dilate
    rect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    frame = cv2.dilate(frame,rect,iterations=1)
    """

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= supermask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',supermask)
    cv2.imshow('res',res)
    
    # Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1

cv2.destroyAllWindows()

"""

#easy way to find hsv value of a colour
blue = np.uint8([[[255,0,0]]])
hsv_blue = cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
print("Blue in HSV:",hsv_blue)
green = np.uint8([[[0,255,0 ]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print("Green in HSV:",hsv_green)

red = np.uint8([[[0,0,255]]])
hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
print("Red in HSV:",hsv_red)

#thresholding
#documentation: http://opencvpython.blogspot.fr/2013/05/thresholding.html

"""

"""
#opencv exemple:
graylama_mb = cv2.medianBlur(graylama,5)
threshold_lama = cv2.adaptiveThreshold(graylama_mb,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
print("Writing a binary version of GrayLama.jpg")
cv2.imwrite('TresholdLama.jpg',threshold_lama)
"""
"""
threshold_lama = graylama
height, width = threshold_lama.shape
seuil = 120
#artisanal example
for i in range (0,height):
	for j in range (0,width):
		if (threshold_lama[i,j] <= seuil):
			threshold_lama[i,j] = 0
		else:
			threshold_lama[i,j] = 255

print("Writing a binary version of GrayLama.jpg")
cv2.imwrite('TresholdLama.jpg',threshold_lama)

#borders binary image
#contours, hierarchy = cv2.findContours(threshold_lama,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 

#contours is a list of points
#drawContours commans takes image, contours , -1,color, thickness (-1 = full)
#contours_lama = cv2.drawContours(threshold_lama,contours,-1,(0,255,0),3) 
#print("Writing ContoursLama.jpg")
#cv2.imwrite('ContoursLama.jpg',contours_lama)

#edge detector opencv
"""

"""
OpenCV puts all the above in single function, cv2.Canny(). We will see how to use it.
First argument is our input image. 
Second and Third arguments are our minVal and maxVal respectively.
Fourth argument is aperture_size (the size of Sobel kernel to find image gradients. By default it is 3).
Last argument is L2gradient which specifies the equation for finding gradient magnitude. (False by default)
	-> True = sqrt(G_x² + G_y²) (more accurate).
	-> False (default) = Edge Gradient = |G_x| + |G_y|.
"""

"""
edgylama = cv2.Canny(graylama,100,200)
print("Writing EdgyLama.jpg")
cv2.imwrite('EdgyLama.jpg',edgylama)

rect = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
edgylama2 = cv2.dilate(edgylama,rect,iterations=2)
print("Dilatation")
edgylama2 = cv2.morphologyEx(edgylama2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
print("Opening")
print("Writing EdgyLama2.jpg")
cv2.imwrite('EdgyLama2.jpg',edgylama2)
"""