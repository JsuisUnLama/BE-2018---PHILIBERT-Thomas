# Import the necessary packages
import numpy as np
import cv2
from PIL import Image
import pytesseract
import math

def preprocess(image):
	# Gray into blur into binary
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Return preprocessed image
	return gray

def order_points(pts):
	# Initialize a list of coordinates (top-left, top-right, bottom-left, bottom-right)
	rect = np.zeros((4, 2), dtype = "float32")
 
	# Top-left should have smallest sum whereas bottom-right point should have largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# Computing the difference between the points (top-right will have the smallest while bottom-left will have the largest)
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# Return packed points
	return rect


def four_point_transform(image, pts):
	# Order and unpack points
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# New width
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# New height
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# Generates "Bird's eye view"
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# Compute warp perspective
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# Return warped image
	return warped

def character_recognition(image):
	# Preprocess
	image = preprocess(image)

	# Do character recognition
	img_array = Image.fromarray(image)
	characters = pytesseract.image_to_string(img_array)

	# Return characters
	return characters


def colour_detection(image):
	# Turn image into HSV
	hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

	# Define range of red colour in HSV
	lower_red = np.array([0,140,140]) 
	upper_red = np.array([18,255,255])

	# Generate red mask
	mask = cv2.inRange(hsv,lower_red,upper_red)

	# Bitwise-AND mask and original image
	image = cv2.bitwise_and(image,image,mask=mask)

	# Check whether or not the card is red. If not, it is black.
	colour = 'b'
	if(has_red(image)):
		colour = 'r'

	# Return colour as a character
	return colour

def contour_detection(image):
	# Canny edge detection
	image = cv2.Canny(image,30,200)

	# Get contours
	contours, _, _ = cv2.findContours(image, 1, 2)

	# Return contours
	return contours

def distance(a, b):
	# Compute distance between 2 2D points
	dis = math.sqrt(math.pow(math.fabs(a[0] - b[0]), 2) + math.pow(math.fabs(a[1] - b[1]), 2))

	# Return distance
	return dis

def updateCorner(point, reference, dmax, corner):
	# Retrieve distance between given point and reference
    d = distance(point, reference)

    # Update distance maximum
    if d > dmax:
        dmax = d
        corner = point

    # Return greater distance and furthest point
    return dmax, corner

# Determines the 4 vertices of the marker.
# To do so we're going to divide the marker into 4 regions using a Bounding Rectangle.
# In each region, we look for the point farthest from the center of the region.
# These 4 points will be the 4 corners of a quadrilateral.
def vertices(contour):
	# Initialize corner points
	V0 = (0.0, 0.0)
	V1 = (0.0, 0.0)
	V2 = (0.0, 0.0)
	V3 = (0.0, 0.0)

	# Retrieve corners of the bounding boxes
	x, y, w, h = cv2.boundingRect(contour)
   	
	# Compute each points coordinates
	A = (x, y)
	B = (x + w, y)
	C = (x + w, y + h)
	D = (x, y + h)
	midX = x + w / 2
	midY = y + h / 2

	# Maximum distances between a point and a region center
	dmax = [(0.0), (0.0), (0.0), (0.0)]

	# We split the bounding box into 4 squares of equal size
	for i in range(0, len(contour)):
		# If top left
		if(contour[i][0][0] < midX and contour[i][0][1] <= midY):
			dmax[0], V0 = updateCorner(contour[i][0], C, dmax[0], V0)

        # If top right
		elif(contour[i][0][0] >= midX and contour[i][0][1] < midY):
			dmax[1], V1 = updateCorner(contour[i][0], D, dmax[1], V1)

        # If bottom right
		elif(contour[i][0][0] > midX and contour[i][0][1] >= midY):
			dmax[2], V2 = updateCorner(contour[i][0], A, dmax[2], V2)

   	    # If bottom left
		elif(contour[i][0][0] <= midX and contour[i][0][1] > midY):
			dmax[3], V3 = updateCorner(contour[i][0], B, dmax[3], V3)

    	# If the point is in the exact middle of the bounding box, nothing is done.

	# Return packed points
	pts = [V0, V1, V2, V3]
	return pts 



if __name__ == '__main__':

	# Configure capture
	cap = cv2.VideoCapture(0)

	# Stop parameter
	cam_quit = 0

	# Take first frame
	_, frame = cap.read(0)

	characters = character_recognition(frame)

	for character in c:
		print(character)

	"""
	# Loop over each frame
	while(cam_quit == 0):

		# Take each frame
		_, frame = cap.read()

		characters = character_recognition(frame)

		for character in c:
			print(character)

		# Detect each card
		#card_contours = contour_detection(frame)

		# Warp each card
		#cards = 

		# Apply fonction for each card
		#for (x,y,w,h) in cards:
			
		
		# Display result
		#cv2.imshow('frame',frame)
		
		# Poll the keyboard. If 'q' is pressed, exit the main loop.
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			cam_quit = 1

	cv2.destroyAllWindows()
	"""