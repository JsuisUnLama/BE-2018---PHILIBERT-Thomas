import cv2
import numpy as np
import copy
import random as rng

def dessin(img,name):

	# Etapes noir et blanc:
	# 1: Négativiser
	# 2: Appliquer un flou gaussien (30px de rayon)
	# 3: Eclaircir
	# 4: Noiréblanchir
	# 5: Epaissir puis Affiner (Dilatation puis Erosion?)
	# 6: Baisser l'opacité à 60 (quoi que cela puisse vouloir dire)
	# 7: 
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
	lowThreshold = 50
	ratio = 4
	edges = cv2.Canny(gray,lowThreshold,ratio*lowThreshold,apertureSize = 3)
	minLineLength = 4
	maxLineGap = 10
	lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
	if lines is not None:
		for line in lines:
			for x1,y1,x2,y2 in line:
				cv2.line(edges,(x1,y1),(x2,y2),(0,255,0),2)

	print("Ecriture de",name)
	cv2.imwrite(name,edges)


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

vintage(image,name_o)