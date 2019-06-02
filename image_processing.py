###############################################
#
# Title:	image_processing
#
# Descr:	Recognize crosses and circles, may further be updated with recognizing different animals.
#
# Author:	Tarllark
#
# Team:		Successful Story
#
###############################################

from webget import download as wget
import matplotlib.pyplot as plt
import numpy as np
import cv2

crossURL = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRgNyfVoZCFaHgVuHn6OCUbV2GPDx3AP9MaKN7EEk4ZK9z_y7Ek'
circleURL = 'https://i.pinimg.com/originals/4f/19/7d/4f197daa37eb39738d30c8bbf56f60cc.jpg'
circle2URL = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSJ3aTjnjcEgnlYKUMhYo9vMw8rg6tCGe1XkpY3EL-C4rXC55H5'
def prep(resources = [crossURL,circleURL, circle2URL]):
	images = []
	for resource in resources:
		images.append(cv2.cvtColor(cv2.imread(wget(resource)), cv2.COLOR_BGR2GRAY))
	 
	return images

def findCross(image):
	templates = [cv2.cvtColor(cv2.imread('./cross_template.jpg'), cv2.COLOR_BGR2GRAY), cv2.cvtColor(cv2.imread('./cross_template2.jpg'), cv2.COLOR_BGR2GRAY), cv2.cvtColor(cv2.imread('./thin_Cross.jpg'), cv2.COLOR_BGR2GRAY), cv2.cvtColor(cv2.imread('./thin_Cross2.jpg'), cv2.COLOR_BGR2GRAY)]
	accuracy = 0.8
	output = image.copy()

	for template in templates:
		w, h = template.shape[::-1]
		res = cv2.matchTemplate(output, template, cv2.TM_CCOEFF_NORMED)
		loc = np.where(res >= accuracy)
		for data in zip(*loc[::-1]):
			cv2.rectangle(output, data, (data[0]+w, data[1]+h), (0, 255, 255), 2)
	cv2.imshow('output', np.hstack([image, output]))
	cv2.waitKey(0)


def findCircles(image):
	minDist = 100 #Minimum distance in pixels
	output = image.copy()
	circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, minDist)
	if circles is not None:
		circles = np.round(circles[0, :]).astype("int")
		for(x, y, r) in circles:
			cv2.circle(output, (x,y), r, (0, 255, 0), 4)
			cv2.rectangle(output, (x -5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
		cv2.imshow("output", np.hstack([image, output]))
		cv2.waitKey(0)
		
	
if __name__ == '__main__':
	for image in prep():
		findCross(image)
		findCircles(image)
		
	