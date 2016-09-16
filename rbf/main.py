# gaussian rbf method
import random, math
import numpy as np
import cv2
from scipy import spatial

SIGMA_X = 0.4
SIGMA_Y = 0.4

def gaussian(x,y,a,sigma_x,sigma_y):
	return a*math.exp(-(x**2/(2*sigma_x**2)+(y**2/(2*sigma_y**2))))


def choose_random_pixels(img, samples):
	width = img.shape[1]
	height = img.shape[0]
	output_samples = []

	for i in range(samples):
		rx = random.randint(0,width-1)
		ry = random.randint(0,height-1)

		output_samples.append(((rx,ry),img[ry][rx]))

	return output_samples


def main():
	img = cv2.imread('landscape.png',1)/255.0

	# img = cv2.resize(img,(img.shape[1]/2,img.shape[0]/2))

	pixel_img = np.zeros(img.shape,np.float32)
	output_img = np.zeros(img.shape,np.float32)
	cv2.imshow("original",img)

	pixel_density = (img.shape[0]*img.shape[1])/50

	pixels = choose_random_pixels(img, pixel_density)
	for pixel in pixels:
		pixel_img[pixel[0][1]][pixel[0][0]] = pixel[1]

	tree = spatial.cKDTree([(pixel[0][0],pixel[0][1]) for pixel in pixels])
	cv2.imshow("pixels", pixel_img)

	for y in range(img.shape[0]):
		for x in range(img.shape[1]):

			points = tree.query_ball_point([x,y],20)

			px_value = 0.0
			factor = 0.0

			for point in points:
				dist = gaussian(abs(pixels[point][0][0]-x),abs(pixels[point][0][1]-y),1.0,2.0,2.0)
				px_value += pixels[point][1]*dist
				factor += dist

			if len(points) > 0:
				output_img[y,x] = px_value/factor

			# point = tree.query([x,y])
			# output_img[y,x] = pixels[point[1]][1]
		print "{0}\r".format(y)

	cv2.imshow("output", output_img)
	cv2.waitKey(0)


main()
