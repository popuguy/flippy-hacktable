from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import os
from os.path import isfile, join

blocks_high = 6
blocks_wide = 4


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	the_min = np.argmin(s)
	the_max = np.argmax(s)
	rect[0] = pts[the_min]
	rect[2] = pts[the_max]

	diff = np.diff(pts, axis = 1)
	last_two_min = np.argmin(diff)
	last_two_max = np.argmax(diff)
	rect[1] = pts[last_two_min]
	rect[3] = pts[last_two_max]

	return rect

# Reference: PyImageSearch.com

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def trim_edges(img):
    y,x, rgb = img.shape
    cropx=x*9//10
    cropy=y*9//10 
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)

    return img[starty:starty+cropy,startx:startx+cropx]



# Reference: https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/ 

def convert_frames_to_video(vid_frames, out_file, fps):
    height, width, _ = vid_frames[0].shape
    size = (width, height)

    out = cv2.VideoWriter(out_file,0x00000021, fps, size)
    for i in range(len(vid_frames)):
        # writing to a image array
        out.write(vid_frames[i])
    out.release()

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
def save_vid_of_corrected(corrected, save_file):
	image = corrected
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)

	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)

	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break

	# apply the four point transform to obtain a top-down
	# view of the original image
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

	# warped is final grid here

	y,x,rgb = warped.shape

	one_frame_height = y//6
	one_frame_width = x//4
	frames = []
	for row in range(6):
		for col in range(4):
			base_height = col*one_frame_height
			extra_height = one_frame_height

			base_width = row*one_frame_width
			extra_width = one_frame_width

			print( "base_height: base_height + extra_height, base_width: base_width + extra_width")
			print(base_height, base_height+extra_height, base_width, base_width+extra_width)
			# frame = warped[base_height: base_height + extra_height, base_width: base_width + extra_width]
			frame = warped[base_width: base_width + extra_width, base_height: base_height + extra_height]
			frames.append(frame)

	# convert_frames_to_video(frames, "diditwork.mp4", 5)
	convert_frames_to_video(frames, save_file, 5)


def save_flipbook(img_url, img_data='None', save_file='thing.mp4'):
	image = cv2.imread(img_url) if img_data=='None' else img_data
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)

	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)

	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break

	# get correct view
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

	warped = trim_edges(warped)

	save_vid_of_corrected(warped, save_file)

if __name__ == '__main__':
	save_flipbook('actual_shot.jpg')