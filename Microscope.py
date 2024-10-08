# from random import randrange
import matplotlib.pyplot as plt
# import numpy as np
# import cv2 as cv
import glob
import pandas as pd

# def plot_image(img, figsize_in_inches=(5,5)):
#     fig, ax = plt.subplots(figsize=figsize_in_inches)
#     ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#     plt.show()
    
# def plot_images(imgs, figsize_in_inches=(5,5)):
#     fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
#     for col, img in enumerate(imgs):
#         axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#     plt.show()


# figsize = (10, 10)

# # rgb_l = cv2.cvtColor(cv2.imread("./photos/2024-09-03-105747.jpg"), cv2.COLOR_BGR2RGB)
# # gray_l = cv2.cvtColor(rgb_l, cv2.COLOR_RGB2GRAY)
# # rgb_r = cv2.cvtColor(cv2.imread("./photos/2024-09-03-105806.jpg"), cv2.COLOR_BGR2RGB)
# # gray_r = cv2.cvtColor(rgb_r, cv2.COLOR_RGB2GRAY)


# from stitching import Stitcher
# stitcher = Stitcher(detector="sift", confidence_threshold=0.1)

# # # 
# panorama = stitcher.stitch(["./photos/2024-09-03-105737.jpg", "./photos/2024-09-03-105747.jpg", "./photos/2024-09-03-105806.jpg", "./photos/2024-09-03-110711.jpg"])
# plot_image(panorama, (20,20))

# # # use orb if sift is not installed
# # feature_extractor = cv2.SIFT_create()

# # # find the keypoints and descriptors with chosen feature_extractor
# # kp_l, desc_l = feature_extractor.detectAndCompute(gray_l, None)
# # kp_r, desc_r = feature_extractor.detectAndCompute(gray_r, None)

# # test = cv2.drawKeypoints(rgb_l, kp_l, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # # plt.figure(figsize=figsize)
# # # plt.imshow(test)
# # # plt.title("keypoints")
# # # plt.show()


# # bf = cv2.BFMatcher()
# # matches = bf.knnMatch(desc_l, desc_r, k=2)

# # # Apply ratio test
# # good_and_second_good_match_list = []
# # for m in matches:
# #     if m[0].distance/m[1].distance < 0.5:
# #         good_and_second_good_match_list.append(m)
# # good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]

# # # show only 30 matches
# # im_matches = cv2.drawMatchesKnn(rgb_l, kp_l, rgb_r, kp_r,
# #                                 good_and_second_good_match_list[0:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # plt.figure(figsize=(20, 20))
# # plt.imshow(im_matches)
# # plt.title("keypoints matches")
# # plt.show()

# # good_kp_l = np.array([kp_l[m.queryIdx].pt for m in good_match_arr])
# # good_kp_r = np.array([kp_r[m.trainIdx].pt for m in good_match_arr])
# # H, masked = cv2.findHomography(good_kp_r, good_kp_l, cv2.RANSAC, 5.0)

# # print(H)

# # rgb_r_warped = cv2.warpPerspective(rgb_r, H, (rgb_l.shape[1] + rgb_r.shape[1], rgb_l.shape[0]))
# # rgb_r_warped[0:rgb_l.shape[0], 0:rgb_l.shape[1]] = rgb_l

# # plt.figure(figsize=figsize)
# # plt.imshow(rgb_r_warped)
# # plt.title("naive warping")
# # plt.show()

# # def warpTwoImages(img1, img2, H):
# #     '''warp img2 to img1 with homograph H
# #     from: https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
# #     '''
# #     h1, w1 = img1.shape[:2]
# #     h2, w2 = img2.shape[:2]
# #     pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
# #     pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
# #     pts2_ = cv2.perspectiveTransform(pts2, H)
# #     pts = np.concatenate((pts1, pts2_), axis=0)
# #     [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
# #     [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
# #     t = [-xmin, -ymin]
# #     Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

# #     result = cv2.warpPerspective(img2, Ht@H, (xmax-xmin, ymax-ymin))
# #     result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
# #     return result


# # result = warpTwoImages(rgb_l, rgb_r, H)

# # plt.figure(figsize=figsize)
# # plt.imshow(result)
# # plt.title("better warping")
# # plt.show()


# from stitch2d import create_mosaic


# from stitch2d import StructuredMosaic

# mosaic = StructuredMosaic(
#     "./photos",
#     dim=4,                  # number of tiles in primary axis
#     origin="upper left",     # position of first tile
#     direction="horizontal",  # primary axis (i.e., the direction to traverse first)
#     pattern="snake"          # snake or raster
#   )


# mosaic.save("mosaic.jpg")





import cv2
import numpy as np
import math

# input image
path = "image1.jpg"
# 1 EUR coin diameter in cm
coinDiameter = 2.325
# real area for the coin in cm^2
coinArea = (coinDiameter/2)**2 * math.pi
# initializing the multiplying factor for real size
realAreaPerPixel = 1


# pixel to cm^2
def pixelToArea(objectSizeInPixel, coinSizeInPixel):
    # how many cm^2 per pixel?
    realAreaPerPixel = coinArea / coinSizeInPixel
    print("realAreaPerPixel: ", realAreaPerPixel)
    # object area in cm^2
    objectArea = realAreaPerPixel * objectSizeInPixel
    return objectArea    


# finding coin and steak contours
def getContours(img, imgContour):
    
    # find all the contours from the B&W image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # needed to filter only our contours of interest
    finalContours = []
    
    # for each contour found
    for cnt in contours:
        # cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 2)
        # find its area in pixel
        area = cv2.contourArea(cnt)
        print("Detected Contour with Area: ", area)

        # minimum area value is to be fixed as the one that leaves the coin as the small object on the scene
        if (area > 5000):
            perimeter = cv2.arcLength(cnt, True)
            
            # smaller epsilon -> more vertices detected [= more precision]
            epsilon = 0.002*perimeter
            # check how many vertices         
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            #print(len(approx))
            
            finalContours.append([len(approx), area, approx, cnt])

    # we want only two objects here: the coin and the meat slice
    print("---\nFinal number of External Contours: ", len(finalContours))
    # so at this point finalContours should have only two elements
    # sorting in ascending order depending on the area
    finalContours = sorted(finalContours, key = lambda x:x[1], reverse=False)
    
    # drawing contours for the final objects
    for con in finalContours:
        cv2.drawContours(imgContour, con[3], -1, (0, 0, 255), 3)

    return imgContour, finalContours

def testThresholds():    
	# sourcing the input image
	# img = cv2.imread('./my_photo-2.jpg')
	# img = cv2.imread('./2024-09-04-144715.jpg')
	img = cv2.imread('./calibration/2024-09-04-145805.jpg')
	Cpy = img.copy()

	# cv2.imshow("image", img)
	# cv2.waitKey()
	# cv2.destroyAllWindows()


	# # blurring
	# imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
	# # graying
	# imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
	# # canny
	# imgCanny = cv2.Canny(imgGray, 255, 195)

	# kernel = np.ones((2, 2))
	# imgDil = cv2.dilate(imgCanny, kernel, iterations = 3)
	# # cv2.imshow("Diluted", imgDil)
	# imgThre = cv2.erode(imgDil, kernel, iterations = 3)

	# imgFinalContours, finalContours = getContours(imgThre, img)

	# # first final contour has the area of the coin in pixel
	# coinPixelArea = finalContours[0][1]
	# print("Coin Area in pixel", coinPixelArea)
	# # second final contour has the area of the meat slice in pixel
	# slicePixelArea = finalContours[1][1]
	# print("Entire Slice Area in pixel", slicePixelArea)

	# # let's go cm^2
	# print("Coin Area in cm^2:", coinArea)
	# print("Entire Slice Area in cm^2:", pixelToArea(slicePixelArea, coinPixelArea))

	# # show  the contours on the unfiltered starting image
	# cv2.imshow("Final External Contours", imgFinalContours)
	# cv2.waitKey()


	# # now let's detect and quantify the lean part

	# # convert to HSV
	# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
	# # set lower and upper color limits
	# lowerVal = np.array([0, 159, 160])
	# upperVal = np.array([101, 255, 253])
	# # Threshold the HSV image to get only red colors
	# mask = cv2.inRange(hsv, lowerVal, upperVal)
	# # apply mask to original image - this shows the red with black blackground
	# final = cv2.bitwise_and(img, img, mask= mask)

	# # show selection
	# cv2.imshow("Lean Cut", final)
	# cv2.waitKey()

	# # convert it to grayscale because countNonZero() wants 1 channel images
	# gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
	# # cv2.imshow("Gray", gray)
	# # cv2.waitKey()
	# meatyPixelArea = cv2.countNonZero(gray)

	# print("Red Area in pixel: ", meatyPixelArea)
	# print("Red Area in cm^2: ", pixelToArea(meatyPixelArea, coinPixelArea))

	# # finally the body-fat ratio calculation
	# print("Body-Fat Ratio: ", meatyPixelArea/slicePixelArea*100, "%")



	# Convert to grayscale 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

	# cv2.imshow('image', gray) 
	# cv2.waitKey(0) 
	# cv2.destroyAllWindows()

	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	# cv2.imshow("blur", blur)
	  
	# #to separate the object from the background 
	# ret, thresh = cv2.threshold(blur, 127, 255, 0) 

	# ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
	# th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	#             cv2.THRESH_BINARY,11,2)
	th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
	            cv2.THRESH_BINARY,7,2)
	th4 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv2.THRESH_BINARY,7,2)
	# th5 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	#             cv2.THRESH_BINARY,15,2)
	# # Otsu's thresholding
	# ret4,th4 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# # Otsu's thresholding with blur
	# ret5,th5= cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	 
	titles = ['Original Image', 'Grayscale',
	            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']#, 'Otsu Thresholding', 'test2']
	images = [img, gray, th3, th4]#, th4, gray]
	 
import os 

def calibrate_and_save_parameters():
	# ------------------------------
	# ENTER YOUR REQUIREMENTS HERE:
	ARUCO_DICT = cv2.aruco.DICT_4X4_250
	SQUARES_VERTICALLY = 10
	SQUARES_HORIZONTALLY = 8
	SQUARE_LENGTH = 3.5E-6
	MARKER_LENGTH = SQUARE_LENGTH/2
	# ...
	PATH_TO_YOUR_IMAGES = './ChArUco_4'
	# ------------------------------
	# Define the aruco dictionary and charuco board
	dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
	board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
	# board.setLegacyPattern(True)

	params = cv2.aruco.DetectorParameters()

	# Load PNG images from folder
	# image_files = [os.path.join(PATH_TO_YOUR_IMAGES, f) for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".png")]
	# image_files.sort()  # Ensure files are in order

	all_charuco_corners = []
	all_charuco_ids = []

	cam = cv2.VideoCapture(0)

# for image_file in image_files:
	while True:
		ret, image = cam.read()
		# image_copy = image.copy()

	
		# image = cv2.imread(image_file)
		image_copy = image.copy()
		# cv2.imshow('image', image)
		# cv2.waitKey()
		marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)

		# If at least one marker is detected
		if len(marker_ids) > 0:
			cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
			cv2.imshow('image_copy', image_copy)
			# cv2.waitKey()
			charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
			# print(charuco_retval)
			# print(charuco_corners)
			if charuco_retval:
				all_charuco_corners.append(charuco_corners)
				all_charuco_ids.append(charuco_ids)

		# Press 'q' to exit the loop
		if cv2.waitKey(1) == ord('q'):
		    break

	# Calibrate camera
	# print(all_charuco_corners)
	# print(all_charuco_ids)
	# print(board)
	retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, image.shape[:2], None, None)
	# print(retval)

	# Save calibration data
	np.save('camera_matrix.npy', camera_matrix)
	np.save('dist_coeffs.npy', dist_coeffs)

	# # Iterate through displaying all the images
	# for image_file in image_files:
	# 	image = cv2.imread(image_file)
	# 	undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
	# 	cv2.imshow('Undistorted Image', undistorted_image)
	# 	cv2.waitKey(0)

	cv2.destroyAllWindows()
	print('---------------------------')


# calibrate_and_save_parameters()

def calibration(img, show_calibration=False):
	# Camera calibration 


	# Set up calibration chessboard

	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	num_rows = 5
	num_cols = 6
	 
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros(((num_cols+1)*(num_rows+1),3), np.float32)
	objp[:,:2] = np.mgrid[0:(num_rows+1),0:(num_cols+1)].T.reshape(-1,2)
	 
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.
	 
	# images = glob.glob('*.jpg')
	 
	# for fname in images:
	# img = cv2.imread('./calibration/2024-09-04-153634.jpg')
	# img = cv2.imread('./calibration/calibration_grid2.png')
	cv2.imshow('img', img)
	# cv2.waitKey()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imshow('gray', gray)
	cv2.waitKey()

	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(gray, ((num_rows+1),(num_cols+1)), None)
	# print(corners)

	# If found, add object points, image points (after refining them)
	if ret == True:
	    objpoints.append(objp)

	    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
	    imgpoints.append(corners2)

	    # Draw and display the corners
	    if show_calibration:
		    cv2.drawChessboardCorners(img, ((num_rows+1),(num_cols+1)), corners2, ret)
		    cv2.imshow('img', img)
		    cv2.waitKey(0)

	cv2.destroyAllWindows()



	#Perform calibration
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)



	# Undistoration correction
	# img = cv2.imread('2024-09-04-145841.jpg')
	img = cv2.imread('./calibration/calibration_test.png')
	if show_calibration:
		cv2.imshow('img', img)
		cv2.waitKey()
	h,  w = img.shape[:2]
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


	# undistort
	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

	# Re-projection error
	mean_error = 0
	for i in range(len(objpoints)):
	    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
	    mean_error += error
	 
	# print( "total error: {}".format(mean_error/len(objpoints)) )
	 
	# crop the image
	x, y, w, h = roi
	dst = dst[y:y+h, x:x+w]
	if show_calibration:
		cv2.imshow('calibresult.png', dst)
		cv2.waitKey()

	cv2.destroyAllWindows()


	# np.savetxt('camera_distortion_coefficients.txt', (ret, mtx, dist, rvecs, tvecs))   # x,y,z equal sized 1D arrays

	return ret, mtx, dist, rvecs, tvecs


# calibration()

# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

# # apply Otsu's automatic thresholding which automatically determines
# # the best threshold value
# (ret, thresh) = cv2.threshold(blur, 0, 255,
# 	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# # cv2.imshow("Threshold", thresh)
# print("[INFO] otsu's thresholding value: {}".format(ret))


# rowSum = Cpy[...,0].sum(axis=0)
# colSum = Cpy[...,0].sum(axis=1)

# rows = np.zeros_like(Cpy)
# cols = np.zeros_like(Cpy) 
# mask = np.zeros_like(Cpy)

# # Not sure if these values will work always
# rows[:, rowSum>100] = 1
# cols[colSum>200, :] = 1

# mask = rows*cols

# y0 = np.min(np.nonzero(mask.sum(axis=1))[0])
# y1 = np.max(np.nonzero(mask.sum(axis=1))[0])

# x0 = np.min(np.nonzero(mask.sum(axis=0))[0])
# x1 = np.max(np.nonzero(mask.sum(axis=0))[0])

# mask[y0:y1, x0:x1] = 1

# mask1 = mask*rows
# mask2 = mask*cols

# mask = np.maximum(mask1, mask2)


def opposingKernels():
	image = cv2.imread('./2024-09-04-144715.jpg')
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,5)

	# Remove horizontal
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
	detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
	cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	for c in cnts:
	    cv2.drawContours(image, [c], -1, (255,255,255), 2)

	# Repair image
	repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
	result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

	cv2.imshow('thresh', thresh)
	cv2.imshow('detected_lines', detected_lines)
	cv2.imshow('image', image)
	cv2.imshow('result', result)
	cv2.waitKey()


# opposingKernels()


# SE = np.ones((16,16))
# dilated = cv2.dilate(mask, SE)
# dilated [...,1:3] = 0

# from skimage.measure import label

# labelled = label(1-dilated [...,0])


def tryHough():
	### LINES WITH HOUGHLINES()

	# Convert the resulting image from previous step (no text) to gray colorspace.
	res2 = cv2.imread('./2024-09-04-144715.jpg')
	gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

	# You can either use threshold or Canny edge for HoughLines().
	# _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	edges = cv2.Canny(gray, 10, 50, apertureSize=3)
	kernel = np.ones((2,2),np.uint8)
	dilation = cv2.dilate(edges,kernel,iterations = 1)

	# Perform HoughLines tranform.  
	lines = cv2.HoughLines(edges,1,np.pi/180,200)
	for line in lines:
	    for rho,theta in line:
	            a = np.cos(theta)
	            b = np.sin(theta)
	            x0 = a*rho
	            y0 = b*rho
	            x1 = int(x0 + 1000*(-b))
	            y1 = int(y0 + 1000*(a))
	            x2 = int(x0 - 1000*(-b))
	            y2 = int(y0 - 1000*(a))

	            cv2.line(res2,(x1,y1),(x2,y2),(0,0,255),2)

	# Find the contours of the object  
	
	# _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,5)

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
	cv2.drawContours(res2, contours, -1, (0,255,0), 3) 

	#Display the result.
	# cv2.imshow('res', res)
	cv2.imshow('res2', res2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# tryHough()


def tryHoughP():
	### LINES WITH HOUGHLINESP()

	# Convert the resulting image from first step (no text) to gray colorspace.
	res3 = cv2.imread('./2024-09-04-144715.jpg')
	gray = cv2.cvtColor(res3, cv2.COLOR_BGR2GRAY) 

	# Use Canny edge detection and dilate the edges for better result.
	edges = cv2.Canny(gray, 50, 150, apertureSize=3)
	kernel = np.ones((4,4),np.uint8)
	dilation = cv2.dilate(edges,kernel,iterations = 1)

	# Perform HoughLinesP tranform.  
	minLineLength = 100
	maxLineGap = 10
	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength, maxLineGap)
	for line in lines:
	    for x1, y1, x2, y2 in line:
	        cv2.line(res3, (x1, y1), (x2, y2), (0, 255, 0), 2)

	#Display the result.
	# cv2.imwrite('h_res1.png', res)
	# cv2.imwrite('h_res2.png', res2)
	cv2.imwrite('h_res3.png', res3)

	# cv2.imshow('res', res)
	# cv2.imshow('res2', res2)
	cv2.imshow('res3', res3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


	# labelled[labelled==1] = 0
	# labelled[labelled >0] = 1

	# labelled = labelled.astype(np.uint8)

	# res = cv2.bitwise_and(img,img,mask = labelled)

	# cv2.namedWindow('Splitted Images', cv2.WINDOW_NORMAL)
	# cv2.imshow('Splitted Images', res)
	# cv2.waitKey(0)

# tryHoughP()

def findLines():
	res3 = cv2.imread('./2024-09-04-144715.jpg')
	gray = cv2.cvtColor(res3, cv2.COLOR_BGR2GRAY) 
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	# thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	#             cv2.THRESH_BINARY,7,2)
	# Detect horizontal lines
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
	detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
	cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	for c in cnts:
		cv2.drawContours(res3, [c], -1, (36,255,12), 2)

	cv2.imshow('res3', res3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
  


# findLines()

# Find the contours of the object  
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
  
# # Draw the contours on the original image 
# cv2.drawContours(img, contours, -1, (0,255,0), 3) 
  
# # # Get the area of the object in pixels 
# # area = cv2.contourArea(contours[0]) 
  
# # # Convert the area from pixels to a real-world unit of measurement (e.g. cm^2) 
# # scale_factor = 0.1 # 1 pixel = 0.1 cm 
# # size = area * scale_factor ** 2
  
# # # Print the size of the object 
# # print('Size:', size) 
  
# # # Display the image with the contours drawn 
# # cv2.imwrite('Object.jpeg', img) 
# cv2.imshow('Countours', img) 
# cv2.waitKey(0) 
  
# # # Save the image with the contours drawn to a file 
# # cv2.imwrite('object_with_contours.jpg', img)

def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    # print(p1)
    # print(p2)
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    # print(dis)
    return dis

def measureGridSize(img, show_plot=True):
	# get the areas of square black areas
	# Mat squaresContours = src.clone();
	# img = cv2.imread('./Camera.png')
	# img = cv2.imread('./test_image.jpeg')


	# # Undistoration correction
	# # img = cv2.imread('2024-09-04-145841.jpg')
	# # img = cv2.imread('./calibration/calibration_test.png')
	# # if show_calibration:
	# # cv2.imshow('img', img)
	# # cv2.waitKey()
	# h,  w = img.shape[:2]
	# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


	# # undistort
	# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

	# # Re-projection error
	# # mean_error = 0
	# # for i in range(len(objpoints)):
	# #     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	# #     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
	# #     mean_error += error
	 
	# # print( "total error: {}".format(mean_error/len(objpoints)) )
	 
	# # crop the image
	# x, y, w, h = roi
	# dst = dst[y:y+h, x:x+w]
	# # if show_calibration:
	# # cv2.imshow('calibresult.png', dst)
	# # cv2.waitKey()




	# squaresContours = img.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	grid_area_threshold = 10
	known_grid_width = 4.91 # mm

	# print(gray.shape)

	outerBox = np.zeros(gray.shape, np.uint8)
	# h,w = vis.shape
	# vis2 = cv.CreateMat(h, w, cv.CV_8UC1)
	# vis0 = cv.fromarray(vis)

	gray = cv2.GaussianBlur(gray,(3,3),0)
	# GaussianBlur(gray, gray, Size(3, 3), 0);

	#     adaptiveThreshold(gray, outerBox, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);
	outerBox = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,2)

	# bitwise_not(outerBox, outerBox);
	outerBox = cv2.bitwise_not(outerBox)

	#     Mat kernel = (Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
	kernel = np.array((
		[0, 1, 0],
		[1, 1, 1],
		[0, 1, 0]), dtype="uint8")

	#     dilate(outerBox, outerBox, kernel);
	outerBox = cv2.dilate(outerBox,kernel,iterations = 1)

	# kernel = np.ones((30,30))
	outerBox = cv2.morphologyEx(outerBox, cv2.MORPH_CLOSE, np.ones((2,2)))
	outerBox = cv2.morphologyEx(outerBox, cv2.MORPH_OPEN, np.ones((3,3)))

	outerBox = cv2.erode(outerBox,np.ones((3,4),np.uint8),iterations = 1)






	# cv2.imshow('original', img)
	# cv2.imshow('gray', outerBox)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


	# # vector areas;
	# # vector<vector > contours;
	# # vector<vector > resContours;
	# # vector hierarchy;

	# contours, hierarchy  = cv2.findContours(outerBox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# contours, hierarchy  = cv2.findContours(outerBox, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours, hierarchy  = cv2.findContours(outerBox, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# findContours(outerBox, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
	area_list = []
	contour_list = []

	# for (int i = 0; i < contours.size(); i++) {
	for contour in contours:
		# cv2.drawContours(img, [contour], -1, (36,255,12), 2)
		# print('h')

		# // find moments of the image
		# Moments m = moments(127, true);
		M = cv2.moments(contour)
		# print(M)

		# Point center (m.m10 / m.m00, m.m01 / m.m00);
		try:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		except ZeroDivisionError:
			# print(M["m10"], M["m00"])
			# print(M["m01"], M["m00"])
			cX = 0
			cY = 0
		# print(outerBox[cY,cX])
		# print(outerBox)

		# if (outerBox.at(center) == 0) {//the center is black so certainly the area
		if (outerBox[cY,cX] == 0) and (cY > 0) and (cX > 0):
			# print("h")
			# cv2.drawContours(img, [contour], -1, (0,255,0), 2)
			# vector approxCurve;
			# approxPolyDP(contours[i], approxCurve, 3, true);
			approxCurve = cv2.approxPolyDP(contour, 50, True)
			# if (approxCurve.size() == 4) {//the area is rectangle
			if len(approxCurve)==4: 
				# print(approxCurve)
				# print('-----')
				# //lengths of sides
				a = distanceCalculate(approxCurve[0][0], approxCurve[1][0])
				b = distanceCalculate(approxCurve[0][0], approxCurve[2][0])
				c = distanceCalculate(approxCurve[1][0], approxCurve[0][0])
				d = distanceCalculate(approxCurve[1][0], approxCurve[2][0])
				(x,y,w,h) = cv2.boundingRect(contour)
				delta = 0.2;
				# double area = contourArea(contours[i]);
				area = area = cv2.contourArea(contour)
				# print(area)
				# print(area,w,h)
				cv2.drawContours(img, [contour], -1, (255,0,0), 3)  # plot all found contours in blue
				cv2.drawContours(outerBox, [contour], -1, (255,0,0), 3)  # plot all found contours in blue


				# print(((np.max([a,b]) / np.min([a,b]) -1 ) <delta ))
				# print(a,b,c,d)

				# if (max(a, b) / min(a, b)-1 < delta && max(a, c) / min(a, c)-1 < delta
					# && max(a, d) / min(a, d)-1 < delta && max(b, c) / min(b, c)-1 < delta
					# && max(b, d) / min(b, d)-1 < delta && max(c, d) / min(c, d)-1 < delta
					# && area>30) {
				if (abs((w/h)-1) < delta) and (area > grid_area_threshold) and (abs((w*h/area)-1) < delta) :
				# if (abs((w/h)-1) < delta) and (area > grid_area_threshold)  :
				# if ((np.max([a,b]) / np.min([a,b]) -1 ) <delta ) and ((np.max([a,c]) / np.min([a,c]) -1 ) <delta ) and ((np.max([a,d]) / np.min([a,d]) -1 ) <delta ) and ((np.max([b,c]) / np.min([b,c]) -1 ) <delta ) and ((np.max([b,d]) / np.min([b,d]) -1 ) <delta ) and ((np.max([c,d]) / np.min([c,d]) -1 ) <delta ) and (area > grid_area_threshold):
						# //the area is square
						# print('test')
						# areas.push_back(area);
						# cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
						if area > 0:

							area_list.append(area)
							# resContours.push_back(contours[i]);
							contour_list.append(contour)
						# }
			#         }
		#     }
	    
	# }
	# drawContours(squaresContours, resContours, -1, Scalar(0, 0, 255));
	# imwrite("squaresContours.jpg", squaresContours);
	# //sort the areas
	# int temp;
	# for (int i = areas.size() - 1; i > 0; i--) {
	#     bool tableau_trie = true;
	#     for (int j = 0; j < i; j++) {
	#         if (areas[j + 1] < areas[j]) {
	#             temp = areas[j];
	#             areas[j] = areas[j + 1];
	#             areas[j + 1] = temp;
	#             tableau_trie = false;
	#         }
	#     }
	#     if (tableau_trie)
	#         break;
	# }

	# print(area_list)

	for (area,contour) in zip(area_list,contour_list):
		cv2.drawContours(img, [contour], -1, (0,255,0), 1)  # plot square-like contours in green
		# cv2.drawContours(outerBox, [contour], -1, (0,255,0), 1)  # plot square-like contours in green
		# continue

	# //take the median
	# int median = areas[areas.size() / 2];
	# cout << "black area median = " << median << endl;
	# median = np.median(area_list)
	area_list.sort()
	if len(area_list) < 2:
		median = np.median(area_list)
	elif len(area_list) < 3:
		median = np.median(area_list[1:-1])
	else:
		median = np.median(area_list[2:-2])
	pixels_per_mm = 0
	# //take the side of a square which is the number of pixels per millimeter
	# cout << "number of pixels per millimeter = " << sqrt(median) << endl;`Preformatted text`
	if np.size(median) == 1:
		pixels_per_mm = np.sqrt(median) / known_grid_width

		# print(pixels_per_mm)
		# # print(area_list)
		# # print('-----------------------')

	# cv2.imshow('original', img)
	# # cv2.waitKey(0)
	# cv2.imshow('grayscale', outerBox)
	# cv2.waitKey(0)
	# # cv2.destroyAllWindows()

	# if len(area_list) < 2:
	# 	print(np.mean(area_list/(pixels_per_mm*pixels_per_mm)))
	# elif len(area_list) < 3:
	# 	print(np.mean(area_list[1:-1]/(pixels_per_mm*pixels_per_mm)))
	# else:
	# 	print(np.mean(area_list[2:-2]/(pixels_per_mm*pixels_per_mm)))


	for contour in contour_list:
		area = cv2.contourArea(contour)
		if area > grid_area_threshold:

			x, y, w, h = cv2.boundingRect(contour) 
			# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 
			cv2.putText(img, '{0:.2f}'.format(area/(pixels_per_mm*pixels_per_mm)), (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2) 

			# cv2.drawContours(img, [contour], -1, (0,255,0), 1)  # plot square-like contours in green
			# cv2.drawContours(outerBox, [contour], -1, (0,255,0), 1)  # plot square-like contours in green
			# print(area)

	# Perspective transform
	# All points are in format [cols, rows]
	pt_A = [18, 8]
	pt_B = [20, 468]
	pt_C = [623, 467]
	pt_D = [625, 15]

	# Here, I have used L2 norm. You can use L1 also.
	width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
	width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
	maxWidth = max(int(width_AD), int(width_BC))


	height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
	height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
	maxHeight = max(int(height_AB), int(height_CD))

	input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
	output_pts = np.float32([[0, 0],
	                        [0, maxHeight - 1],
	                        [maxWidth - 1, maxHeight - 1],
	                        [maxWidth - 1, 0]])

	# Compute the perspective transform M
	M = cv2.getPerspectiveTransform(input_pts,output_pts)

	return pixels_per_mm#, M, maxWidth, maxHeight


# measureGridSize(cv2.imread('./Camera.png'))







def measureArea(img, pixels_per_mm):
	# img = cv2.imread('./Camera4.png')
	# cv2.imshow('original', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	area_threshold = 100


	# Convert to grayscale 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

	# cv2.imshow('image', gray) 
	# cv2.waitKey(0) 
	# cv2.destroyAllWindows()


	#
	# Hough Line detection
	#

	# # Find the edges in the image using canny detector
	# edges = cv2.Canny(gray, 50, 200)
	# # Detect points that form a line
	# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)
	# # Draw lines on the image
	# for line in lines:
	#     x1, y1, x2, y2 = line[0]
	#     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

	# # Show result
	# # img = cv2.resize(img, dsize=(600, 600))
	# cv2.imshow("Result Image", img)
	# cv2.waitKey(0) 
	# cv2.destroyAllWindows()



	#
	# Threshold detection
	#

	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	# cv2.imshow("blur", blur)
	  
	# #to separate the object from the background 
	# ret, thresh = cv2.threshold(blur, 127, 255, 0) 

	ret,th1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)
	# th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.THRESH_BINARY,11,2)
	# th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY,7,2)
	# th4 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.THRESH_BINARY,7,2)
	# th5 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.THRESH_BINARY,15,2)
	# # Otsu's thresholding
	# ret6,th6 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# # Otsu's thresholding with blur
	# ret7,th7= cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	 
	# titles = ['Original Image', 'Grayscale', 'Binary ',
	#             'Adaptive Gaussian (11) ', 'Adaptive Mean', 'Adaptive Gaussian (7) ', 'Adaptive Gaussian (15) ', 'Otsu ', 'Otsu w/ Blur']
	# images = [img, gray, th1, th2, th3, th4, th5, th6, th7]


	# for i in range(9):
	#     plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
	#     plt.title(titles[i])
	#     plt.xticks([]),plt.yticks([])
	# plt.show(block=False)
	# plt.pause(0.05)
	# plt.close()



	contours,hierarchy = cv2.findContours(th1, 1, 2)
	# print(len(contours))
	for contour in contours:
		area = cv2.contourArea(contour)
		if area > area_threshold:

			x, y, w, h = cv2.boundingRect(contour) 
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 
			cv2.putText(img, '{0:.3f} cm2'.format(area/(pixels_per_mm*pixels_per_mm*100)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 

			cv2.drawContours(img, [contour], -1, (0,255,0), 1)  # plot square-like contours in green
			# cv2.drawContours(outerBox, [contour], -1, (0,255,0), 1)  # plot square-like contours in green
			# print(area)

	# print('---------------')
	# return images, titles



	#
	# Fast Line detection
	#

	# # Init. the fast-line-detector (fld)
	# fld = cv2.ximgproc.createFastLineDetector().detect(gray)

	# # Detect the lines
	# for line in fld:

	#     # Get current coordinates
	#     x1 = int(line[0][0])
	#     y1 = int(line[0][1])
	#     x2 = int(line[0][2])
	#     y2 = int(line[0][3])

	#     # Draw the line
	#     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 5)

	# # Display
	# cv2.imshow("img", img)
	# cv2.waitKey(0)



def running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0)) 
	# print(cumsum)
	return (cumsum[N:] - cumsum[:-N]) / N




def acquireFromCamera(show_calibration=False):
	# Open the default camera (default was 0)
	cam = cv2.VideoCapture(0)

	# Get the default frame width and height
	frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# # Define the codec and create VideoWriter object
	# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

	# ret, frame = cam.read()

	# ret, mtx, dist, rvecs, tvecs = calibration(frame,True)

	pixel_scale_list = []
	# i=0

	# Load calibration data
	camera_matrix = np.load('camera_matrix.npy')
	dist_coeffs = np.load('dist_coeffs.npy')

	# Perspective transform
	# All points are in format [cols, rows]
	pt_A = [18, 8]
	pt_B = [20, 468]
	pt_C = [623, 467]
	pt_D = [625, 15]

	# Here, I have used L2 norm. You can use L1 also.
	width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
	width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
	maxWidth = max(int(width_AD), int(width_BC))


	height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
	height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
	maxHeight = max(int(height_AB), int(height_CD))

	input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
	output_pts = np.float32([[0, 0],
	                        [0, maxHeight - 1],
	                        [maxWidth - 1, maxHeight - 1],
	                        [maxWidth - 1, 0]])

	# Compute the perspective transform M
	M = cv2.getPerspectiveTransform(input_pts,output_pts)

	while True:
		ret, frame = cam.read()
		frame = cv2.warpPerspective(frame,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
		frame_copy = frame.copy()

		# Write the frame to the output file
		# out.write(frame)
		# calibrate_and_save_parameters(frame)

		pixels_per_mm = measureGridSize(frame)#,ret, mtx, dist, rvecs, tvecs)
		if not np.isnan(pixels_per_mm):
			pixel_scale_list.append(pixels_per_mm)

		avg_pixel_scale2 = np.convolve(pixel_scale_list, np.ones(len(pixel_scale_list))/len(pixel_scale_list), mode='valid')

		if len(pixel_scale_list) <25:
			avg_pixel_scale = running_mean(pixel_scale_list, len(pixel_scale_list))[0]
		else:
			avg_pixel_scale = running_mean(pixel_scale_list, 20)[-1]

		# print('Convolve average', avg_pixel_scale2)
		# print('Running mean average', avg_pixel_scale)
		measureArea(frame_copy, avg_pixel_scale)
		# out_frame = cv2.warpPerspective(frame,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
		# cv2.imshow('Perspective Transform', frame_copy)

		# if show_calibration:
		cv2.imshow('Area Detection', frame_copy)


		# undistorted_image = cv2.undistort(frame_copy, camera_matrix, dist_coeffs)
		# cv2.imshow('undistorted_image', undistorted_image)

		# for i in range(9):
		# 	plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
		# 	plt.title(titles[i])
		# 	plt.xticks([]),plt.yticks([])


		# Display the captured frame
		# if show_plot:
		if show_calibration:
			cv2.imshow('Camera (Press q to exit)', frame)
		# cv2.imwrite('out.jpeg',frame)

		# Press 'q' to exit the loop
		if cv2.waitKey(1) == ord('q'):
			break

		# i = i+1

	# Release the capture and writer objects
	cam.release()
	# out.release()
	cv2.destroyAllWindows()

	print('-------------')
	print(avg_pixel_scale)

acquireFromCamera(True)

def testPerspective(image):
	# Load calibration data
	camera_matrix = np.load('camera_matrix.npy')
	dist_coeffs = np.load('dist_coeffs.npy')
	img = cv2.imread(image)
	frame_copy = img.copy()
	# Perspective transform
	# All points are in format [cols, rows]
	pt_A = [18, 8]
	pt_B = [20, 468]
	pt_C = [623, 467]
	pt_D = [625, 15]

	# Here, I have used L2 norm. You can use L1 also.
	width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
	width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
	maxWidth = max(int(width_AD), int(width_BC))


	height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
	height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
	maxHeight = max(int(height_AB), int(height_CD))

	input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
	output_pts = np.float32([[0, 0],
	                        [0, maxHeight - 1],
	                        [maxWidth - 1, maxHeight - 1],
	                        [maxWidth - 1, 0]])

	# Compute the perspective transform M
	M = cv2.getPerspectiveTransform(input_pts,output_pts)

	transformed = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)

	undistorted_image = cv2.undistort(frame_copy, camera_matrix, dist_coeffs)

	undistorted_image2 = cv2.undistort(transformed, camera_matrix, dist_coeffs)
	

	cv2.imshow('transformed image', transformed)
	cv2.imshow('original image', img)
	cv2.imshow('undistorted_image', undistorted_image)
	cv2.imshow('transformed and undistorted', undistorted_image2)
	cv2.waitKey(0)

	
# testPerspective('original_image.png')

def testDistortion(image):
	# Load calibration data
	# camera_matrix = np.load('camera_matrix.npy')
	# dist_coeffs = np.load('dist_coeffs.npy')
	img = cv2.imread(image)
	frame_copy = img.copy()
	# # Perspective transform
	# # All points are in format [cols, rows]
	# pt_A = [18, 8]
	# pt_B = [20, 468]
	# pt_C = [623, 467]
	# pt_D = [625, 15]

	# # Here, I have used L2 norm. You can use L1 also.
	# width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
	# width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
	# maxWidth = max(int(width_AD), int(width_BC))


	# height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
	# height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
	# maxHeight = max(int(height_AB), int(height_CD))

	# input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
	# output_pts = np.float32([[0, 0],
	#                         [0, maxHeight - 1],
	#                         [maxWidth - 1, maxHeight - 1],
	#                         [maxWidth - 1, 0]])

	# # Compute the perspective transform M
	# M = cv2.getPerspectiveTransform(input_pts,output_pts)

	# transformed = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)

	undistorted_image = cv2.undistort(frame_copy, np.load('camera_matrix.npy'), np.load('dist_coeffs.npy'))

	undistorted_image2 = cv2.undistort(frame_copy, np.load('./backup/camera_matrix.npy'), np.load('./backup/dist_coeffs.npy'))

	undistorted_image3 = cv2.undistort(frame_copy, np.load('./backup2/camera_matrix.npy'), np.load('./backup2/dist_coeffs.npy'))

	

	# cv2.imshow('transformed image', transformed)
	cv2.imshow('original image', img)
	cv2.imshow('matrix', undistorted_image)
	cv2.imshow('backup', undistorted_image2)
	cv2.imshow('backup2', undistorted_image3)
	cv2.waitKey(0)

	
# testDistortion('original_image.png')

# a = [[[392,  61]],
#  [[483,  64]],
#  [[479, 153]],
#  [[391, 152]]]

# print(a[0][0])

# a = [[[223, 277]],
#  [[314, 352]],
#  [[237, 445]],
#  [[148, 369]]]

# a = [[[498, 222]],
#  [[594, 295]],
#  [[519, 396]],
#  [[427, 322]]]

# plt.plot(a[0][0][0],a[0][0][1],'o')
# plt.plot(a[1][0][0],a[1][0][1],'o')
# plt.plot(a[2][0][0],a[2][0][1],'o')
# plt.plot(a[3][0][0],a[3][0][1],'o')
# plt.gca().set_aspect('equal')

# plt.show()










# while True:
# 	measureArea(cv2.imread)