# from random import randrange
import matplotlib.pyplot as plt
# import numpy as np
# import cv2 as cv
import glob

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
 








# Camera calibration 
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

num_rows = 3
num_cols = 4
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros(((num_cols+1)*(num_rows+1),3), np.float32)
objp[:,:2] = np.mgrid[0:(num_rows+1),0:(num_cols+1)].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
# images = glob.glob('*.jpg')
 
# for fname in images:
img = cv2.imread('./calibration/2024-09-04-153634.jpg')
# cv2.imshow('img', img)
# cv2.waitKey()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img', gray)
# cv2.waitKey()

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, ((num_rows+1),(num_cols+1)), None)
# print(corners)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    cv2.drawChessboardCorners(img, ((num_rows+1),(num_cols+1)), corners2, ret)
    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()


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





# SE = np.ones((16,16))
# dilated = cv2.dilate(mask, SE)
# dilated [...,1:3] = 0

# from skimage.measure import label

# labelled = label(1-dilated [...,0])





# labelled[labelled==1] = 0
# labelled[labelled >0] = 1

# labelled = labelled.astype(np.uint8)

# res = cv2.bitwise_and(img,img,mask = labelled)

# cv2.namedWindow('Splitted Images', cv2.WINDOW_NORMAL)
# cv2.imshow('Splitted Images', res)
# cv2.waitKey(0)

  
# Find the contours of the object  
contours, hierarchy = cv2.findContours(th4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
  
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