import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv.imread('img\\IMG_2110.jpg', cv.IMREAD_GRAYSCALE)  #Calibration img with object to grey scale
height, width = img1.shape[:2]

#Object dimensions (cm)
object_width_cm = 8.5
object_height_cm = 5.5

#Initiate SIFT
sift = cv.SIFT_create()

#Resize img1 (so its not tiny or huge)
def resize_img(img, target_width, target_height):
    return cv.resize(img, (target_width, target_height))

#Capture video from camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

while True:
    ret, img2 = cap.read()  #Read a frame from the camera
    gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)  #Convert frame to grayscale

    #Call resize img1
    img1_resized = resize_img(img1, img2.shape[1], img2.shape[0])

    #Find the keypoints with SIFT on both images (img1 and frame)
    kp1, des1 = sift.detectAndCompute(img1_resized, None)
    kp2, des2 = sift.detectAndCompute(gray, None)
    #Define and save keypoints 
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) 
    search_params = dict(checks=50) #Times it rewatch the tree

    #Run the matcher with the flann and get matching points from the existent keypoints
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    #Store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    #Run rest of the code only if enough matches has been found
    if len(good) > MIN_MATCH_COUNT:
        #3trasnform points to lists of 2D coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        #Find homography and so calibration criteria
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1_resized.shape
        
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2) #Reshape poitns to be used for perspective
        dst = cv.perspectiveTransform(pts, M) #Get perspective points alrady calibrated

        img2 = cv.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA) #Sorround object within a blue mark

        #Calculate distance
        #pixel_width = np.linalg.norm(dst[0][0] - dst[1][0])
        pixel_height = np.linalg.norm(dst[0][0] - dst[3][0])
        #distance_width = (object_width_cm * width) / (2 * pixel_width * np.tan(np.pi / 6))  # Assuming a 60 degree field of view
        distance_height = (object_height_cm * height) / (2 * pixel_height * np.tan(np.pi / 6))
        #print("Distance to object (width):", distance_width, "centimeters")
        print("Distance to object (height):", distance_height, "centimeters")

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  #Draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  #Draw only inliers
                       flags=2)

    #Prepare what we seeing on screen
    img3 = cv.drawMatches(img1_resized, kp1, img2, kp2, good, None, **draw_params)

    key = cv.waitKey(1) & 0xFF
    cv.imshow('frame', img3)
    if key == ord('q'):
        break
    elif key == ord('c'):
        print("homography Matrix")
        print(M)
        print("calibration gotten")

cap.release()
cv.destroyAllWindows()