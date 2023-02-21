import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# filtering image using canny function
def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) # grayscalling
    blur = cv.GaussianBlur(gray, (5,5), 0) # using (5,5) kernel and deviation as 0
    canny = cv.Canny(blur, 50, 150) # to get gradient image - (low_threshold, high_threshold)
    return canny

# getting region of interest from image using coordinate values
def region_of_interest(image):
    height = image.shape[0] # gets
    triangle = np.array([
        [(200, height), (1100, height), (550, 250)] # single polygon as fill_poly fills with multiple polygons
        ])
    mask = np.zeros_like(image) # same size image (no. of pixels as original image) but with the values of 0 (black)
    cv.fillPoly(mask, triangle, 255) # applying the triangle contour on mask image with values of 255 
    masked_image = cv.bitwise_and(image, mask) # applying bitwise_and to get lane lines
    return masked_image


# optimizing the generated lines
def average_slope_intercept(image, lines):
    left_fit = [] # negative slope
    right_fit = [] # positive slope
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1) # returns the slope and y-intercept of the line
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0) # axis = 0 - operate through the rows of the list (final result: rows = 1 and cols = same)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

# make coordinates for new line
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*3/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])


# display lines on the black image
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10) # builds a line, 10 - thickness
    return line_image

# read the image 
image = cv.imread('test_image.jpg')

#creating a copy to avoid changes in the original image
lane_image = np.copy(image)

canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)

# 2 - pixel value (rho), radian value = 1 radian, 100 - threshold which is min no. of intersections a bins should have in order for it to be accepted as a line
lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # gives line coordinates (x1, y1, x2, y2)
averaged_lines = average_slope_intercept(lane_image, lines)

# get the lines on the black image
line_image = display_lines(lane_image, averaged_lines)

blended_image = cv.addWeighted(lane_image, 0.8, line_image, 1, 1) # adds the pixels values - hence the reason as why lines were generated on the black image

# display the image (heading/name of the window, image to be displayed)
cv.imshow('Result', blended_image)
cv.waitKey(0)
