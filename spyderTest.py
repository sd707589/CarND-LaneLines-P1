# -*- coding: utf-8 -*-
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    imshape = img.shape
    y_bottom = imshape[0]
    left_lines=[]
    right_lines=[]
    
    #first, separate the left and right lane lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            if y1 == y2:
                continue
            elif x1 == x2:
                if x1 > imshape[0]/2:
                    right_line=[x1,min(y1,y2),x2,y_bottom]
                    right_lines.append(right_line)
                else:
                    left_line=[x1,min(y1,y2),x2,y_bottom]
                    left_lines.append(left_line)
            else:
                # calculate model: y=k*x+b
                k= (y2-y1)/(x2-x1)
                b= y2-k*x2
                x_bottom= (y_bottom-b)/k
                if abs(k)< 0.47:
                    continue
                if k <0:
                    if y1>y2 :
                        left_line=[x2,y2,x_bottom,y_bottom]
                    else:
                        left_line=[x1,y1,x_bottom,y_bottom]
                    left_lines.append(left_line) 
                    
                else:
                    if y1>y2 :
                        right_line=[x2,y2,x_bottom,y_bottom]
                    else:
                        right_line=[x1,y1,x_bottom,y_bottom]
                    right_lines.append(right_line) 
                    
    def draw_one_line(line,color):
        cv2.line(img, (int(line[0]), int(line[1])), 
                   (int(line[2]),int(line[3])), color, thickness)
        return
    # filter lines into one line           
    def get_one_line(Lines):
        if len(Lines) == 1:
            return Lines[0]
        sum_bottom=0
        x_toppest=0
        y_toppest=y_bottom
        for Line in Lines:
#            draw_one_line(Line,[0,255,0])
            sum_bottom +=Line[2]
            if Line[1] < y_toppest:
                y_toppest=Line[1]
                x_toppest=Line[0]
#        print("Final Line",[x_toppest,y_toppest, sum_bottom/len(Lines),y_bottom])
        return [x_toppest,y_toppest, sum_bottom/len(Lines),y_bottom]
    
        
    sorted(left_lines,key=lambda x: x[2])
    len_left=len(left_lines)
    left_lines[:]=left_lines[int(len_left/3):int(len_left*2/3)]
#    left_lines[:int(len_left/3)]=left_lines[int(len_left*2/3):]=[]
    if len(left_lines)>0:
        one_line_left=get_one_line(left_lines)
        draw_one_line(one_line_left,[255,0,0])
    
    sorted(right_lines,key=lambda x: x[2])
    len_right=len(right_lines)
    right_lines[:]=right_lines[int(len_right/3):int(len_right*2/3)]
#    right_lines[:int(len_right/3)]=right_lines[int(len_right*2/3):]=[]
    if len(right_lines)>0:
        one_line_right=get_one_line(right_lines)
        draw_one_line(one_line_right,[0,0,255])
#    # draw lines
#    cv2.line(img, (int(one_line_left[0]), int(one_line_left[1])), 
#                   (int(one_line_left[2]),int(one_line_left[3])), [255,0,0], thickness)
#    cv2.line(img, (int(one_line_right[0]), int(one_line_right[1])), 
#                   (int(one_line_right[2]),int(one_line_right[3])), [0,0,255], thickness)
    
    #--------------old version------------
#    imshape = img.shape
#    y_bottom = imshape[1]
#    
#    for line in lines:
#        for x1,y1,x2,y2 in line:
#            if y1 == y2:
#                continue
#            elif x1 == x2:
#                color=[0, 255, 0]
#                cv2.line(img, (x1, min(y1,y2)), (x2, int(y_bottom)), color, thickness)
#            else:
#                # calculate model: y=k*x+b
#                k= (y2-y1)/(x2-x1)
#                b= y2-k*x2
#                x_bottom= (y_bottom-b)/k
#                if abs(k)< 0.47:
#                    continue
#                if k >0:
#                    color=[255, 0 ,0] #left line color
#                else:
#                    color=[0, 0, 255] #right line color
#                if y1 > y2:
#                    cv2.line(img, (x2, y2), (int(x_bottom), int(y_bottom)), color, thickness)
#                else:
#                    cv2.line(img, (x1, y1), (int(x_bottom), int(y_bottom)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho,
                            theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(initial_img, img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
import os
dirs = os.listdir("test_images/")
#--------------------------------------------------

for imgName in dirs:
    image=mpimg.imread("test_images/"+imgName)
    plt.imshow(image)
    print ("test_images/"+imgName)
    gray=grayscale(image)
    kernel_size = 5
    blur_gray=gaussian_blur(gray,kernel_size)
    low_threshold = 50
    high_threshold = 150
    edges=canny(blur_gray,low_threshold,high_threshold)
    imshape = image.shape
    vertices = np.array([[(imshape[1]*0.05,imshape[0]),(imshape[1]*0.48,imshape[0]*0.51),
                       (imshape[1]*0.52,imshape[0]*0.51), (imshape[1]*0.95,imshape[0])]], dtype=np.int32) #[x,y]
    masked_edges=region_of_interest(edges,vertices)
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold =8     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20 #minimum number of pixels making up a line
    max_line_gap = 1    # maximum gap in pixels between connectable line segments
    line_image=hough_lines(masked_edges,rho,theta,threshold,min_line_length,max_line_gap)
    color_edges = np.dstack((edges, edges, edges))
    lines_edges=weighted_img(line_image,image,1.,0.8,0.)
    plt.imshow(lines_edges)
#
## Import everything needed to edit/save/watch video clips
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
#def process_image(image):
#    # NOTE: The output you return should be a color image (3 channel) for processing video below
#    # TODO: put your pipeline here,
#    # you should return the final output (image where lines are drawn on lanes)
#    gray=grayscale(image)
#    blur_gray=gaussian_blur(gray,kernel_size)
#    edges=canny(blur_gray,low_threshold,high_threshold)
#    imshape = image.shape
#    vertices = np.array([[(imshape[1]*0.05,imshape[0]),(imshape[1]*0.48,imshape[0]*0.51),
#                       (imshape[1]*0.52,imshape[0]*0.51), (imshape[1]*0.95,imshape[0])]], dtype=np.int32) #[x,y]
#    masked_edges=region_of_interest(edges,vertices)
#    line_image=hough_lines(masked_edges,rho,theta,threshold,min_line_length,max_line_gap)
#    result=weighted_img(line_image,image,0.8,1.,0.)
#    return result
#white_output = 'test_videos_output/solidWhiteRight.mp4'
### To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
### To do so add .subclip(start_second,end_second) to the end of the line below
### Where start_second and end_second are integer values representing the start and end of the subclip
### You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
###clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
## %time white_clip.write_videofile(white_output, audio=False)
#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(white_output))
#yellow_output = 'test_videos_output/solidYellowLeft.mp4'
### To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
### To do so add .subclip(start_second,end_second) to the end of the line below
### Where start_second and end_second are integer values representing the start and end of the subclip
### You may also uncomment the following line for a subclip of the first 5 seconds
#clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
###clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
#yellow_clip = clip2.fl_image(process_image)
## %time yellow_clip.write_videofile(yellow_output, audio=False)
#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(yellow_output))
#challenge_output = 'test_videos_output/challenge.mp4'
### To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
### To do so add .subclip(start_second,end_second) to the end of the line below
### Where start_second and end_second are integer values representing the start and end of the subclip
### You may also uncomment the following line for a subclip of the first 5 seconds
#clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
###clip3 = VideoFileClip('test_videos/challenge.mp4')
#challenge_clip = clip3.fl_image(process_image)
## %time challenge_clip.write_videofile(challenge_output, audio=False)
#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(challenge_output))