---
typora-root-url: page_pic
---

# **Finding Lane Lines on the Road** 
## Allen's answer

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # "Image References"

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

![original img](https://github.com/sd707589/CarND-LaneLines-P1/blob/master/page_pic/0.png)

First, I converted the images to grayscale, and dealt it with Gaussian Blur.

![Gray_blur img](https://github.com/sd707589/CarND-LaneLines-P1/blob/master/page_pic/1.png)

Second, use Canny Method to find edges.

![edge img](https://github.com/sd707589/CarND-LaneLines-P1/blob/master/page_pic/2.png)

Third, I created a mask to the edges of lane line on the road.

![edge of lane line](https://github.com/sd707589/CarND-LaneLines-P1/blob/master/page_pic/3.png)

Forth, get the lines by the way of Houph.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by tree steps:

 	1. Separate left an right lane lines by their slope rate.
    2. Sort the lines by each line's bottom-x posiotn, and choose only middle 1/3 samples.
    3. For the top point, choose the toppest point; For the bottom point, calculate the average x value. Then, the final one line is made.

![hough line img](https://github.com/sd707589/CarND-LaneLines-P1/blob/master/page_pic/4.png)

### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be that the recognized lane lines are shot and unstable when vehicle turns a corner.

Another shortcoming could be that my algorithm is too depending on lane line's color. If in dark or flickering conditions, I wonder whether the algorithm could work well.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to set up a local lane map to avoid getting lost when algorithm can't recognize any lane.
Another potential improvement could be to use AI technology to get better lane recognition.
