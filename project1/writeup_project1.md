#**Finding Lane Lines on the Road** 

##Project Report

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on the work in a written report


[//]: # (Image References)

[image1]: ./i1.png "Original RGB image"
[image2]: ./i2.png "Grayscale image"
[image3]: ./i3.png "Gaussian smoothing"
[image4]: ./i4.png "Canny filtering"
[image5]: ./i5.png "Area of interest and Hough lines"
[image6]: ./i6.png "Superimposed lane lines on original image"


---

### Reflection

###1. Pipeline Description
* Start with RGB image
* Convert to grayscale
![alt text][image1]
* Apply Gaussian filtering
![alt text][image2]
* Apply Canny filtering
![alt text][image3]
* Generate Hough Lines
![alt text][image4]
* Draw Hough lines on Canny image
![alt text][image5]
* Merge Canny image with original RGB image
![alt text][image6]


###2. Identify potential shortcomings with your current pipeline
* Clearly it did not work with the challenge video (the last optional video). It means the tuning parameters are very much tied to the images and videos presented as part of exercise. 
* There were too many hough lines generated. I managed to cut those by: area of interest, and then find a best fit line via linear regression. It will be good to be able to better optimize hough parameters.


###3. Suggest possible improvements to your pipeline
* Recognize curved lanes. 
* Solve the optional challenge problem, without breaking other problems.