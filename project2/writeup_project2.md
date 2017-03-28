#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The dataset was already available in 3 different files - for training, validation and test data. 

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

Some of the classes have limited sample sizes. 


####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Picked 100 random images out of those 35k image samples and displayed along with labels.
Additionally displayed the bar chart that shows discrepency of not having enough input data.


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Preprocessing is the key to a good model. I adopted a mix of cropping (from 32x32 to 28x28) and applying histogram normalization. I was not quite convinced that histogram normalization resulted in any better results looking at 100's of visual samples. Maybe I could have kept orginal samples (by scaling to 28x28) for training as well, alongside transformed samples.

I decided to keep RGB color and not opt of grayscale. 


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)


The good thing is that the data were already separated when available to us.

Regarding the need for augmentation: the reference paper cited in the problem explains it succinctly: "ConvNets architectures have built-in invariance to small translations, scaling and rotations. When a dataset does not naturally contain those deformations, adding them synthetically will yield more robust learning to potential deformations in the test set. We demonstrate the error rate gain on the validation set in table I. Other realistic perturbations would probably also increase robustness such as other affine transformations, brightness, contrast and blur."

I decided to apply a mix of contrast, rotation and sharpness. For every sample (in training and validation set) was augmented to create 10x more samples. 

Lesson learnt - exposure.equalize_adapthist from skimage is terribly slower. I was not using GPU.



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


My final model consisted of the following layers -  LeNet model with minor changes.
    


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 28x28x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 24x24x6 	|
| Max pooling	      	| 2x2 stride,  outputs 12x12x6   				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 8x8x16		|
| Max pooling	      	| 2x2 stride,  outputs 4x4x16    				|
| Fully connected		| 256x120        								|
| Fully connected		| 120x84        								|
| Fully connected		| 84x43        									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I found that beyond epoch 10, I stopped seeing major improvements.  Also learning rate of 0.0005 did help, but 0.001 was not much different. Optimizer is ADAMoptimizer.

I could have tried with convolution layers of higher count (say 16 and 24) as opposed to 6 and 16. But not being able to get the GPU running was discouraging.

Batch size of 128 or 256 gave me identical results.

The remaining code for the training and model was similar to the one given as part of the course.

Also I did shuffle the images before every epoch.


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I was initially stuck at <90% and the challenge was the image samples I was generating from transformation and augmentation. At the end, I could reach 92.7% test classification score.

Then I was stuck until decided to crop (makes sense because points of interest are towards the center). I could have cropped to 26x26 as well. Cropping did improve the training accuracy.

I picked LeNet because it is already well known for image classification. Next time, I am going to try the GoogleNet.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I picked 6 traffic signs (road work, do not enter, stop sign etc.). 4 out of those 6 images had mixed background images to easily confuse the model. I also realized that scaling down the image size and transforming it altered the image quite a bit.


####2. Discuss the models predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Unfortunately the model managed to get only 50% accuracy.

The images were: [ 1 18 17 12 25 14]
Top Predictions ended up to be: [10 18 17 12 4 17]

Top 5 predictation for each image:
[[10  5  7  9  1]
 [18 31 38  0  1]
 [17 14  0  1  2]
 [12  9 32 10 40]
 [ 4 18  1 20  2]
 [17 14  8  3 13]]
 


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


It is interesting that the model a converging heaviliy on a result with high probability, even if the classification is wrong. For example, in all the wrong cases of classification (image #1, 5 and 6), the probabilities of classifying the WRONG class is  (0.998813987, 0.998223841, 1.0) respectively. This tells me that the quality of image is important - for for training and testing. Good, intelligent preprocessing can significantly improve accuracy without a lot of compute power.


Top Probabilites [[  9.98813987e-01   1.17811037e-03   7.89814476e-06   4.11538856e-08
    1.28228210e-08]
 [  1.00000000e+00   1.64055824e-31   4.69142881e-32   0.00000000e+00
    0.00000000e+00]
 [  1.00000000e+00   4.39419625e-32   0.00000000e+00   0.00000000e+00
    0.00000000e+00]
 [  1.00000000e+00   1.73755380e-31   2.65772190e-32   4.64738730e-36
    1.30627765e-37]
 [  9.98223841e-01   1.20255968e-03   3.46825691e-04   1.11701760e-04
    1.07698055e-04]
 [  1.00000000e+00   1.97235685e-16   7.63156859e-26   4.98850492e-29
    2.85317862e-33]]