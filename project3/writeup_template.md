#**Behavioral Cloning** 


[//]: # (Image References)

[image1]: ./examples/image4.png "Steering angles and speeds in my own training data"
[image2]: ./examples/image1.png "A normal image"
[image3]: ./examples/image2.png "A cropped Image"
[image4]: ./examples/image3.png "A flipped Image"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used NVIDIA model for training. Cropping was applied inside the model layer itself, so did not have to change drive.py.

___________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
___________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 80, 280, 3)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 80, 280, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
color_conv (Convolution2D)       (None, 80, 280, 3)    12          lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 38, 138, 24)   1824        color_conv[0][0]                 
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 37, 137, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 67, 36)    21636       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 32, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 30, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 28, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 5376)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          6258828     flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
___________________________________________________________________________________________________
Total params: 6,512,259
Trainable params: 6,512,259
Non-trainable params: 0
____________________________________________________________________________________________________



####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

This took a lot of time. Getting the Udacity provided data itself took time to work. Once that worked, my own training data worked flawlessless. When I increased the epochs, at one point, the car crossed to side of the road and then came back to the center. A clear indication was how validation loss was fluctuating up/down beyond certain # of epochs.

- Cropping was the key. I took out 20 pixels from each sides, 20 pixels from bottom (car front) and 60 from top (distant view/hills/water).
- Flipping helped in increasing the sample and providing richness. 
- I drove BOTH ways, ensuring the model has a rich set of images


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

- Drove BOTH ways
- Added flipped image for every image
- Shuffled the images in the beginning

I could have had a better mix of training data if the "flipped images" could be added into original set of images. As you can see, I am doing the flipping in the very stage while feeding the batch into the model. So N images/angles result in 2N images into the training model.


###Model Architecture and Training Strategy

Used NVIDIA model. Used 0.21 for adjusting the side camera angles. 


####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

__________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
___________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 80, 280, 3)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 80, 280, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
color_conv (Convolution2D)       (None, 80, 280, 3)    12          lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 38, 138, 24)   1824        color_conv[0][0]                 
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 37, 137, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 67, 36)    21636       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 32, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 30, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 28, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 5376)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          6258828     flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
___________________________________________________________________________________________________
Total params: 6,512,259
Trainable params: 6,512,259
Non-trainable params: 0
____________________________________________________________________________________________________




####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving (one in each direction).


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by increasing validation loss... I used an adam optimizer so that manually training the learning rate wasn't necessary.
