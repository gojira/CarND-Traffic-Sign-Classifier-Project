#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report-images/explore-training-set.png "Training set distribution"
[image2]: ./report-images/explore-validation-set.png "Validation set distribution"
[image3]: ./report-images/explore-test-set.png "Test set distribution"

[image102]: ./examples/grayscale.jpg "Grayscaling"
[image103]: ./examples/random_noise.jpg "Random Noise"
[image104]: ./examples/placeholder.png "Traffic Sign 1"
[image105]: ./examples/placeholder.png "Traffic Sign 2"
[image106]: ./examples/placeholder.png "Traffic Sign 3"
[image107]: ./examples/placeholder.png "Traffic Sign 4"
[image108]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! Here is:

* [Github repo](https://github.com/gojira/CarND-Traffic-Sign-Classifier-Project)
* [Project code](https://github.com/gojira/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
* [Jupyter notebook output saved as HTML](https://github.com/gojira/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here are exploratory visualizations of the data sets.

First, in the [code](https://github.com/gojira/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html), I output a random set of 25 images with their labels to familiarize myself with the image set.

The labels were loaded from signnames.csv into a pandas data frame for easy access.

Next, I plotted the class distributions in each data set - training, validation, and test.  We can see that the datasets are not particularly balanced.  We see that the training and test sets are very well matched in class distribution. The validation set is roughly similar, but appears tob e quantized into increments of roughly 25.

![alt text][image1]
![alt text][image2]
![alt text][image3]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I approached data transformation & augmentation based on whether it was needed by the model.

The steps I took are the following.

1. Resize image to 32x32x3 as needed.  Clearly this was not needed for the provided dataset, but it is needed for most sample images obtained from the web.  This is the first step before any other image processing.
2. Normalize pixel value ranges to [-1,1]
3. Experiment with data augmentation such as flipping, rotating, shifting images.  Since I am using Keras (see more below in model architecture), I used its built-in support for image data generation.

In the end, I did not spend a lot of time with data augmentation as model test accuracy reached 0.98 with only normalization.

If I had needed to, I would have taken these further steps.

1. Test grayscale.
2. Manually generate flipped, translated (shifted), and rotated images to balance out the class distribution.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is a straightforward convolutional neural network similar to the basic
Keras CIFAR-10 model.

It is loosely based on Alexnet but sized for 32x32x3 images that is suitable for the
traffic sign data set as well as for CIFAR-10.

The model has the following properties

* Classic convnet with convolution and max pooling
* 2 convolution + 1 max pooling, repeated twice
* Dropout after max pooling and the first fully connected layer
* Batch norm before each activation layer
* Relu activation
* L2 regularization

The model consisted of the following layers. The notebook and HTML output have diagrams of the network.

| Layer				| Output Shape	| Description |
|:-----------------:|:--------------:|:-----------:|
| Input				| 32x32x3			| Input RGB image | 
| 3x3 Convolution	| 32x32x32		| 32 filters, stride=1, same padding |
| Batch Norm			| 32x32x32		| |
| Activation			| 32x32x32		| Relu |
| 3x3 Convolution	| 30x30x32		| 32 filters, stride=1 |
| Batch Norm			| 30x30x32		| |
| Activation			| 30x30x32		| Relu |
| Max Pooling			| 15x15x32		| 2x2 pooling |
| Dropout				| 15x15x32		| Keep = 0.25 |
| 3x3 Convolution	| 15x15x64		| 64 filters, stride=1, same padding |
| Batch Norm			| 15x15x64		| |
| Activation			| 15x15x64		| Relu |
| 3x3 Convolution	| 13x13x64		| 64 filters, stride=1 |
| Batch Norm			| 13x13x64		| |
| Activation			| 13x13x64		| Relu |
| Max Pooling			| 6x6x64			| 2x2 pooling |
| Dropout				| 6x6x64			| Keep = 0.25 |
| Flatten				| 2304				| |
| Fully connected	| 512				| |
| Batch Norm			| 512				| |
| Activation			| 512				| Relu |
| Dropout				| 512				| Keep = 0.25 |
| Fully connected	| 43				| Fully connected layer with outputs = # of classes |
| Batch Norm			| 43				| |
| Activation			| 43				| Softmax for classification |

 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The final training uses the following.

1. Adam Optimizer with Keras defaults, including learning rate of 0.01
2. Batch size: 128
3. Number of epochs: The final model uses 30 epochs with optional early stopping
4. L2 regularization at 0.001

There is some discussion on approach to arrive at the hyperparameters in next section.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 0.99
* validation set accuracy of 0.98
* test set accuracy of 0.98


I started by adapting the sample LeNet model and code.  I got reasonable results of over 0.95 validation accuracy.  However, I found LeNet a little cumbersome to modify and experiment with.

As a result, I decided to start again from scratch using Keras with a standard CIFAR-10 CNN model.  I immediately got better results with the Keras model so I switched all subsequent efforts to Keras.

For Keras, I started with Keras 1 with Tensorflow 0.12 in the carnd-term1 environment but encountered errors.  Because Keras 2 is better maintained, I switched to Keras 2 with Tensorflow 1.2 in its own environment.  I got much better results with this compared to trying with the carnd-term1 environment.

With Keras, it was very quick and easy to experiment with dropout, regularization, and batch normalization.  In my experiements, I found dropout gave the biggest bang in terms of adding generalization to the model by better validation and test set results.  My model includes L2 regularization as it also improved results although less so than dropout.

I did manual tuning of the following:

* Optimizer: I tried standard SGD and Adam.  SGD was considerably slower (it took an order or two magnitude slower at first and rather than searching for better hyperparameters with SGD, I spent more time tuning Adam)
* Learning rate: I experimented with learning rate with my initial LeNet model as well as with SGD, but found Adam converged fast and with good accuracy.
* Dropout keep rate: I tried several different dropout percentages including 0, 0.25, 0.5, 0.8, and 1.0.  In the end the standard 0.5 and 0.25 worked well.  Dropout had the biggest contribution to reducing overfitting and improving test set accuracy.
* L2 regularization: I experimented both with and without and found using parameters of about 0.001 worked well.
* Train with and without batch normalization.  Batch normalization generally led to faster improvements in test set accuracy, and better accuracy in validation and test set.

My result notebook includes 3 different models:

* Normalized dataset without batch normalization
* Normalized dataset with batch normalization
* Normalized AND augmented dataset without batch normalization

The best results come from normalized dataset with batch normalization.  That approach generally, although not always, yielded better validation and test set accuracy compared to the same without batch normalization.  The accuracy difference was around 0.01.

The data augmentation results using Keras image data support yielded lower accuracy than the unaugmented dataset.  It may be promising to investigate this further, but given that my unaugmented data model yielded test set accuracy of up to 0.98, I did not pursue this.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


