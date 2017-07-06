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


[//]: # (Image References)

[histogram]: ./images/data_histogram.png "Visualization"
[accuracy]: ./images/accuracy.png "Accuracy"
[sign1]: ./signs-photos/1.png "Speed limit (50km/h)"
[sign2]: ./signs-photos/2.png "Australia's NSW Stop (if the traffic lights are not working)"
[sign3]: ./signs-photos/3.png "No entry"
[sign4]: ./signs-photos/4.png "No entry"
[sign5]: ./signs-photos/5.png "Stop sign"
[sign6]: ./signs-photos/6.png "No entry"
[sign7]: ./signs-photos/7.png "Speed limit (50km/h) covered by leaves"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/clockworkbits/CarND-Traffic-Sign-Classifier-Project/blob/solution/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used standard Python functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

I created a normalized histogram with all three data sets (training, validation, testing) in it.

![Data visualization][histogram]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to normalized the data and shuffle the data set.

The other options I considered were using grascale images or augmenting the images by applying a random patches on them.

I did not used the additional thechniques because the model was working pretty well without them. Converting to grayscale removes some information. Generationg more samples would be needed if the model had a tendency to overfit, but I used diffrent approch to avoid it (dropout).

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x6x16 				    |
| Fully connected		| 400 to 1200        							|
| RELU					|												|
| Dropout               | 50% during training                           |
| Fully connected		| 1200 to 84        							|
| RELU					|												|
| Dropout               | 50% during training                           |
| Output layer          | 84 to 43                                      |


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam Optimizer. The batch size was 128, 30 epochs. The learning rate was 0.001. Dropout during training 0.5.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.949
* test set accuracy of 0.950

![Accuracy during training][accuracy]

I have chosen the LeNet-5 network as a base for this project because it worked pretty well for the digits recognition. I started with the same size of the layers as for the digits recognition, however it was performing worse than the required minimum (with the accuracy of about 0.85 on the validation set). To improve this result I started experimenting with the size of the network, initially with making it bigger, because I thought a traffic sign is more complicated than a digit. This resulted with two problems, one is the computation time increased and the other, was that model was overfitting giving quite good results for the test set, but not so much for the validation set. I also tried to decrease the size of the model but it didn't perform well either.
Then I experimented with expanding single layers over the original hand written digits model, especially adding extra dimensions to the convolution filter. This did not help. The final result was the initial LeNet architecture with the first connected layer much wider (1200 vs oryginal 400). To avoid overfitting I used dropout with the 50% probabliity. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five traffic signs that I found on the web:

![sign1]

![sign2]

![sign3] 

![sign4]

![sign5]

![sign6]

![sign7]

I was curious how the model will perform with a sign that was not included in the original training set, but is quite similar so I added sign 2, which is Australian Stop sign for the cases when the traffic lights is not working. The last sign is the speed limit covered by the leaves.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)  | Speed limit (50km/h)   						|
| Australian Stop       | Stop                                          |
| No Entry #1     		| No Entry										|
| No Entry #2           | No Entry                                      |
| Stop					| Stop											|
| No Entry #3           | No Entry                                      |
| 50 km/h covered  		| 30 km/h   					 				|

The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.7%. This is a lower result than the test set, which had the accuracy of 95%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

In general the predictions were very strong (>0.99), if the image was clear or from the test set. The one that was recognized incorrectly, was the 50 km/h speed limit covered by leaves, and the model clasified it as the 30 km/h speed limit with very high probability (>0.98).

For the first image, the prediction is very strong.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.9995915293693542     | Speed limit (50km/h)                          |
|0.0004084741813130677  | Speed limit (30km/h)                          |
|2.6031414979317546e-13 | Speed limit (70km/h)                          |
|6.972745406237657e-14  | Speed limit (80km/h)                          |
|5.380127970195346e-15  | Speed limit (100km/h)                         |


For the second image, the Australian stop sign, the prediction it is the Stop sign is very high.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.9936254024505615     | Stop                                          |
|0.004622188396751881   | No vehicles                                   |
|0.0007988493307493627  | Children crossing                             |
|0.0003835259412880987  | Yield                                         |
|0.0003486303612589836  | Right-of-way at the next intersection         |


The probability table for the third image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.0 | No entry|
|1.4753689825397487e-08 | Stop|
|3.612041915568298e-11 | No passing|
|4.8130483973340965e-12 | Bicycles crossing|
|2.2339378610847227e-12 | Yield|


The probability table for the forth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
|1.0 | No entry|
|4.675549010457747e-12 | Stop|
|2.9800301434163874e-24 | Yield|
|4.246246892405432e-26 | No passing|
|3.7646506994047066e-26 | Bicycles crossing|


The probability table for the fifth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
|1.0 | Stop|
|7.91101303414131e-31 | Road work|
|7.959577602442434e-36 | No vehicles|
|7.546735434253067e-36 | Yield|
|0.0 | Speed limit (20km/h)|


The probability table for the sixth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
|1.0 | No entry|
|3.745169307777113e-19 | Stop|
|2.349077576631819e-23 | No passing|
|6.099569277889269e-26 | Yield|
|1.0714994060585995e-27 | Bicycles crossing|


The probability table for the seventh image, the speed limit covered by the leaves.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
|0.98102205991745 | Speed limit (30km/h)|
|0.016811365261673927 | Speed limit (50km/h)|
|0.0005423730472102761 | Speed limit (60km/h)|
|0.0004377235600259155 | Speed limit (80km/h)|
|0.00032083169207908213 | End of speed limit (80km/h)|



