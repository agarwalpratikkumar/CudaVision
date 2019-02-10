__convolutional neural networks__ <br>
CNNs, are a specialized kind of neural network for processing data that has a known grid-like topology. __For Eg:__ time-series data, which canbe thought of as a 1-D grid taking samples at regular time intervals, and image data, which can be thought of as a 2-D grid of pixels.Building invariance directly into the network structure is the basis for convolutional neural networks. <br>
__Eg:__ For recognizing handwritten digits systems should be invariant to translations, scaling and (small) rotations. <br>
<br>
In CNN we do 3 operations : <br>__1.Convolution <br> 2.Apply a non-linear activation (ReLU) <br> 3.Pooling__ <br>
<br>
In __convolutional__:  the __ﬁrst argument__ is often referred to __as__ the __input__, the __second argument as__ the __kernel__ and the __output is sometimes referred to as the feature map.__ In machine learning applications, the input is usually a multidimensional array of data, and the kernel is usually a multidimensional array of parameters that are adapted by the learning algorithm. we usually assume that these functions (i.e input and kernel) are zero everywhere but in the ﬁnite set of points for which we store the values.<br>
<br>
__Working__ <br>

<p align="left">
  <img width="660" height="300" src="https://user-images.githubusercontent.com/41232373/52537778-9b7db180-2d6a-11e9-87d0-a95a5baed6df.jpg">
</p>  

__Note: Here the depth of an image is 3 represents RGB. We call it channels.__ <br>

<p align="left">
  <img width="660" height="300" src="https://user-images.githubusercontent.com/41232373/52537840-4bebb580-2d6b-11e9-87d3-492d56e76a6c.jpg">
</p>  

__Note: Here the size of filter is 5x5x3. 3 because in our input we have 3 channels. If our image had 1 channel then the filter size would be 5x5x1.__

<p align="left">
  <img width="660" height="300" src="https://user-images.githubusercontent.com/41232373/52537989-4e4f0f00-2d6d-11e9-8171-27d5fe0e6571.jpg">
</p> 

Here we are taking the dot product of the filter with the small 5x5x3 portion of the image. And it will give us one value. Now are we are sliding this filter over the images its size gets reduced. <br>
Output Size Formula: ((Size_of_image i.e 32) - (size_of_filter i.e 5)) / Stride + 1  <br> 
Hence we will get: (32-5)/1 + 1 = 28   (Here stride is 1 because we are sliding the filter pixel by pixel)

__Important thing to notice is at the output (28x28x1) has 1 channel because we have used only one filter till now.__

<p align="left">
  <img width="660" height="300" src="https://user-images.githubusercontent.com/41232373/52538158-793a6280-2d6f-11e9-9213-05b595cd48a4.jpg">
</p> 

Now after applying Convolution we apply non-linearity (ReLU). And after ReLU we will apply Pooling.
<br>
__There are two types of Pooling that we can apply: Max Pooling or Average Pooling.__

<p align="left">
  <img width="360" height="300" src="https://user-images.githubusercontent.com/41232373/52538257-c0752300-2d70-11e9-8bae-6291cf7bddb5.png">
</p>

A pooling function replaces the output at a certain location with some summary statistic of nearby outputs.
Pooling helps makes the representation become approximately invariant to small translations of the input. Invariance to local translation is very useful if we care about whether some feature is present rather than exactly where it is.
__Note: Since pooling summarizes the response over an entire neighbourhood, it is possible to use fewer pooling units.__

<p align="left">
  <img width="660" height="300" src="https://user-images.githubusercontent.com/41232373/52538343-d0413700-2d71-11e9-9618-1be8496892e7.png">
</p>

We repeate these 3 operations until our numbers of parameters gets reduced and finnally pass the output from a fully connected feed-forward neural network.
<br>
<br>
__Now Let's Understand the Code:__
<br>
I have used PyTorch to implement this. We will apply CNN on CIFAR10 dataset.<br>
<br>
After all the necessary imports I have downloaded CIFAR10 dataset from torchvision datasets class. And using "torch.utils.data.DataLoader" I have loaded the dataset. For more details about the function we can be follow it this link:   __https://pytorch.org/docs/stable/torchvision/datasets.html#cifar__  <br> 
__https://pytorch.org/docs/stable/data.html__ <br>
<br>
 I have divided the dataset into batches of 100.
When we checked the size of our input data it was torch.Size ([100, 3, 32, 32]). Here,
__100 is the batch size__ , 
__3 is the number of channels of the image__ , 
__32,32 is the width and height of the image__. 
Similarly the size of the target or labels is torch.Size([100]). 100 because we have 100 trainig data in a batch. <br>
<br>
Now inside class Net() we are defining our network architecture. And in the forward function we are actually passing our input data (i.e 'x' which is an image) through the defined network. <br>
This is our convolution operation: nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1, padding=0)<br>
In the first convolution layer the in_channels is 3 because our image has 3 channels. 
out_channels=20 means we are taking 20 different kernels/filters.
kernel_size=5 means our kernel will of shape 5x5x3 (as shown in diagram) and then we defined the stride and padding. Padding means to add layer of zeros surrounding the image. We dopadding mostly because to use the corner pixel value more number of times so that we don't miss any important feature of the image which might be present at the corners.
After we use ReLU and then Max Pooling. <br>
We made one more convolution layer and then pass its result to the neural network. One thing we need to notice is that in the second nn.conv2d() our in_channel is 20 because in the previous layer we have used 20 different krnels so they will all stacked up to form the final shape. <br>
While training we have used CrossEntropyLoss as our loss function and SGD(stochastic Gradient Descent as our optimizer). In the train() function we are passing the image (in batch) to our model and calculating the loss. And then tested our model on test dataset and measured the accuracy. And the end we have just visualized how our kernel looks or what is its value at different convolution layer.
<br>
<br>
__Some really helpful references are:__ https://pytorch.org/docs/stable/nn.html#conv2d https://pytorch.org/docs/stable/nn.html#maxpool2d  
https://pytorch.org/docs/stable/nn.html#relu 
http://www.deeplearningbook.org/contents/convnets.html

 

 
