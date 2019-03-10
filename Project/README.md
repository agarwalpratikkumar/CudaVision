**Detection and Tracking of the soccer ball**
we are implementing the SweatyNet1 model,
from Humanoids RoboCup Workshop 2017 (the base paper mentioned below) for detection
and localization of a Soccer ball with feedforward Fully Convolutional
Neural Networks (FCNN). The network architecture has a
contracting path to capture the context and a symmetric expanding
path that enables precise localization. We show that the network can
be trained in few hours. After training the base model i.e SweatyNet1,
a ConvLSTM layer is added on top of it and trained taking continuous
sequence of frames. We show that adding convLSTM layer has a
significant improvement in localizing the ball. And finally in the post
processing step we are calculating Recall and False Detection Rate to
check the accuracy of the models.

**The base paper is :** Detection and Localization of Features on a Soccer
Field with Feedforward Fully Convolutional Neural
Networks (FCNN) for the Adult-Size Humanoid
Robot Sweaty

And for implementing the convLSTM part we followed: https://github.com/ndrplz/ConvLSTM_pytorch
