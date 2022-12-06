American SignLanguage Classifier

Authors

Jacqueline Lin, Valerie Wang, Weining Mai

Introduction

The problem that we tried to solve is the American sign language classifier. The input is an image of a certain sign pose. Giving a certain sign pose, the model will try to predict the correct sign pose. We chose 10 signs that we want to classify: Hey, How are you, Name, Nice to meet you, No, Run, Sit, Swim, What, and Yes. We have an interest in solving sign language problems because there has not been a ton of research done on this topic. Having a model that classifies American Sign Language is relatively a hard problem since there exist many meanings with different hand signs and facial expressions. Our project will mainly focus on classifying the ten hand signs as stated above. Not only that,  this problem will be fun to solve since none of us know anything about American sign language, so we need to also learn the poses of the American sign language system.

Methodology

In order to classify the ten sign poses, we will first need to create our datasets. To collect our images, we will be using opencv to turn on the camera. Next, we have our groupmates to perform the hand signs in front of the web camera. OpenCV will collect images of us frame by frame and we will use these images to store in 10 different label folders. Tensorflow will build a dataset with the data we have. The process of collecting images was challenging since we need to perform the hand signs and the webcam in OpenCV has low frames.  Lastly, we use two different neural network architectures: ResNet50, and EfficientNet to perform transfer learning. In order to train the model, the training of both notebooks: ResNet50, EfficientNet run on a local GPU: Nvidia GTX 3070 8GB. Here are the following of our parameters…




ResNet50 notebook:

For this notebook, since resnet50 is a relatively large neural network architecture with 50 layers, we used early stopping to stop training early when we find no improvements while running though epochs.
Num_classes = 10
Batch_size = 16, though can change depending on the the memory of local machine
Img_shape = (224,224,3)
Training_size = 87 files
Validing_size = 21 files
Testing_size = 30 files
Validation_split = 0.2 for both train_ds and valid_ds
Output_layer_activation = softmax activation
Loss_function = categorical_crossentropy
Optimizer = adam with learning_rate = 0.01
Metrics = accuracy
Epochs = 50
Early_stopping_patience = 3, number of epochs with no improvement, after that, training stop









EfficientNet notebook:

For this notebook, we added an image augmentation layer to see whether it will improve accuracy. For this notebook, we didn’t add early stopping since we want to keep on training to see whether accuracy will improve.
Num_classes = 10
Batch_size = 32, though can change depending on the the memory of local machine
Img_shape = (224,224,3)
Training_size = 108 files
Testing_size = 30 files
Image_Augmentation:
RandomFlip = horizontal
RandomRotation = 0.2
RandomHeight = 0.2
RandomWidth  = 0.2
RandomZoom = 0.2
Epochs = 200
Output_layer_activation = softmax activation
Loss_function = categorical_crossentropy
Optimizer = adam with default learning rate
Metrics = accuracy
Results:
ResNet50
ResNet50 Model val loss: 1.6906
ResNet50 Model train loss: 0.0129
Accuracy on test_ds: 0.7667
Accuracy on train_ds: 0.9885







EfficientNetB0
EfficientNetB0 Model val loss: 0.6908
EfficientNetB0 Model train loss: 0.1308
Accuracy on test_ds: 0.80
Accuracy on train_ds: 0.9907




Conclusions

By doing this project, we built an American Sign Language classifier using computer vision. We noticed that computer vision in American Sign Language is not as easy as it sounds. For example, we need to consider that some meaning has a sequence of hand and face movements, and a picture cannot fully represent that. In order to improve on our American Sign Language project, here are some thoughts. We need to combat bias-variance tradeoff which is a constant problem throughout machine learning problems. In addition,we need to consider the domain adaptation of the model. For instance, our model is trained on our hand color, but we need the model to adapt to different hand colors as well. Lastly, to improve on this American Sign Language project, we can perhaps include a GAN model. Why? Suppose someone is designing images of hand signs intentionally to fool our model, how can we design such a model and do we combat such images?
