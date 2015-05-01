# Question 2 - Autoencoder #

## Part 1 ##

**Determine experimentally what is the minimal number of hidden units required for training a 16-H-16 network successfully**

According to Tom Mitchell (1997), when backpropagation is applied to the autoencoder training task, the values of the hidden layers become similar to a binary encoding based on the number of hidden units. For an 8 input autoencoder, the network assigns values that when rounded, form the binary encodings 000 to 111 (i.e. 0 to 7). From here, it can be inferred that, in general, successfully training a neural network requires a single layer of lg(2) hidden units. Hence for the task of training a 16-H-16 network, we should only require 4 hidden units on a singple sigmoid layer, to perform this task. 

For the initial experiment, we examined the behaviour of the network when given 16 Hidden Nodes on a simgle Sigmoid layer. As expected, the training error per training epoch decreased rapidly, and in the first 100 training iterations, resulted in the following activations:

![Base Line Activations](images/Q2/Q2Task1-TrainedNN-2015-04-29_22-41-03/activations.png)
Auto Encoder Base Line Activations (100)

In the interest of avoiding overfitting the network to the dataset (considering that the only inputs the network will ever expect are the inputs it is trained on), we reduced the number of training iterations considerably, from 100 to 10, producing the following result

![Base Line Activations](images/Q2/Q2Task1-TrainedNN-2015-04-29_22-41-20/activations.png)
Auto Encoder Base Line Activations (10)

From here, we began by finding an optimal adjustment of the learning rate, learning decay, and momentum, which would give the smallest possible training error towards the end of training the network. Once we were satisfied with the results, we reduced the number of hidden nodes in the sigmoid layer and repeated the process. As the reduced the number of hidden nodes, the final training error toward the end of training began to increase (as expected), and the process of slightly modifying the parameters became less effective.

![Manual FeedForward 8 Hidden Errors](images/Q2/Q2Task1-TrainedNN-2015-04-29_22-53-02/errors.png)
Auto Encoder 8 Hidden Nodes with manual FeedForward Network (Errors)

![Manual FeedForward 8 Hidden Activations](images/Q2/Q2Task1-TrainedNN-2015-04-29_22-53-02/activations.png)
Auto Encoder 8 Hidden Nodes with manual FeedForward Network (Activations)

```
HIDDEN_LAYERS = 8
LEARNING_DECAY = 0.9999
LEARNING_RATE = 0.359
MOMENTUM = 0.11649
TRAINING_ITERATIONS = 5000
BATCH_LEARNING = False
VALIDATION_PROPORTION = 0.0

```

Upon retesting using pybrain's buiilt in shortcut ``buildNetwork(inDimension, hiddenNodes, outDimension)``, the we were able to achieve a better result with some further tweaking of parameters.

![BuildNetwork 10 Hidden Errors](images/Q2/Q2Task1-TrainedNN-2015-05-01_10-59-59/errors.png)
Auto Encoder 10 Hidden Nodes using ``buildNetwork()`` (Errors)

![BuildNetwork 10 Hidden Activations](images/Q2/Q2Task1-TrainedNN-2015-05-01_10-59-59/activations.png)
Auto Encoder 10 Hidden Nodes using ``buildNetwork()`` (Activations)

```
HIDDEN_LAYERS = 10
LEARNING_DECAY = 0.999501
LEARNING_RATE = 0.324501
MOMENTUM = 0.101
TRAINING_ITERATIONS = 1000
BATCH_LEARNING = False
VALIDATION_PROPORTION = 0.0
```

![BuildNetwork 8 Hidden Errors](images/Q2/Q2Task1-TrainedNN-2015-05-01_10-36-08/errors.png)
Auto Encoder 8 Hidden Nodes using ``buildNetwork()`` (Errors)

![BuildNetwork 8 Hidden Activations](images/Q2/Q2Task1-TrainedNN-2015-05-01_10-36-08/activations.png)
Auto Encoder 8 Hidden Nodes using ``buildNetwork()`` (Activations)

```
HIDDEN_LAYERS = 8
LEARNING_DECAY = 0.999501
LEARNING_RATE = 0.324501
MOMENTUM = 0.101
TRAINING_ITERATIONS = 1000
BATCH_LEARNING = False
VALIDATION_PROPORTION = 0.0
```

With the increase in misclassifiations and training error, it became too difficult to create an auto encoder that was able to correctly replicate the inputs using a four hidden node layer.

## Part 2 ##

**Conduct experiments on 16-h1-h2-16 ANNs**