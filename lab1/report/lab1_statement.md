# Assignment 1: Neural Networks, Convolutions and TensorFlow

:warning: The deadline for the assignment is the **November 14th at 23:59** :warning:

## Introduction
This first assignment is an introduction to neural networks in Python and TensorFlow. It contains three main parts.

1. In the first part you will work through the mathematical aspects of neural networks and implement a multi-layer neural network using [NumPy](http://www.numpy.org/) routines to learn the fundamentals.

2. The main goal of the second part is to make you familiar with [TensorFlow](https://www.tensorflow.org/). In this part you will implement multi-layer neural network from the first part but now in TensorFlow. TensorFlow is a widely used open-source library from Google for numerical computation using data flow graphs such as neural networks.
Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (i.e. tensors) transfered between the nodes. As a TensorFlow user, you define the computational architecture of your predictive model, combine that with your objective function, and just feed data to your mode -- TensorFlow can automatically compute the corresponding derivatives for you.    

3. The third part of the assignment is dedicated to [Convolutional Neural Networks (CNNs)](http://cs231n.github.io/convolutional-networks/). This type of networks has become the most widely used of all neural network structures and finds it applications mostly in processing visual input such as images. You will start by implementing a baseline CNN and then extend it to achieve the best possible performance.

All three parts of the assignment use the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), a widely used image dataset for benchmarking CNNs. It consists of 60000 32x32 color images equally divided into 10 classes. The data is split into 50000 training and 10000 test images.

As a result of this assignment you will learn the mathematical aspects of neural networks and how to implement them in Python and TensorFlow. For assignment to be graded you need to submit a code with all your implementation and report in PDF format describing all your experiments and observations. The format of the report is specified at the end of this document.

## Prerequisites
For those who are not familiar with Python and NumPy it is highly recommended to get through [NumPy tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html). For this course we will use TensorFlow v1.3. Please make sure you have this particular version of TensorFlow installed, you will run into problems otherwise. To install TensorFlow on a Linux, Mac or Windows, we refer to the following guide:

- [**Installing TensorFlow on Linux, Mac or Windows**](https://www.tensorflow.org/install/)

*Note: TensorFlow has changed its API significantly in the transition from v0.9 to v1.0. If you have used TensorFlow prior to version 1.0 you might want to read through the following page: [Transitioning to TensorFlow 1.0](https://www.tensorflow.org/install/migration).*

## Information Sources
Brief overview of TensorFlow programming fundamentals can be found here:
- [Getting Started with TensorFlow](https://www.tensorflow.org/get_started/get_started)

These three tutorials are very important to help you to get through the basics of TensorFlow:

1. [MNIST for ML Beginners](https://www.tensorflow.org/get_started/mnist/beginners)
2. [Deep MNIST for Experts](https://www.tensorflow.org/get_started/mnist/pros)
3. [TensorFlow Mechanics 101](https://www.tensorflow.org/get_started/mnist/mechanics)

We highly encourage you to get familiar with **TensorBoard**. This utility ships with TensorFlow and is very useful for visualizing many aspects during the training of neural networks. During training you write log files to disk containing information like classification loss, accuracy or gradient information. TensorBoard visualizes this information in a convenient web-interface that allows monitoring the training progress. You can also download the data from TensorBoard in CSV format in case you want the data (e.g. learning curves etc) accessible via other tools for visualization purposes such as `matplotlib`. These guides explain what is TensorBoard and how to use it:
1. [TensorBoard: Visualizing Learning](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
2. [TensorBoard: Graph Visualization](https://www.tensorflow.org/get_started/graph_viz)

We also recommend you to have a look at the following links:
1. [TensorFlow Programmer Guidelines](https://www.tensorflow.org/programmers_guide/)
2. [TensorFlow Python API](https://www.tensorflow.org/api_docs/python/)
3. [TensorFlow Community](https://www.tensorflow.org/community/)
4. [TensorFlow on GitHub](https://github.com/tensorflow/tensorflow)
5. [TensorFlow Model Zoo](https://github.com/tensorflow/models)

TensorFlow has a large community of people eager to help other people. If you have coding related questions: (1) read the documentation, (2) search on Google and StackOverflow, (3) ask your question on StackOverflow or Piazza and finally (4) ask the teaching assistants.

## Overview of Files and Usage

To simplify implementation and testing we have provided to you several files:

- `cifar10_utils.py`: This file contains utility functions that you can use to read CIFAR-10 data. Read through this file to get familiar with the interface of the `Dataset` class. The main goal of this class is to sample new batches, so you don't need to worry about it. To encode labels we are using [one-hot encoding of labels ](https://en.wikipedia.org/wiki/One-hot). *Please do not change anything in this file.*

  Usage examples:

    - Prepare CIFAR10 data:

    ```python
    import cifar10_utils
    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    ```
    - Get a new batch with the size of batch_size from the train set:

    ```python
    x, y = cifar10.train.next_batch(batch_size)
    ```
    Variables `x` and `y` are numpy arrays. The shape of `x` is `[batch_size, 32, 32, 3]`, the shape of `y` is `[batch_size, 10]`.

    - Get test images and labels:

    ```python
    x, y = cifar10.test.images, cifar10.test.labels
    ```
    Variables `x` and `y` are numpy arrays. The shape of `x` is `[batch_size, 32, 32, 3]`, the shape of `y` is `[batch_size, 10]`.

    ***Note***: For multi-layer perceptron you will need to reshape `x` before inference.

- `mlp_numpy.py` : This file contains interface of the `MLP` class to be implemented using NumPy routines. There are five methods of this class: `__init__(*args)`, `inference(*args)`, `loss(*args)`, `train_step(*args)`, `accuracy(*args)`. Implement these methods by strictly following the interfaces of these methods, otherwise we will not be able to check your implementation.

- `mlp_tf.py` : This file contains interface of the `MLP` class to be implemented in TensorFlow. There are five methods of this class: `__init__(*args)`, `inference(*args)`, `loss(*args)`, `train_step(*args)`, `accuracy(*args)`. Implement these methods by strictly following the interfaces of these methods, otherwise we will not be able to check your implementation.

- `convnet_tf.py` : This file contains an interface for the `ConvNet` class. There are five methods of this class: `__init__(*args)`, `inference(*args)`, `loss(*args)`, `train_step(*args)` and `accuracy(*args)`. Implement these methods by strictly following the interfaces given.

- `train_mlp_numpy.py` : This file contains two main functions. The function `train()` is a function where you need to implement training and testing procedure of the network on CIFAR-10 dataset.

  Another function is the `main()` which calls your `train()` function. Carefully go through all possible command line parameters and their possible values for running `train_mlp_numpy.py`. You are going to implement each of these into your code. Otherwise we can not test your code.

- `train_mlp_tf.py` : This file contains two main functions. The function `train()` is a function where you need to implement training and testing procedure of the network on CIFAR-10 dataset.

  Another function is the `main()` function which gets rid of already existing log directory and reinitializes one. Then it calls your `train()` function. Carefully go through all possible command line parameters and their possible values for running `train_mlp_tf.py`. You are going to implement each of these into your code. Otherwise we can not test your code.

- `train_convnet_tf.py` : This file contains two main functions. The function `train()` is a function where you need to implement training and testing procedure of the network on CIFAR-10 dataset using `ConvNet` class. Carefully examine the provided code documentation and implement your code where it is asked to be. Finally, carefully go through and get familiar with all possible command line parameters and their possible values for running `train_convnet_tf.py`. You are going to implement each of these into your code. Otherwise we can not test your code.

You are free to add other methods to the classes. Please, following coding guidelines for Python. For example, prefix private methods with an underscores (i.e. `_private_method`) and enclose the print statement with parenthesis i.e. `print('you shall code correctly')`. Check the mighty [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for more information on coding conventions (must read for every Python programmer).

You also need to save summaries which can be used for getting insights of your model on TensorBoard. Please check [TensorFlow summary operations](https://www.tensorflow.org/api_guides/python/summary) for more information on how to save log files to disk.

### Task 1: Multi-Layer Perceptron in NumPy [50 points]

1. [30 points] Before implementing anything you are required to work through a number of pen-and-paper assignments that will make you more familiar with the mathematical aspects of neural networks. The assignment can be found in the file `assignment_1_pen_paper.pdf`. Put your solution into separate pdf file with the name: `pen_paper_lastname.pdf` where `lastname` is your last name.

2. [20 points] Implement multi-layer perceptron using purely NumPy routines. The network should consist of `N` linear layers with [ReLU](https://goo.gl/NGdBmf) activation functions followed by a final linear layer. Number of hidden layers and hidden units in each layer are specified through the command line argument `dnn_hidden_units`. As loss function, use the common [cross-entropy loss](https://en.wikipedia.org/wiki/Cross_entropy) for classification loss. To optimize your network you will use [mini-batch stochastic gradient descent algorithm](http://sebastianruder.com/optimizing-gradient-descent/index.html#minibatchgradientdescent). Please implement all your code in the file `mlp_numpy.py`:
  - In `__init__(*args)` method you should implement initialization of parameters of the network.
  - In `inference(*args)` you should implement inference (forward pass) of the network.
  - In `loss(*args)` you should implement computation of multiclass cross-entropy.
  - In `train_step(*args)` you should implement backward pass of the network.
  - In `accuracy(*args)` you should implement computation of mean accuracy.

Part of the success of neural networks is the high efficiency on graphical processing units (GPUs) through matrix multiplications. Therefore, all your code should make use of matrix multiplications rather than iterating over samples in the batch or weight rows/columns. Doing multiplication by iteration will result in penalty.

Implement training and testing of the MLP inside `train_mlp_numpy.py`. With default parameters provided in this file you should get an accuracy of around 0.46 the entire *test* set for an MLP with one hidden layer of 100 units.

### Task 2: Multi-layer Perceptron in TensorFlow [15 points]

1. Implement the MLP in the `mlp_tf.py` file by following the instructions in the **Overview** section and inside the `mlp_tf.py` file itself. The interface is the same as in the first part. Implement training and testing procedures for your model in `train_mlp_tf.py` by following instructions in the **Overview** section and inside the file. Using the same parameters as Task 1, you should get similar accuracy on the test set.

2. Before proceeding with this question, convince yourself that your MLP implementation is correct. For this question you need to perform a number of experiments on your MLP to get familiar with several parameters and their effect on training and performance. For example you may want to try different regularization types, run your network for more iterations, add more layers, change the learning rate and other parameters as you like. Your goal is to get the best test accuracy you can. You should be able to get *at least 0.53* accuracy on the test set but we challenge you to improve this. Explain in the report how you are choosing new parameters to test. Do you have any particular strategy? Study your best model by plotting accuracy and loss curves. Also, plot a confusion matrix for this model. You can also visualize 4-5 examples per class for which your model makes confident wrong decisions.

### Task 3: Convolutional Neural Networks [35 points]

Now that you are familiar with multi-layer perceptions we will move on to the more interesting task of processing visual input. In order to do so, you are going to build a Convolutional Neural Network (CNN) using TensorFlow. Use the file `convnet_tf.py` for instructions and implementing your code. Designing a CNN requires some slight of hand due to its large amount of parameters; for this assignment you are asked to implement **[this network structure](https://gist.github.com/tomrunia/ae96f44eafa8398d3bd07d29db39dd6a)**.

The optimization part of your CNN should go in `train_convnet_tf.py`. Use TensorBoard for monitoring the training progress and train the model until convergence with a cross-entropy classification objective. In your report explicitly mention the *train and test accuracy* along with the hyperparameters used for obtaining these results. You should expect a classification accuracy on the test set *at least 0.70* with the given architecture.

Once your CNN obtains the results as expected improve the results of the baseline network by implementing at least 2 of the following ideas:

1. [Batch Normalization](https://arxiv.org/abs/1502.03167)
2. Skip Connections as in [ResNet](https://arxiv.org/abs/1512.03385) or [DenseNet](https://arxiv.org/abs/1608.06993)
3. Data augmentation -- augment your training images (maybe test images as well) with some perturbations like scaling, shifting, color jittering to increase dataset variability.   
4. [Depthwise separable convolutions](https://arxiv.org/abs/1610.02357)
5. [Steerable Convolutions](https://openreview.net/pdf?id=rJQKYt5ll)

## Report
You should write a small report about your study of neural networks models on CIFAR10 dataset using the provided template for [NIPS papers](https://nips.cc/Conferences/2016/PaperInformation/StyleFiles). Please, make your report to be self-contained without this README.md file.

The report should contain the following sections:

- **Abstract** : Should contain information about the current task and the summary of the study of the CNN models on CIFAR10 dataset.
- **Task 1** : Should contain all needed information about Task 1 and report of all your experiments for that task.  
- **Task 2** : Should contain all needed information about Task 2 and report of all your experiments for that task.
- **Task 3** : Should contain all needed information about Task 3 and report of all your experiments for that task.
- **Conclusion** : Should contain conclusion of this study.
- **References** : Reference section if needed.

## Submission

Create ZIP archive with the following structure:

```
lastname_assignment_1.zip
│   report_lastname.pdf
|   pen_paper_lastname.pdf
│   mlp_numpy.py
|   mlp_tf.py
│   convnet_tf.py
|   train_mlp_numpy.py
|   train_mlp_tf.py
|   train_convnet_tf.py
```
Replace `lastname` with your last name.

:warning: The deadline for the assignment is the **14th of November, 23:59** :warning:

:exclamation: We will update the way how you can submit this assignment later :exclamation:

:exclamation: Late submissions will not be graded :exclamation: