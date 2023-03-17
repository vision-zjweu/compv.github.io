---
layout: spec
permalink: /hw4
latex: true

title: Homework 4 – Machine Learning
due: 5 p.m. on Friday March 31st, 2023
---

<link href="style.css" rel="stylesheet">
<div style="display:none">
    <!-- Define LaTeX commands here -->
    \(
        \DeclareMathOperator*{\argmin}{arg\,min}

        \newcommand{\DB}{\mathbf{D}}

        \newcommand{\RR}{\mathbb{R}}

        \newcommand{\pd}[2]{\frac{\partial #1}{\partial #2}}
    \)

</div>

{% capture code %}<i class="fa fa-code icon-large"></i>{% endcapture %}
{% capture autograde %}<i class="fa fa-robot icon-large"></i>{% endcapture %}
{% capture report %}<i class="fa fa-file icon-large"></i>{% endcapture %}

# Homework 4 – Machine Learning

## Instructions

This homework is **due at {{ page.due }}**.

This homework is divided into two major sections based on how you're expected to write code:
1. **Section 1**:
    
    You'll be writing the code in the same way you've been doing until now, i.e., in simple python files.

2. **Section 2**:

    - We are going to use  [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) or local [Jupyter Notebook](https://jupyter.org/) on your on machine (both GPU and CPU) to run our code. For more information on using Colab, please see the [official Colab tutorial](https://colab.research.google.com/?utm_source=scs-index#). The whole assignment is designed to be **CPU friendly**, but we still strongly encourage you try with Colab first.
    - We have also provided you with the python file version of the assignment in `python_backup` folder, but since the assignment is originally designed for Jupyter Notebook only, **we strongly suggest you do this assignment in Jupyter Notebook**. This option is provided only to backup the case of Colab failure or local Jupyter Notebook problem. {{ report }} <span class="report">If you're doing the homework in the python files, please attach your terminal output to the report</span>.
    - To do the homework on Colab, you just need to login to Colab with your Google/UMich account and upload corresponding notebook to the Colab (`File -> Upload notebook`), then you can get started.

The submission includes two parts:
1. **To Canvas**: submit a `zip` file of all of your code.

    {{ code }} - 
    <span class="code">We have indicated questions where you have to do something in code in red</span>  
    {{ autograde }} - 
    <span class="autograde">We have indicated questions where we will definitely use an autograder in purple</span>

    Please be especially careful on the autograded assignments to follow the instructions. Don't swap the order of arguments and do not return extra values. If we're talking about autograding a filename, we will be pulling out these files with a script. Please be careful about the name.

    Your zip file should contain a single directory which has the same name as your uniqname. If I (David, uniqname `fouhey`) were submitting my code, the zip file should contain a single folder `fouhey/` containing all required files.  
        
    <div class="primer-spec-callout info" markdown="1">
      **Submission Tip:** Use the [Canvas Submission Checklist](#canvas-submission-checklist) and [Gradescope Submission Checklist](#gradescope-submission-checklist) at the end of this homework. We also provide a script that validates the submission format [here](https://raw.githubusercontent.com/eecs442/utils/master/check_submission.py).

      If we don't ask you for it, you don't need to submit it; while you should clean up the directory, don't panic about having an extra file or two.
    </div>

2. **To Gradescope**: submit a `pdf` file as your write-up, including your answers to all the questions and key choices you made.

    {{ report }} - 
    <span class="report">We have indicated questions where you have to do something in the report in blue.</span>

    <div class="primer-spec-callout info" markdown="1">
      **Changes in format requirements:** 
      - Put your name and uniqname on the first page of your report.
      - In addition to submitting your code files on Canvas, please also put *readable* screenshots of your code in your report, labeling the respective questions they belong to.  
      <br>
    </div>

    You might like to combine several files to make a submission. Here is an example online [link](https://combinepdf.com/) for combining multiple PDF files. The write-up must be an electronic version. **No handwriting, including plotting questions.** $$$$\LaTeX$$$$ is recommended but not mandatory.

### Python Environment

The autograder uses Python 3.7. Consider referring to the [Python standard library docs](https://docs.python.org/3.7/library/index.html) when you have questions about Python utilties.

To make your life easier, we recommend you to install the latest [Anaconda](https://www.anaconda.com/download/) for Python 3.7. This is a Python package manager that includes most of the modules you need for this course. We will make use of the following packages extensively in this course:
- [Numpy](https://numpy.org/doc/stable/user/quickstart.html)
- [Matplotlib](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)
- [OpenCV](https://opencv.org/)

#### Local Development

 If you're doing this homework on your local machine instead of Colab, then other than the packages you should have already installed in previous homework, you will also need: `tqdm`, `pytorch>=1.8.0`, `torchvision` and `torchsummary` of the corresponding version. You may install these packages using `anaconda` or `pip`. Notice that some of the packages may need to be downloaded from certain anaconda channel, you may need to search on the [Anaconda](https://anaconda.org/) official website for more instructions.

 # Section 1

## Computational Graphs and Backprop
We have seen that representing mathematical expressions as *computational graphs* allows us to easily compute gradients using backpropagation. After writing a mathematical expression as a computational graph, we can easily translate it into code. In this problem you'll gain some experience with backpropagation in a simplified setting where all of the inputs, outputs, and intermediate values are all scalar values instead vectors, matrices, or tensors.

In the *forward pass* we receive the inputs (leaf nodes) of the graph and compute the output. The output is typically a scalar value representing the loss $$L$$ on a minibatch of training data.

In the *backward pass* we compute the derivative of the graph's output $$L$$ with respect to each input of the graph. There is no need to reason *globally* about the derivative of the expression represented by the graph; instead when using backpropagation we need only think *locally* about how derivatives flow backward through each node of the graph. Specifically, during backpropagation a node that computes $$y=f(x_1, \ldots, x_N)$$ receive an *upstream gradient* $$\pd{L}{y}$$ giving the derivative of the loss with respect the the node output and computes *downstream gradients* $$\pd{L}{x_1},\ldots,\pd{L}{x_N}$$ giving the derivative of the loss with respect to the node inputs.

Here's an example of a simple computational graph and the corresponding code for the forward and backward passes. Notice how each \textcolor{RoyalBlue}{\bf outgoing edge} from an operator gives rise to one line of code in the forward pass, and each \textcolor{ForestGreen}{\bf ingoing edge} to an operator gives rise to one line of code in the backward pass.

<embed src="{{site.url}}/assets/hw4/figures/f1.pdf" height="200">

<div>
    Hello World
    ```python
        for i in range(3)
    ```
</div>

\begin{figure*}[h!]
  \centering
  \begin{minipage}{0.3\textwidth}
    \includegraphics[width=\textwidth]{figures/graph1.pdf}
  \end{minipage}%
  \begin{minipage}{0.5\textwidth}
    \begin{minted}[fontsize=\scriptsize]{python}
    def f(a, b, c):
      d = a * b       # Start forward pass
      L = c + d

      grad_L = 1.0    # Start backward pass
      grad_c = grad_L
      grad_d = grad_L
      grad_a = grad_d * b
      grad_b = grad_d * a

      return L, (grad_a, grad_b, grad_c)
    \end{minted}
  \end{minipage}
\end{figure*}

Sometimes you'll see computational graphs where one piece of data is used as input to multiple operations. In such cases you can make the logic in the backward pass cleaner by rewriting the graph to include an explicit `copy` operator that returns multiple copies of its input. In the backward pass you can then compute separate gradients for the two copies, which will sum when backpropagating through the copy operator:

\begin{figure*}[h!]
  \centering
  \begin{minipage}{0.3\textwidth}
    \centering
    \includegraphics[width=0.9\textwidth]{figures/graph2.pdf}%
    \\*[2mm]
    $$\downarrow$$ Add copy operator
    \\*[2mm]
    \includegraphics[width=0.9\textwidth]{figures/graph2_v2.pdf}
  \end{minipage}% 
  \begin{minipage}{0.4\textwidth}
    \begin{minted}[fontsize=\scriptsize]{python}
    def f(a, b, c):
      b1 = b          # Start forward pass
      b2 = b
      d = a * b1
      e = c * b2
      L = d + e

      grad_L = 1.0    # Start backward pass
      grad_d = grad_L
      grad_e = grad_L
      grad_a = grad_d * b1
      grad_b1 = grad_d * a
      grad_c = grad_e * b2
      grad_b2 = grad_e * c
      grad_b = grad_b1 + grad_b2  # Sum grads for copies

      return L, (grad_a, grad_b, grad_c)
    \end{minted}

  \end{minipage}
\end{figure*}

### Task 1: Implementing Computational Graphs

Below we've drawn three computational graphs for you to practice implementing forward and backward passes. The functions `f1` and `f2` are optional, and the function `f3` is required. The file `backprop/functions.py` contains stubs for each of these computational graphs. You can use the driver program `backprop/backprop.py` to check your implementation.

 {{ autograde }} <span class="autograde">Implement the forward and backward passes for the computational graph `f3` below.</span>

The file `backprop/backprop-data.pkl` contains sample inputs and outputs for the three computational graphs; the driver program loads inputs from this file for you when checking your forward passes.

To check the backward passes, the driver program implements *numeric gradient checking*. Given a function $$f:\RR\to\RR$$, we can approximate the gradient of $$f$$ at a point $$x_0\in\RR$$ as:

$$\pd{f}{x}(x_0) \approx \frac{f(x_0 + h) - f(x_0 - h)}{2h}$$

Each of these computational graphs implements a function or operation commonly used in machine learning. Can you guess what they are? (This is just for fun, not required).

\begin{figure}[h!]
  \centering
  \begin{minipage}{0.42\textwidth}
    \includegraphics[width=\textwidth]{figures/f1.pdf}
  \end{minipage}%
  \hspace{2mm}
  \begin{minipage}{0.4\textwidth}
    **`f1`: (OPTIONAL)** \\*[2mm]
    The subtraction node computes $$d = \hat y - y$$ \\*[2mm]
    The \verb|^2| node computes $$L = d^2$$
  \end{minipage}
  \vspace{-2mm}
\end{figure}


\begin{figure}[h!]
  \centering
  \begin{minipage}{0.42\textwidth}
    \includegraphics[width=\textwidth]{figures/f2.pdf}
  \end{minipage}%
  \hspace{2mm}
  \begin{minipage}{0.4\textwidth}
    **`f2`: (OPTIONAL)** \\*[2mm]
    The $$\times2$$ node computes $$d = 2x$$ \\*[2mm]
    The $$\div$$ node computes $$y = t / b$$
  \end{minipage}
  \vspace{2mm}
\end{figure}

\begin{figure}[h!]
  \centering
  \begin{minipage}{0.42\textwidth}
    \includegraphics[width=\textwidth]{figures/f3.pdf}
  \end{minipage}%
  \hspace{2mm}
  \begin{minipage}{0.4\textwidth}
    **`f3`: (REQUIRED [10 points])** \\*[2mm]
    $$y$$ is an integer equal to either 1 or 2. \\*
    You don't need to compute a gradient for $$y$$. \\*[2mm]
    The $$\div$$ nodes compute $$p_1 = e_1 / d$$ and \\* 
    $$p_2 = e_2 / d$$. \\*[2mm]
    The \verb|choose| node outputs outputs $$p_1$$ if $$y=1$$,
    and outputs $$p_2$$ if $$y=2$$.
  \end{minipage}
  \vspace{-2mm}
\end{figure}

### Write Your Own Graph (Optional)

 {{ report }} <span class="report">In your report, draw a computational graph for any function of your choosing.</span> It should have at least five operators. (You can hand-draw the graph and include a picture of it in your report.)

 {{ code }} <span class="code">In the file `backprop/functions.py`, implement a forward and backward pass through your computational graph in the function `f4`.</span> You can modify the function to take any number of input arguments. After implementing `f4`, you can use the driver script to perform numeric gradient checking. Depending on the functions in your graph, you may see errors $$\geq10^{-8}$$ even with a correct backward pass. This is ok!

## Fully-Connected Neural Networks

In this question you will implement and train a fully-connected neural network to classify images.

**For this question you cannot use any deep learning libraries such as PyTorch or TensorFlow**.

### Task 2: Modular Backprop API
In the previous questions on this assignment you used backpropagation to compute gradients by implementing monolithic functions that combine the forward and backward passes for an entire graph. As we've discussed in lecture, this monolithic approach to backpropagation isn't very modular -- if you want to change some component of your graph (new loss function, different activation function, etc) then you need to write a new function from scratch.

Rather than using monolithic backpropagation implementations, most modern deep learning frameworks use a *modular API* for backpropagation. Each primitive operator that will be used in a computational graph implements a *forward* function that computes the operator's output from its inputs, and a *backward* function that receives upstream gradients and computes downstream gradients. Deep learning libraries like PyTorch or TensorFlow provide many predefined operators with corresponding forward and backward functions.

To gain experience with this modular approach to backpropagation, you will implement your own miniature modular deep learning framework. The file `neuralnet/layers.py` defines forward and backward functions for several common operators that we'll need to implement our own neural networks.

Each forward function receives one or more numpy arrays as input, and returns: 
1. A numpy array giving the output of the operator;
2. A *cache* object containing values that will be needed during the backward pass. The backward function receives a numpy array of upstream gradients along with the cache object, and must compute and return downstream gradients for each of the inputs passed to the forward function.

Along with forward and backward functions for operators to be used in the middle of a computational graph, we also define functions for *loss functions* that will be used to compute the final output from a graph. These loss functions receive an input and return both the loss and the gradient of the loss with respect to the input.

This modular API allows us to implement our operators and loss functions once, and reuse them in different computational graphs. For example, we can implement a full forward and backward pass to compute the loss and gradients for linear regression in just a few lines of code:

```python
  from layers import fc_forward, fc_backward, l2_loss

  def linear_regression_step(X, y, W, b):
      y_pred, cache = fc_forward(X, W, b)
      loss, grad_y_pred = l2_loss(y_pred, y)
      grad_X, grad_W, grad_b = fc_backward(grad_y_pred, cache)
      return grad_W, grad_b
```

In the file `neuralnet/layers.py` you need to complete the implementation of the following:

1. {{ autograde }} 
    <span class="autograde">Fully-connected layer</span>: `fc_forward` and `fc_backward`.

2. {{ autograde }} 
    <span class="autograde">ReLU nonlinearity</span>: `relu_forward` and `relu_backward` which applies the function $$ReLU(x_i) = \max(0, x)$$ elementwise to its input.

3. {{ autograde }} 
    <span class="autograde">Softmax Loss Function</span>:`softmax_loss`.

    The softmax loss function receives a matrix $$x\in\RR^{N\times C}$$ giving a batch of classification scores for $$N$$ elements, where for each element we have a score for each of $$C$$ different categories. The softmax loss function first converts the scores into a set of $$N$$ probability distributions over the elements, defined as:     
    
    $$
    p_{i,c} = \cfrac{\exp(x_{i,c})}{\sum_{j=1}^C \exp(x_{i,j})}
    $$     
    
    The output of the softmax loss is then given by:
    
    $$
    L = -\frac{1}{N} \sum_{i=1}^N \log(p_{i,y_i})
    $$     
    
    where $$y_i\in\{1,\ldots,C\}$$ is the ground-truth label for the $$i$$th element.

    A naive implementation of the softmax loss can suffer from *numeric instability*. More specifically, large values in $$x$$ can cause overflow when computing $$\exp$$. To avoid this, we can instead compute the softmax probabilities as:
    
    $$
    p_{i,c} = \frac{\exp(x_{i,c} - M_i)}{\sum_{j=1}^C \exp(x_{i,j} - M_i)}
    $$     
    
    where $$M_i = \max_c x_{i,c}$$.     
    
    This ensures that all values we exponentiate are $$<0$$, avoiding any potential overflow. It's not hard to see that these two formulations are equivalent, since 
    
    $$
    \frac{\exp(x_{i,c} - M_i)}{\sum_{j=1}^C\exp(x_{i,j} - M_i)} = \frac{\exp(x_{i,c})\exp(-M_i)}{\sum_{j=1}^C \exp(x_{i,j})\exp(-M_i)}       = \frac{\exp(x_{i,c})}{\sum_{i=1}^C \exp(x_{i,j})}
    $$     
    
    **Your softmax implementation should use this max-subtraction trick for numeric stability.** You can run the script `neuralnet/check_softmax_stability.py` to check the numeric stability of your softmax loss implementation.

4. {{ autograde }} 
    <span class="autograde">L2 Regularization</span>: `l2_regularization` which implements the L2 regularization loss
    
    $$
    L(W) = \frac{\lambda}{2}\|W\|^2 = \frac{\lambda}{2} \sum_i W_i^2
    $$

    where the sum ranges over all scalar elements of the weight matrix $$W$$ and $$\lambda$$ is a hyperparameter controlling the regularization strength.

After implementing all functions above, you can use the script `neuralnet/gradcheck_layers.py`  to perform numeric gradient checking on your implementations.  The difference between all numeric and analytic gradients should be less than $$10^{-9}$$.

Keep in mind that numeric gradient checking does not check whether you've correctly implemented  the forward pass; it only checks whether the backward pass you've implemented actually computes the  gradient of the forward pass that you implemented.

### Task 3: Implement a Two-Layer Network

Your next task is to implement a two-layer fully-connected neural network using the modular forward  and backward functions that you just implemented.

In addition to using a modular API for individual layers, we will also adopt a modular API for  classification models as well. This will allow us to implement multiple different types of  image classification models, but train and test them all with the same training logic.

The file `neuralnet/classifier.py` defines a base class for image classification models.  You don't need to implement anything in this file, but you should read through it to familiarize  yourself with the API. In order to define your own type of image classification model, you'll  need to define a subclass of `Classifier` that implements the `parameters`,  `forward`, and `backward` methods.

In the file `neuralnet/linear_classifier.py` we've implemented a `LinearClassifier`  class that subclasses `Classifier` and implements a linear classification model using the  modular layer API from the previous task together with the modular classifier API.  Again, you don't need to implement anything in this file but you should read through it to get a  sense for how to implement your own classifiers.

Now it's your turn! In the file `neuralnet/two_layer_net.py` we've provided the start to  an implementation of a `TwoLayerNet` class that implements a two-layer neural network  (with ReLU nonlinearity).

{{ autograde }} 
<span class="autograde">Complete the implementation of the `TwoLayerNet` class.</span>  Your implementations for the `forward` and `backward` methods should use the modular  forward and backward functions that you implemented in the previous task.

After completing your implementation, you can run the script `gradcheck_classifier.py` to  perform numeric gradient checking on both the linear classifier we've implemented for you as well  as the two-layer network you've just implemented. You should see errors less than $$10^{-10}$$ for  the gradients of all parameters.

### Task 4: Training Two-Layer Networks
You will train a two-layer network to perform image classification on the CIFAR-10 dataset.  This dataset consists of $$32\times 32$$ RGB images of 10 different categories.  It provides 50,000 training images and 10,000 test images.  Here are a few example images from the dataset:

\begin{figure*}[h!]
  \centering
  \includegraphics[width=0.5\textwidth]{figures/p2/CIFAR10.png}
\end{figure*}

You can use the script `neuralnet/download_cifar.sh` to download and unpack the CIFAR10 dataset.

The file `neuralnet/train.py` implements a training loop.
We've already implemented a lot of the logic here for you.
You don't need to do anything with the following files, but you can look through them to see how
they work:
- `neuralnet/data.py` provides a function to load and preprocess the CIFAR10 dataset,
    as well as a `DataSampler` object for iterating over the dataset in minibatches.
- `neuralnet/optim.py` defines an `Optimizer` interface for objects that
    implement optimization algorithms, and implements a subclass `SGD` which implements
    basic stochastic gradient descent with a constant learning rate.

{{ autograde }} 
<span class="autograde">Implement the `training_step` function in the file `neuralnet/train.py`</span>.

This function inputs the model, a minibatch of data, and the regularization strength;
it computes a forward and backward pass through the model and returns both the loss and the
gradient of the loss with respect to the model parameters.
The loss should be the sum of two terms:
1. A *data loss* term, which is the softmax loss between the model's predicted scores
    and the ground-truth image labels
2. A *regularization loss* term, which penalizes the L2 norm of the weight matrices of
    all the fully-connected layers of the model. You should not apply L2 regularization to the
    biases.

 Now it's time to train your model! Run the script `neuralnet/train.py` to train a two-layer network on the CIFAR-10 dataset. The script will print out training losses and train and val set accuracies as it trains. After training concludes, the script will also mke a plot of the training losses as well as the training and validation-set accuracies of the model during training; by default this will be saved in a file `plot.pdf`, but this can be customized with the flag `--plot-file`. You should see a plot that looks like this:

\begin{figure*}[h!]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/loss-plot.pdf}
\end{figure*}

Unfortunately, it seems that your model is not training very effectively -- the training loss has
not decreased much from its initial value of $$\approx2.3$$, and the training and validation
accuracies are very close to 10\% which is what we would expect from a model that randomly guesses
a category label for each input.

You will need to tune the hyperparameters of your model in order to improve it.
Try changing the hyperparameters of the model in the provided space of the
`main` function of `neuralnet/train.py`.
You can consider changing any of the following hyperparameters:

- `num_train`: The number of images to use for training
- `hidden_dim`: The width of the hidden layer of the model
- `batch_size`: The number of examples to use in each minibatch during SGD
- `num_epochs`: How long to train the model.
    An *epoch* is a single pass through the training set.
- `learning_rate`: The learning rate to use for SGD
- `reg`: The strength of the L2 regularization term


You should tune the hyperparameters and train a model that achieves at least 40\% on the validation set. After tuning your model, run your best model **exactly once** on the test set using the script `neuralnet/test.py`.

{{ report }} 
<span class="report">In your report, include the loss / accuracy plot for your best model, describe the hyperparameter settings you used, and give the final test-set performance of your model.</span>

You may not need to change all of the hyperparameters; some are fine at their default values. Your model shouldn't take an excessive amount of time to train. For reference, our hyperparameter settings achieve $$\approx45\%$$ accuracy on the validation set in $$\approx5$$ minutes of training on a 2019 MacBook Pro.

To gain more experience with hyperparameters, you should also tune the hyperparameters to find a setting that results in an *overfit model* that achieves $$\geq75\%$$ accuracy on the *training set*.

{{ report }} 
<span class="report">In your report, include the loss / accuracy plot for your overfit model and describe the hyperparameter settings you used.</span>

As above, this should not take an excessive amount of training time -- we are able to train an overfit model that achieves $$\approx80\%$$ accuracy on the training set within about a minute of training.

HINT: It's easier to overfit a smaller training set.


