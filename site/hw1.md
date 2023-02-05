---
layout: spec
permalink: /hw1
title: Homework 1 – Numbers and Images
latex: true
---
<link href="style.css" rel="stylesheet">

## Instructions

- This homework is due at **5:00 p.m. on Wednesday January 25, 2022**.
- The submission includes two parts:
    1. **To Gradescope**: a pdf file as your write-up, including your answers to all the questions.

        Please do not handwrite and please mark the location of each answer in gradescope. 
        The write-up must be electronic. No handwriting! You can use Word, Open Office, Notepad, Google Docs, LATEX(try [overleaf](https://www.overleaf.com/)!), or any other form. You might like to combine several files. Here is an example online [link](https://combinepdf.com/) for combining multiple PDF files. Try also looking at this stack overflow [post](https://askubuntu.com/questions/2799/how-to-merge-several-pdf-files).

        <span class="blue">We have marked things you should provide as a write-up in blue.</span>

    2. **To Canvas**: a zip file including all your code.

        This should contain a single directory which has the same name as your uniqname. If I (David, uniqname fouhey) were submitting my code, the zip file should contain a single folder `fouhey/` containing the folders from the homework. So, for instance, my zip file should contain: `fouhey/numpy/run.py` and `fouhey/dither/dither.py`. We provide a script that validates the submission format [here](https://raw.githubusercontent.com/eecs442/utils/master/check_submission.py).

        <span class="red">We have marked things that involve submitting code in red.</span>


## Overview

In this assignment, you’ll work through three tasks that help set you up for success in the class as well as a short assignment involving playing with color. The document seems big, but most of it is information, advice, and hand-holding. Indeed, one section was a bit more open-ended and terse in the past, and has been expanded to four pages that walk you through visualizing images. The assignment has three goals.

1. **The assignment will show you bugs in a low-stakes setting**. You’ll encounter a lot of programming mistakes in the course. If you have buggy code, it could be that the concept is incorrect (or incorrectly understood) or it could be that the implementation in code is incorrect. You’ll have to sort out the difference. It’s a lot easier if you’ve seen the bugs in a controlled environment where you know what the answer is. Here, the programming problems are deliberately easy and we even provide the solution for one!
2. **The assignment incentivizes you to learn how to write reasonably good python and numpy code**. You should learn to do this anyway, so this gives you credit for doing it and incentivizes you to learn things in advance.
3. **You don’t need to worry about having similar solutions for coding in this homework**. The code in this homework (especially warmups, test, and dither) are going to be very similar across students becauseeach function is so short. Please do not worry that we will be looking to find people who copied off each other’s code.

The assignment has four parts and corresponding folders in the starter code:
- Numpy (Section 2 – folder `numpy/`)
- Data visualization (Section 3 – folder `visualize/`)
- Image dithering (Section 4 – folder `dither/`)
- Looking at color (Section 5)

Here’s my recommendation for how to approach this homework:
- If you have not had any experience with Numpy, read this [tutorial](http://cs231n.github.io/python-numpy-tutorial/). Numpy is like a lot of other high- level numerical programming languages. Once you get the hang of it, it makes a lot of things easy. However, you need to get the hang of it and it won’t happen overnight!
- You should then do Section 2.
- You should then read our description about images in Section A. Some will make sense; some may not. That’s OK! This is a bit like learning to ride a bike, swim, cook a new recipe, or play a new game by being told by someone. A little teaching in advance helps, but actually doing it yourself is crucial. Then, once you’ve tried yourself, you can revisit the instructions (which might make more sense).
If you haven’t recently thought much about the difference between an integer and a floating point number, or thought about multidimensional arrays, it might be worth brushing up on both.
- You should then do Section 3 and then Section 4. Both are specifically designed to produce common bugs, issues, and challenges that you will likely run into the course.
- Finally, conclude with Section 5. 

### Python Environment
We are using Python 3.7. We will make use of the following packages extensively in this course:
- [Numpy](https://numpy.org/doc/stable/user/quickstart.html).
- [Matplotlib](https://matplotlib.org/2.0.2/users/pyplot tutorial.html). 
- [OpenCV](https://opencv.org/).

## Numpy Intro

All the code/data for this is located in the folder `numpy/`. Each assignment requires you to fill in the blank in a function (in `tests.py` and `warmup.py`) and return the value described in the comment for the function. There’s driver code you do not need to read in `run.py` and `common.py`.

**Note**: All the `python` below refer to `python3`. As we stated earlier, we are going to use Python 3.7 in this assignment. Python 2 was [sunset](https://www.python.org/doc/sunset-python-2/) on January 1, 2022.

<span class="blue">Question 1.1 (the only question): Fill in the code stubs in tests.py and warmups.py. Put the terminal output in your pdf from</span>:
```
python run.py --allwarmups
python run.py --alltests
```

**Do I have to get every question right?** We give partial credit: each warmup exercise is worth 2% of the total grade for this question and each test is worth 3% of the total grade for this question.

### Tests Explained

When you open one of these two files, you will see starter code that looks like this:
```python
def sample1(xs):
    """
    Inputs:
    - xs: A list of values
    Returns:
    The first entry of the list
    """
    return None
```
You should fill in the implementation of the function, like this:
```python
def sample1(xs):
    """
    Inputs:
    - xs: A list of values
    Returns:
    The first entry of the list
    """
    return xs[0]
```
You can test your implementation by running the test script:
```bash
python run.py --test w1     # Check warmup problem w1 from warmups.py
python run.py --allwarmups  # Check all the warmup problems
python run.py --test t1     # Check the test problem t1 from tests.py
python run.py --alltests    # Check all the test problems

# Check all the warmup problems; if any of them fail, then launch the pdb
# debugger so you can find the difference
python run.py --allwarmups --pdb
```
If you are checking all the warmup problems (or test problems), the perfect result will be:
```bash
python run.py --allwarmups
Running w1
Running w2
...
Running w20
Ran warmup tests
20/20 = 100.0
```

### Warmup Problems

You need to solve all 20 of the warmup problems in `warmups.py`. They are all solvable with one line of code.

### Test Problems

You need to solve all 20 problems in `tests.py`. Many are not solvable in one line. You may not use a loop to solve any of the problems, although you may want to first figure out a slow for-loop solution to make sure you know what the right computation is, before changing the for-loop solution to a non for-loop solution. The one exception to the no-loop rule is t10 (although this can also be solved without loops).

Here is one example:
```python
def t4(R, X):
    """
    Inputs:
    - R: A numpy array of shape (3, 3) giving a rotation matrix
    - X: A numpy array of shape (N, 3) giving a set of 3-dimensional vectors
    Returns:
    A numpy array Y of shape (N, 3) where Y[i] is X[i] rotated by R
    Par: 3 lines
    Instructor: 1 line
    Hint:
    1) If v is a vector, then the matrix-vector product Rv rotates the vector
       by the matrix R.
    2) .T gives the transpose of a matrix
    """
    return None
```

### What We Provide

For each problem, we provide:
- **Inputs**: The arguments that are provided to the function
- **Returns**: What you are supposed to return from the function
- **Par**: How many lines of code it should take. We don’t grade on this, but if it takes more lines than this, there is probably a better way to solve it. Except for t10, you should not use any explicit loops.
- **Instructor**: How many lines our solution takes. Can you do better? Hints: Functions and other tips you might find useful for this problem.

### Walkthroughs and Hints


