---
layout: spec
permalink: /hw3
title: Homework 3 – Fitting Models and Image Warping
latex: true
---
<link href="style.css" rel="stylesheet">
<div style="display:none">
    <!-- Define LaTeX commands here -->
    \(
        \DeclareMathOperator*{\argmin}{arg\,min}

        \newcommand{\AB}{\mathbf{A}}
        \newcommand{\HB}{\mathbf{H}}
        \newcommand{\MB}{\mathbf{M}}
        \newcommand{\SB}{\mathbf{S}}
        
        \newcommand{\bB}{\mathbf{b}}
        \newcommand{\hB}{\mathbf{h}}
        \newcommand{\mB}{\mathbf{m}}
        \newcommand{\pB}{\mathbf{p}}
        \newcommand{\sB}{\mathbf{s}}
        \newcommand{\tB}{\mathbf{t}}
        \newcommand{\vB}{\mathbf{v}}
    \)
</div>

# Homework 3 – Fitting Models and Image Warping

## Instructions

- This homework is **due at 11:59:59 p.m. on Tuesday, March 9, 2022**.
- The submission includes two parts:
    1. **To Canvas**: submit a `zip` file of all of your code.

        <span class="red">We have indicated questions where you have to do something in code in red.</span>  
        <span class="purple">We have indicated questions where we will definitely use an autograder in purple</span>

        Please be especially careful on the autograded assignments to follow the instructions. Don't swap the order of arguments and do not return extra values. If we're talking about autograding a filename, we will be pulling out these files with a script. Please be careful about the name.

        Your zip file should contain a single directory which has the same name as your uniqname. If I (David, uniqname `fouhey`) were submitting my code, the zip file should contain a single folder `fouhey/` containing all required files.  
            
        **What should I submit? At the end of the homework, there is a canvas submission checklist provided.** We provide a script that validates the submission format [here](https://raw.githubusercontent.com/eecs442/utils/master/check_submission.py). If we don't ask you for it, you don't need to submit it; while you should clean up the directory, don't panic about having an extra file or two.

    2. **To Gradescope**: submit a `pdf` file as your write-up, including your answers to all the questions and key choices you made.

        <span class="blue">We have indicated questions where you have to do something in the report in blue.</span>

        You might like to combine several files to make a submission. Here is an example online [link](https://combinepdf.com/) for combining multiple PDF files. The write-up must be an electronic version. **No handwriting, including plotting questions.** $$\LaTeX$$ is recommended but not mandatory.

### Python Environment

We are using Python 3.7. You can find references for the Python standard library [here](https://docs.python.org/3.7/library/index.html). To make your life easier, we recommend you to install the latest [Anaconda](https://www.anaconda.com/download/) for Python 3.7. This is a Python package manager that includes most of the modules you need for this course. We will make use of the following packages extensively in this course:
- [Numpy](https://numpy.org/doc/stable/user/quickstart.html)
- [Matplotlib](https://matplotlib.org/2.0.2/users/pyplot tutorial.html)
- [OpenCV](https://opencv.org/)

## RANSAC and Fitting Models

### Task 1: RANSAC Theory (9 points)

In this section, suppose we are fitting a 3D plane (i.e., $$ax + by + cz + d = 0$$). A 3D plane can be defined by 3 points (2 points define a line). Plane fitting happens when people analyze point clouds to reconstruct scenes from laser scans. To distinguish from other notations that you may find elsewhere, we will refer to the model that is fit within the loop of RANSAC (covered in the lecture) as the *putative* model.

1. (3 points) <span class="blue">Write in your report</span> the minimum number of 3D points needed to sample in an iteration to compute a putative model.
2. (3 points) [REPORT] <span class="blue">Determine the probability</span> that the data picked for to fit the putative model in a single iteration fails, assuming that the outlier ratio in the dataset is $$0.5$$ and we are fitting 3D planes.
3. (3 points) [REPORT] <span class="blue">Determine the minimum number of RANSAC trials</span> needed to have $$\geq 98\%$$ chance of success, assuming that the outlier ratio in the dataset is $$0.5$$ and we are fitting planes.

*Hint*: You can do this by explicit calculation or by search/trial and error with numpy.

### Task 2: Fitting Linear Transformations (6 points)

Throughout, suppose we have a set of 2D correspondences ($$[x_i',y_i'] \leftrightarrow [x_i,y_i]$$) for $$1 \le i \le N$$.

1. (3 points) Suppose we are fitting a linear transformation, which can be parameterized by a matrix $$\MB \in \mathbb{R}^{2\times 2}$$ (i.e., $$[x',y']^T = \MB [x,y]^T$$).

    <span class="blue">Write in your report</span> the number of degrees of freedom $$\MB$$ has and the minimum number of 2D correspondences that are required to fully constrain or estimate $$\MB$$.

2. (3 points) Suppose we want to fit $$[x_i',y_i']^T = \MB [x_i,y_i]^T$$. We would like you formulate the fitting problem in the form of a least-squares problem of the form:

    $$
    \argmin_{m \in \mathbb{R}^4} \|\AB \mB - \bB\|_2^2
    $$

    where $$\mB \in \mathbb{R}^4$$ contains all the parameters of $$\MB$$, $$\AB$$ depends on the points $$[x_i,y_i]$$ and $$\bB$$ depends on the points $$[x'_i, y'_i]$$.

    <span class="blue">Write the form of $$\AB$$, $$\mB$$, and $$\bB$$ in your report.</span> 

### Task 3: Fitting Affine Transformations (11 points)

Throughout, again suppose we have a set of 2D correspondences $$[x_i',y_i'] \leftrightarrow [x_i,y_i]$$ for $$1 \le i \le N$$.

**Files**: We give an actual set of points in `task3/points_case_1.npy` and `task3/points_case_2.npy`: each row of the matrix contains the data $$[x_i,y_i,x'_i,y'_i]$$ representing the correspondence. **You do not need to turn in your code but you may want to write some file** `task3.py` **that loads and plots data.**

1. Fit a transformation of the form:

    $$
    [x',y']^T = \SB [x,y]^T + \tB, ~~~~~ \SB \in \mathbb{R}^{2 \times 2}, \tB \in \mathbb{R}^{2 \times 1}
    $$

    by setting up a problem of the form:

    $$
    \argmin_{\vB \in \mathbb{R}^6} \|\AB \vB - \bB\|^2_2
    $$

    and solving it via least-squares.

    <span class="blue">Report ($$\SB$$,$$\tB$$) in your report for `points_case_1.npy`.</span>

    *Hint*:} There is no trick question -- use the setup from the foreword. Write a small amount of code that does this by loading a matrix, shuffling the data around, and then calling `np.linalg.lstsq`.

2. Make as scatterplot of the points $$[x_i,y_i]$$, $$[x'_i,y'_i]$$ and $$\SB[x_i,y_i]^T+\tB$$ in one figure with different colors. Do this for both `points_case_1.npy` and `point_case_2.npy`. In other words, there should be two plots, each of which contains three sets of $$N$$ points.

    <span class="blue">Save the figures and put them in your report</span>

    *Hint*: Look at `plt.scatter` and `plt.savefig`. For drawing the scatterplot, you can do `plt.scatter(xy[:,0],xy[:,1],1)`. The last argument controls the size of the dot and you may want this to be small so you can set the pattern. As you ask it to scatterplot more plots, they accumulate on the current figure. End the figure by `plt.close()`.

3. (5 points) <span class="blue">Write in the report your answer to</span> how well does an affine transform describe the relationship between $$[x,y] \leftrightarrow [x',y']$$ for `points_case_1.npy` and `points_case_2.npy`? You should describe this in two to three sentences.

    *Hint*: What properties are preserved by each transformation?

### Task 4: Fitting Homographies (11 points)

**Files**: We have generated 9 cases of correspondences in `task4/`. These are named `points\_case\_k.npy` for $$1 \le k \le 9$$. All are the same format as the previous task and are matrices where each row contains $$[x_i,y_i,x'_i,y'_i]$$. Eight are transformed letters $$M$$. The last case (case 9) is copied from task 3. You can use these examples to verify your implementation of `fit_homography`.

1. (5 points) <span class="purple">Fill in `fit_homography`</span> in `homography.py`.

    This should fit a homography mapping between the two given points. Remembering that $$\pB_i \equiv [x_i, y_i, 1]$$ and $$\pB'_i \equiv [x'_i, y'_i, 1]$$, your goal is to fit a homography $$\HB \in \mathbb{R}^{3}$$ that satisfies:

    $$\pB'_i \equiv \HB \pB_i.$$

    Most sets of correspondences are not exactly described by a homography, so your goal is to fit a homography using an optimization problem of the form:

    $$
    \argmin_{\|\hB\|_2^2=1} \|\AB \hB\|,~~~\hB \in \mathbb{R}^{9}, \AB \in \mathbb{R}^{2N \times 9}
    $$

    where $$\hB$$ has all the parameters of $$\HB$$.

    *Hint*: Again, this is not meant to be a trick question -- use the setup from the foreword. 

    **Important**: This part will be autograded. Please follow the specifications precisely.

2. (3 points) <span class="blue">Report $$\HB$$</span> for cases `points_case_1.npy` and `points_case_4.npy`. You must normalize the last entry to $$1$$.

3. (3 points) Visualize the original points $$[x_i,y_i]$$,  target points $$[x'_i,y'_i]$$ and points after applying a homography transform $$T(H,[x_i,y_i])$$ in one figure. Please do this for `points_case_5.npy` and `points_case_9.npy`. Thus there should be two plots, each of which contains 3 sets of `N` points.

    <span class="blue">Save the figure and put it in the report.</span>