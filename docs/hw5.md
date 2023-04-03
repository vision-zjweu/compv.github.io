---
layout: spec
permalink: /hw5
latex: true

title: Homework 5 – Cameras
due: 5 p.m. on Monday April 17th, 2023
---

<link href="style.css" rel="stylesheet">
<div style="display:none">
    <!-- Define LaTeX commands here -->
    \(
		\newcommand{\EB}{\mathbf{E}}
		\newcommand{\FB}{\mathbf{F}}
		\newcommand{\IB}{\mathbf{I}}
		\newcommand{\KB}{\mathbf{K}}
        \newcommand{\MB}{\mathbf{M}}
		\newcommand{\RB}{\mathbf{R}}
		\newcommand{\XB}{\mathbf{X}}
        
		\newcommand{\pB}{\mathbf{p}}
		\newcommand{\tB}{\mathbf{t}}

		\newcommand{\zeroB}{\mathbf{0}}
    \)
</div>

{% capture code %}<i class="fa fa-code icon-large"></i>{% endcapture %}
{% capture autograde %}<i class="fa fa-robot icon-large"></i>{% endcapture %}
{% capture report %}<i class="fa fa-file icon-large"></i>{% endcapture %}

# Homework 5 - Cameras

## Instructions

This homework is **due at {{ page.due }}**.

The submission includes two parts:
1. **To Canvas**: submit a `zip` file of all of your code.

    {{ code }} - 
    <span class="code">We have indicated questions where you have to do something in code in red</span>  
    {{ autograde }} - 
    <span class="autograde">We have indicated questions where we will definitely use an autograder in purple</span>

    Please be especially careful on the autograded assignments to follow the instructions. Don't swap the order of arguments and do not return extra values. If we're talking about autograding a filename, we will be pulling out these files with a script. Please be careful about the name.

    Your zip file should contain a single directory which has the same name as your uniqname. If I (David, uniqname `fouhey`) were submitting my code, the zip file should contain a single folder `fouhey/` containing all required files.  
        
    <div class="primer-spec-callout info" markdown="1">
      **Submission Tip:** Use the [Tasks Checklist](#tasks-checklist) and [Canvas Submission Checklist](#canvas-submission-checklist) at the end of this homework. We also provide a script that validates the submission format [here](https://raw.githubusercontent.com/eecs442/utils/master/check_submission.py){:target="_blank"}.

      If we don't ask you for it, you don't need to submit it; while you should clean up the directory, don't panic about having an extra file or two.
    </div>

2. **To Gradescope**: submit a `pdf` file as your write-up, including your answers to all the questions and key choices you made.

    {{ report }} - 
    <span class="report">We have indicated questions where you have to do something in the report in green.</span>

    You might like to combine several files to make a submission. Here is an example online [link](https://combinepdf.com/){:target="_blank"} for combining multiple PDF files. The write-up must be an electronic version. **No handwriting, including plotting questions.** $$$$\LaTeX$$$$ is recommended but not mandatory.

### Python Environment

The autograder uses Python 3.7. Consider referring to the [Python standard library docs](https://docs.python.org/3.7/library/index.html){:target="_blank"} when you have questions about Python utilties.

To make your life easier, we recommend you to install the latest [Anaconda](https://www.anaconda.com/download/){:target="_blank"} for Python 3.7. This is a Python package manager that includes most of the modules you need for this course. We will make use of the following packages extensively in this course:
- [Numpy](https://numpy.org/doc/stable/user/quickstart.html){:target="_blank"}
- [Matplotlib](https://matplotlib.org/stable/tutorials/introductory/pyplot.html){:target="_blank"}
- [OpenCV](https://opencv.org/){:target="_blank"}

## Camera Calibration

<figure class="figure-container">
    <div class="flex-container">
        <figure>
            <img src="{{site.url}}/assets/hw5/figures/temple_us.png" alt="Temple" width="300">
            <figcaption>Temple</figcaption>
        </figure>
        <figure>
            <img src="{{site.url}}/assets/hw5/figures/zrtrans_us.png" alt="ztrans" width="300">
            <figcaption>ztrans</figcaption>
        </figure>
        <figure>
            <img src="{{site.url}}/assets/hw5/figures/reallyInwards_us.png" alt="reallyInwards" width="300">
            <figcaption>reallyInwards</figcaption>
        </figure>
    </div>
    <figcaption>Figure 1: Epipolar lines for some of the datasets</figcaption>
</figure>


### Task 1: Estimating $$\MB$$

We will give you a set of 3D points $$\{\XB_i\}_i$$ and corresponding 2D points $$\{\pB_i\}_i$$. %In part 1, you're given corresponding point locations in `pts2d-norm-pic.txt` and `pts3d-norm.txt`, which corresponds to a camera projection matrix. **Solve** the projection matrix $$P$$ and **include** it in your report. The goal is to compute the projection matrix $$\MB$$ that maps from world 3D coordinates to 2D image coordinates. Recall that 

$$
\pB \equiv \MB \XB
$$ 

and (see foreword) by deriving an optimization problem. The script `task1.py` shows you how to load the data. The data we want you to use is in `task1/`, but we show you how to use data from Task 2 and 3 as well. **Credit:** The data from task 1 and an early version of the problem comes from James Hays's Georgia Tech CS 6476. 

1.  *(15 points)* {{ code }} <span class="code">Fill in `find_projection` in `task1.py`.</span>

2.  *(5 points)* {{ report }} <span class="report">Report $$\MB$$</span> for the data in `task1/`.

3.  *(10 points)* {{ code }} <span class="code">Fill in `compute_distance` in `task1.py`.</span> 
	
	In this question, you need to compute the average distance in the image plane (i.e., pixel locations) between the homogeneous points $$\MB \XB_i$$ and 2D image coordinates $$\pB_i$$, or
	$$
	\label{eqn:projectionError}
	\frac{1}{N} \sum_{i}^{N} ||\textrm{proj}(\MB\XB_i) - \pB_i||_2 .
	$$
	where $$\textrm{proj}([x,y,w]) = [x/w, y/w]$$.
	The distance quantifies how well the projection maps the points $$\XB_i$$ to $$\pB_i$$. You should use `find_projection` from part a).
	Note: You should feel good about the distance if it is **less than 0.01** for the given sample data. If you plug in different data, this threshold will of course vary.

4.  *(5 points)* {{ report }} <span class="report">Describe what relationship, if any, there is between Equation \ref{eqn:projectionError</span> and Equation 6 in the HW5 Notes} 
	
	Note that the points we've given you are well-described by a linear projection -- there's no noise in the measurements -- but in practice, there will be an error that has to minimize. Both equations represent objectives that could be used. If they are the same, show it; if they are not the same, report which one makes more sense to minimize. Things to consider include whether the equations directly represent anything meaningful.


## Estimation of the Fundamental Matrix and Reconstruction

**Data:** we give you a series of datasets that are nicely bundled in the folder `task23/`. Each dataset contains two images `img1.png` and `img2.png` and a numpy file `data.npz` containing a whole bunch of variables. The script `task23.py` shows how to load the data.

**Credit:** `temple` comes from Middlebury's Multiview Stereo dataset. The images shown in the synthetic images are described in HW1's credits. 


### Task 2: Estimating $$\FB$$

1.  *(15 points)* {{ code }} <span class="code">Fill in `find_fundamental_matrix`</span> in `task23.py`. You should implement the eight-point algorithm. Remember to normalize the data  and to reduce the rank of $$\FB$$. For normalization,
you can scale the image size and center the data at 0.

2.  *(10 points)* {{ code }} <span class="code">Fill in `compute_epipoles`.</span> This should return the homogeneous coordinates of the epipoles -- remember they can be infinitely far away!

3.  *(5 points)* {{ report }} <span class="report">Show epipolar lines for `temple`, `reallyInwards`, and another dataset of your choice.</span>

4.  *(5 points)* {{ report }} <span class="report">Report the epipoles for `reallyInwards` and `xtrans`</span>.


### Task 3: Triangulating $$\XB$$

<figure class="figure-container">
  <img src="{{site.url}}/assets/hw5/figures/reallyInwards_rec.png" alt="reallyInwards_rec" width="500">
  <figcaption>Figure 2: Visualizations of reallyInwards reconstructions</figcaption>
</figure>

The next step is extracting 3D points from 2D points and camera matrices, which is called triangulation. Let $$\XB$$ be a point in 3D.

$$
\pB = \MB_1 \XB ~~~~ \pB' = \MB_2 \XB
$$

Triangulation solves for $$\XB$$ given $$\pB, \pB', \MB_1, \MB_2$$. We'll use OpenCV's algorithms to do this.

1.  *(5 points)* {{ report }} <span class="report">Compute the Essential Matrix $$\EB$$ for the Fundamental Matrix $$\FB$$.</span> You should do this for the dataset `reallyInwards`. Recall that

	$$
	\FB = \KB'^{-T} \EB \KB^{-1}
	$$

	and that $$\KB, \KB'$$ are always invertible (for reasonable cameras), so you can compute $$\EB$$ straightforwardly.

2.  *(15 points)* {{ code }} <span class="code">Fill in `find_triangulation` in `task23.py`.</span> 

	The first camera's projection matrix is $$\KB[\IB,\zeroB]$$. The second camera's projection matrix can be obtained by decomposing $$\EB$$ into a rotation and translation via `cv2.decomposeEssentialMat`. (Note: $$\EB$$ can be obtained using the formula from part a) This function returns two matrices $$\RB_1$$ and $$\RB_2$$ and a translation $$\tB$$. The four possible camera matrices for $$\MB_2$$ are: 	
	
	$$ 	
	\MB_2^1 = \KB' [\RB_1, \tB],~~~~\MB_2^2 = \KB' [\RB_1, -\tB],~~~~\MB_2^3 = \KB' [\RB_2, \tB],~~~~\MB_2^4 = \KB' [\RB_2, -\tB] 	
	$$
	
	You can identify which projection is correct by picking the one for which the most 3D points are in front 	of both cameras. This can be done by checking for the positive depth, which can be done by looking at the last entry of the homogeneous coordinate: the extrinsics put the 3D point in the camera's frame, where $$z<0$$ is behind the camera, and the last row of $$\KB$$ is $$[0,0,1]$$ so this does not change things.

	Finally, triangulate the 2D points using `cv2.triangulatePoints`.

3.  *(10 points)* {{ report }} <span class="report">Put a visualization of the point cloud for `reallyInwards` in your report.</span> You can use `visualize_pcd` in `utils.py` or implement your own.

 
# Tasks Checklist

This section is meant to help you keep track of the many things that go in the report:
- [ ] **Estimating $$M$$**
	- [ ] 1.1 - {{ code }} `find_projections`
	- [ ] 1.2 - {{ report}} Report $$M$$ for `task1/`
	- [ ] 1.3 - {{ code }} `compute_distance`
	- [ ] 1.4 - {{ report }} Relationship between equation 2 and 6
- [ ] **Estimating $$F$$**
	- [ ] 2.1 - {{ code }} `find_fundamental_matrix`
	- [ ] 2.2 - {{ code }} `compute_epipoles`
	- [ ] 2.3 - {{ report }} Epipolar lines for `temple`, `reallyInwards`, and your choice
	- [ ] 2.4 - {{ report }} Epipoles for `reallyInwards` and `xtrans`
- [ ] **Triangulating $$X$$**
	- [ ] 3.1 - {{ report }} Compute $$E$$
	- [ ] 3.2 - {{ code }} `find_triangulation`
	- [ ] 3.3 - {{ report }} Point cloud for `reallyInwards`


# Canvas Submission Checklist

In the `zip` file you submit to Canvas, the directory named after your uniqname should include the following files: