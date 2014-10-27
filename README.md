# Fisherfaces on Hadoop #

## Introduction ##

This projects implements the well-known Fisherfaces face recognition
algorithm on the Hadoop Map-Reduce framework.

Several implementations of the Fisherfaces algorithm (including one from
the OpenCV project) are available. However, all of them are single machine
implementations, and thus cannot scale well in case the number of input
images is large. One alternative could be to create the model (i.e. the
Fisherfaces eigenvectors) from a smaller set of images and then create the
database by applying the model on each image of the larger set. The latter
step can be easily parallelized. However, depending on how good and diverse
the training set is, the accuracy of the model can vary significantly.

This project is an attempt at parallelizing and scaling the Fisherfaces
algorithm to a large number of input images, using the Hadoop Map-Reduce
framework. Several steps in the algorithm have been implemented using
Hadoop Map-Reduce. A detailed description of the algorithm steps is given
later in this document.

This implementation is inspired by the Python based Fisherface
implementation by Phillipp Wagner
[https://github.com/bytefish/facerec](https://github.com/bytefish/facerec)

## Algorithm overview ##

### Pre-requisites ###
The Fisherfaces algorithm requires a set of input images which are:

* in grayscale
* of exactly the same size -- the algorithm is sensitive to scale
* cropped to include only the face and as little background as possible
* aligned with each other, i.e.:
    * the face should be upright (two eyes should lie on the same horizontal line) -- rotate the original image if necessary
    * the eyes should be at approximately the same height in all images
* tagged with some ID indicating the person shown in the image

Most face databases available for research use are annotated with these
details (face coordinates, eye coordinates, head tilt angle, etc.) and this
information can be used for cropping and aligning the input images. If
however, the intent is to use un-annotated images, a good face and face
landmark detection system is necessary. We used the
[Rekognition APIs](https://rekognition.com/) for this purpose. Our
experiments showed that the [OpenCV face and eye detection models](http://docs.opencv.org/trunk/doc/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html) are woefully inadequate. 

The cropped and aligned images should be written into a file such that each
line of the file contains an encoded image. Please see the code in
Rekognition.py and yale.py to understand how to create such a file.

This file is copied to the HDFS before processing starts.

### Algorithm steps ###

Assume that we have N images in total of C people. Each image is labeled
with an ID which indicates the person whose image it is. No two persons
have the same ID. Also assume that each image has n pixels. n is typically
a few thousands (e.g. for a 70X80 face image, n = 5600), whereas N can be
quite large.

Let this N x n matrix be called 'X'

#### Step 1: Compute the mean of the input images ####

N images, each with n pixels results in an N X n matrix (a matrix with N
rows and n columns). The first step is to compute the 'mean image', i.e.
the n-dimensional vector where each element is the mean of the corresponding
column of the input N X n matrix. In addition, we also compute the mean
image for each class (i.e. images belonging to the same person), though
this step is not strictly necessary at this stage.

A simple map-reduce job does this task. The map phase outputs the partial
mean and record count for each split, while the reduce phase (with a single
reducer) computes the global mean and count.

#### Step 2: Compute the mean-diff matrix ####

Once the mean image is computed, we create a matrix U by subtracting the
mean image from each image. U, like X, is an N X n matrix. This is done by
a map-only job where the global mean data is passed on to the mappers via
the Hadoop distributed cache.

#### Step 3: Compute the covariance matrix S ####

Next step is to compute the covariance matrix S of U, i.e. S = (U^T * U)/N

Some philosophical discussion is in order here. Computation of PCA involves
computing the covariance matrix (S) of the mean-diff matrix (U) and then
computing the eigenvalues and eigenvectors of the covariance matrix. S is
of dimension n X n. Since traditional implementations assume n to be much
larger than N, they compute the eigenvalues in a round-about way by
computing the SVD of the mean-diff matrix and then deriving the
eigenvalues/vectors from the SVD. This is advantageous since the dimension
of U * U^T is N X N, assumed to be far less than n X n.

However, when we try to parallelize the process, we are faced with
some problems:

* In our case N can potentially be far more than n (which is the motivation behind parallelizing the algorithm in the first place)
* Computing the largest K eigenvectors using the map-reduce implementation of SVD (the Mahout library offers such an implementation) can take a very long time as it is an iterative algorithm.

In our case, n can be in the order of a few thousands (typically less than
10,000-12,000 if the images are suitably resized). Our experiments showed
that using images of size less than 80 X 80 causes no degradation in
recognition accuracy. On the other hand, N can be several tens of thousands
or more. On modern computers, computing the eigenvectors of a n X n
symmetric matrix is not an intractable problem.

Hence, we adopt a different approach.

* Compute the covariance matrix (this can be parallelized very easily)
* Compute the eigenfaces from the covariance matrix on a single node.

Multiplying a matrix with its transpose is a simpler problem than matrix
multiplication and can be easily ported to the Map-Reduce framework. The
map phase computes a n X n matrix by multiplying each row with its
transpose and accumulating the sum. Once this is over, we sum the matrices
output by each mapper to get the final U^T * U matrix.
[The HIPI page](http://hipi.cs.virginia.edu/examples/pca.html) nicely
illustrates this process.

#### Step 4: Compute eigenvalues and eigenvectors of S #####

This step is done on a single node. We compute the eigenvalues and
eigenvectors of the S matrix. We then sort the eigenvalues in descending
order and select the K1 largest eigenvalues and their corresponding
eigenvectors. Let them be denoted by e (eigenvalues) and E (eigenvectors)
respectively. E is an n X K1 matrix (i.e. each column is an eigenvector).
These eigenvectors are also called eigenfaces.

For a pure Eigenfaces implementation (i.e. without the LDA stage),
a small number of eigenvectors (typically < 100) are sufficient for
face detection. However, in case of Fisherfaces, usually a larger
number of eigenvectors are used. The original paper [Belheumer et al. 1997]
takes K1 = N - C. However, in case of large N,
N - C can actually be larger than
n! We experimented with this number, and found that it can be reduced.
We have kept this parameter (K1), as configurable. Please see the
section on '*Choosing K1 and K2*' for details.

#### Step 5: Compute the eigenfaces representation of each image ####

Given an image x, its eigenface representation is given by
y = (x - m) * E, where m is the 'mean image' computed in step 1 above.
Note that the quantity x - m is already computed in step 2. Thus, in this
stage, we simply multiply each row of U by E in a map-only job. E is passed
to the mappers via distributed cache. Let the resultant matrix (where each
row is a PCA transformed image) be denoted by Y. Y is a N X K1 matrix.

This step concludes the Eigenfaces computation, which is the first phase of
the Fisherfaces algorithm.

#### Step 6: Compute the mean for the eigenfaces representation of each image ####

This is done in a manner similar to step 1. We compute both the overall mean
m as well as the mean for each class c, m\_c.

#### Step 7: Compute the between class scatter ####

The between-class-scatter matrix S\_b is defined as follows:

    <LaTeX> :-)
    \begin{equation}
    S_b = \sum_{i=1}^{C} N_i(m_i - m)^T(m_i - m)
    \end{equation}
where m\_i is the mean for class i and m is the overall mean.

Each mean vector is of dimension K1, and this sum is computed as a
single machine operation. However, in case of large C, we can choose to
run this as a map-reduce job as well. The code used in step 5 can be
reused for this. Feel free to modify this.

S\_b is a K1 X K1 matrix.

#### Step 8: Compute the within class scatter ####

The within-class-scatter matrix S\_w is defined as follows:

    <LaTeX> :-)
    \begin{equation}
    S_w = \sum_{i=1}^{C} \sum_{y_k \in C_i} (x_k - m_i)^T(x_k - m_i)
    \end{equation}

Here the *y*'s are PCA-transformed (step 5) images of dimension 1 X K.
This operation is similar to multiplying a
matrix with its transpose, and thus we reuse the same code as in step 3 to
compute S\_w. We use the Y (PCA transformed image) matrix, subtract the mean
of the corresponding class from each row, and then multiply with its
transpose.

S\_w is also a K1 X K1 matrix.

#### Step 9: Compute the LDA ####

This step is the most compute intensive step of the process. We compute
the LDA eigenvalues and eigenvectors as follows:

    #!/usr/bin/env python
    import numpy as np
    ...
    T_1 = np.linalg.inv(S_w)
    T_2 = T_1 * S_b
    evalues, evectors = np.linalg.eig(T_2)

S\_w and S\_b are K1 X K1 matrices. The original paper takes
K1 = N - C, and in case of large N, this step can become a bottleneck
since it is performed on a single machine. However, our experiments show
that using a smaller K1 is also possible. This is discussed in detail
later.

As in the PCA computation stage, we sort the eigenvalues in descending order
and select the top K2 eigenvalues and their corresponding eigenvectors.

K2 is taken as C - 1 in the original paper. However, we have found that
a much smaller K2 can be used. This parameter is also kept as configurable.

#### Step 10: Compute the Fisherfaces eigenvectors ####

The PCA eigenvectors (n X K1) were computed in step 4 and the LDA
eigenvectors (K1 X K2) were computed in step 9. The Fisherfaces
eigenvectors matrix is the product of these two and has dimension n X K2.

This operation is done in a single node and the results are saved for later
use.

#### Step 11: Compute the Fisherfaces database ####
For each input image, the Fisherface representation of it is obtained by
multiplying it with the Fisherfaces eigenvectors. A map-only job is run
on the input images (the process is the same as in step 5) with the
Fisherfaces eigenvectors passed via distributed cache.

That's all, folks!

## Choosing K1 and K2 ##

As mentioned earlier the original paper takes K1 (number of PCA
eigenvectors to choose) as N - C and K2 (number of LDA eigenvectors to
choose) as C - 1. We also mentioned that for large N, this can lead to
several problems (N - C can be potentially larger than n!).

Our experiments showed that reducing K1, without reducing K2, leads to
sub optimal matching accuracy. However, if both are reduced simultaneously,
the accuracy goes up. As of now, we do not have an empirical formula for
choosing K1 and K2. But K1 in the range of 1000-1500 and K2 in the range of
50-300 worked very well. Please feel free to experiment with these, and 
any contribution towards a formula for selecting K1 and K2 is most welcome!.

## Sample usage ##

Now lets get to the code.

The distribution (so far) consists of three main Python files
* Rekognition.py -- Code for cropping and aligning images using the Rekognition API output.
* Fisherface\_Hadoop.py -- Implements the overall control flow of the algorithm and the single machine steps.
* Hadoop\_Functions.py -- Implements the Map-Reduce functions needed for the algorithm implementation.

The last part of the Fisherface\_Hadoop.py file shows some sample usage.

There is also an example search server - Server.py - which implements a
simple HTTP based server for querying face images. Please see the file
Server.py for further details.

To run the algorithm, you first need to create a configuration file in the JSON format. A sample configuration file (example\_config.py) is provided in this distribution. The important elements in the configuration file are as follows:

* 'input': The file containing the input images. This file will be copied to the HDFS. *TODO*: Make it possible to directly specify an HDFS file.
* 'local\_path': The local work directory where all temporary/intermediate results and log files will be stored.
* 'hdfs\_path': The HDFS work directory path. _Caution_: This directory will be *deleted* and recreated before the process starts, so make sure this path either does not exist or is empty.

Optional parameters:
* 'hadoop\_nn': The Hadoop name node URI. To be used to generate the distributed cache location. If not specified, defaults to 'hdfs://localhost:9000/'
* 'k1': The number of eigenvectors to choose in the PCA stage. If not specified, then k1 will be taken as min((N-C), 2000).
* 'k2': The number of eigenvectors to choose in the LDA stage. If not specified, then k1 will be taken as min(C-1), 1000).
* 'num\_splits': The number of splits to be generated for the final Fisherfaces DB. Defaults to '4'.

'Use at your own risk' parameters:
* 'script': Name of the auxillary script for running the Hadoop streaming jobs. Defaults to 'Hadoop\_Functions.py' in the same directory as the main script.
* 'format': '0' indicates base64 encoded I/O (recommended) and any other value indicates ascii I/O. Defaults to '0'.

To create a model from a training set of images:

    #!/usr/bin/env python
    import json
    import sys
    from Fisherfaces_Hadoop import *

    cfg = json.loads(open(sys.argv[1]).read())
    f = Fisherfaces()
    f.create(cfg)

Once the process completes, there will be two main output files in the work
directory:

* eigenvectors.out: The Fisherfaces eigenvectors, to be used for
    getting the Fisherfaces representation of any new image. This is an
    n X (C-1) matrix and each line contains one row of the matrix.
* fisherfaces\_db.out: The Fisherfaces DB, to be loaded for lookup. Each
    line contains the Fisherfaces representation of the corresponding input
    image.

The numerical data parts in both files are by default base64 encoded numpy
arrays (data type numpy.float64) and can be decoded in Python as follows:

    #!/usr/bin/env python
    import numpy as np
    import base64

    with open(data_file) as fin:
        for line in fin:
            line = line.rstrip()
            b = base64.decodestring(line)
            a = np.frombuffer(b, dtype=np.float64)
            ...

It is also possible (though not recommended) to use plain ascii text for
I/O. Set the variable 'format' to '1' in the configuration file and make
sure that in the input file, the image pixel values are written out
separated by a single blank space (' ') between values.

Fisherfaces\_Hadoop.py also contains a simple lookup and matching code at
the end.

## Contributors ##

* Tathagato Rai Dastidar, trdastidar AT gmail dot com
* Krishanu Seal, krishanu dot seal AT gmail dot com

## References: ##
* P. N. Belhumeur, J. P. Hespanha, D. J. Kriegman: 'Eigenfaces vs.  Fisherfaces: Recognition Using Class Specific Linear Projection' IEEE Trans. on Pattern Analysis and Machine Intelligence, vol. 19, No. 7, Jul 1997, p 711-720
* P. Wagner: Fisherfaces http://www.bytefish.de/blog/fisherfaces/
* Principal Component Analysis: http://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/IDAPILecture15.pdf
* Linear Discriminant Analysis: http://research.cs.tamu.edu/prism/lectures/pr/pr\_l10.pdf
