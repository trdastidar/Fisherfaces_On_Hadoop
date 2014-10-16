#!/usr/bin/env python

'''

Software License Agreement (New BSD)

Copyright (c) 2014, Tathagato Rai Dastidar & Krishanu Seal
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.
 * Neither the name of the author nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Authors:
Tathagato Rai Dastidar - trdastidar AT gmail dot com
Krishanu Seal - krishanu dot seal AT gmail dot com


'''

import numpy as np
import sys
import os
import logging
import json
import base64
from scipy.spatial.distance import *

class Fisherfaces:
    '''
    Class for implementing the Fisherfaces algorithm on a Hadoop map-reduce
    framework using Python and Hadoop streaming.

    Three main functions:
    create_model(cfg) --> To create a model by running the Fisherfaces
                          algorithm
    load_model(cfg) --> Loads an already created model (and database).
    match(image) --> Matches a given image against the current database
                     and returns the nearest matches.

    cfg elements:
    - input: The file containing the input images, to be used in
             create_model. Each line should be of the following format
             img_label \t image_id (unique) \t base64 (numpy uint8) image repr.
    - local_path: Local work directory.
    - hdfs_path: Work directory in HDFS.
    - hadoop_home: The installation directory of hadoop. The 'hadoop'
        executable should be in the bin/ directory under this directory.
    - hadoop_streaming: The path to the hadoop streaming jar file

    Requires an auxillary Python script to run the Hadoop streaming
    jobs. The script is in the same directory - Hadoop_Functions.py

    Use the create function to create the Fisherfaces database from an
    input set of images. The output database is created on HDFS and
    then copied to the local disk for convenience. This database can be
    split into smaller parts, since each line of the output file is
    independent. Thus, the lookup mechanism (match()) can be parallelized
    to any degree. The parallelized lookup code is not yet part of this
    distribution.

    The local work directory will contain the following elements:
    Final output files:
    - eigenvectors.out: The Fisherfaces eigenvectors, to be used for
        getting the Fisherfaces representation of any new image.
    - fisherfaces_db.out: The Fisherfaces DB, to be loaded for lookup.

    There will be some temporary/intermediate output files and log files
    in this directory as well. Please see the code for more details on
    what they are. The main log file will be in the run directory, the
    defaulting to fisherfaces.log

    The HDFS output directory will be <HDFS path>/fisherfaces_db which
    will contain many part files.

    Some philosophical discussion is in order. Traditional implementations
    of the Fisherfaces algorithm (or Eigenfaces implementations) assume that
    the total number of images (N) is far less than the number of pixels
    in each image (n).

    Computation of PCA involves computing the covariance matrix of the
    mean-diff matrix (U) and then computing the eigenvalues and eigenvectors
    of the covariance matrix (S). S is of dimension n X n.
    Since n is assumed to be large, traditional implementations compute the
    eigenvalues in a round-about way by computing the SVD of the mean-diff
    matrix and then deriving the eigenvalues/vectors from the SVD. This
    is advantageous since the dimension of U*U^T is N X N, far less than
    n X n.

    However, when we try to parallelize the process, we are faced with
    a problem:
    - In our case N can potentially be far more than n (which is the
      motivation behind parallelizing the algorithm in the first place)
    - If the number of classes (C) is large, computing C eigenvectors
      using the map-reduce implementation of SVD can take a very long time
      as it is an iterative algorithm.

    In our case, n can be in the order of a few thousands (typically less
    than 15,000 if the images are suitably resized), while N can be a
    few millions. On modern computers, computing the eigenvectors of
    a n X n symmetric matrix is not an intractable problem.

    Hence, we adopt a different approach.
    - Compute the covariance matrix (this can be parallelized very easily)
    - Compute the eigenfaces from the covariance matrix on a single node.

    Note:
    Computing the LDA requires eigen computation of a N-C X N-C matrix
    where N is the number of images and C is the number of classes. For
    large N and a lot of images per class, N-C can be quite large. This
    part of the computation is done on a single node, and can thus be a
    potential bottleneck. TODO: Revisit this.

    This implementation builds on the Python based Fisherface implementation
    by Phillipp Wagner (https://github.com/bytefish/facerec)

    References:
    P. N. Belhumeur, J. P. Hespanha, D. J. Kriegman: 'Eigenfaces vs.
        Fisherfaces: Recognition Using Class Specific Linear Projection'
        IEEE Trans. on Pattern Analysis and Machine Intelligence,
        vol. 19, No. 7, Jul 1997, p 711-720

    P. Wagner: Fisherfaces http://www.bytefish.de/blog/fisherfaces/

    Principal Component Analysis: http://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/IDAPILecture15.pdf

    Linear Discriminant Analysis: http://research.cs.tamu.edu/prism/lectures/pr/pr_l10.pdf

    '''
    def __init__(self):
        pass

    def create_model(self, cfg):
        '''
        Runs the fisherfaces algorithm on a set of input images and creates
        the fisherface implementation of each image.
        '''
        assert 'local_path' in cfg, \
            'Local work directory path not specified in config. Cannot proceed!'

        self.local_path = cfg['local_path']
        try:
            os.makedirs(self.local_path)
        except:
#            print 'Directory ' + self.local_path + ' already exists'
            pass

        assert 'input' in cfg, \
            'Input file path not specified in config. Cannot proceed!'
        assert 'hdfs_path' in cfg, \
            'HDFS base path not specified in config. Cannot proceed!'
        assert 'hadoop_home' in cfg, \
            'Hadoop installation directory not specified. Cannot proceed!'
        assert 'hadoop_streaming' in cfg, \
            'Hadoop streaming jar not specified. Cannot proceed!'

        self.infile = cfg['input']
        self.hdfs_path = cfg['hdfs_path']
        self.hadoop = cfg['hadoop_home'] + '/bin/hadoop'
        self.hadoop_streaming = cfg['hadoop_streaming']
        self.hadoop_nn = cfg.get('hadoop_nn', 'hdfs://localhost:9000')
        self.script_path = cfg.get('script', os.path.dirname(sys.argv[0]) + '/Hadoop_Functions.py')
        self.script_name = os.path.basename(self.script_path)
        self.delim = '\t'
        self.total_tag = cfg.get('total_tag', '__total__')
        self.fmt = cfg.get('format', '0')
        self.d_format = int(self.fmt)

        self.logfile = cfg.get('logfile', 'fisherfaces.log')

        logging.basicConfig(filename=self.logfile,
                format='%(asctime)s %(levelname)s > %(message)s',
                level=logging.INFO)

        self.logger = logging.getLogger("FisherFaces_Hadoop");

        self.logger.info('Successfully initialized Fisherfaces object')

        # Start the fisherfaces algorithm. The first stage is to compute
        # the PCA.
        self.stage = 'pca'

        # Create the HDFS work directory.
        self.__create_hdfs_path()

        # Copy the input file to HDFS
        self.__copy_input_to_hdfs()

        # Compute the mean image and the mean of each class.
        self.__compute_mean()

        # Subtract the overall mean image from each image.
        self.__mean_diff()

        # Compute the covariance matrix of the mean-diff output
        self.__compute_covariance()

        # Compute the eigenvalues and eigenvectors of the covariance
        # matrix.
        self.__compute_pca()

        # Now get the eigenfaces representation of each image. This is
        # done by multiplying each mean-diff image with the eigenvector
        # matrix computed in the above step.
        self.__pca_project()

        # PCA computation is over. Now compute the LDA.
        self.stage = 'lda'

        # Compute the mean of the PCA projected image.
        self.__compute_mean()

        # Compute the between class scatter for the PCA projected images.
        self.__compute_sb()

        # Compute the within class scatter for the PCA projected images.
        self.__compute_sw()

        # Compute the LDA
        self.__compute_lda()

        # Compute and save the combined eigenvectors for later use.
        self.__save_eigenvectors()

        # Get the fisherface representation of each image and save for later
        # use.
        self.__lda_project()


    def load_model(self, cfg):

        '''
        Loads an already computed Fisherfaces module from disk into memory.
        The model consists of a set of eigenvectors (which are used to
        derive the Fisherfaces representation of any given image), and
        a database of face images (in the Fisherfaces reduced-dimension
        representation).

        '''
        assert 'local_path' in cfg, \
            'Local work directory path not specified in config. Cannot proceed!'
        assert 'input' in cfg, \
            'Input file path not specified in config. Cannot proceed!'

        self.local_path = cfg['local_path']
        self.infile = cfg['input']
        self.delim = '\t'
        self.total_tag = cfg.get('total_tag', '__total__')
        self.__fisherfaces_out = cfg.get('database', self.local_path + '/fisherfaces_db.out')
        self.__eigenvectors_out = cfg.get('eigenvectors', self.local_path + '/eigenvectors.out')

        self.logfile = cfg.get('logfile', 'fisherfaces.log')

        logging.basicConfig(filename=self.logfile,
                format='%(asctime)s %(levelname)s > %(message)s',
                level=logging.DEBUG)

        self.logger = logging.getLogger("Fisherfaces_Hadoop");

        # Load the eigenvectors into memory
        self.__load_eigenvectors()

        # Load the database into memory.
        self.__load_database()

    def match(self, x, num_matches=10):
        '''
        Matches a given image with those in the database and returns the top
        N matches (num_matches).

        x is assumed to be a numpy array of type uint8 (i.e. the single
        dimensional representation of the input image).
        '''

        # Project the input image to a lower dimension by multiplying with
        # the eigenvector matrix.
        xp = self.__project(x)

        # Now iterate over all images in the database and store the euclidean
        # distance with the projected image.
        distances = []
        for i in range(len(self.__features)):
#            distances.append(euclidean(xp, self.__features[i]))
            distances.append(np.linalg.norm(xp-self.__features[i]))

        # Sort the distances in ascending order and select top N
        distances = np.asarray(distances)
        idx = np.argsort(distances)
        sorted_distances = distances[idx]

        sorted_distances = sorted_distances[0:num_matches]
        predicted_labels = []
        predicted_names = []
        for i in range(num_matches):
            predicted_labels.append(self.__labels[idx[i]])
            predicted_names.append(self.__img_names[idx[i]])

        match = {}
        match['distances'] = sorted_distances
        match['labels'] = predicted_labels
        match['names'] = predicted_names
        return match

    def __project(self, x):
        '''
        Projects an input image to a lower dimension by multiplying with
        the eigenvector matrix.
        '''
        return np.dot(self.__eigenvectors, x.reshape(-1,1))

    def __execute_cmd(self, cmd, force_fail=1):
        '''
        Utility function to execute a shell command.
        '''
        self.logger.info('Executing command ' + cmd)
        status = os.system(cmd)
        if status:
            self.logger.error('Command ' + cmd + ' exited with status ' + str(status))
            if force_fail:
                sys.exit(1)
            return
        self.logger.info('Command ' + cmd + ' executed successfully')

    def __write_matrix(self, ofile, mat):
        '''
        Utility function to write a given matrix in row major format into
        a given file. Each line of the file will be a row of the matrix.
        '''
        fout = open(ofile, 'w')
        for row in mat:
            if self.d_format == 0:
                b = base64.b64encode(np.ravel(row))
                fout.write(b + '\n')
            else:
                fout.write(' '.join(map(str, np.ravel(row))) + '\n')
        fout.close()

    def __create_hdfs_path(self):
        # Create the HDFS work directory
        self.logger.info('Creating HDFS directory ' + self.hdfs_path)
        cmd = self.hadoop + ' fs -rmr ' + self.hdfs_path
        self.__execute_cmd(cmd, force_fail=0)

        cmd = self.hadoop + ' fs -mkdir ' + self.hdfs_path
        self.__execute_cmd(cmd)

        # Create the cache directory as well.
        cmd = self.hadoop + ' fs -mkdir ' + self.hdfs_path + '/cache'
        self.__execute_cmd(cmd)
        self.logger.info('Created HDFS directory ' + self.hdfs_path)

    def __copy_input_to_hdfs(self):
        self.logger.info('Copying file ' + self.infile + ' to HDFS ' + self.hdfs_path + '/input')
        cmd = self.hadoop + ' fs -mkdir ' + self.hdfs_path + '/input'
        self.__execute_cmd(cmd)

        cmd = self.hadoop + ' fs -copyFromLocal ' + self.infile + ' ' + self.hdfs_path + '/input'
        self.__execute_cmd(cmd)

        self.input_path = self.hdfs_path + '/input'
        self.logger.info('Finished copying input file to HDFS')

    def __compute_mean(self):
        '''
        Function to compute the mean.
        Depending on which stage of the algorithm we are in (pca or lda)
        the input changes. In case of pca stage, we compute the mean of the
        input images. In case of lda, we compute the mean of the PCA
        projected images.
        '''
        self.logger.info('Started computing means ...')
        if self.stage == 'pca':
            data = self.input_path
            output_dir = self.hdfs_path + '/input_mean'
            local_mean = self.local_path + '/input_mean.out'
            dtype = 'int'

        elif self.stage == 'lda':
            data = self.__pca_transformed_data
            output_dir = self.hdfs_path + '/pca_mean'
            local_mean = self.local_path + '/pca_mean.out'
            dtype = 'float'

        else:
            return

        # Execute the hadoop streaming command to compute mean.
        cmd = self.hadoop + ' jar ' + self.hadoop_streaming + ' \\\n' \
            + '-Dmapred.reduce.tasks=1 \\\n' \
            + '-input ' + data + ' \\\n' \
            + '-output ' + output_dir + ' \\\n' \
            + '-file ' + self.script_path + ' \\\n' \
            + '-mapper \'python ' + self.script_name + ' mean map ' + dtype + ' ' + self.total_tag + ' ' + self.fmt + '\' \\\n' \
            + '-reducer \'python ' + self.script_name + ' mean reduce ' + dtype + ' ' + self.total_tag + ' ' + self.fmt + '\' > ' \
            + self.local_path + '/mean_' + self.stage + '.log 2>&1'
        self.__execute_cmd(cmd)

        # Computation over. Now copy the Hadoop output into a local file
        # for later use.
        cmd = self.hadoop + ' fs -cat ' + output_dir + '/part\\* > ' + local_mean
        self.__execute_cmd(cmd)

        self.logger.info('Completed mean computation ...')
        self.__mean_output = output_dir
        self.__local_mean = local_mean
        return

    def __mean_diff(self):
        '''
        Function to subtract the overall mean image from each image.
        '''
        data = self.input_path
        output = self.hdfs_path + '/input_mean_diff'
        cache_local = 'input_mean.txt'
        cache = self.hdfs_path + '/cache/input_mean.txt'

        # Copy the hadoop mean output into a local cache.
        self.logger.info('Creating a cache from mean computation output ...')
        cmd = self.hadoop + ' fs -put ' + self.__local_mean + ' ' + cache
        self.__execute_cmd(cmd)

        self.logger.info('Started computing mean diff ...')
        cmd = self.hadoop + ' jar ' + self.hadoop_streaming + ' \\\n' \
            + '-Dmapred.reduce.tasks=0 \\\n' \
            + '-files \'' + self.hadoop_nn + cache + '\' \\\n' \
            + '-file ' + self.script_path + ' \\\n' \
            + '-mapper \'python ' + self.script_name + ' center ' + cache_local + ' ' + self.total_tag + ' ' + self.fmt + '\' \\\n' \
            + '-input ' + data + ' \\\n' \
            + '-output ' + output \
            + ' > ' + self.local_path + '/mean_diff_' + self.stage + '.log 2>&1'

        self.__execute_cmd(cmd)
        self.logger.info('Completed mean diff computation ...')
        self.__mean_diff = output

    def __read_mean_file(self):
        '''
        Function to read the mean computation output file and load into
        memory as a dict.
        '''
        self.logger.info('Started reading mean output file ' + self.__local_mean + ' ...')

        self.__mean = {}
        self.__num_records = {}
        with open(self.__local_mean) as fin:
            for line in fin:
                line = line.rstrip()
                toks = line.split(self.delim)
                if self.d_format == 0:
                    b = base64.decodestring(toks[2])
                    self.__mean[toks[0]] = np.frombuffer(b, dtype=np.float64)
                else:
                    vals = map(float, toks[2].split(' '))
                    self.__mean[toks[0]] = np.asarray(vals, dtype=np.float64)

                self.__num_records[toks[0]] = int(toks[1])
        self.logger.info('Finished reading mean output file ' + self.__mean_output + ' ...')

    def __compute_covariance(self):
        '''
        Compute the covariance matrix by multiplying the transpose of
        the mean-diff matrix with the mean-diff matrix
        S' = U^T * U

        Note that the actual covariance matrix S = S'/N. We are not
        performing the division here. Any later stage can do it while
        reading the matrix.
        '''

        # Compute the covariance matrix by multiplying the transponse of the
        # mean-diff matrix with the mean-diff matrix. This gives a matrix of
        # dimension nXn where n is the number of pixels in each image.

        self.logger.info('Computing the covariance matrix ...')
        self.__covariance_hdfs = self.hdfs_path + '/covariance'
        self.__covariance_out = self.local_path + '/covariance.out'
        covariance_tmp = self.local_path + '/covariance_map.out'
        cmd = self.hadoop + ' jar ' + self.hadoop_streaming + ' \\\n' \
            + '-Dmapred.reduce.tasks=0 \\\n' \
            + '-file ' + self.script_path + ' \\\n' \
            + '-input ' + self.__mean_diff + ' \\\n' \
            + '-output ' + self.__covariance_hdfs + ' \\\n' \
            + '-mapper \'python ' + self.script_name + ' mult_transpose map ' + self.fmt + '\'' \
            + ' > ' + self.local_path + '/compute_covariance.log 2>&1'
        self.__execute_cmd(cmd)

        # Now we execute the reducer part of the algorithm on the local disk
        # This is done so that the order of the mapper outputs are
        # maintained (Hadoop will shuffle the mapper output, and thus
        # order can get awry). As we would have used a single reducer in
        # any case, this does not lead to any performance degradation.
        cmd = self.hadoop + ' fs -cat ' + self.__covariance_hdfs + '/part\\* >' + covariance_tmp
        self.__execute_cmd(cmd)

        cmd = 'python ' + self.script_path + ' mult_transpose reduce ' + self.fmt + ' < ' + covariance_tmp + ' > ' + self.__covariance_out
        self.__execute_cmd(cmd)

        self.logger.info('Finished computing the covariance matrix ...')


    def __compute_pca(self):
        '''
        Compute the eigenfaces (i.e. the eigenvectors of the covariance matrix)
        '''
        self.logger.info('Started computing PCA ...')
        # Compute the number of images and number of classes from the mean output
        self.__read_mean_file()
        self.num_classes = len(self.__num_records)-1
        self.num_images = 0
        for (k,v) in self.__num_records.items():
            if k == self.total_tag: continue
            self.num_images += v

        self.logger.debug('Started reading covariance data ..')

        X = []
        with open(self.__covariance_out) as fin:
            for line in fin:
                line = line.rstrip()
                if self.d_format == 0:
                    b = base64.decodestring(line)
                    data = np.frombuffer(b, dtype=np.float64)
                else:
                    data = np.asarray(map(float, line.split(' ')), dtype=np.float64)
                # The covariance computation above just did the matrix
                # multiplication part. We need to divide by the number
                # of images
                data = data / self.num_images
                X.append(data)

        X = np.asmatrix(X, dtype=np.float64)
        self.logger.debug('Finished reading covariance data ..')

        self.logger.debug('Started eigen computation ...')
        self.__pca_eigenvalues, self.__pca_eigenvectors = \
                np.linalg.eigh(X)
        self.logger.debug('Finished eigen computation ...')

        # Select the largest N-C (C == number of classes) eigenvalues and
        # their corresponding eigenvectors as the eigenfaces.
        self.N_c = self.num_images - self.num_classes
        self.logger.debug('Sorting eigenvalues and selecting top ' + str(self.N_c) + ' ...')
        idx = np.argsort(-self.__pca_eigenvalues)

        self.__pca_eigenvalues, self.__pca_eigenvectors = \
            self.__pca_eigenvalues[idx], self.__pca_eigenvectors[:,idx]

        self.__pca_eigenvalues = self.__pca_eigenvalues[0:self.N_c].copy()
        self.__pca_eigenvectors = self.__pca_eigenvectors[0:,0:self.N_c].copy()

        # Write the eigenvalues and eigenvectors (reduced dimension)
        # into a file for later use.
        self.__pca_eigenvalues_out = self.local_path + '/pca_eigenvalues.out'
        self.logger.debug('Writing eigenvalues into ' + self.__pca_eigenvalues_out)
        fout = open(self.__pca_eigenvalues_out, 'w')
        if self.d_format == 0:
            b = base64.b64encode(self.__pca_eigenvalues)
            fout.write(b + '\n')
        else:
            fout.write(' '.join(map(str, self.__pca_eigenvalues)) + '\n')
        fout.close()

        self.__pca_eigenvectors_out = self.local_path + '/pca_eigenvectors.out'
        self.logger.debug('Writing eigenvectors into ' + self.__pca_eigenvectors_out)
        self.__write_matrix(self.__pca_eigenvectors_out, self.__pca_eigenvectors)
        self.logger.debug('Eigenvectors.shape = ' + str(self.__pca_eigenvectors.shape))
        self.logger.info('Finished PCA computation ...')


    def __pca_project(self):

        '''
        Create the eigenface representation of each mean-diff image
        multiplying it with the eigenvectors.
        '''

        self.logger.info('Started PCA-projecting the input images ...')
        eigen_local = 'pca_eigenvectors.txt'
        eigen = self.hdfs_path + '/cache/' + eigen_local
        cmd = self.hadoop + ' fs -put ' + self.__pca_eigenvectors_out + ' ' + eigen
        self.__execute_cmd(cmd)

        self.__pca_transformed_data = self.hdfs_path + '/pca_transformed'
        cmd = self.hadoop + ' jar ' + self.hadoop_streaming + ' \\\n' \
            + '-Dmapred.reduce.tasks=0 \\\n' \
            + '-files \'' + self.hadoop_nn + eigen + '\' \\\n' \
            + '-file ' + self.script_path + ' \\\n' \
            + '-mapper \'python ' + self.script_name + ' project ' + eigen_local + ' float ' + self.fmt + '\' \\\n' \
            + '-input ' + self.__mean_diff + ' \\\n' \
            + '-output ' + self.__pca_transformed_data + ' ' \
            + ' > ' + self.local_path + '/pca_project.log 2>&1'
        self.__execute_cmd(cmd)
        self.logger.info('Finished PCA-projecting the input images ...')

    def __compute_sb(self):
        '''
        Computes the between-class scatter (Sb) in the LDA stage.
        This runs on a single node. Sb formula (in LaTeX notation):

        S_B = \sum_{i=1}^{C} { N_i  (\mu_i - \mu)  (\mu_i - \mu)^T }

        where \mu_i is the mean of class i, \mu is the overall mean,
        N_i is the number of images in class i
        '''

        self.logger.info('Started computing SB (LDA stage) ...')

        self.__read_mean_file()

        mt = self.__mean[self.total_tag]
        dim = len(mt)

        self.__Sb = np.zeros((dim,dim), dtype=np.float64)

        for (k,v) in self.__mean.items():
            if k == self.total_tag: continue
            # m --> mu_i
            # mt --> mu
            # num_records[k] --> N_i
            m = self.__mean[k] - mt
            self.__Sb = self.__Sb + \
                self.__num_records[k]*m.reshape(-1,1)*m

        # Store the Sb matrix for later use. This is not strictly
        # necessary.
        self.__Sb_out = self.local_path + '/Sb.out'
        self.logger.info('Finished computation. Writing into file ' + self.__Sb_out + ' ...')
        self.__write_matrix(self.__Sb_out, self.__Sb)
        self.logger.info('Finished computing SB (LDA stage) ...')
        return

    def __compute_sw(self):

        '''
        Computes the within class scatter matrix (Sw) in the LDA stage.
        Formula:

        S_W = \sum_{i=1}^{C}{\sum_{x_k \in X_i}{(x_k-\mu_i)(x_k-\mu_i)^T}}
        '''
        self.logger.info('Started computing Sw (LDA stage) ...')
        self.__Sw_hdfs = self.hdfs_path + '/Sw'
        self.__Sw_out = self.local_path + '/Sw.out'
        Sw_tmp = self.local_path + '/Sw_map.out'

        mean_file = 'pca_mean.txt'
        pca_mean = self.hdfs_path + '/cache/' + mean_file
        cmd = self.hadoop + ' fs -put ' + self.__local_mean + ' ' + pca_mean
        self.__execute_cmd(cmd)

        # We use the same utility to compute the scatter as we used for
        # computing the covariance matrix. However, in this case, we also
        # pass along the mean file, containing the mean of each class,
        # which is subtracted from the vector before multiplication.
        cmd = self.hadoop + ' jar ' + self.hadoop_streaming + ' \\\n' \
            + '-Dmapred.reduce.tasks=0 \\\n' \
            + '-files ' + self.hadoop_nn + pca_mean + ' \\\n' \
            + '-file ' + self.script_path + ' \\\n' \
            + '-input ' + self.__pca_transformed_data + ' \\\n' \
            + '-output ' + self.__Sw_hdfs + ' \\\n' \
            + '-mapper \'python ' + self.script_name + ' mult_transpose map ' + self.fmt + ' ' + mean_file + '\'' \
            + ' > ' + self.local_path + '/compute_sw.log 2>&1'
        self.__execute_cmd(cmd)

        # As before, we do the reduce part of the algorithm on the local
        # machine.
        cmd = self.hadoop + ' fs -cat ' + self.__Sw_hdfs + '/part\\* > ' + Sw_tmp
        self.__execute_cmd(cmd)

        cmd = 'python ' + self.script_path + ' mult_transpose reduce ' + self.fmt + '< ' + Sw_tmp + ' > ' + self.__Sw_out
        self.__execute_cmd(cmd)

        self.logger.info('Finished computing Sw (LDA stage) ...')

    def __compute_lda(self):
        '''
        Computes the LDA eigenvectors from the between-class and within-class
        scatter matrices.

        Sb and Sw are both of dimension n X n. This step involves matrix
        inversion, multiplication, and finally eigen computation. We need
        to revisit this at a later stage to see if we can parallelize
        (at least parts of) this step.
        '''
        self.logger.info('Started computing LDA ...')

        self.logger.debug('Reading SW matrix from file ' + self.__Sw_out + ' ...')
        self.__Sw = []
        with open(self.__Sw_out) as fin:
            for line in fin:
                line = line.rstrip()
                if self.d_format == 0:
                    b = base64.decodestring(line)
                    row = np.frombuffer(b, dtype=np.float64)
                else:
                    row = np.asarray(map(float, line.split(' ')), dtype=np.float64)
                self.__Sw.append(row)

        self.__Sw = np.asmatrix(self.__Sw, dtype=np.float64)
        self.logger.debug('Finished reading SW matrix from file ...')

        self.logger.debug('Computing eigenvalues and eigenvectors ...')

        T = np.linalg.inv(self.__Sw)
        self.__lda_eigenvalues, self.__lda_eigenvectors = \
            np.linalg.eig(T*self.__Sb)

        # Now select the top C-1 eigenvalues and corresponding eigenvectors
        idx = np.argsort(-self.__lda_eigenvalues)

        self.__lda_eigenvalues, self.__lda_eigenvectors = \
            self.__lda_eigenvalues[idx], self.__lda_eigenvectors[:,idx]

        C = self.num_classes-1
        self.__lda_eigenvalues = np.array(self.__lda_eigenvalues[0:C].real, \
                dtype=np.float64, copy=True)
        self.__lda_eigenvectors = np.matrix(self.__lda_eigenvectors[0:,0:C].real, \
                dtype=np.float64, copy=True)
        self.logger.debug('Finished computing and sorting eigenvalues and eigenvectors ...')

        # Write the eigenvalues and eigenvectors (reduced dimension)
        # into a file for later use.
        self.__lda_eigenvalues_out = self.local_path + '/lda_eigenvalues.out'
        self.logger.debug('Writing eigenvalues into ' + self.__lda_eigenvalues_out)
        fout = open(self.__lda_eigenvalues_out, 'w')
        if self.d_format == 0:
            b = base64.b64encode(self.__lda_eigenvalues)
            fout.write(b + '\n')
        else:
            fout.write(' '.join(map(str, self.__lda_eigenvalues)) + '\n')
        fout.close()

        self.__lda_eigenvectors_out = self.local_path + '/lda_eigenvectors.out'
        self.logger.debug('Writing eigenvectors into ' + self.__lda_eigenvectors_out)
        self.__write_matrix(self.__lda_eigenvectors_out, self.__lda_eigenvectors)

    def __save_eigenvectors(self):
        '''
        Computes the Fisherfaces eigenvectors (product of PCA eigenvectors
        and LDA eigenvectors) and saves into a file for later use.
        '''
        # Compute and save eigenvectors for later use.
        self.__eigenvectors = np.dot(self.__pca_eigenvectors, self.__lda_eigenvectors)
        self.__eigenvectors_out = self.local_path + '/eigenvectors.out'
        self.__write_matrix(self.__eigenvectors_out, self.__eigenvectors)

    def __lda_project(self):
        '''
        Computes the Fisherfaces representation of each input image
        by multiplying with the eigenvectors.
        '''

        self.logger.info('Started LDA-projecting the input images ...')
        C = self.num_classes-1
        eigen_local = 'eigenvectors.txt'
        hdfs_eigen = self.hdfs_path + '/cache/' + eigen_local
        cmd = self.hadoop + ' fs -put ' + self.__eigenvectors_out + ' ' + hdfs_eigen
        self.__execute_cmd(cmd)

        self.__fisherfaces_out = self.local_path + '/fisherfaces_db.out'
        output_dir = self.hdfs_path + '/fisherfaces_db'

        cmd = self.hadoop + ' jar ' + self.hadoop_streaming + ' \\\n' \
            + '-Dmapred.reduce.tasks=0 \\\n' \
            + '-files \'' + self.hadoop_nn + hdfs_eigen + '\' \\\n' \
            + '-file ' + self.script_path + ' \\\n' \
            + '-input ' + self.input_path + ' \\\n' \
            + '-output ' + output_dir + ' \\\n' \
            + '-mapper \'python ' + self.script_name + ' project ' + eigen_local + ' int ' + self.fmt + '\' \\\n' \
            + ' > ' + self.local_path + '/lda_project.log 2>&1'
        self.__execute_cmd(cmd)

        cmd = self.hadoop + ' fs -cat ' + output_dir + '/part\\* > ' + self.__fisherfaces_out
        self.__execute_cmd(cmd)

        self.logger.info('Finished LDA-projecting the input images ...')

    def __load_eigenvectors(self):
        '''
        To be used in the match() stage. Loads the Fisherfaces eigenvectors
        (created in the 'create' stage) into memory.
        '''
        self.logger.info('Started loading eigenvectors from file ' + self.__eigenvectors_out)
        self.__eigenvectors = []
        with open(self.__eigenvectors_out) as fin:
            for line in fin:
                line = line.rstrip()
                if self.d_format == 0:
                    b = base64.decodestring(line)
                    row = np.frombuffer(b, dtype=np.float64)
                else:
                    row = np.asarray(map(float, line.split(' ')), dtype=np.float64)
                self.__eigenvectors.append(row)

        self.__eigenvectors = np.asmatrix(self.__eigenvectors, dtype=np.float64)
        self.__eigenvectors = self.__eigenvectors.T
        self.logger.debug('Eigenvectors shape = ' + str(self.__eigenvectors.shape))
        self.logger.info('Finished reading eigenvectors ...')

    def __load_database(self):
        '''
        To be used in the 'match' stage. Loads the Fisherfaces database
        (created in the 'create' stage) into memory.
        '''
        self.logger.info('Started reading face database from file ' + \
            self.__fisherfaces_out + ' ...')
        self.__features = []
        self.__labels = []
        self.__img_names = []
        with open(self.__fisherfaces_out) as fin:
            for line in fin:
                line = line.rstrip()
                toks = line.split('\t')
                self.__labels.append(toks[0])
                self.__img_names.append(toks[1])
                if self.d_format == 0:
                    b = base64.decodestring(toks[2])
                    row = np.frombuffer(b, dtype=np.float64)
                else:
                    row = np.asarray(map(float, toks[2].split(' ')), dtype=np.float64)
                self.__features.append(row)
        self.__labels = np.asarray(self.__labels)
        self.__img_names = np.asarray(self.__img_names)

        self.num_images = len(self.__img_names)
        self.num_classes = len(np.unique(self.__labels))

        self.logger.debug('Each face has ' + str(len(self.__features[0])) + ' components')
        self.logger.debug('Num images = ' + str(self.num_images))
        self.logger.debug('Num classes = ' + str(self.num_classes))
        self.logger.info('Finished reading face database ...')


if __name__ == "__main__":
    # Just a simple example of model creation and matching usage.
    if len(sys.argv) < 3:
        print 'Usage: ' + sys.argv[0] + ' <cfg file> <mode>'
        print '\twhere \'mode\' is one of \'create\' or \'match\''
        sys.exit(1)
    cfg = json.loads(open(sys.argv[1]).read())

    f = Fisherfaces()
    if sys.argv[2] == 'create':
        f.create_model(cfg)
        sys.exit(0)

    # A simple validation utility.
    test_images = cfg['test_images']
    match_file = cfg.get('match_file', './match.tsv')
    dist_file = cfg.get('dist_file', './dist.tsv')
    num_test = int(cfg.get('num_test', '200'))
    fmt = int(cfg.get('format', '0'))

    f.load_model(cfg)
    num_test = 200
    X = []
    y = []
    z = []
    j = 0
    # In our case, we typically use a subset of the training images
    # as the test images.
    # Read the test images into memory.
    with open(test_images) as fin:
        for line in fin:
            line = line.rstrip()
            toks = line.split('\t')
            if fmt == 0:
                i = base64.decodestring(toks[2])
                img = np.frombuffer(i, dtype=np.uint8)
            else:
                img = np.asarray(map(int, toks[2].split(' ')), dtype=np.uint8)
            X.append(img)
            # Record the label/class of each test image as well.
            y.append(toks[0])
            z.append(toks[1])
            j = j+1
            if j > num_test: break
    sys.stderr.write('Finished reading test images ...')

    # We compute the following statistics:
    # p1 - 1 if the closest match is a positive match, else 0.
    # p5 - Percentage of positive matches in the first 5 closest matches.
    # p10 - Percentage of positive matches in the first 10 closest matches.
    # These statistics are calculated for each test image, as well as the
    # overall test.
    counts = {}
    num_images = 0
    total_p1 = 0
    total_p5 = 0
    total_p10 = 0
    # Compute the number of image in each class of the test images.
    for l in y:
        if not l in counts:
            counts[l] = 0
        counts[l] += 1

    fout = open(match_file, 'w')
    fout2 = open(dist_file, 'w')

    # Iterate over the test images and match each against the database.
    for i in range(min(len(X),num_test)):
        num_images += 1
        if num_images%20 == 0:
            sys.stderr.write('Image ' + z[i] + '...\n')
        fout.write(str(y[i]) + '\t' + z[i])
        fout2.write(str(y[i]) + '\t' + z[i])

        predictions = f.match(X[i],15)

        num_matched = 0
        num_compared = 0

        count = counts[y[i]]; # Total number of images available for this class.

        distances = predictions['distances']
        labels = predictions['labels']
        names = predictions['names']
        fout2.write('\t' + str(labels) + '\t' + str(distances) + '\t' + str(names) + '\n')
        for l in labels:
            num_compared += 1
            # Ignore the first match. Since the test images are a subset
            # of the training images, and assuming the math is correct,
            # the first match will always be the image itself, with
            # distance = 0. Thus, we are more interested in the later
            # matches.
            if num_compared == 1: continue
            if num_compared > 11: break

            # The predicted label == test image label. => Match.
            if l == y[i]: num_matched += 1

            # All available images for this label have already been
            # extracted. A negative result now doesn't matter. So, increment
            # the number of matches.
            if num_compared > count: num_matched += 1

            if num_compared == 2:
                fout.write('\t' + str(num_matched))
                total_p1 += num_matched
            elif num_compared == 6:
                fout.write('\t' + str(num_matched*0.2))
                total_p5 += num_matched
            elif num_compared == 11:
                fout.write('\t' + str(num_matched*0.1))
                total_p10 += num_matched
        fout.write('\n')
    if num_images > 0:
        fout.write('Total\tMatches\t' + str(total_p1*1.0/num_images) + '\t' + str(total_p5*0.2/num_images) + '\t' + str(total_p10*0.1/num_images) + '\n')
    fout.close()
    fout2.close()
