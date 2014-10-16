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

Codes several utility functions to be run as streaming jobs on a
Hadoop system. This is part of the Fisherfaces-on-Hadoop package.
Please see the file Fisherfaces_Hadoop.py for more information on
the algorithm, citations and prior work.

Authors
- Tathagato Rai Dastidar - trdastidar AT gmail dot com
- Krishanu Seal - krishanu dot seal AT gmail dot com

'''

import sys
import numpy as np
import base64

def compute_mean(mode, delim, dtype, total_tag, d_format=0):
    '''
    Mapper and reducer function for mean computation. We compute the
    overall mean for each column of the data, as well as label-wise means.

    mode - 'map' or 'reduce'
    delim - separator used in data.
    dtype - numpy datatype to be used for input data
    total_tag - the tag to use for the overall mean.
    d_format - format of the input/output data. Defaults to 0 which results
            in base64 encoded I/O. Any other value will result in normal
            ascii I/O.

    If d_format is 0 (default) this assumes that the data is base64
    encoded numpy array.
    Otherwise, the data is assumed to be a set of numbers (ascii) separated
    by space (' ').

    Mapper output:
    tag \t num_rows \t sum of each column (for the tag) in base64 or ascii.

    Reducer output:
    tag \t num_rows \t mean of each column (for the tag) in base64 or ascii.
    '''
    sums = {}
    num_records = {}
    num_records[total_tag] = 0

    i = 0
    for line in sys.stdin:
        line = line.rstrip()
        toks = line.split(delim)
        if len(toks) != 3: continue

        # Increment row count for the overall mean.
        if mode == 'map':
            # For mapper, increment by 1
            N = 1
            # Also increment the total tag as this label will not be
            # present in input data.
            num_records[total_tag] += 1
        else:
            # For reducer, increment by num_rows output by mapper.
            N = int(toks[1])

        # Now increment row count for the current tag
        if toks[0] not in num_records:
            num_records[toks[0]] = 0
        num_records[toks[0]] += N

        # Add current row to the sum for the current label.
        data = None
        if d_format == 0:
            b = base64.decodestring(toks[2])
            data = np.frombuffer(b, dtype=dtype)
        else:
            if dtype == np.uint8:
                data = np.asarray(map(int, toks[2].split(' ')), dtype=dtype)
            else:
                data = np.asarray(map(float, toks[2].split(' ')), dtype=dtype)
        if mode == 'map':
            if i == 0:
                sums[total_tag] = np.array(data, dtype=np.float64)
            else:
                # Since mapper input will not have the 'total_tag' label, we
                # need to add the data to the total sum separately.
                sums[total_tag] = sums[total_tag] + data

        if toks[0] not in sums:
            sums[toks[0]] = np.array(data, dtype=np.float64)
        else:
            # add the data to current label sums.
            sums[toks[0]] = sums[toks[0]] + data
        i = i+1

    for key in sums:
        if mode == 'reduce':
            # In reducer, divide by row count to get the mean.
            sums[key] = sums[key] / num_records[key]
        if d_format == 0:
            b = base64.b64encode(np.ravel(sums[key]))
            sys.stdout.write(key + delim + str(num_records[key]) + delim + b + '\n')
        else:
            sys.stdout.write(key + delim + str(num_records[key]) + delim + ' '.join(map(str, np.ravel(sums[key]))) + '\n')


def mean_center(mean_file, delim, total_tag, d_format=0):
    '''
    Function to subtract the column means from each row of data to make
    the data mean centered. Map-only job.

    mean_file - file which stores the mean for each column. Generated
        by the compute_mean() function above.
    delim - separator used in data.
    total_tag - the tag to use for the overall mean.
    d_format - format of the input/output data. Defaults to 0 which results
            in base64 encoded I/O. Any other value will result in normal
            ascii I/O.
    '''
    # Read the column means from the file which will be passed via
    # distributed cache.
    with open(mean_file) as fin:
        for line in fin:
            line = line.rstrip()
            toks = line.split(delim)
            if toks[0] != total_tag:
                # We are interested in only the overall mean. So, ignore
                # the class means.
                continue
            mean = None
            if d_format == 0:
                b = base64.decodestring(toks[2])
                mean = np.frombuffer(b, dtype=np.float64)
            else:
                mean = np.asarray(map(float, toks[2].split(' ')), dtype=np.float64)
            break

    for line in sys.stdin:
        line = line.rstrip()
        toks = line.split(delim)
        if len(toks) != 3:
            continue
        data = None
        if d_format == 0:
            b = base64.decodestring(toks[2])
            data = np.frombuffer(b, dtype=np.uint8)
        else:
            data = np.asarray(map(int, toks[2].split(' ')), dtype=np.uint8)
        data = data - mean
        if d_format == 0:
            be = base64.b64encode(data)
            sys.stdout.write(toks[0] + delim + toks[1] + delim + be + '\n')
        else:
            sys.stdout.write(toks[0] + delim + toks[1] + delim + ' '.join(map(str, np.ravel(data))) + '\n')

def project(eigenfile, dtype, delim, d_format=0):
    '''
    Function to reduce the dimension of a given data by multiplying with
    a set of eigenvectors. Map-only job.

    The eigenvectors are passed via distributed cache.

    eigenfile - File storing the eigenvectors. Eigenvectors are stored as
        columns in this file. Thus, this file should have number of lines
        == size of input images, and number of columns == reduced dimension.
    dtype - numpy datatype to be used for input data
    delim - separator used in data.
    d_format - format of the input/output data. Defaults to 0 which results
            in base64 encoded I/O. Any other value will result in normal
            ascii I/O.
    '''

    # Each column in eigenfile is an eigenvector. Assume there are C
    # eigenvectors.
    E = []
    with open(eigenfile) as fin:
        for line in fin:
            line = line.rstrip()
            if d_format == 0:
                b = base64.decodestring(line)
                row = np.frombuffer(b, dtype=np.float64)
            else:
                row = np.asarray(map(float, line.split(' ')), dtype=np.float64)
            E.append(row)

    E = np.asmatrix(E)
    # E is now a N X C matrix, where N is the number of pixels in the input
    # images.
    E = E.T
    # Now E has become C X N (after transposing it).
    # Multiplying with N X 1 images gives a vector of dimension C X 1. Thus
    # this achieves dimensionality reduction.

    # Multiply each line in the input (an image) with the transposed
    # eigenvectors matrix.
    for line in sys.stdin:
        line = line.rstrip()
        toks = line.split(delim)
        if len(toks) != 3:
            continue
        x = None
        if d_format == 0:
            b = base64.decodestring(toks[2])
            x = np.frombuffer(b, dtype=dtype)
        else:
            if dtype == np.uint8:
                x = np.asarray(map(int, toks[2].split(' ')), dtype=dtype)
            else:
                x = np.asarray(map(float, toks[2].split(' ')), dtype=dtype)

        xp = np.dot(E, x)

        if d_format == 0:
            be = base64.b64encode(xp)
            sys.stdout.write(toks[0] + delim + toks[1] + delim + be + '\n')
        else:
            sys.stdout.write(toks[0] + delim + toks[1] + delim + ' '.join(map(str, np.ravel(xp))) + '\n')
    

def read_mean_file(mean_file, delim, d_format=0):
    '''
    Function to parse a file generated by the compute_mean() function above.
    Stores the total and class specific mean and N in a dict.
    '''
    mean = {}
    num_records = {}
    with open(mean_file) as fin:
        for line in fin:
            line = line.rstrip()
            toks = line.split(delim)
            if d_format == 0:
                b = base64.decodestring(toks[2])
                mean[toks[0]] = np.frombuffer(b, dtype=np.float64)
            else:
                mean[toks[0]] = np.asarray(map(float, toks[2].split(' ')), dtype=np.float64)
            num_records[toks[0]] = int(toks[1])
    return [mean, num_records]

def multiply_with_transpose(mode, delim, mean_file='', d_format=0):
    '''
    Function to multiply a square matrix A with its transpose, i.e. computes
    A^T * A

    The map multiplies each row in the input matrix with its transpose
    and keeps adding the result to a matrix. Optionally, if the mean_file
    is specified, it subtracts the mean of the class (corresponding to
    the current row) before multiplying.

    The reduce phase adds the matrices output by each mapper. Needs to
    run on a single reducer.

    mode - 'map' or 'reduce'
    mean_file - file which stores the mean for each column in each class.
        Generated by the compute_mean() function above.
    delim - separator used in data.
    total_tag - the tag to use for the overall mean.
    d_format - format of the input/output data. Defaults to 0 which results
            in base64 encoded I/O. Any other value will result in normal
            ascii I/O.
    '''

    Sw = None
    mean = None
    n = None
    dim = 0
    if mode == 'map':
        if mean_file != '':
            [mean, n] = read_mean_file(mean_file, delim)

        for line in sys.stdin:
            line = line.rstrip()
            toks = line.split(delim)
            xi = None
            if d_format == 0:
                b = base64.decodestring(toks[2])
                xi = np.frombuffer(b, dtype=np.float64)
            else:
                xi = np.asarray(map(float, toks[2].split(' ')), dtype=np.float64)
            if Sw is None:
                # Initialize the result matrix to all 0's.
                dim = len(xi)
                Sw = np.zeros((dim,dim), dtype=np.float64)

            if mean_file != '':
                # Mean file is specified. Subtract the class mean from the
                # data before multiplication.
                xi = xi - mean[toks[0]]

            # Multiply with transpose and add to result.
            Sw = Sw + xi.reshape(-1,1)*xi

    else:
        Sw_t = []
        j = 0
        for line in sys.stdin:
            line = line.rstrip()
            if d_format == 0:
                b = base64.decodestring(line)
                x = np.frombuffer(b, dtype=np.float64)
            else:
                x = np.asarray(map(float, line.split(' ')), dtype=np.float64)
            if Sw is None:
                # Initialize the result matrix to all 0's.
                dim = len(x)
                Sw = np.zeros((dim,dim), dtype=np.float64)

            # Keep accumulating the output of a single mapper to form an
            # intermediate result matrix.
            Sw_t.append(x)
            j = j+1
            if j == dim:
                # We have completed reading the output of a single mapper.
                # Add the mapper output matrix to the result matrix.
                Sw_t = np.asmatrix(Sw_t, dtype=np.float64)
                Sw = Sw + Sw_t
                Sw_t = []
                j = 0

    for i in range(dim):
        be = None
        if d_format == 0:
            be = base64.b64encode(np.ravel(Sw[i]))
        else:
            be = ' '.join(map(str, np.ravel(Sw[i])))
        sys.stdout.write(be + '\n')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print 'Usage: ' + sys.argv[0] + ' <operation> [other options]'
        print '\twhere operation is one of: \'mean\', \'center\', \'project\' or \'mult_transpose\''
        sys.exit(1)

    operation = sys.argv[1]
    if operation == 'mean':
        if len(sys.argv) < 3:
            print 'Usage: ' + sys.argv[0] + ' mean <mode> [data_type] [total_tag] [data_format]'
            print '\tmode -- One of \'map\' or \'reduce\''
            print '\tdata_type -- (Optional) One of \'int\' or \'float\''
            print '\ttotal_tag -- (Optional - default: __total__) Tag to be used for overall mean'
            print '\tdata_format -- (Optional - default: 0) If 0, assumes base64 encoded data, ascii otherwise'
            sys.exit(1)

        total_tag = '__total__'
        dtype = np.uint8
        if sys.argv[2] == 'reduce':
            dtype = np.float64
        if len(sys.argv) >= 4:
            if sys.argv[3] == 'float':
                dtype = np.float64
        if len(sys.argv) >= 5:
            total_tag = sys.argv[4]
        d_format = 0
        if len(sys.argv) >= 6:
            d_format = int(sys.argv[5])
        compute_mean(sys.argv[2], '\t', dtype, total_tag, d_format=d_format)

    elif operation == 'center':
        if len(sys.argv) < 3:
            print 'Usage: ' + sys.argv[0] + ' center <mean_file> [total_tag] [data_format]'
            print '\tmean_file -- Mean output file generated by the \'mean\' operation'
            print '\ttotal_tag -- (Optional - default: __total__) Tag to be used for overall mean'
            print '\tdata_format -- (Optional - default: 0) If 0, assumes base64 encoded data, ascii otherwise'
            sys.exit(1)
        total_tag = '__total__'
        if len(sys.argv) >= 4:
            total_tag = sys.argv[3]
        d_format = 0
        if len(sys.argv) >= 5:
            d_format = int(sys.argv[4])
        mean_center(sys.argv[2], '\t', total_tag, d_format=d_format)

    elif operation == 'project':
        if len(sys.argv) < 3:
            print 'Usage: ' + sys.argv[0] + ' project <eigenfile> [data_type] [data_format]'
            print '\teigenfile -- File containing the eigen vectors stored in column format (each column is an eigenvector)'
            print '\tdata_type -- (Optional) One of \'int\' or \'float\''
            print '\tdata_format -- (Optional - default: 0) If 0, assumes base64 encoded data, ascii otherwise'
            sys.exit(1)
        dtype = np.float64
        if len(sys.argv) >=4 and sys.argv[3] == 'int':
            dtype = np.uint8
        d_format = 0
        if len(sys.argv) >= 5:
            d_format = int(sys.argv[4])
        project(sys.argv[2], dtype, '\t', d_format=d_format)

    elif operation == 'mult_transpose':
        if len(sys.argv) < 3:
            print 'Usage: ' + sys.argv[0] + ' mult_transpose <mode> [data_format] [<mean_file>]'
            print '\tmode -- One of \'map\' or \'reduce\''
            print '\tdata_format -- (Optional - default: 0) If 0, assumes base64 encoded data, ascii otherwise'
            print '\tmean_file -- (Optional) Mean output file generated by the \'mean\' operation. If specified, the mean of corresponding class is subtracted from each row before multiplication'
            sys.exit(1)
        mean_file = ''
        d_format = 0
        if len(sys.argv) >= 4:
            d_format = int(sys.argv[3])
        if len(sys.argv) >= 5:
            mean_file = sys.argv[4]
        multiply_with_transpose(sys.argv[2], '\t', d_format=d_format, mean_file=mean_file)

    else:
        print 'Invalid operation ' + operation
        print 'Options are: mean, center, project and mult_transpose'
        sys.exit(1)

