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

A simple example script to create the Hadoop friendly input from the
Yale face database, to be used by the Fisherfaces algorithm. Please see
the Fisherfaces_Hadoop.py file for more information on the algorithm,
citations, and prior work.

The Extended Yale Face Database B: http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html

This script reads a set of images from disk, and creates a file where
each line of the file is a base64 encoded image binary.

'''

import sys
import os
import numpy as np
import base64
import Image
import re

def process_images(path, ofile, sz=None):
    fout = open(ofile, 'w')
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in sorted(os.listdir(subject_path)):
                if re.search('Ambient', filename):
                    continue
                img = None
                try:
                    img = Image.open(os.path.join(subject_path, filename))
                except:
                    print 'Could not open image ' + filename
                    continue
                img = img.convert('L')
                if sz is not None:
                    img = img.resize(sz, Image.ANTIALIAS)
                a = np.asarray(img, dtype=np.uint8).flatten()
                b = base64.b64encode(a)
                fout.write(subdirname + '\t' + filename + '\t' + b + '\n')

    fout.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: ' + sys.argv[0] + ' <database root directory> <output file>'
        sys.exit(1)
    path = sys.argv[1]
    ofile = sys.argv[2]
    process_images(path, ofile, (70,80))
