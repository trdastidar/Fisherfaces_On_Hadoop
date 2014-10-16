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
import os
import logging
try:
    from PIL import Image
except:
    import Image
import math
import json
import re
import base64
import numpy as np

class Rekognition:
    '''
    Reads the output of the Rekognition API, rotates and crops images to
    extract the face (along with aligning the eyes to the center)

    The grayscale converted and cropped image is then written out either as
    a standalone image file, or a base64 encoded string. Optionally, the
    image is also resized to a given size.

    For each face, it prints out the face bounding box coordinates in the
    original image (after rotation to make the eyes horizontal), and the
    left eye, right eye, nose and mouth bounding boxes w.r.t. the
    cropped image (if those landmarks are available in the input file).

    Typical usage:
    cfg = json.loads(cfg_file)
    r = Rekognition(cfg)
    r.run()

    Description of cfg:
        - path: Input directory containing all images. The image names
            should be in the format <tag>_<suffix>.<extension>. The part
            preceding the first '_' will be interpreted as the 'label' for
            that image.  So, images of the same person should have the same
            'label'
        - infile: The input file containing the face, eye, etc. coordinates.
            The current implementation assumes that each line in this file
            is a json string returned by the Rekognition API. However, we
            can read any other file as well and populate a dict according to
            the Rekognition API format
            Please see the 'placeholder' comment in the code below - it
            indicates where this new function should go in.
        - input_mode: Defaults to 'rekognition'. Can be used to determine
            which reader to use for reading the input file.
        - mode: Output mode. Defaults to 'normal'. In this case, a separate
            cropped image will be written to the output directory. If set to
            'hadoop', it creates a file suitable for hadoop i/o. Each line
            in the file is of the format:
            label \t img_file_name \t base64_encoded_image
        - output: The file where all the face and landmark coordinates
            will be written.
        - cropped_path: In case of mode 'normal', this should be a
            directory. In case of hadoop, this should be a file name. If
            not specified, cropped images will not be produced.
        - height,width: If specified, the cropped image will be of this
            dimension. Otherwise the image will not be resized, and the
            cropped aspect ratio will be 1:1.25 (width:height)
        - min_face_dim: The minimum dimension for a face to be useable.
            Defaults to 50.
        - pattern: Specific to rekognition API output. This should be a
            regular expression encoding the initial part of the image URL.
            The part of the URL matching this pattern will be removed, and
            the remaining part will be treated as the image filename.
            Defaults to '^.*:8000'
        - logfile: Name of logfile. Defaults to 'rekognition.log'. Created
            under the 'path' directory.

    '''

    def __init__(self, cfg):
        
        assert 'path' in cfg, \
            'Base path not specified in config. Cannot proceed!'
        assert 'infile' in cfg, \
            'Rekognition output file not specified. Cannot proceed!'
        assert 'output' in cfg, \
            'Cropping output file not specified. Cannot proceed!'

        self.path = cfg['path']
        self.infile = cfg['infile']
        self.output = cfg['output']
        self.cropped_path = cfg.get('face_images', '')
        self.input_mode = cfg.get('input_mode', 'rekognition')

        self.min_face_dim = int(cfg.get('min_face_dim', '50'))
        self.pattern = cfg.get('pattern', '^.*:8000')
        self.orig_margin = float(cfg.get('margin', '0.01'))
        self.height = int(cfg.get('height', '0'))
        self.width = int(cfg.get('width', '0'))
        self.mode = cfg.get('mode', 'normal')

        if self.mode != 'normal' and self.mode != 'hadoop':
            sys.stderr.write('Invalid mode ' + mode +
                ' specified. Exiting!\n')
            sys.exit(1)

        # Setup logging.
        self.logfile = cfg.get('logfile', 'rekognition.log')
        self.logfile = self.path + '/' + self.logfile

        logging.basicConfig(filename=self.logfile,
                format='%(asctime)s %(levelname)s > %(message)s',
                level=logging.DEBUG)

        self.logger = logging.getLogger("Rekognition");

    def run(self):
        '''
        Runs the face cropping/aligning/rotation process on the input
        images.
        '''
        # Open the output file where face and other landmark coordinates
        # will be written into.
        self.fout = open(self.output, 'w')

        # In case of hadoop friendly output, open the output file
        # (cropped_path) for writing.
        self.hadoop_out = None
        if self.mode == 'hadoop' and self.cropped_path != '':
            self.hadoop_out = open(self.cropped_path, 'w')

        # Read the image information from the input file.
        with open(self.infile) as fin:
            for line in fin:
                line = line.rstrip()
                if self.input_mode == 'rekognition':
                    # Get the image information
                    (cfg, img_path, src_path) = \
                        self.__process_rekognition_api(line)
                    # Now crop the image.
                    self.__crop_image(cfg, src_path, img_path)
                else:
                    # Placeholder for implementing other type of
                    # input formats.
                    continue

        self.fout.close()

        if self.hadoop_out is not None:
            self.hadoop_out.close()

    def __process_rekognition_api(self, line):
        '''
        Processes a single Rekognition API output.
        '''
        # Load the API json into a dict.
        cfg = json.loads(line)

        # Get the image path from the dict. Assumption, the image URL
        # will end with the actual file name on local disk.
        # Also note that the image file name should be of the form
        # <tag>_<suffix>.<extension>
        img_path = re.sub(self.pattern, '', cfg['url'])
        img_path = re.sub('^[/]*', '', img_path)
        src_path = self.path + '/' + img_path
        return (cfg, img_path, src_path)

    def __save_cropped_image(self, img, img_path, index=0):
        '''
        Method to save cropped image to disk. Used in case of 'normal'
        output mode.

        Parameters:
            img - PIL image (face crop of the original image)
            img_path - basename of the image file on disk
            index - an optional suffix. Used in case multiple faces are
                detected in the same original image. Each cropped face
                can then be stored with different suffix with the original
                image file name.
        '''
        # Get the name of the image till the last '.'. We call it the
        # 'prefix'. The 'suffix' is the extension (i.e. the part after the
        # last '.').
        parts = img_path.split('.')
        prefix = '.'.join(parts[0:-1])
        suffix = parts[-1]
        # Destination path is <prefix>_<index>.<suffix>
        dst_path = self.cropped_path + '/' + prefix + '_' + str(index) + \
            '.' + suffix
        if self.cropped_path == '': return dst_path

        dirname = os.path.dirname(dst_path)
        try:
            os.makedirs(dirname)
        except:
            self.logger.debug('Dir ' + dirname + ' already exists.')
        try:
            img.save(dst_path, quality=100)
            self.logger.debug('Written image ' + dst_path)
        except:
            self.logger.info('Failed to write image ' + dst_path)
        return dst_path
        
    def __save_hadoop_image(self, img, label, img_path, index=0):
        '''
        Method to save a cropped image in the hadoop friendly format.
        <label> \t <file name> \t <image data in base64 encoded format>

        The encoded string should be decoded as follows:
        b = base64.decodestring(encoded_string)
        data = np.frombuffer(b, dtype=np.uint8)

        This returns the image data as a single array in row major
        format. This can be made into a 2-dimensional array using
        np.reshape()

        Parameters:
            img - PIL image (face crop of the original image)
            label - tag for the image
            img_path - basename of the image file on disk
            index - an optional suffix. Used in case multiple faces are
                detected in the same original image. Each cropped face
                can then be stored with different suffix with the original
                image file name.
        '''

        if self.hadoop_out is None: return

        # Get the name of the image till the last '.'. We call it the
        # 'prefix'. The 'suffix' is the extension (i.e. the part after the
        # last '.').
        parts = img_path.split('.')
        prefix = '.'.join(parts[0:-1])
        suffix = parts[-1]

        # Destination path is <prefix>_<index>.<suffix>
        img_path = prefix + '_' + str(index) + '.' + suffix
        self.hadoop_out.write(label + '\t' + img_path + '\t')
        a = np.asarray(img, dtype=np.uint8).flatten()
        s = base64.b64encode(a)
        self.hadoop_out.write(s + '\n')

    def __crop_image(self, cfg, src_path, img_path):
        '''
        Given the original image and the Rekognition API output for that
        image, writes the landmark coordinates for each face. The
        landmark coordinates are with respect to the cropped image (not
        the original one).
        '''
        self.logger.debug('Reading image ' + img_path)
        img = Image.open(src_path)
        img = img.convert('L')
        (cols,rows) = img.size

        index = 0
        # Iterate over each face in the API output.
        for face in cfg['face_detection']:
            angle = self.__rotation_angle(face)
            if angle > 360: continue

            # Get the landmark coordinates (w.r.t cropped image)
            landmarks = self.__find_landmarks(face, angle, rows, cols)
            if landmarks is None:
                continue

            if angle != 0:
                # Rotate image to align the eyes to horizontal.
                self.logger.debug('Rotating image %s by angle %f' % (img_path, angle))
                img = img.rotate(angle)

            # Now crop the image. Note that the 'boundingbox' is the only
            # set of coordinates which are with respect to the original
            # image.
            (x,y,w,h) = landmarks['boundingbox']
            img = img.crop((int(x),int(y),int(x+w),int(y+h)))

            if self.height > 0 and self.width > 0:
                # Resize the image if necessary.
                img = img.resize((self.width,self.height), Image.ANTIALIAS)

            if self.mode == 'normal':
                # Write the image into a separate file on disk.
                dst_path = self.__save_cropped_image(img, img_path, index)
                self.fout.write(dst_path)

            else:
                # Write it in hadoop friendly format.
                label = re.sub('_.*$', '', img_path)
                self.__save_hadoop_image(img, label, img_path, index)
                self.fout.write(img_path)

            # Increment the index so that the next face in the same image
            # is written with a different suffix in the file name.
            index += 1

            # Now write the landmark coordinates in the output file.
            for (k,v) in landmarks.items():
                self.fout.write('\t' + k + ':' + str(v))
            self.fout.write('\n')

    def __rotation_angle(self, face):
        '''
        Finds the roll angle of the face by detecting the angle between
        the line joining left and right eye centers and the horizontal.

        Note that the Rekognition API outputs the roll pitch and yaw angles
        and we could have used the roll angle from there as well.
        TODO: Revisit this.

        The yaw and pitch are not used as there is no way to compensate
        for those.
        '''

        angle = 0
        if 'eye_left' in face and 'eye_right' in face:
            x_l = float(face['eye_left']['x'])
            y_l = float(face['eye_left']['y'])
            x_r = float(face['eye_right']['x'])
            y_r = float(face['eye_right']['y'])

            angle = (y_r - y_l)/(x_r - x_l)
            angle = math.atan(angle)*180.0/math.pi
        else:
            # Use an invalid angle value. This indicates that two eyes
            # are not visible in the image.
            angle = 370

        return angle

    def __rotate(self, x, y, H, W, sint, cost):
        '''
        Find the coordinates of a point (x,y) in an image of dimension (H,W)
        when the image is rotated by an angle theta (sint = sin theta, cost
        = cos theta).
        '''

        # These could be written as a single step as well, but separating
        # each operation for clarity.

        # Tranlate origin to center and reverse direction of y
        x = x - W/2.0
        y = H/2.0 - y
        # Rotate at angle t (sint and cost given) (affine transform)
        xr = x*cost - y*sint
        yr = y*cost + x*sint
        # Translate origin back to top left and reverse direction of y
        xr = W/2.0 + xr
        yr = H/2.0 - yr
        return (xr,yr)

    def __padding_left(self, x, y):
        x -= self.margin
        if x < 0: x = 0
        y -= self.margin
        if y < 0: y = 0
        return (x,y)

    def __padding_right(self, x, y, H, W):
        x += self.margin
        if x > W: x = W
        y += self.margin
        if y > H: y = H
        return (x,y)

    def __find_landmarks(self, face, angle, H, W):
        '''
        Method to translate the coordinates of the face landmark to the
        face cropped and rotated image.
        '''

        sint = math.sin(angle/180.0*math.pi)
        cost = math.cos(angle/180.0*math.pi)

        landmarks = {}
        x = float(face['boundingbox']['tl']['x'])
        y = float(face['boundingbox']['tl']['y'])
        w = float(face['boundingbox']['size']['width'])
        h = float(face['boundingbox']['size']['height'])
        if w < self.min_face_dim and h < self.min_face_dim:
            self.logger.debug('Face (%f,%f,%f,%f) too small. Rejecting.' % (x,y,w,h))
            return None

        m = max(w,h)
        self.margin = self.orig_margin*m
        if self.margin < 1: self.margin = 1

        # Find the coordinates of the top left corner in the rotated image.
        (xr,yr) = self.__rotate(x,y,H,W,sint,cost)

        # Align the crop so that the midpoint between the two eyes lies
        # along the center vertical divider of the cropped image.
        if 'eye_left' in face and 'eye_right' in face:
            x1 = float(face['eye_left']['x'])
            y1 = float(face['eye_left']['y'])
            x2 = float(face['eye_right']['x'])
            y2 = float(face['eye_right']['y'])
            (xr1,yr1) = self.__rotate(x1,y1,H,W,sint,cost)
            (xr2,yr2) = self.__rotate(x2,y2,H,W,sint,cost)
            # xm, ym is the center of the two eyes in the rotated image.
            xm = (xr1 + xr2)/2.0
            ym = (yr1 + yr2)/2.0

            # mid is the horizontal distance between the center and the
            # left edge of the image
            mid = xm - xr

            # if mid > width/2, shift the crop to the right.
            # else shift it to the left.
            if mid > w/2.0:
                xr += mid-w/2.0
            else:
                xr -= w/2.0-mid

            # Now make the width = height/ratio
            ratio = 1.25
            if self.height > 0 and self.width > 0:
                ratio = self.height*1./self.width
            w = h/ratio
            xr = xr + (h-w)/2.0

            # Move the box vertically so that the eyes are at w/2 (=h*0.4)
            ym = (yr1 + yr2)/2.0 - yr
            d = ym - w/2.
            yr = yr + d

        landmarks['boundingbox'] = (xr,yr,w,h)

        keys = []
        values = []
        # Compute the coordinates of the four important landmarks in the
        # cropped image.
        # Left eye.
        if 'e_ll' in face and 'e_lu' in face and 'e_lr' in face and 'e_ld' in face:
            x1 = float(face['e_ll']['x'])
            y1 = float(face['e_lu']['y'])
            x2 = float(face['e_lr']['x'])
            y2 = float(face['e_ld']['y'])
            keys.append('left_eye')
            values.append((x1,y1,x2,y2))

        # Right eye
        if 'e_rl' in face and 'e_ru' in face and 'e_rr' in face and 'e_rd' in face:
            x1 = float(face['e_rl']['x'])
            y1 = float(face['e_ru']['y'])
            x2 = float(face['e_rr']['x'])
            y2 = float(face['e_rd']['y'])
            keys.append('right_eye')
            values.append((x1,y1,x2,y2))

        # Mouth
        if 'mouth_l' in face and 'm_u' in face and 'mouth_r' in face and 'm_d' in face:
            x1 = float(face['mouth_l']['x'])
            y1 = float(face['m_u']['y'])
            x2 = float(face['mouth_r']['x'])
            y2 = float(face['m_d']['y'])
            keys.append('mouth')
            values.append((x1,y1,x2,y2))

        # Nose
        if 'n_l' in face and 'e_lr' in face and 'n_r' in face:
            x1 = float(face['e_lr']['x'])
            y1 = float(face['e_lr']['y'])
            x2 = float(face['n_r']['x'])
            y2 = float(face['n_r']['y'])
            keys.append('nose')
            values.append((x1,y1,x2,y2))

        # For each landmark, first rotate the coordinates and then
        # translate to the coordinate system of the cropped image.
        for i in range(len(keys)):
            key = keys[i]
            (x1,y1,x2,y2) = values[i]
            (xr1,yr1) = self.__rotate(x1,y1,H,W,sint,cost)
            (xr2,yr2) = self.__rotate(x2,y2,H,W,sint,cost)
            (xr1,yr1) = self.__padding_left(xr1,yr1)
            (xr2,yr2) = self.__padding_right(xr2,yr2,H,W)

            landmarks[key] = (xr1-xr,yr1-yr,xr2-xr1,yr2-yr1)

        return landmarks

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print 'Usage: ' + sys.argv[0] + ' <cfg file>'
        sys.exit(1)
    print 'Arg = ' + sys.argv[1] 
    cfg = json.loads(open(sys.argv[1]).read())

    r = Rekognition(cfg)
    r.run()

