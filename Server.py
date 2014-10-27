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
from tornado.ioloop import IOLoop
from tornado.web import RequestHandler, Application, url
from StringIO import StringIO
import urllib
import Image
from Fisherfaces_Hadoop import *
import json
import base64
import numpy as np
import sys
import hashlib

class Searcher(RequestHandler):
    '''
    This is a toy example of a search mechanism for matching images in
    a corpus.
    We use the Tornado (http://tornado.readthedocs.org/) framework to
    create a webserver. This server, on startup, loads an already created
    Fisherfaces model (created by Fisherfaces_Hadoop class). Given a query
    image, it returns the top N closest images from the corpus, N being
    configurable. The query image can be passed as either:
    - A hosted URL via a GET request
    - Or as the resized base64 encoded image as part of the body in a POST
      request.

    Theoretically, the Fisherfaces face images corpus can be split into
    several parts (this is already being done by the Fisherfaces_Hadoop
    class) and each part loaded into a separate server. In that case, one
    would require another HTTP server in front of these server which will
    pass on the query image to each split, and then merge the results from
    each split by distance. Please feel free to contribute.

    The choice of Tornado as the webserver is entirely arbitrary. We
    do not use any Tornado-specific features.  The plain HTTPServer of
    Python (and its subclasses) would have also done fine. In fact, it
    should be fairly simple to modify this code to use the built-in
    HTTP server of Python.

    Finally, remember that this is just a toy example ;-)

    '''

    def post(self):
        global cache
        global fisherfaces

        self.set_header("Content-Type", "text/plain")
        i = self.get_body_argument('image', '')
        if i == '':
            self.set_status(400)
            self.finish('<html><body>Mandatory argument "image" missing in request.</body></html>')
        num_match = int(self.get_body_argument("num_matches", "15"))

        # We implement a very naive caching. Cache is never updated or
        # cleared. Feel free to contribute!
        m = hashlib.md5()
        m.update(i)
        m.update(str(num_match))
        md = m.hexdigest()
        
        if md in cache:
            self.write(cache[md])
        else:
            try:
                b = base64.b64decode(i)
            except Exception as inst:
                self.set_status(400)
                self.finish('<html><body>Could not decode image. Error message: ' + str(inst) + '</html></body>')
                return
            img = np.frombuffer(b, dtype=np.uint8)
            predictions = fisherfaces.match(img, num_match)
            j = json.dumps(predictions)
            cache[md] = j
            self.write(j)

    def get(self):
        global cache
        global fisherfaces
        global cfg

        self.set_header("Content-Type", "text/plain")
        url = self.get_argument('url', '')
        if url == '':
            self.set_status(400)
            self.finish('<html><body>Mandatory argument "url" missing in request.</body></html>')
        img = None
        data = None
        try:
            data = urllib.urlopen(url)
        except Exception as inst:
            self.status(400)
            self.finish('<html><body>Could not open url ' + url + '</body></html>')
        try:
            img = Image.open(StringIO(data.read()))
            h = int(cfg.get('height', '75'))
            w = int(cfg.get('width', '60'))
            img = img.resize((w,h), Image.ANTIALIAS)
        except Exception as inst:
            self.status(400)
            self.finish('<html><body>Error reading data from URL ' + url + '</body></html>')
        num_match = int(self.get_argument("num_matches", default="15"))
        m = hashlib.md5()
        m.update(url)
        m.update(str(num_match))
        md = m.hexdigest()
        
        if md in cache:
            self.write(cache[md])
        else:
            nimg = np.asarray(img, dtype=np.uint8).flatten()
            predictions = fisherfaces.match(nimg, num_match)
            j = json.dumps(predictions)
            cache[md] = j
            self.write(j)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print 'Usage: ' + sys.argv[0] + ' <config file> [port]'
        sys.exit(1)

    port = 8888
    if len(sys.argv) >= 3:
        port = int(sys.argv[2])

    cfg = json.loads(open(sys.argv[1]).read())
    fisherfaces = Fisherfaces()

    fisherfaces.load_model(cfg)
    cache = {}
    sys.stderr.write('Initialized searcher object.\n')

    app = Application([url(r"/", Searcher),])

    app.listen(port)
    IOLoop.current().start()
