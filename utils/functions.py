#!/usr/bin/env python

# this script contains all functions called from demo web
# application.

#from vr.FeatureVector import FeatureVector

#import os
import wave
import struct
import cPickle
#import numpy
#import gzip

def save(object, filename, bin = 1):
	"""Saves a compressed object to disk
        """
	#file = gzip.GzipFile(filename, 'wb')
	file = open(filename, 'wb')
	file.write(cPickle.dumps(object, bin))
	file.close()


def load(filename):
	"""Loads a compressed object from disk
        """
	#file = gzip.GzipFile(filename, 'rb')
	file = open(filename, 'rb')
	buffer = ""
	while 1:
		data = file.read()
		if data == "":
			break
		buffer += data
	object = cPickle.loads(buffer)
	file.close()
	return object


def extract_audio_samples(filePath):
    #print 'Reading wav file...'
    wav = wave.open (filePath, "r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams ()
    '''
    print 'samples:',nframes
    print 'audio length:',1.*nframes/framerate,'s'
    print 'frame rate:',framerate
    '''
    frames = wav.readframes (nframes * nchannels)
    out = struct.unpack_from ("%dh" % nframes * nchannels, frames)
    wav.close()
    return out

def write_wav(samples,filepath):
    wav = wave.open (filepath, "w")
    wav.setnframes(len(samples))
    wav.setframerate(8000)
    wav.setnchannels(1)
    wav.setsampwidth(2)
    
    for s in samples:
        p = struct.pack('1h',int(s))
        wav.writeframes(p)

    wav.close()

