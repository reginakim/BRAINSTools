#! /usr/bin/env python
"""
WMHCreateFeatureVector.py
=========
This program is used to generate a feature vector file, using "BRAINSCut --createVectors" option.

Usage:
  WMHCreateFeatureVector.py [--rewrite-datasinks]  --inputFLAIR inputFLAIRVolume --inputReference inputReferenceVolume --inputMoreFeatures inputMoreFeatureVolumes --inputTargetLabel inputTargetLabelVolume
  WMHCreateFeatureVector.py -v | --version
  WMHCreateFeatureVector.py -h | --help

Arguments:

Options:
  -h, --help                                    Show this help and exit
  -v, --version                                 Print the version and exit
  --rewrite-datasinks                           Turn on the Nipype option to overwrite all files in the 'results' directory
  --inputFLAIR inputFLAIRVolume                FLAIR volume for white matter hyperintensity (WMH) feature extraction
  --inputReference inputReferenceVolume        A reference volume for the FLAIR. Usually T1-weighted image
  --inputMoreFeatures inputMoreFeatureVolumes  More volumes for feature extraction
  --inputTargetLabel inputTargetLabelVolume    A target white matter hyperintensity label volume


Examples:
  $ WMHCreateFeatureVector.py
  $ WMHCreateFeatureVector.py
  $ WMHCreateFeatureVector.py

"""
from __future__ import absolute_import
from __future__ import print_function

def _wmh_createFeatureVector(**_args):
    print("_wmh_createFeatureVector starts...")
    print(_args)
    return 0

# #####################################
# Set up the environment, process command line options, and start processing
#
if __name__ == '__main__':
    import sys
    import os

    from docopt import docopt

    argv = docopt(__doc__, version='1.1')
    print(argv)
    print('=' * 100)



    exit = _wmh_flairThreshold(
            outputPrefix,
            CACHEDIR,
            inputFLVolume,
            inputT1Volume,
            LabelMapImage,
            processingType="hyper",
            inputIntensityReference = "/Users/eunyoungkim/src/NamicBuild_20171031/bin/Atlas/Atlas_20131115/template_t2_clipped.nii.gz",
            thresholdList = [0.3, 0.4, 0.45, 0.5],
            )
    sys.exit(exit)
