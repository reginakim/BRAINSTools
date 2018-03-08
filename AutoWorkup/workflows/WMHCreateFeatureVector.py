#! /usr/bin/env python
"""
WMHCreateFeatureVector.py
=========
This program is used to generate a feature vector file, using "BRAINSCut --createVectors" option.

Usage:
  WMHCreateFeatureVector.py [--rewrite-datasinks]  --inputFLAIR inputFLAIRVolume --inputReference inputReferenceVolume --inputT1Volume inputT1Volume 
  WMHCreateFeatureVector.py -v | --version
  WMHCreateFeatureVector.py -h | --help

Arguments:

Options:
  -h, --help                                   Show this help and exit
  -v, --version                                Print the version and exit
  --rewrite-datasinks                          Turn on the Nipype option to overwrite all files in the 'results' directory
  --inputFLAIR inputFLAIRVolume                FLAIR volume for white matter hyperintensity (WMH) feature extraction
  --inputT1Volume inputT1Volume                T1 volume for white matter hyperintensity (WMH) feature extraction


Examples:
  $ WMHCreateFeatureVector.py
  $ WMHCreateFeatureVector.py
  $ WMHCreateFeatureVector.py

"""
from __future__ import absolute_import
from __future__ import print_function

import SimpleITK as sitk
import nipype
from nipype.interfaces import ants
from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory
from nipype.interfaces.base import traits, isdefined, BaseInterface
from nipype.interfaces.utility import Merge, Split, Function, Rename, IdentityInterface
import nipype.interfaces.io as nio   # Data i/oS
import nipype.pipeline.engine as pe  # pypeline engine
from nipype.interfaces.freesurfer import ReconAll
from nipype.interfaces.semtools import GradientAnisotropicDiffusionImageFilter, BRAINSFit

from utilities.distributed import modify_qsub_args

def getMeanVolume(inputFilename, outputFilename, radius=3):
    inputVolume = sitk.ReadImage( inputFilename )
    outputVolume = sitk.Mean( inputVolume, radius )

    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getMaximumVolume(inputFilename, outputFilename, radius=3):
    inputVolume = sitk.ReadImage( inputFilename )
    outputVolume = sitk.Maximum( inputVolume, radius )
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getMinimumVolume(inputFilename, outputFilename, radius=3):
    inputVolume = sitk.ReadImage( inputFilename )
    outputVolume = sitk.Minimum( inputVolume, radius )
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getCannyEdge(inputFilename, outputFilename):
    inputVolume = sitk.ReadImage( inputFilename )
    outputVolume = sitk.CannyEdgeDetection( inputVolume )
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getGradientMagnitude(inputFilename, outputFilename):
    inputVolume = sitk.ReadImage( inputFilename )
    outputVolume = sitk.GradientMagnitudeRecursiveGaussian( inputVolume )
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getSobelEdge(inputFilename, outputFilename):
    inputVolume = sitk.ReadImage( inputFilename )
    outputVolume = sitk.CannyEdgeDetection( inputVolume )
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getSquaredDifference(inputFilename1, inputFilename2, outputFilename):
    inputVolume1 = sitk.ReadImage( inputFilename1 )
    inputVolume2 = sitk.ReadImage( inputFilename2 )
    outputVolume = sitk.SquaredDifference( inputVolume1, 
                                           inputVolume2)
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def generate1stFeatureVolumeWF( inputVolumeDict ):
    print("generateFeatureVolumeWF starts...")

    import SimpleITK as sitk
    import nipype
    from nipype.interfaces import ants
    from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory
    from nipype.interfaces.base import traits, isdefined, BaseInterface
    from nipype.interfaces.utility import Merge, Split, Function, Rename, IdentityInterface
    import nipype.interfaces.io as nio   # Data i/oS
    import nipype.pipeline.engine as pe  # pypeline engine
    from nipype.interfaces.freesurfer import ReconAll
    from nipype.interfaces.semtools import GradientAnisotropicDiffusionImageFilter, BRAINSFit

    from utilities.distributed import modify_qsub_args

    # #####################################
    # ######## Workflow ###################
    #
    WFname = 'genFeatureWF'
    genFeatureWF = pe.Workflow(name=WFname)
    genFeatureWF.config['execution'] = {'remove_unnecessary_outputs': 'False',
                                        'hash_method': 'timestamp'}
    """
    input spec TODO: may not be needed
    """
    #inputsSpec = pe.Node(interface=IdentityInterface(fields=['inputFLVolume',
    #                                                         'inputT1Volume']),
    #                     name='inputspec')

    # Denoised input for Feature Creation
    denosingTimeStep = 0.0625
    denosingConductance = 0.4
    denosingIteration = 5

    DenoiseInput = pe.Node(interface=GradientAnisotropicDiffusionImageFilter(), name="DenoiseInput")
    DenoiseInput.inputs.timeStep = denosingTimeStep
    DenoiseInput.inputs.conductance = denosingConductance
    DenoiseInput.inputs.numberOfIterations = denosingIteration
    DenoiseInput.inputs.outputVolume = "DenoiseInput.nii.gz"

    DenoiseInput.interables( 'inputVolume', inputVolumeDict.values() )

    """
    Mean
    """
    meanFeatureNode = pe.Node(interface=Function(['inputVolume', 'radius', 'outputFilename'],
                                                 ['outputVolume'],
                                                 function=getMeanVolume),
                              name="meanFeatureNode")
    meanFeatureNode.inputs.outputFilename = 'outputMeanVolume.nii.gz'
    meanFeatureNode.inputs.radius= 3 
    cutWF.connect( DenoiseInput, 'outputVolume',
                   meanFeatureNode, 'inputVolume')
    """
    Max 
    """
    maximumFeatureNode = pe.Node(interface=Function(['inputVolume', 'radius', 'outputFilename'],
                                                    ['outputVolume'],
                                                 function=getMaximumVolume),
                              name="maximumFeatureNode")
    maximumFeatureNode.inputs.outputFilename = 'outputMaximumVolume.nii.gz'
    maximumFeatureNode.inputs.radius= 3 
    cutWF.connect( DenoiseInput, 'outputVolume',
                   maximumFeatureNode, 'inputVolume')
    """
    Min
    """
    minimumFeatureNode = pe.Node(interface=Function(['inputVolume', 'radius', 'outputFilename'],
                                                    ['outputVolume'],
                                                    function=getCannyEdge()),
                              name="minimumFeatureNode")
    minimumFeatureNode.inputs.outputFilename = 'outputMinimumVolume.nii.gz'
    minimumFeatureNode.inputs.radius= 3 
    cutWF.connect( DenoiseInput, 'outputVolume',
                   minimumFeatureNode, 'inputVolume')

    """
    Grad. Mag.
    """
    GMFeatureNode = pe.Node(interface=Function(['inputVolume', 'radius', 'outputFilename'],
                                                ['outputVolume'],
                                                function=getGradientMagnitude()),
                              name="GMFeatureNode")
    GMFeatureNode.inputs.outputFilename = 'outputGradMagVolume.nii.gz'
    cutWF.connect( DenoiseInput, 'outputVolume',
                   GMFeatureNode, 'inputVolume')
    
    """
    Edge Potential
    """
    EdgePotentialFeatureNode = pe.Node(interface=Function(['inputVolume', 'outputFilename'],
                                                ['outputMinimumFilename'],
                                                function=getGradientMagnitude()),
                              name="EdgePotentialFeatureNode")
    EdgePotentialFeatureNode.inputs.outputFilename = 'outputEdgePotentialVolume.nii.gz'
    cutWF.connect( DenoiseInput, 'outputVolume',
                   EdgePotentialFeatureNode, 'inputVolume')

    """
    Canny
    """
    cannyNode = pe.Node(interface=Function(['inputVolume', 'outputFilename'],
                                           ['outputEdgeFilename'],
                                           function=getMinimumVolume),
                              name="cannyNode")
    cannyNode.inputs.outputFilename = 'outputCannyEdgeVolume.nii.gz'
    cutWF.connect( DenoiseInput, 'outputVolume',
                   cannyNode, 'inputVolume')

    """
    Sobel 
    """
    sobelNode = pe.Node(interface=Function(['inputVolume', 'outputFilename'],
                                           ['outputEdgeFilename'],
                                           function=getMinimumVolume),
                              name="sobelNode")
    sobelNode.inputs.outputFilename = 'outputSobelEdgeVolume.nii.gz'
    cutWF.connect( DenoiseInput, 'outputVolume',
                   sobelNode, 'inputVolume')
    return genFeatureWF

def generate2ndFeatures( featureVolumeDict )
    """
    SG
    """
    # Sum the gradient images for BRAINSCut
    WFname = "gen2ndFeatureWF"
    gen2ndFeatureWF = pe.Workflow(name=WFname)
    gen2ndFeatureWF.config['execution'] = {'remove_unnecessary_outputs':'False',
                                           'hash_method':'timestamp'}

    SGI = pe.Node(interface=GenerateSummedGradientImage(), name="SGI")
    SGI.inputs.outputFileName = "SummedGradImage.nii.gz"
    SGI.inputs.inputVolume1 = featureVolumeDict['T1']
    SGI.inputs.inputVolume2 = featureVolumeDict['FL']


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

