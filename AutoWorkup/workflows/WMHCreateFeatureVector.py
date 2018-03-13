#! /usr/bin/env python
"""
WMHCreateFeatureVector.py
=========
This program is used to generate a feature vector file, using "BRAINSCut --createVectors" option.

Usage:
  WMHCreateFeatureVector.py   --inputFLAIRVolume inputFLAIRVolume --inputT1Volume inputT1Volume --outputDirectory outputDirectory [--python_aux_paths python_aux_paths] [--binary_paths binary_paths] [--inputOtherVolumes inputOtherVolumes]... 
  WMHCreateFeatureVector.py -v | --version
  WMHCreateFeatureVector.py -h | --help

Options:
  -h, --help                                   Show this help and exit
  -v, --version                                Print the version and exit
  --inputFLAIRVolume inputFLAIRVolume          FLAIR volume for white matter hyperintensity (WMH) feature extraction
  --inputT1Volume inputT1Volume                T1 volume for white matter hyperintensity (WMH) feature extraction
  --inputOtherVolumes inputOtherVolumes        Any other volumes for white matter hyperintensity (WMH) feature extraction
  --outputDirectory outputDirectory            Output directory 
  --python_aux_paths python_aux_paths          Python Aux path
  --binary_paths binary_paths                  Binary path to execute


"""
from __future__ import absolute_import
from __future__ import print_function

import SimpleITK as sitk
import nipype
from nipype.interfaces import ants
from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory
from nipype.interfaces.base import traits, isdefined, BaseInterface
from nipype.interfaces.utility import Merge, Split, Function, Rename, IdentityInterface
#import nipype.interfaces.io as nio   # Data i/oS
import nipype.pipeline.engine as pe  # pypeline engine
#from nipype.interfaces.freesurfer import ReconAll
#from nipype.interfaces.semtools import GradientAnisotropicDiffusionImageFilter, BRAINSFit

def getMeanVolume(inputFilename, outputFilename, radius=3):
    import SimpleITK as sitk
    import os
    inputVolume = sitk.ReadImage( inputFilename, sitk.sitkFloat32 )
    outputVolume = sitk.Mean( inputVolume, [radius,radius,radius] )

    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getMaximumVolume(inputFilename, outputFilename, radius=3):
    import SimpleITK as sitk
    import os
    inputVolume = sitk.ReadImage( inputFilename, sitk.sitkFloat32 )
    outputVolume = sitk.Maximum( inputVolume, [radius, radius, radius])
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getMinimumVolume(inputFilename, outputFilename, radius=3):
    import SimpleITK as sitk
    import os
    inputVolume = sitk.ReadImage( inputFilename, sitk.sitkFloat32 )
    outputVolume = sitk.Minimum( inputVolume, [radius, radius, radius] )
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getCannyEdge(inputFilename, outputFilename):
    import SimpleITK as sitk
    import os
    inputVolume = sitk.ReadImage( inputFilename, sitk.sitkFloat32 )
    outputVolume = sitk.CannyEdgeDetection( inputVolume )
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getGradientMagnitude(inputFilename, outputFilename):
    import SimpleITK as sitk
    import os
    inputVolume = sitk.ReadImage( inputFilename, sitk.sitkFloat32 )
    outputVolume = sitk.GradientMagnitudeRecursiveGaussian( inputVolume )
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getEdgePotential(inputFilename, outputFilename):
    import SimpleITK as sitk
    import os
    inputVolume = sitk.ReadImage( inputFilename, sitk.sitkUInt16 )
    outputVolume = sitk.EdgePotential( inputVolume )
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getSobelEdge(inputFilename, outputFilename):
    import SimpleITK as sitk
    import os
    inputVolume = sitk.ReadImage( inputFilename, sitk.sitkFloat32 )
    outputVolume = sitk.SobelEdgeDetection( inputVolume )
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getZeroCrossingEdge(inputFilename, outputFilename):
    import SimpleITK as sitk
    import os
    inputVolume = sitk.ReadImage( inputFilename, sitk.sitkFloat32 )
    outputVolume = sitk.ZeroCrossingBasedEdgeDetection( inputVolume )
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def getSquaredDifference(inputFilename1, inputFilename2, outputFilename):
    import SimpleITK as sitk
    import os
    inputVolume1 = sitk.ReadImage( inputFilename1, sitk.sitkFloat32 )
    inputVolume2 = sitk.ReadImage( inputFilename2, sitk.sitkFloat32 )
    outputVolume = sitk.SquaredDifference( inputVolume1, 
                                           inputVolume2)
    
    returnFilename = os.path.abspath( outputFilename ) 
    sitk.WriteImage( outputVolume, returnFilename)
    return returnFilename

def generate1stFeatures( inputVolumeList, outputDirectory ):
    print("generate1stFeatures starts...")

    import SimpleITK as sitk
    import nipype
    from nipype.interfaces import ants
    from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory
    from nipype.interfaces.base import traits, isdefined, BaseInterface
    from nipype.interfaces.utility import Merge, Split, Function, Rename, IdentityInterface
    import nipype.interfaces.io as nio   # Data i/oS
    import nipype.pipeline.engine as pe  # pypeline engine
    from nipype.interfaces.freesurfer import ReconAll
    from nipype.interfaces.semtools import GradientAnisotropicDiffusionImageFilter, BRAINSFit, GenerateSummedGradientImage

    from utilities.distributed import modify_qsub_args

    # #####################################
    # ######## Workflow ###################
    #
    WFname = 'genFeatureWF'
    genFeatureWF = pe.Workflow(name=WFname)
    genFeatureWF.config['execution'] = {'remove_unnecessary_outputs': 'False',
                                        'hash_method': 'timestamp',
                                        'overwrite':'True'}
    """
    DataSink 
    """
    DS = pe.Node( nio.DataSink(), name='sinker')
    DS.inputs.base_directory=  outputDirectory 
    DS.inputs.regexp_substitutions = [ (r'genFeatureWF/',r''),
                                       (r'_inputVolume_.*\.\.',r''),
                                       (r'.nii.gz/',r'_'),
                                       (r'Node/',r'_')]
    
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

    print( inputVolumeList )
    DenoiseInput.iterables = ( 'inputVolume', inputVolumeList )

    """
    Mean
    """
    meanFeatureNode = pe.Node(interface=Function(['inputFilename', 'radius', 'outputFilename'],
                                                 ['outputVolume'],
                                                 function=getMeanVolume),
                              name="meanFeatureNode")
    meanFeatureNode.inputs.outputFilename = 'featureMean.nii.gz'
    meanFeatureNode.inputs.radius= 3 
    genFeatureWF.connect( DenoiseInput, 'outputVolume',
                   meanFeatureNode, 'inputFilename')

    genFeatureWF.connect( meanFeatureNode, 'outputVolume',
                          DS, '@meanFeature')
    """
    Grad. Mag.
    """
    GMFeatureNode = pe.Node(interface=Function(['inputFilename', 'outputFilename'],
                                                ['outputVolume'],
                                                function=getGradientMagnitude),
                              name="GMFeatureNode")
    GMFeatureNode.inputs.outputFilename = 'featureGradMag.nii.gz'
    genFeatureWF.connect( DenoiseInput, 'outputVolume',
                   GMFeatureNode, 'inputFilename')
    genFeatureWF.connect( GMFeatureNode, 'outputVolume',
                          DS, '@GMFeature')
    
    """
    Edge Potential
    """
    """
    EdgePotentialFeatureNode = pe.Node(interface=Function(['inputFilename', 'outputFilename'],
                                                          ['outputVolume'],
                                                 function=getEdgePotential),
                              name="EdgePotentialFeatureNode")
    EdgePotentialFeatureNode.inputs.outputFilename = 'outputEdgePotentialVolume.nii.gz'
    genFeatureWF.connect( GMFeatureNode, 'outputVolume',
                          EdgePotentialFeatureNode, 'inputFilename')
    """

    """
    Canny
    """
    cannyNode = pe.Node(interface=Function(['inputFilename', 'outputFilename'],
                                           ['outputEdgeFilename'],
                                           function=getCannyEdge),
                              name="cannyNode")
    cannyNode.inputs.outputFilename = 'featureCannyEdge.nii.gz'
    genFeatureWF.connect( DenoiseInput, 'outputVolume',
                   cannyNode, 'inputFilename')
    genFeatureWF.connect( cannyNode, 'outputEdgeFilename',
                          DS, '@CannyFeature')

    """
    Sobel 
    """
    sobelNode = pe.Node(interface=Function(['inputFilename', 'outputFilename'],
                                           ['outputEdgeFilename'],
                                           function=getSobelEdge),
                              name="sobelNode")
    sobelNode.inputs.outputFilename = 'featureSobelEdge.nii.gz'
    genFeatureWF.connect( DenoiseInput, 'outputVolume',
                   sobelNode, 'inputFilename')
    genFeatureWF.connect( sobelNode, 'outputEdgeFilename',
                          DS, '@SobelFeature')

    """
    Zero Crossing
    """
    zeroCrossingNode = pe.Node(interface=Function(['inputFilename', 'outputFilename'],
                                              ['outputEdgeFilename'],
                                              function=getZeroCrossingEdge),
                               name="zeroCrossingNode")
    zeroCrossingNode.inputs.outputFilename = 'featureZeroCrossingEdge.nii.gz'
    genFeatureWF.connect( DenoiseInput, 'outputVolume',
                   zeroCrossingNode, 'inputFilename')
    genFeatureWF.connect( zeroCrossingNode, 'outputEdgeFilename',
                          DS, '@ZeroCrossingFeature')


    """
    Variance
    """
    varFeatureNode = pe.Node(interface=Function(['inputFilename1', 'inputFilename2', 'outputFilename'],
                                                ['outputFilename'],
                                                function=getSquaredDifference),
                             name="varFeatureNode")
    varFeatureNode.inputs.outputFilename = 'featureVariance.nii.gz'
    genFeatureWF.connect( DenoiseInput, 'outputVolume',
                          varFeatureNode, 'inputFilename1')
    genFeatureWF.connect( meanFeatureNode, 'outputVolume',
                          varFeatureNode, 'inputFilename2')
    genFeatureWF.connect( varFeatureNode, 'outputFilename',
                          DS, '@SGIFeature')
    return genFeatureWF

def generate2ndFeatures( inputVolume1, inputVolume2, outputDirectory):

    from nipype.interfaces.semtools import GradientAnisotropicDiffusionImageFilter, BRAINSFit, GenerateSummedGradientImage
    WFname = "gen2ndFeatureWF"
    gen2ndFeatureWF = pe.Workflow(name=WFname)
    gen2ndFeatureWF.config['execution'] = {'remove_unnecessary_outputs':'True',
                                           'hash_method':'timestamp'}

    """
    DataSink 
    """
    import nipype.interfaces.io as nio   # Data i/oS
    DS = pe.Node( nio.DataSink(), name='sinker')
    DS.inputs.base_directory=  outputDirectory 
    DS.inputs.regexp_substitutions = [ (r'gen2ndFeatureWF/',r''),
                                       (r'_inputVolume_.*\.\.',r''),
                                       (r'.nii.gz/',r'_'),
                                       (r'Node/',r'_')]
    """
    Denoised input for Feature Creation
    """
    denosingTimeStep = 0.0625
    denosingConductance = 0.4
    denosingIteration = 5

    DenoiseInput1 = pe.Node(interface=GradientAnisotropicDiffusionImageFilter(), name="DenoiseInput1")
    DenoiseInput1.inputs.timeStep = denosingTimeStep
    DenoiseInput1.inputs.conductance = denosingConductance
    DenoiseInput1.inputs.numberOfIterations = denosingIteration
    DenoiseInput1.inputs.outputVolume = "DenoiseInput1.nii.gz"
    DenoiseInput1.inputs.inputVolume = inputVolume1


    DenoiseInput2 = pe.Node(interface=GradientAnisotropicDiffusionImageFilter(), name="DenoiseInput2")
    DenoiseInput2.inputs.timeStep = denosingTimeStep
    DenoiseInput2.inputs.conductance = denosingConductance
    DenoiseInput2.inputs.numberOfIterations = denosingIteration
    DenoiseInput2.inputs.outputVolume = "DenoiseInput2.nii.gz"
    DenoiseInput2.inputs.inputVolume = inputVolume2

    """
    Summed Gradient
    """
    SGI = pe.Node(interface=GenerateSummedGradientImage(), name="SGI")
    SGI.inputs.outputFileName = "SummedGradImage.nii.gz"
    gen2ndFeatureWF.connect( DenoiseInput1, 'outputVolume',
                             SGI, 'inputVolume1')
    gen2ndFeatureWF.connect( DenoiseInput2, 'outputVolume',
                             SGI, 'inputVolume2')
    
    gen2ndFeatureWF.connect( SGI, 'outputFileName',
                             DS, '@SGFeature')
  

    return gen2ndFeatureWF



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

    PYTHON_AUX_PATHS= argv['--python_aux_paths']
    PYTHON_AUX_PATHS = PYTHON_AUX_PATHS.split(':')
    PYTHON_AUX_PATHS.extend(sys.path)
    sys.path = PYTHON_AUX_PATHS
    
    PROGRAM_PATHS = argv['--binary_paths'] 
    PROGRAM_PATHS = PROGRAM_PATHS.split(':')
    PROGRAM_PATHS.extend(os.environ['PATH'].split(':'))
    os.environ['PATH'] = ':'.join(PROGRAM_PATHS)

    inputFLAIRVolume = os.path.abspath( argv['--inputFLAIRVolume'])
    inputT1Volume = os.path.abspath( argv['--inputT1Volume'])
    outputDirectory = os.path.abspath( argv['--outputDirectory'])

    inputF1VolList = []
    inputF1VolList.append( inputFLAIRVolume )
    inputF1VolList.append( inputT1Volume )
    
    if argv['--inputOtherVolumes']:
        for fn in argv['--inputOtherVolumes']:
            inputF1VolList.append( os.path.abspath( fn ))

    print( "input Volumes:")
    print( inputF1VolList )

    """
    1st Features
    """
    local_feature1WF = generate1stFeatures( inputF1VolList,
                                            outputDirectory)
    local_feature1WF.base_dir = outputDirectory.rstrip('/')  + '_CACHE'
    local_feature1WF.run()


    """
    2nd Features
    """
    local_feature2WF = generate2ndFeatures( inputFLAIRVolume,
                                            inputT1Volume,
                                            outputDirectory)
    local_feature2WF.base_dir = outputDirectory.rstrip('/')  + '_CACHE'
    local_feature2WF.run()
