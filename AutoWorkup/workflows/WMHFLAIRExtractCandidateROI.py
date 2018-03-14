#! /usr/bin/env python
"""
WMHFLAIRExtractCandidateROI.py
==============
Over-sample the candidate regions of T2-FLAIR hyperintensity:
input:
    T2-FLAIR Image
    T1(Reference) volume
    Brain mask (labelmap) volume
output (to outputDirectory):
    output mask file
    co-registered input FL image (?)
Processing steps:
    1. Co-register input images to the reference (T1) image
    2. Threshold T2-FLAIR

Usage:
  WMHFLAIRExtractCandidateROI.py --inputFLVolume inputFLVolume --inputT1Volume inputT1Volume --inputBrainLabelsMapImage BLMImage --thresholdValue thresholdValue  --program_paths PROGRAM_PATHS [--processingType hypo|hyper] [--useIntensityReference=<BOOL>] [--inputIntensityReference inputIntensityReference] [--python_aux_paths PYTHON_AUX_PATHS] [--cacheDir cacheDir] [--outputDirectory outputDirectory] 
  WMHFLAIRExtractCandidateROI.py -v | --version
  WMHFLAIRExtractCandidateROI.py -h | --help

Options:
  -h --help                                         Show this help and exit
  -v --version                                      Print the version and exit
  --inputFLVolume inputFLVolume                     Path to the input MRI scan for further processing
  --inputT1Volume inputT1Volume                     Path to the input T1 Volume
  --thresholdValue thresholdValue                   A value for thresholding 
  --processingType processingType                   Either [hyper|hypo] [default: hyper]
  --useIntensityReference=BOOL                      useIntensityReference [default: False]
  --inputIntensityReference inputIntensityReference Path to the input Intensity Reference. Brain-Clipped T2 Image was used for testing.
  --inputBrainLabelsMapImage BLMImage               Path to the input brain labels map image
  --program_paths PROGRAM_PATHS                     Path to the directory where binary files are places
  --python_aux_paths PYTHON_AUX_PATHS               Path to the AutoWorkup directory
  --cacheDir cacheDir                       Base directory that cache outputs of workflow will be written to [default: ./]
  --outputDirectory outputDirectory                             Outputs of dataSink will be written to a sub directory under the outputDirectory  [default: cacheDir]
"""
from __future__ import print_function
import os
import glob
import sys

from docopt import docopt

def Threshold(inputVolume, outputVolume, thresholdLower=-0.1, thresholdUpper=1.1):
    import os
    import sys
    import SimpleITK as sitk
    ## Now clean up the posteriors based on anatomical knowlege.
    ## sometimes the posteriors are not relevant for priors
    ## due to anomolies around the edges.
    inputImg = sitk.Cast(sitk.RescaleIntensity( sitk.ReadImage(inputVolume), 0,1), sitk.sitkFloat32)
    outputImg = sitk.BinaryThreshold( inputImg, thresholdLower, thresholdUpper)

    sitk.WriteImage(outputImg, outputVolume)
    outputVolume = os.path.realpath(outputVolume)
    return  outputVolume

def HistogramMatching(inputVolume, refVolume, outputVolume):
    import os
    import sys
    import SimpleITK as sitk
    ## Now clean up the posteriors based on anatomical knowlege.
    ## sometimes the posteriors are not relevant for priors
    ## due to anomolies around the edges.
    inputImg = sitk.Cast(sitk.ReadImage(inputVolume), sitk.sitkFloat32)
    refImg = sitk.Cast( sitk.ReadImage(refVolume), sitk.sitkFloat32)
    outputImg = sitk.HistogramMatching( inputImg, refImg )
    outputImg = sitk.Cast( sitk.RescaleIntensity( outputImg, 0, 1), sitk.sitkFloat32)

    sitk.WriteImage(outputImg, outputVolume)
    outputVolume = os.path.realpath(outputVolume)
    return outputVolume

def HistogramEqualizer(inputVolume, outputVolume):
    import os
    import sys
    import SimpleITK as sitk
    ## Now clean up the posteriors based on anatomical knowlege.
    ## sometimes the posteriors are not relevant for priors
    ## due to anomolies around the edges.
    inputImg = sitk.Cast(sitk.ReadImage(inputVolume), sitk.sitkFloat32)
    outputImg = sitk.AdaptiveHistogramEqualization( inputImg )
    outputImg = sitk.Cast( sitk.RescaleIntensity( outputImg, 0, 1), sitk.sitkFloat32)

    sitk.WriteImage(outputImg, outputVolume)
    outputVolume = os.path.realpath(outputVolume)
    return outputVolume

def ClipVolumeWithBinaryMask(inputVolume, inputBinaryVolume, outputVolume):
    import os
    import sys
    import SimpleITK as sitk
    ## Now clean up the posteriors based on anatomical knowlege.
    ## sometimes the posteriors are not relevant for priors
    ## due to anomolies around the edges.
    inputImg = sitk.Cast(sitk.ReadImage(inputVolume), sitk.sitkFloat32)
    inputMsk = sitk.Cast(sitk.ReadImage(inputBinaryVolume), sitk.sitkFloat32)
    inputMskBinary = sitk.Cast( (inputMsk > 0 ), sitk.sitkFloat32)
    outputImg = inputImg * inputMskBinary
    sitk.WriteImage(outputImg, outputVolume)
    outputVolume = os.path.realpath(outputVolume)
    return outputVolume


def wmh_roiExtractor( processingType, outputDirectory ):
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
    #\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
    ####### Workflow ###################
    WFname = 'LesionDetector'  
    LesionDetectorWF = pe.Workflow(name=WFname)
    LesionDetectorWF.config['execution'] = {'remove_unnecessary_outputs': 'False',
                                            'hash_method': 'timestamp'}
    #
    # I/O Specification
    #
    inputsSpec = pe.Node(interface=IdentityInterface(fields=['inputFLVolume', 
                                                             'inputT1Volume',
                                                             'inputIntensityReference', 
                                                             'inputBrainMask',
                                                             'thresholdValue']),
                         name='inputspec')

    outputsSpec = pe.Node(interface=IdentityInterface(fields=['input2T1_transformed', 'input2T1_transform']),
                          name='outputsSpec')

    #
    # Denoise Image
    #
    denosingTimeStep = 0.0625
    denosingConductance = 0.4
    denosingIteration = 5

    DenoisedInput = pe.Node(interface=GradientAnisotropicDiffusionImageFilter(), name="DenoisedInput")
    DenoisedInput.inputs.timeStep = denosingTimeStep
    DenoisedInput.inputs.conductance = denosingConductance
    DenoisedInput.inputs.numberOfIterations = denosingIteration
    DenoisedInput.inputs.outputVolume = "DenoisedInput.nii.gz"

    LesionDetectorWF.connect(inputsSpec, 'inputFLVolume', DenoisedInput, 'inputVolume')

    #
    # N4 Bias Correction
    #
    from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection

    N4BFC = pe.Node(interface=N4BiasFieldCorrection(),
                    name='N4BFC')
    N4BFC.inputs.num_threads = -1
    N4BFC.inputs.dimension = 3
    N4BFC.inputs.bspline_fitting_distance = 200
    N4BFC.inputs.shrink_factor = 3
    N4BFC.inputs.n_iterations = [50,50,30,20]
    N4BFC.inputs.convergence_threshold = 1e-6

    LesionDetectorWF.connect( DenoisedInput, 'outputVolume',
                              N4BFC, 'input_image')

    #
    # Intra-subject registration
    #
    BFit_Input2T1 = pe.Node(interface=BRAINSFit(), name="BFit_Input2T1")
    BFit_Input2T1.inputs.costMetric = "MMI"
    BFit_Input2T1.inputs.numberOfSamples = 100000
    BFit_Input2T1.inputs.numberOfIterations = [1500]
    BFit_Input2T1.inputs.numberOfHistogramBins = 50
    BFit_Input2T1.inputs.maximumStepLength = 0.2
    BFit_Input2T1.inputs.minimumStepLength = [0.00005]
    BFit_Input2T1.inputs.useRigid = True
    BFit_Input2T1.inputs.useAffine = True
    BFit_Input2T1.inputs.maskInferiorCutOffFromCenter = 65
    BFit_Input2T1.inputs.maskProcessingMode = "ROIAUTO"
    BFit_Input2T1.inputs.ROIAutoDilateSize = 13
    BFit_Input2T1.inputs.backgroundFillValue = 0.0
    BFit_Input2T1.inputs.initializeTransformMode = 'useCenterOfHeadAlign'
    BFit_Input2T1.inputs.outputTransform= "FLToT1_RigidTransform.h5"
    #BFit_Input2T1.inputs.strippedOutputTransform = "InputToT1_RigidTransform.h5"
    BFit_Input2T1.inputs.writeOutputTransformInFloat = True
    BFit_Input2T1.inputs.outputVolume = "DenoisedFLinT1.nii.gz"

    LesionDetectorWF.connect(inputsSpec, 'inputT1Volume', BFit_Input2T1, 'fixedVolume')
    LesionDetectorWF.connect(N4BFC, 'output_image', BFit_Input2T1, 'movingVolume')

    LesionDetectorWF.connect( BFit_Input2T1, 'outputTransform',
                              outputsSpec, 'input2T1_transform')
    LesionDetectorWF.connect( BFit_Input2T1, 'outputVolume',
                              outputsSpec, 'input2T1_transformed')
    ## Write all outputs with DataSink
    LesionDetectorDS = pe.Node(interface=nio.DataSink(), name='LDDataSink')
    LesionDetectorDS.inputs.base_directory = outputDirectory
    #LesionDetectorDS.inputs.container = outputPrefix

    LesionDetectorWF.connect(BFit_Input2T1, 'outputTransform',
                             LesionDetectorDS, 'Transform.@input2T1_transform')
    LesionDetectorWF.connect(BFit_Input2T1, 'outputVolume',
                             LesionDetectorDS, 'Transform.@input2T1_transformed')

    ## Skull Striping Input
    BrainClip_Input = pe.Node(interface=Function(function=ClipVolumeWithBinaryMask,
                                             input_names=['inputVolume',
                                                          'inputBinaryVolume',
                                                          'outputVolume'],
                                             output_names=['outputVolume']),
                          name="BrainClip_Input")
    BrainClip_Input.inputs.outputVolume= 'BrainClip_Input.nii.gz'

    LesionDetectorWF.connect( BFit_Input2T1, 'outputVolume',
                              BrainClip_Input, 'inputVolume')
    LesionDetectorWF.connect( inputsSpec, 'inputBrainMask',
                              BrainClip_Input, 'inputBinaryVolume')

    ## Threshold
    threshold = pe.Node( interface = Function( function= Threshold,
                                               input_names = ['inputVolume', 'outputVolume','thresholdLower','thresholdUpper'],
                                               output_names = ['outputVolume']),
                         name="threshold")

    threshold.inputs.outputVolume = 'lesionCandidateDetectedLabel.nii.gz'


    ## Skull Striping Input
    if( useIntensityReference ):
        if processingType == "hyper" :
            LesionDetectorWF.connect( inputsSpec, 'thresholdValue',
                                      threshold, 'thresholdLower')
            #threshold.iterables = ( 'thresholdLower', thresholdList )
            #threshold.iterables = ( 'thresholdLower', [0.3, 0.4, 0.45, 0.5] )
        else:
            LesionDetectorWF.connect( inputsSpec, 'thresholdValue',
                                      threshold, 'thresholdUpper')
            #threshold.iterables = ( 'thresholdUpper', thresholdList )
            #threshold.iterables = ( 'thresholdUpper', [025,0.26,0.27,0.28,0.29] )
        print( """
        *** Use the input Intensity Reference Volume for the Histogram Matching""")
        histMatching = pe.Node( interface = Function( function = HistogramMatching,
                                                      input_names = ['inputVolume','refVolume','outputVolume'],
                                                      output_names = ['outputVolume'] ),
                                                      name = "inputHistogramMatching")
        histMatching.inputs.outputVolume = "inputVolume_HistogramMatching.nii.gz"
        LesionDetectorWF.connect( BrainClip_Input, 'outputVolume',
                                  histMatching, 'inputVolume' )
        LesionDetectorWF.connect( inputsSpec, 'inputIntensityReference',
                                  histMatching, 'refVolume')
        LesionDetectorWF.connect( histMatching, 'outputVolume',
                                  threshold, 'inputVolume' )


    else:
        if processingType == "hyper" :
            LesionDetectorWF.connect( inputsSpec, 'thresholdValue',
                                      threshold, 'thresholdLower')
            #threshold.iterables = ( 'thresholdLower', thresholdList  )
            #threshold.iterables = ( 'thresholdLower', [0.7, 0.725, 0.75, 0.775, 0.8] )
        else:
            LesionDetectorWF.connect( inputsSpec, 'thresholdValue',
                                      threshold, 'thresholdUpper')
            #threshold.iterables = ( 'thresholdUpper', thresholdList )
            #threshold.iterables = ( 'thresholdUpper', [0.4, 0.45, 0.5] )
        print( """
        *** Use the Histogram Equalizer""")
        ## Histogram Equalaizer
        histEq = pe.Node( interface = Function( function= HistogramEqualizer,
                                                input_names = ['inputVolume','outputVolume'],
                                                output_names = ['outputVolume']),
                          name = "inputHistogramEqualizer")
        histEq.inputs.outputVolume = 'inputVolume_HistogramEqualized.nii.gz'
        LesionDetectorWF.connect( BrainClip_Input, 'outputVolume',
                                  histEq, 'inputVolume' )

        LesionDetectorWF.connect( histEq, 'outputVolume',
                                  threshold, 'inputVolume' )

    if processingType == "hypo":
        ## Skull Striping the Label
        BrainClip_Label = pe.Node(interface=Function(function=ClipVolumeWithBinaryMask,
                                                 input_names=['inputVolume',
                                                              'inputBinaryVolume',
                                                              'outputVolume'],
                                                 output_names=['outputVolume']),
                              name="BrainClip_Label")
        BrainClip_Label.inputs.outputVolume= 'BrainClip_Label.nii.gz'

        LesionDetectorWF.connect( threshold, 'outputVolume',
                                  BrainClip_Label, 'inputVolume')
        LesionDetectorWF.connect( inputsSpec, 'LabelMapVolume',
                                  BrainClip_Label, 'inputBinaryVolume')
        LesionDetectorWF.connect(BrainClip_Label, 'outputVolume',
                                 LesionDetectorDS, 'Threshold.@threshold')
        LesionDetectorDS.inputs.substitutions = [('/_thresholdUpper_','_'),
                                             ('/BrainClip_Label.nii.gz','/lesionCandidateDetectedLabel.nii.gz'),
                                             ('Transform/','')]

    else:
        LesionDetectorWF.connect(threshold, 'outputVolume', LesionDetectorDS, 'Threshold.@threshold')
        LesionDetectorDS.inputs.substitutions = [('/_thresholdLower_','_'),
                                             ('/lesionCandidateDetectedLabel.nii.gz','_labelmap.nii.gz'),
                                             ('Transform/',''),
                                             ('Threshold/','')]
    LesionDetectorWF.connect( DenoisedInput, 'outputVolume',
                              LesionDetectorDS, 'Threshold.@inputFLAIR')


    print("Return the workflow ...")
    return LesionDetectorWF

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

    #
    # command line argument treatment
    #
    argv = docopt(__doc__, version='1.0')
    print(argv)

    inputFLVolume = argv['--inputFLVolume']
    assert os.path.exists(inputFLVolume), "Input Volume scan is not found: %s" % inputFLVolume

    inputT1Volume = argv['--inputT1Volume']
    assert os.path.exists(inputT1Volume), "Input T2 scan is not found: %s" % inputT1Volume

    thresholdValue = argv['--thresholdValue']

    LabelMapImage = argv['--inputBrainLabelsMapImage']
    assert os.path.exists(LabelMapImage), "Input Brain labels map image is not found: %s" % LabelMapImage

    PROGRAM_PATHS = argv['--program_paths']

    if argv['--processingType'] != None:
        assert argv['--processingType'] in ['hyper','hypo'], "processing type should be either 'hyper', or 'hypo'."
        processingType = argv['--processingType']
    else:
        processingType = 'hyper'

    if argv['--useIntensityReference'] != None:
        useIntensityReference= argv['--useIntensityReference']
    else:
        useIntensityReference=False

    if argv['--inputIntensityReference'] != None:
        inputIntensityReference = argv['--inputIntensityReference']
        assert os.path.exists(inputIntensityReference), "Input Intensity Reference (E.g., Brain Clipped T2 Volume) scan is not found: %s" % inputIntensityReference
        useIntensityReference = True
    else:
        inputIntensityReference = None

    if argv['--python_aux_paths'] == None:
        PYTHON_AUX_PATHS = '/Users/eunyoungkim/anaconda/envs/namicAnacondaEnv/bin/'
    else:
        PYTHON_AUX_PATHS = argv['--python_aux_paths']

    if argv['--cacheDir'] == None:
        print("*** workflow cache directory is set to current working directory.")
        cacheDir = os.getcwd()
    else:
        cacheDir = os.path.abspath( argv['--cacheDir'] )
        if not os.path.exists(cacheDir):
            os.makedirs( cacheDir )
        assert os.path.exists(cacheDir), "Cache directory is not found: %s" % cacheDir

    if argv['--outputDirectory'] == None:
        print("*** data sink result directory is set to the neighbor to the cache directory.")
        outputDirectory = os.path.join( os.path.dirname( os.path.abspath( cacheDir )), "LD_Result")
        print("    :{0}".format( outputDirectory ) )
    else:
        outputDirectory = os.path.abspath(argv['--outputDirectory'])
        if not os.path.exists( outputDirectory ):
            os.makedirs( outputDirectory )
        assert os.path.exists(outputDirectory), "Results directory is not found: %s" % outputDirectory

    print('=' * 100)

    #####################################################################################
    ### start workflow
    ### Prepend the shell environment search paths
    PROGRAM_PATHS = PROGRAM_PATHS.split(':')
    PROGRAM_PATHS.extend(os.environ['PATH'].split(':'))
    os.environ['PATH'] = ':'.join(PROGRAM_PATHS)

    CUSTOM_ENVIRONMENT=dict()

    #####################################################################################
    ### Platform specific information
    ### Prepend the python search paths
    PYTHON_AUX_PATHS = PYTHON_AUX_PATHS.split(':')
    PYTHON_AUX_PATHS.extend(sys.path)
    sys.path = PYTHON_AUX_PATHS


    import nipype.pipeline.engine as pe  # pypeline engine
    from nipype.interfaces.utility import Merge, Split, Function, Rename, IdentityInterface

    wmh_localWF =  wmh_roiExtractor( processingType, outputDirectory)
    wmh_localWF.base_dir = cacheDir

    wmh_localWF_inputspec = wmh_localWF.get_node('inputspec')
    wmh_localWF_inputspec.inputs.inputFLVolume = inputFLVolume
    wmh_localWF_inputspec.inputs.inputT1Volume = inputT1Volume
    wmh_localWF_inputspec.inputs.inputIntensityReference = inputIntensityReference 
    wmh_localWF_inputspec.inputs.inputBrainMask = LabelMapImage

    wmh_localWF.run()

""" 
script example
"""
# inputFL="/Volumes/KOGES_MRI/MRI_Rawdata_repository/convert_file/2003459/137577_20150709/FLAIR.nii.gz"
# inputT1="/Volumes/KOGES_MRI/DataRepository/20160928_KoGES_base_Results/KoGES/2003459/137577_20150709/TissueClassify/t1_average_BRAINSABC.nii.gz"
# inputBinary="/Volumes/KOGES_MRI/DataRepository/20160928_KoGES_base_Results/KoGES/2003459/137577_20150709/JointFusion/JointFusion_HDAtlas20_2015_dustCleaned_label.nii.gz"
# 
# python ~/src/BRAINS_FLAIRWorkflow/BRAINSTools/AutoWorkup/workflows/WMHFLAIRExtractCandidateROI.py \
#   --inputFLVolume  $inputFL\
#   --inputT1Volume $inputT1\
#   --thresholdValue 0.45\
#   --inputBrainLabelsMapImage $inputBinary\
#   --program_paths /Users/eunyoungkim/src/NamicBuild_20180124/bin/\
#   --processingType hyper \
#   --inputIntensityReference /Users/eunyoungkim/src/NamicBuild_20171031/bin/Atlas/Atlas_20131115/template_t2_clipped.nii.gz\
#   --python_aux_paths ".:/Users/eunyoungkim/src/BRAINS_FLAIRWorkflow/BRAINSTools/AutoWorkup/" \
#   --cacheDir my_CACHE \
#   --outputDirectory my_Result


