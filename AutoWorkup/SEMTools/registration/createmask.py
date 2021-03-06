# -*- coding: utf8 -*- 
"""Autogenerated file - DO NOT EDIT
If you spot a bug, please report it on the mailing list and/or change the generator."""

from nipype.interfaces.base import CommandLine, CommandLineInputSpec, SEMLikeCommandLine, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os


class CreateMaskInputSpec(CommandLineInputSpec):
    inputVolume = File(desc="Input Image", exists=True, argstr="--inputVolume %s")
    closingSize = traits.Int(desc="Closing Size", argstr="--closingSize %d")
    threshold = traits.Float(desc="otsuPercentileThreshold", argstr="--threshold %f")
    outputVolume = traits.Either(traits.Bool, File(), hash_files=False, desc="Output Image", argstr="--outputVolume %s")


class CreateMaskOutputSpec(TraitedSpec):
    outputVolume = File(desc="Output Image", exists=True)


class CreateMask(SEMLikeCommandLine):
    """title: Create Mask

category: Registration

description: 
This programs creates synthesized average brain.
  

version: 0.1

documentation-url: http:://mri.radiology.uiowa.edu/mriwiki

license: NEED TO ADD

contributor: This tool was developed by Yongqiang Zhao.

"""

    input_spec = CreateMaskInputSpec
    output_spec = CreateMaskOutputSpec
    _cmd = " CreateMask "
    _outputs_filenames = {'outputVolume':'outputVolume.nii'}
