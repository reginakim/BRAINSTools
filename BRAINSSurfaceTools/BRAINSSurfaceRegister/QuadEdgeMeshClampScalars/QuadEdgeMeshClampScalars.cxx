/*=========================================================================

 Program:   BRAINS (Brain Research: Analysis of Images, Networks, and Systems)
 Module:    $RCSfile: $
 Language:  C++
 Date:      $Date: 2011/07/09 14:53:40 $
 Version:   $Revision: 1.0 $

 Copyright (c) University of Iowa Department of Radiology. All rights reserved.
 See GTRACT-Copyright.txt or http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt
 for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 =========================================================================*/

#include "itkQuadEdgeMeshVTKPolyDataReader.h"
#include "itkQuadEdgeMeshScalarDataVTKPolyDataWriter.h"

#include "itkQuadEdgeMesh.h"
#include "itkQuadEdgeMeshClampScalarsFilter.h"

#include "QuadEdgeMeshClampScalarsCLP.h"

int main( int argc, char * argv [] )
{
  PARSE_ARGS;

  if( inputSurfaceFile == "" )
    {
    std::cerr << "No input file specified" << std::endl;
    return 1;
    }
  if( outputSurfaceFile == "" )
    {
    std::cerr << "No output file specified" << std::endl;
    return 1;
    }

  std::cout << "---------------------------------------------------" << std::endl;
  std::cout << "Input Surface: " << inputSurfaceFile << std::endl;
  std::cout << "Output Surface: " << outputSurfaceFile << std::endl;
  std::cout << "Clamp the scalar values into: " << std::endl;
  std::cout << "[ " << outputMin << " " << outputMax << " ]" << std::endl;
  std::cout << "---------------------------------------------------" << std::endl;

  typedef float MeshPixelType;
  const unsigned int Dimension = 3;

  typedef itk::QuadEdgeMesh<MeshPixelType, Dimension> MeshType;

  typedef itk::QuadEdgeMeshVTKPolyDataReader<MeshType> ReaderType;

  ReaderType::Pointer inputReader = ReaderType::New();
  inputReader->SetFileName( inputSurfaceFile.c_str() );
  inputReader->Update();

  typedef itk::QuadEdgeMeshClampScalarsFilter<MeshType, MeshType> FilterType;

  FilterType::Pointer filter  = FilterType::New();

  filter->SetInput(inputReader->GetOutput() );

  filter->ClampMinOn();
  filter->SetOutputMinimum(outputMin);

  filter->ClampMaxOn();
  filter->SetOutputMaximum(outputMax);

  filter->Update();

  typedef itk::QuadEdgeMeshScalarDataVTKPolyDataWriter<MeshType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput( filter->GetOutput() );
  writer->SetFileName( outputSurfaceFile.c_str() );
  writer->Update();

  return 0;
}
