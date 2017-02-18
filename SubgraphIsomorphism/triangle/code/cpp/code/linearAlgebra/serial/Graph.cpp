//@HEADER
// ************************************************************************
// 
//                        miniTri v. 1.0
//              Copyright (2016) Sandia Corporation
// 
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  Jon Berry (jberry@sandia.gov)
//                     Michael Wolf (mmwolf@sandia.gov)
// 
// ************************************************************************
//@HEADER

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// File:      Graph.cc                                                      //
// Project:   miniTri                                                       //   
// Author:    Michael Wolf                                                  //
//                                                                          //
// Description:                                                             //
//              Source file for graph class.                                //
//////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <cassert>
#include <cstdlib>
#include <sys/time.h>

#include "Graph.hpp"
#include "mmio.h"


//////////////////////////////////////////////////////////////////////////////
// Enumerate triangles in graph
//////////////////////////////////////////////////////////////////////////////
void Graph::triangleEnumerate()
{
  struct timeval t1, t2;
  double eTime;

  std::cout << "************************************************************"
            << "**********" << std::endl;
  std::cout << "Enumerating triangles ....." << std::endl;
  std::cout << "************************************************************" 
            << "**********" << std::endl;

  ///////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////
  std::cout << "--------------------" << std::endl;
  std::cout << "Permuting matrix ...";

  gettimeofday(&t1, NULL);
  //  mMatrix.permute();
  gettimeofday(&t2, NULL);

  std::cout << " done" <<std::endl;

  eTime = t2.tv_sec - t1.tv_sec + ((t2.tv_usec-t1.tv_usec)/1000000.0);
  std::cout << "TIME - Time to permute  matrix: " << eTime << std::endl;

  std::cout << "--------------------" << std::endl;
  ///////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////
  // Form B
  ///////////////////////////////////////////////////////////////////////
  std::cout << "--------------------" << std::endl;
  std::cout << "Creating incident matrix B...";

  gettimeofday(&t1, NULL);

  CSRMat B(INCIDENCE);
  B.createIncidentMatrix(mMatrix,mEdgeIndices);

  gettimeofday(&t2, NULL);

  std::cout << " done" <<std::endl;

  //mMatrix.print();
  //B.print();

  eTime = t2.tv_sec - t1.tv_sec + ((t2.tv_usec-t1.tv_usec)/1000000.0);
  std::cout << "TIME - Time to create B: " << eTime << std::endl;

  std::cout << "--------------------" << std::endl;
  ///////////////////////////////////////////////////////////////////////




  ///////////////////////////////////////////////////////////////////////
  // C = A*B
  ///////////////////////////////////////////////////////////////////////
  std::cout << "--------------------" << std::endl;

  boost::shared_ptr<CSRMat> C(new CSRMat(mMatrix.getM(),B.getN(),true));

  std::cout << "C = A*B: " << std::endl;

  gettimeofday(&t1, NULL);
  C->matmat(mMatrix,B);
  gettimeofday(&t2, NULL);

  //C.print();

  eTime = t2.tv_sec - t1.tv_sec + ((t2.tv_usec-t1.tv_usec)/1000000.0);
  std::cout << "TIME - Time to compute C = L*B: " << eTime << std::endl;

  std::cout << "--------------------" << std::endl;

  std::cout << "NNZ: " << B.getNNZ() << std::endl;
  mNumTriangles = C->getNNZ() / 3;
  ///////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////
  // Save triangle information                                           
  ///////////////////////////////////////////////////////////////////////
  mTriMat = C;
  ///////////////////////////////////////////////////////////////////////

  std::cout << "************************************************************"
            << "**********" << std::endl;
  std::cout << "Finished triangle enumeration" << std::endl;
  std::cout << "************************************************************" 
            << "**********" << std::endl;
}
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
// Calculate triangle degrees                                                 
//////////////////////////////////////////////////////////////////////////////
void Graph::calculateTriangleDegrees()
{
  struct timeval t1, t2;
  double eTime;

  std::cout << "************************************************************"
            << "**********" << std::endl;
  std::cout << "Calculating triangle degrees ....." << std::endl;
  std::cout << "************************************************************"
            << "**********" << std::endl;

  ///////////////////////////////////////////////////////////////////////
  // Compute triangle vertex degrees                                     
  //     dv = C * 1                                                      
  ///////////////////////////////////////////////////////////////////////
  std::cout << "--------------------" << std::endl;
  std::cout << "Computing triangle vertex degrees ...";

  gettimeofday(&t1, NULL);

  mVTriDegrees.resize(mNumVerts);

  mTriMat->SpMV1(false, mVTriDegrees);

  gettimeofday(&t2, NULL);

  std::cout << " done" <<std::endl;

  eTime = t2.tv_sec - t1.tv_sec + ((t2.tv_usec-t1.tv_usec)/1000000.0);
  std::cout << "TIME - Time to compute vertex degrees: " << eTime << std::endl;
  ///////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////
  // Compute triangle edge degrees                                       
  //     de = C' * 1                                                     
  ///////////////////////////////////////////////////////////////////////
  std::cout << "--------------------" << std::endl;
  std::cout << "Computing triangle edge degrees ...";

  gettimeofday(&t1, NULL);

  mETriDegrees.resize(mNumEdges);
  mTriMat->SpMV1(true, mETriDegrees);

  gettimeofday(&t2, NULL);

  std::cout << " done" <<std::endl;

  eTime = t2.tv_sec - t1.tv_sec + ((t2.tv_usec-t1.tv_usec)/1000000.0);
  std::cout << "TIME - Time to compute edge degrees: " << eTime << std::endl;
  ///////////////////////////////////////////////////////////////////////

  std::cout << "************************************************************"
            << "**********" << std::endl;
  std::cout << "Finished calculating triangle degrees" << std::endl;
  std::cout << "************************************************************"
            << "**********" << std::endl;

}
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
// Prints triangles in graph
//////////////////////////////////////////////////////////////////////////////
void Graph::printTriangles() const
{
  std::cout << "Triangles: " << std::endl;

  std::list<int> triangles = mTriMat->getSumElements();

  //Iterate through list and output triangles
  std::list<int>::const_iterator iter;
  for (iter=triangles.begin(); iter!=triangles.end(); iter++)
    {
      int v1 = *iter+1;
      iter++;
      int v2 = *iter+1;
      iter++;
      int v3 = *iter+1;

      if(v1>v2 && v1>v3)
        {
	  std::cout << "(" << v1;
	  std::cout << ", " << v2;
	  std::cout << ", " << v3 << ")" << std::endl;
        }
    }

}
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
// Calculate Kcounts                                                          
//////////////////////////////////////////////////////////////////////////////
void Graph::calculateKCounts()
{
  std::cout << "************************************************************"
            << "**********" << std::endl;
  std::cout << "Calculating K-counts ....." << std::endl;
  std::cout << "************************************************************"
            << "**********" << std::endl;

  mTriMat->computeKCounts(mVTriDegrees,mETriDegrees,mEdgeIndices,mKCounts);

  std::cout << "************************************************************"
            << "**********" << std::endl;
  std::cout << "Finished calculating K-counts" << std::endl;
  std::cout << "************************************************************"
            << "**********" << std::endl;

}
//////////////////////////////////////////////////////////////////////////////                                                                                         

//////////////////////////////////////////////////////////////////////////////
// Print kcounts                                                              
//////////////////////////////////////////////////////////////////////////////
void Graph::printKCounts()
{
  std::cout << "K-Counts: " << std::endl;
  for(unsigned int i=3; i<mKCounts.size(); i++)
    {
      std::cout << "K[" << i << "] = " <<mKCounts[i] << std::endl;
    }
}
//////////////////////////////////////////////////////////////////////////////

