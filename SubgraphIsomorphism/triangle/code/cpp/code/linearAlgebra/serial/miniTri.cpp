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
// File:      miniTri.cpp                                                   //
// Project:   miniTri                                                       //
// Author:    Michael Wolf                                                  //
//                                                                          //
// Description:                                                             //
//              Enumerates triangles in graphs                              //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cstdlib>
#include <cassert>

#include "Graph.hpp"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
  if(argc!=2 && argc!=3)
  {
    std::cerr << "Usage: miniTri matrixFile [fileformat ={MM || Bin}]" << std::endl;
    exit(1);
  }

  std::string mat = argv[1];
  bool isBinFile = false;

  if(argc==3)
  {
    std::string fileFormat = std::string(argv[2]);
    if(fileFormat == "MM")
    {
      isBinFile=false;
    }
    else if(fileFormat == "Bin")
    {
      isBinFile=true;
    }
    else
    {
      std::cerr << "File format must be MM or Bin" << std::endl;
      exit(1);
    }
  }


  Graph g(mat,isBinFile);
  g.triangleEnumerate();
  std::cout << "Number of Triangles: " << g.getNumTriangles() << std::endl;
  //   g.printTriangles();

  g.calculateTriangleDegrees();
  g.calculateKCounts();
  g.printKCounts();


}
//////////////////////////////////////////////////////////////////////////////

