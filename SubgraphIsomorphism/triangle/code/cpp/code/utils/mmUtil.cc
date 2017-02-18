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
// File:      mmUtil.h                                                      //
// Project:   miniTri                                                       //
// Author:    Michael Wolf                                                  //
//                                                                          //
// Description:                                                             //
//              Source file for Matrix Market utilities.                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
#include <list>
#include <cstdlib>
#include <cassert>

#include <vector>

#include "mmUtil.h"

#include "mmio.h"

bool rowOnProc(int rowi,int numLocRows, int startRow);


//////////////////////////////////////////////////////////////////////////////
// Writes the matrix market banner to a file
//////////////////////////////////////////////////////////////////////////////
void writeMMBanner(std::ofstream &ofs, int m, int n, int nnz)
{
  MM_typecode matcode;

  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_coordinate(&matcode);
  mm_set_real(&matcode);

  mm_write_banner(ofs,matcode);
  mm_write_mtx_crd_size(ofs,m,n,nnz);

}
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
// Matrix market file to edge list conversion
//////////////////////////////////////////////////////////////////////////////
void buildEdgeListFromMM(const char *fname, int &numVerts, int &numEdges, std::vector<edge_t> &edgeList)
{
  //handle symmetric
  MM_typecode matcode;
  FILE *fp;

  if ((fp = fopen(fname, "r")) == NULL) 
  {
      std::cerr << "Cannot open filename" << fname << std::endl;
      exit(1);
  }

  if (mm_read_banner(fp, &matcode) != 0)
  {
      std::cerr << "Cannot process MM banner" << std::endl;
      exit(1);
  }

  checkMatrixType(matcode);

  int numRows,numCols;
  int nnzToRead;
  // determine matrix dimensions
  if ( mm_read_mtx_crd_size(fp, &numRows, &numCols, &nnzToRead) !=0 )
  {
    std::cout << "Cannot read in matrix dimensions" << std::endl;
    exit(1);
  }

  if(numRows!=numCols)
  {
    std::cout << "Invalid graph: Rows must equal Columns" << std::endl;
    exit(1);
  }
  numVerts = numRows;
  //////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////
  // Allocate memory for edgeList
  //////////////////////////////////////////////////////////////
  if(mm_is_symmetric(matcode)!=0)
  {
    numEdges = nnzToRead*2;
  }
  else
  {
    numEdges = nnzToRead;
  }

  edgeList.resize(numEdges);
  //////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////
  // read data from file and insert into edge List 
  //////////////////////////////////////////////////////////////

  // read in matrix entries from file
  int tmprow,tmpcol;
  int tmpval;

  for (int i=0; i<nnzToRead; i++)
  {
     if(mm_is_pattern(matcode)!=0)
     {
       fscanf(fp, "%d %d\n", &tmprow, &tmpcol);
      }
     else
     {
       fscanf(fp, "%d %d %d\n", &tmprow, &tmpcol, &tmpval);
     }

     if(mm_is_symmetric(matcode)!=0)
     {
       edgeList[2*i].v0 = tmprow;
       edgeList[2*i].v1 = tmpcol;

       edgeList[2*i+1].v0 = tmpcol;
       edgeList[2*i+1].v1 = tmprow;
     }
     else
     {
       edgeList[i].v0 = tmprow;
       edgeList[i].v1 = tmpcol;
     }
    
  }

  fclose(fp);
}
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
// Matrix market file to edge list conversion
//////////////////////////////////////////////////////////////////////////////
void buildDistEdgeListFromMM(const char *fname, int worldsize, int myrank,
			     int &numGlobVerts, int &numLocVerts, int &startVert, 
			     std::vector<edge_t> &edgeList)
{
  //handle symmetric
  MM_typecode matcode;
  FILE *fp;

  if ((fp = fopen(fname, "r")) == NULL) 
  {
      std::cerr << "Cannot open filename" << fname << std::endl;
      exit(1);
  }

  if (mm_read_banner(fp, &matcode) != 0)
  {
      std::cerr << "Cannot process MM banner" << std::endl;
      exit(1);
  }

  checkMatrixType(matcode);

  int numRows,numCols;
  int nnzToRead;
  // determine matrix dimensions
  if ( mm_read_mtx_crd_size(fp, &numRows, &numCols, &nnzToRead) !=0 )
  {
    std::cout << "Cannot read in matrix dimensions" << std::endl;
    exit(1);
  }

  if(numRows!=numCols)
  {
    std::cout << "Invalid graph: Rows must equal Columns" << std::endl;
    exit(1);
  }
  numGlobVerts = numRows;
  //////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////
  // partition problem
  /////////////////////////////////////////////////////////////////////////
  int locNumRows, startRow;
  partitionMatrix(numRows,worldsize,myrank,locNumRows,startRow);
  numLocVerts = locNumRows;
  startVert = startRow;
  /////////////////////////////////////////////////////////////////////////


  /////////////////////////////////////////////////////////////////////////
  // Allocate memory for edgeList
  /////////////////////////////////////////////////////////////////////////
  // Need to find a way to get an initial estimate and make this more efficient

//   if(mm_is_symmetric(matcode)!=0)
//   {
//     numEdges = nnzToRead*2;
//   }
//   else
//   {
//     numEdges = nnzToRead;
//   }

//   edgeList.resize(numEdges);
  /////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////
  // read data from file and insert into edge List 
  /////////////////////////////////////////////////////////////////////////
  int tmprow,tmpcol;
  int tmpval;

  for (int i=0; i<nnzToRead; i++)
  {
     if(mm_is_pattern(matcode)!=0)
     {
       fscanf(fp, "%d %d\n", &tmprow, &tmpcol);
     }
     else
     {
       fscanf(fp, "%d %d %d\n", &tmprow, &tmpcol, &tmpval);
     }

     if(rowOnProc(tmprow-1,locNumRows,startRow))
     {
       edge_t e1;
       e1.v0 = tmprow;
       e1.v1 = tmpcol;
       edgeList.push_back(e1);
     }

     // if symmetric format, store symmetric nonzero
     if(mm_is_symmetric(matcode)!=0 && rowOnProc(tmpcol-1,locNumRows,startRow))
     {
       edge_t e2;
       e2.v0 = tmpcol;
       e2.v1 = tmprow;
       edgeList.push_back(e2);
     } 
  }
  /////////////////////////////////////////////////////////////////////////





  fclose(fp);
}
//////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void checkMatrixType(const MM_typecode &matcode)
{
  if (mm_is_complex(matcode))
  {
    std::cout << "We do not support complex matrices." << std::endl;
    exit(1);
  }

  if (mm_is_coordinate(matcode)==0)
  {
    std::cout << "Matrix must be in coordinate format." << std::endl;
    exit(1);
  }
}
//////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void partitionMatrix(int gNumRows,int P, int rank, int &locNumRows,
                     int &startrow)
{
  /////////////////////////////////////////////////////////////////////////
  // Local number of rows is gNumRows/P + 1 if rank < gNumRows%P
  // Otherwise: gNumRows/P
  /////////////////////////////////////////////////////////////////////////
  locNumRows = (gNumRows/P);
  if(rank < gNumRows%P)
  {
    locNumRows++;
  }
  /////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////
  // If rank < than number of parts that are size gNumRows/P+1             
  // (or gNumRows%P), the start row is rank * gNumRow/P+1.                 
  //                                                                       
  // Otherwise, start row calculated as: rank*(gNumRows/P) + (gNumRows%P); 
  //    First term, counts gNumRows/P for each of the parts up to rank     
  //    Second term, adds 1 for each of the larger parts                   
  /////////////////////////////////////////////////////////////////////////
  if(rank < gNumRows%P)
  {
    startrow = rank * (gNumRows/P + 1);
  }
  else
  {
    startrow = rank*(gNumRows/P) + (gNumRows%P);
  }
  /////////////////////////////////////////////////////////////////////////

}
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
// returns true if row is on this process                                                                                                    
//////////////////////////////////////////////////////////////////////////////
bool rowOnProc(int rowi,int numLocRows, int startRow)
{
  if(rowi >= startRow && rowi < startRow+numLocRows)
  {
    return true;
  }
  return false;
}
//////////////////////////////////////////////////////////////////////////////
