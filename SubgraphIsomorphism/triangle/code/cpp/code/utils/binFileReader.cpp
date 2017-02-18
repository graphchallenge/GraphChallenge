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

#include <inttypes.h> /* PRId64 */
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "miniTriDefs.h"

////////////////////////////////////////////////////////////////////////////////
//readBinEdgeFile - read in binary edge file 
////////////////////////////////////////////////////////////////////////////////
int readBinEdgeFile(char const * edgesFname, int64_t &numVerts, int64_t &numEdges,
                    std::vector<edge_t> &edges)
{
  int rc = 0;

  numVerts =0;

  if(numEdges>0)
  {
    edges.reserve(numEdges);
  }

  std::ifstream file(edgesFname, std::ios::binary);

  edge_t e;
  while( file.read( reinterpret_cast<char*>( &e ), sizeof(e) ) )
  {
    edges.push_back(e);

    numVerts = std::max(numVerts, std::max(e.v0,e.v1) );
  }

  numEdges = edges.size();
  numVerts = numVerts+1;

  return rc;

}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//readBinEdgeFileMPI - read in binary edge file using MPI
////////////////////////////////////////////////////////////////////////////////
#ifdef USE_MPI
int readBinEdgeFileMPI(char const * edgesFilename, int64_t const numEdges,
                       int64_t &numLocalEdges, int myrank, int worldsize,
		       edge_t * &edges)
{
  int rc = 0;

  int64_t start_ei;
  int64_t stop_ei;

  MPI_File     edges_file;

  static MPI_Datatype edgeMPIType;
  MPI_Type_contiguous(2, MPI_INT64_T, &edgeMPIType);
  MPI_Type_commit(&edgeMPIType);

  /* Calculate local edge block start and stop indices */
  {
    int64_t const num_chunk_edges = numEdges / worldsize;
    int64_t const numEdges_remaining = numEdges % worldsize;

    start_ei = myrank * num_chunk_edges;
    if (myrank <= numEdges_remaining)
    {
      start_ei += myrank;
    }
    else
    {
      start_ei += numEdges_remaining;
    }

    stop_ei = (myrank + 1) * num_chunk_edges;
    if ((myrank + 1) <= numEdges_remaining)
    {
      stop_ei += myrank + 1;
    } 
    else 
    {
      stop_ei += numEdges_remaining;
    }

    numLocalEdges = stop_ei - start_ei;
  }

  /* Allocate edges buffer */
  edges = (edge_t *)malloc(numLocalEdges * sizeof(edge_t));
  if (NULL == edges) 
  {
    fprintf(stderr, "Error: insufficient memory for edges buffer\n");
    exit(1);
  }

  /* Read in local edges from shared file */
  MPI_File_set_errhandler(MPI_FILE_NULL, MPI_ERRORS_ARE_FATAL);
  MPI_File_open(MPI_COMM_WORLD, (char *)edgesFilename, MPI_MODE_RDONLY, MPI_INFO_NULL, &edges_file);
  MPI_File_set_view(edges_file, 0, edgeMPIType, edgeMPIType, "native", MPI_INFO_NULL);
  MPI_File_read_at_all(edges_file, start_ei, edges, numLocalEdges, edgeMPIType, MPI_STATUS_IGNORE);
  MPI_File_close(&edges_file);

  return rc;

}
#endif
////////////////////////////////////////////////////////////////////////////////


