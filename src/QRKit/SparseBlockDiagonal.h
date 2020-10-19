// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SPARSE_BLOCK_DIAGONAL_H
#define SPARSE_BLOCK_DIAGONAL_H

#include <Eigen/Core>
#include "SparseBlockCOO.h"
#include "SparseQRUtils.h"

namespace QRKit {

  /*
  * Container for memory efficient storing block diagonal matrices.
  * Each block on the diagonal can have different size.
  * Block angular matrix is a matrix of the following form:
  *
  *  XXX
  *  XXX 
  *     XXX          
  *     XXX          
  *        XXX        
  *        XXX        
  *           XXX      
  *           XXX      
  *              XXX    
  *              XXX    
  *                 XXX  
  *                 XXX  
  *                    XXX
  *                    XXX
  *
  * The matrix is typically very sparse.
  * This container provides also method for constructing block diagonal matrices from Eigen::SparseMatrix.
  */
  template <typename BlockMatrixType, typename _StorageIndex = int>
  class SparseBlockDiagonal {
  public:
    typedef std::vector<BlockMatrixType> BlockVec;
    typedef typename BlockMatrixType::RealScalar RealScalar;
    typedef typename BlockMatrixType::Scalar Scalar;
    typedef typename _StorageIndex StorageIndex;
    typedef typename BlockMatrixType::Index Index;
    typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

    /*
    * Stores information about a dense block in a block sparse matrix.
    * Holds the position of the block (row index, column index) and its size (number of rows, number of columns).
    */
    typedef typename SparseQRUtils::BlockBandedMatrixInfo<StorageIndex, 3> BlockDiagonalMatrixInfo;

    SparseBlockDiagonal()
      : nRows(0), nCols(0) {
    }

    SparseBlockDiagonal(const StorageIndex &rows, const StorageIndex &cols)
      : nRows(rows), nCols(cols) {
    }

    /*
    * This method is expecting that the given input matrix is already in block diagonal format,
    * having equally-sized diagonal blocks of size (blockRows, blockcols)
    */
    template <typename MatrixType>
    void fromBlockDiagonalPattern(const MatrixType& mat, const StorageIndex blockRows, const StorageIndex blockCols) {
      this->clear();
      this->nRows = mat.rows();
      this->nCols = mat.cols();
      
      // Get matrix block info
      BlockDiagonalMatrixInfo blockInfo;
      blockInfo.fromBlockDiagonalPattern(mat.rows(), mat.cols(), blockRows, blockCols);

      // Feed matrix into the memory efficient storage
      int numBlocks = blockInfo.blockOrder.size();
      BlockDiagonalMatrixInfo::MatrixBlockInfo bi;
      for (int i = 0; i < numBlocks; i++) {
        bi = blockInfo.blockMap.at(blockInfo.blockOrder.at(i));

        this->insertBack(BlockMatrixType(mat.block(bi.idxRow, bi.idxCol, bi.numRows, bi.numCols)));
      }
    }

    /*
    * If row permutation is needed to obtain block diagonal matrix out of the input sparse matrix, 
    * this permutation is returned as reference via an optional parameter for further use.
    */
    template <typename MatrixType>
    void fromSparseMatrix(const MatrixType& mat, PermutationType &rowPerm = PermutationType()) {
      typedef SparseMatrix<Scalar, RowMajor, typename MatrixType::StorageIndex> RowMajorMatrixType;
      
      this->clear();
      this->nRows = mat.rows();
      this->nCols = mat.cols();

      // Get matrix block info
      BlockDiagonalMatrixInfo blockInfo;

      /* We need to do generic analysis of the input matrix
      Looking for as-banded-as-possible structure in the matrix
      Expecting to form something block diagonal in this case
      */
      SparseQROrdering::AsBandedAsPossible<IndexType> abapOrdering;
      RowMajorMatrixType rmMat(mat);
      abapOrdering(rmMat, rowPerm);

      // Permute if permutation found
      if (abapOrdering.hasPermutation) {
        rmMat = rowPerm * rmMat;
      }

      // Compute matrix block structure
      blockInfo(rmMat);

      // Feed matrix into the memory efficient storage
      int numBlocks = blockInfo.blockOrder.size();
      BlockDiagonalMatrixInfo::MatrixBlockInfo bi;
      for (int i = 0; i < numBlocks; i++) {
        bi = blockInfo.blockMap.at(blockInfo.blockOrder.at(i));

        this->insertBack(BlockMatrixType(mat.block(bi.idxRow, bi.idxCol, bi.numRows, bi.numCols)));
      }
    }

    void insertBack(const BlockMatrixType &elem) {
      this->blocks.push_back(elem);
    }

    StorageIndex size() const {
      return this->blocks.size();
    }

    void clear() {
      this->blocks.clear();
    }

    BlockMatrixType& operator[](StorageIndex i) {
      return this->blocks[i];
    }
    const BlockMatrixType& operator[](StorageIndex i) const {
      return this->blocks[i];
    }

    StorageIndex rows() const {
      return this->nRows;
    }
    StorageIndex cols() const {
      return this->nCols;
    }

  protected:
    BlockVec blocks;

    StorageIndex nRows;
    StorageIndex nCols;
  };
}

#endif

