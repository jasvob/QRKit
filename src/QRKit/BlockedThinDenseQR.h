// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef QRKIT_BLOCKED_THIN_DENSE_QR_H
#define QRKIT_BLOCKED_THIN_DENSE_QR_H

#include <ctime>
#include <typeinfo>
#include <shared_mutex>
#include "BlockedThinQRBase.h"

namespace QRKit {
    template<typename MatrixType, int SuggestedBlockCols> class BlockedThinDenseQR;
}

namespace Eigen {
    namespace internal {
        // BlockedThinDenseQR_traits
        template <typename T> struct BlockedThinDenseQR_traits {  };
        template <class T, int Rows, int Cols, int Options> struct BlockedThinDenseQR_traits<Matrix<T, Rows, Cols, Options>> {
            typedef Matrix<T, Rows, 1, Options> Vector;
        };
        template <class Scalar, int Options, typename Index> struct BlockedThinDenseQR_traits<SparseMatrix<Scalar, Options, Index>> {
            typedef SparseVector<Scalar, Options> Vector;
        };
    } // End namespace internal
} // End namespace Eigen

namespace QRKit {
    /**
    * \ingroup BlockedThinDenseQR_Module
    * \class BlockedThinDenseQR
    * \brief Sparse Householder QR Factorization for banded matrices operating on dense matrix.
    *
    * In some cases, it is faster to represent the input matrix as dense matrix and treat its sparsity pattern internally. 
    * Obviously, such approach requires more memory, it is however computationally much more efficient.
    *
    * This implementation is not rank revealing and uses Eigen::HouseholderQR for solving the dense blocks.
    *
    * Q is the orthogonal matrix represented as products of Householder reflectors.
    * Use matrixQ() to get an expression and matrixQ().transpose() to get the transpose.
    * You can then apply it to a vector.
    *
    * R is the sparse triangular or trapezoidal matrix. The later occurs when A is rank-deficient.
    * matrixR().topLeftCorner(rank(), rank()) always returns a triangular factor of full rank.
    *
    * \tparam _MatrixType The type of the sparse matrix A, must be a column-major SparseMatrix<>
    * \implsparsesolverconcept
    *
    * \warning The input sparse matrix A must be in compressed mode (see SparseMatrix::makeCompressed()).
    *
    */

  template<typename _MatrixType, int _SuggestedBlockCols = 2>
  class BlockedThinDenseQR : public BlockedThinQRBase<_MatrixType, Eigen::HouseholderQR<Eigen::Matrix<typename _MatrixType::Scalar, Dynamic, Dynamic>>,
    Eigen::Matrix<typename _MatrixType::Scalar, Dynamic, Dynamic>, _SuggestedBlockCols>
  {
  protected:
    typedef BlockedThinQRBase<_MatrixType, Eigen::HouseholderQR<Eigen::Matrix<typename _MatrixType::Scalar, Dynamic, Dynamic>>, 
      Eigen::Matrix<typename _MatrixType::Scalar, Dynamic, Dynamic>, _SuggestedBlockCols> Base;
  
  public:
    typedef _MatrixType MatrixType;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;
    typedef typename Base::StorageIndex StorageIndex;
    typedef typename Base::IndexVector IndexVector;
    typedef typename Base::DenseVectorType DenseVectorType;
    typedef typename Base::DenseMatrixType DenseMatrixType;

    typedef typename Base::MatrixQType MatrixQType;
    typedef typename Base::MatrixRType MatrixRType;
    typedef typename Base::PermutationType PermutationType;

    typedef typename Base::DenseBlockQR DenseBlockQR;

    typedef typename Base::BlockBandedMatrixInfo BlockBandedMatrixInfo;

    enum {
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

  public:
    BlockedThinDenseQR() : Base() { 
    }

    explicit BlockedThinDenseQR(const MatrixType& mat) : Base(mat) {
    }

    /** Computes the QR factorization of the sparse matrix \a mat.
    *
    * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
    *
    * \sa analyzePattern(), factorize()
    */
    virtual void compute(const MatrixType& mat)
    {
      // Reset variables in case this method is called multiple times
      m_isInitialized = false;
      m_factorizationIsok = false;
      m_blocksYT.clear();
      
      // Analyze input matrix pattern and perform row and column permutations
      // Stores input matrix to m_pmat
      analyzePattern(mat);
  
      // Create dense version of the already permuted input matrix
      // It is much faster to do the permutations on the sparse version
      this->m_R = mat;

      // And start factorizing block-by-block
      Index solvedCols = 0;
      Index cntr = 0;
      // As long as there are some unsolved columns
      while (solvedCols < this->m_R.cols()) {
        // Get next block info
        this->updateBlockInfo(solvedCols, this->m_R, 0, _SuggestedBlockCols);
   
        // Factorize current block
        factorize(this->m_R);
        solvedCols += this->denseBlockInfo.numCols;
      }

      this->m_nonzeroPivots = this->m_R.cols();
      m_isInitialized = true;
      m_info = Success;
    }
    virtual void analyzePattern(const MatrixType& mat, bool rowPerm = true, bool colPerm = true) {
      // No column permutation here
      this->m_outputPerm_c.setIdentity(mat.cols());

      // And no row permutatio neither
      this->m_rowPerm.setIdentity(mat.rows());

      m_analysisIsok = true;
    }

    virtual void updateBlockInfo(const Index solvedCols, const MatrixType& mat, const Index newPivots, const Index blockCols = -1) {
      Index newCols = (blockCols > 0) ? blockCols : _SuggestedBlockCols;
      Index colIdx = solvedCols + newCols;
      Index numRows = mat.rows() - solvedCols;
      if (colIdx >= mat.cols()) {
        colIdx = mat.cols() - 1;
        newCols = mat.cols() - solvedCols;
        numRows = mat.rows() - solvedCols;
      }

      this->denseBlockInfo = BlockBandedMatrixInfo::MatrixBlockInfo(solvedCols, numRows, newCols);
    }

    virtual void factorize(DenseMatrixType& mat) {
      // Dense QR solver used for each dense block 
      DenseBlockQR houseqr;

      /*********** Process the block ***********/
      // 1) Factorize the block
      houseqr.compute(mat.block(this->denseBlockInfo.idxRow, this->denseBlockInfo.idxRow, this->denseBlockInfo.numRows, this->denseBlockInfo.numCols));

      // 2) Create matrices Y and T
      DenseMatrixType Y, T;
      this->computeBlockedRepresentation(houseqr, Y, T);

      // Save current Y and T. The block YTY contains a main diagonal and subdiagonal part separated by (numZeros) zero rows.
      m_blocksYT.insert(SparseBlockYTYType::Element(this->denseBlockInfo.idxRow, this->denseBlockInfo.idxRow, BlockYTY<Scalar, StorageIndex>(Y, T, this->denseBlockInfo.idxRow, this->denseBlockInfo.idxCol, 0)));

      // Update the trailing columns of the matrix block
      this->updateMat(this->denseBlockInfo.idxRow, mat.cols(), mat, this->m_blocksYT.size() - 1);
    }

  protected:
  };

} // end namespace QRKit

#endif
