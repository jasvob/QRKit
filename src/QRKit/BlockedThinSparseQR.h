// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef QRKIT_BLOCKED_THIN_SPARSE_QR_H
#define QRKIT_BLOCKED_THIN_SPARSE_QR_H

#include <ctime>
#include <typeinfo>
#include <shared_mutex>
#include "BlockedThinQRBase.h"

namespace QRKit {

  template<typename MatrixType, int SuggestedBlockCols> class BlockedThinSparseQR;

  namespace internal {
    // BlockedThinSparseQR_traits
    template <typename T> struct BlockedThinSparseQR_traits {  };
    template <class T, int Rows, int Cols, int Options> struct BlockedThinSparseQR_traits<Matrix<T, Rows, Cols, Options>> {
      typedef Matrix<T, Rows, 1, Options> Vector;
    };
    template <class Scalar, int Options, typename Index> struct BlockedThinSparseQR_traits<SparseMatrix<Scalar, Options, Index>> {
      typedef SparseVector<Scalar, Options> Vector;
    };
  } // End namespace internal

    /**
    * \ingroup BlockedThinSparseQR_Module
    * \class BlockedThinSparseQR
    * \brief Sparse Householder QR Factorization for banded matrices
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
    *
    * \implsparsesolverconcept
    *
    * \warning The input sparse matrix A must be in compressed mode (see SparseMatrix::makeCompressed()).
    *
    */
  template<typename _MatrixType, int _SuggestedBlockCols = 2>
  class BlockedThinSparseQR : public BlockedThinQRBase<_MatrixType, Eigen::ColPivHouseholderQR<Eigen::Matrix<typename _MatrixType::Scalar, Dynamic, Dynamic>>,  
    Eigen::SparseMatrix<typename _MatrixType::Scalar, ColMajor, typename _MatrixType::StorageIndex>, _SuggestedBlockCols>
  {
  protected:
    typedef BlockedThinQRBase<_MatrixType, Eigen::ColPivHouseholderQR<Eigen::Matrix<typename _MatrixType::Scalar, Dynamic, Dynamic>>, 
      Eigen::SparseMatrix<typename _MatrixType::Scalar, ColMajor, typename _MatrixType::StorageIndex>, _SuggestedBlockCols> Base;

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

    typedef Eigen::SparseMatrix<Scalar, RowMajor, StorageIndex> RowMajorMatrixType;

    enum {
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

  public:
    BlockedThinSparseQR() : Base() {
    }

    explicit BlockedThinSparseQR(const MatrixType& mat) : Base(mat) {
    }

    /** Computes the QR factorization of the sparse matrix \a mat.
    *
    * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
    *
    * If input pattern analysis has been successfully performed before, it won't be run again by default.
    * forcePatternAnalysis - if true, forces reruning pattern analysis of the input matrix
    * \sa analyzePattern(), factorize()
    */
    virtual void compute(const MatrixType& mat)
    {
      // Reset variables in case this method is called multiple times
      m_isInitialized = false;
      m_factorizationIsok = false;
      m_blocksYT.clear();
      this->m_nnzColPermIdxs.clear();
      this->m_zeroColPermIdxs.clear();

      // Analyze input matrix pattern and perform row and column permutations
      // Stores input matrix to m_pmat
      analyzePattern(mat);
    
      // Create dense version of the already permuted input matrix
      // It is much faster to do the permutations on the sparse version
      this->m_pmatDense = this->m_pmat.toDense();

      // Initialize householder permutation matrix
      this->m_houseColPerm.setIdentity(this->m_pmatDense.cols());

      // Reset nonzero pivots count
      this->m_nonzeroPivots = 0;

      // Prepare m_R to be filled in
      m_R.resize(this->m_pmatDense.rows(), this->m_pmatDense.cols());
      m_R.setZero();
      // Reserve number of elements needed in case m_R is full upper triangular
      m_R.reserve(this->m_pmatDense.cols() * this->m_pmatDense.cols() / 2.0);

      // And start factorizing block-by-block
      Index solvedCols = 0;
      Index cntr = 0;
      // As long as there are some unsolved columns
      Index newPivots = 0;
      while (solvedCols < this->m_pmatDense.cols()) {
        // Get next block info
        this->updateBlockInfo(solvedCols, this->m_pmat, newPivots, _SuggestedBlockCols);

        // Factorize current block
        newPivots = this->m_nonzeroPivots;
        factorize(this->m_pmatDense);
        newPivots = this->m_nonzeroPivots - newPivots;
        solvedCols += this->denseBlockInfo.numCols;
      }

      // Set computed Householder column permutation
      for (int c = 0; c < this->m_nnzColPermIdxs.size(); c++) {
        this->m_houseColPerm.indices()(c) = this->m_nnzColPermIdxs[c];
      }
      for (int c = 0; c < this->m_zeroColPermIdxs.size(); c++) {
        this->m_houseColPerm.indices()(this->m_nnzColPermIdxs.size() + c) = this->m_zeroColPermIdxs[c];
      }

      // Combine the two column permutation matrices together
      this->m_outputPerm_c = this->m_outputPerm_c * this->m_houseColPerm;

      // Don't forget to finalize m_R
      m_R.finalize();
            
      m_isInitialized = true;
      m_info = Success;
    }

    virtual void analyzePattern(const MatrixType& mat, bool rowPerm = true, bool colPerm = true) {
      /******************************************************************/
      // Create column permutation (according to the number of nonzeros in columns)
      if (colPerm) {
        ColumnDensityOrdering<StorageIndex> colDenOrdering;
        colDenOrdering(mat, this->m_outputPerm_c);

        m_pmat = mat * this->m_outputPerm_c;
      }
      else {
        this->m_outputPerm_c.setIdentity(mat.cols());

        // Don't waste time calling matrix multiplication if the permutation is identity
        m_pmat = mat;
      }

      /******************************************************************/
      // Compute and store band information for each row in the matrix
      if (rowPerm) {
        RowMajorMatrixType rmMat(m_pmat);
        AsBandedAsPossibleOrdering<StorageIndex> abapOrdering;
        abapOrdering(rmMat, this->m_rowPerm);

        m_pmat = this->m_rowPerm * m_pmat;
      }
      else {
        this->m_rowPerm.setIdentity(m_pmat.rows());

        // Don't waste time calling matrix multiplication if the permutation is identity
      }
      /******************************************************************/

      m_analysisIsok = true;
    }

    virtual void updateBlockInfo(const Index solvedCols, const MatrixType& mat, const Index newPivots, const Index blockCols = -1) {
      Index newCols = (blockCols > 0) ? blockCols : _SuggestedBlockCols;
      Index colIdx = solvedCols + newCols;
      Index numRows = 0;
      if (colIdx >= m_pmat.cols()) {
        colIdx = m_pmat.cols() - 1;
        newCols = m_pmat.cols() - solvedCols;
        numRows = m_pmat.rows() - this->m_nonzeroPivots;
      }
      else {
        typename MatrixType::StorageIndex biggestEndIdx = 0;
        for (int c = 0; c < newCols; c++) {
          typename MatrixType::InnerIterator colIt(m_pmat, solvedCols + c);
          typename MatrixType::StorageIndex endIdx = 0;
          if (colIt) {
            endIdx = colIt.index();
          }
          while (++colIt) { endIdx = colIt.index(); }

          if (endIdx > biggestEndIdx) {
            biggestEndIdx = endIdx;
          }
        }

        numRows = biggestEndIdx - this->m_nonzeroPivots + 1;
        if (numRows < (this->denseBlockInfo.numRows - newPivots)) {
          // In the next step we need to process at least all the rows we did in the last one
          // Even if the next block would be "shorter"
          numRows = this->denseBlockInfo.numRows - newPivots;
        }
      }

      this->denseBlockInfo = BlockBandedMatrixInfo::MatrixBlockInfo(this->m_nonzeroPivots, solvedCols, numRows, newCols);
    }

    virtual void factorize(DenseMatrixType& mat) {
      // Dense QR solver used for each dense block 
      //Eigen::HouseholderQR<DenseMatrixType> houseqr;
      DenseBlockQR houseqr;

      // Prepare the first block
      DenseMatrixType Ji = m_pmatDense.block(this->denseBlockInfo.idxRow, this->denseBlockInfo.idxCol, this->denseBlockInfo.numRows, this->denseBlockInfo.numCols);

      /*********** Process the block ***********/
      // 1) Factorize the block
      houseqr.compute(Ji);

      // Update column permutation according to ColPivHouseholderQR
      for (Index c = 0; c < houseqr.nonzeroPivots(); c++) {
        this->m_nnzColPermIdxs.push_back(this->denseBlockInfo.idxCol + houseqr.colsPermutation().indices()(c));
      }
      for (Index c = houseqr.nonzeroPivots(); c < this->denseBlockInfo.numCols; c++) {
        this->m_zeroColPermIdxs.push_back(this->denseBlockInfo.idxCol + houseqr.colsPermutation().indices()(c));
      }

      // 2) Create matrices Y and T
      DenseMatrixType Y, T;
      this->computeBlockedRepresentation(houseqr, Y, T);

      // Save current Y and T. The block YTY contains a main diagonal and subdiagonal part separated by (numZeros) zero rows.
      m_blocksYT.insert(typename SparseBlockYTYType::Element(this->denseBlockInfo.idxRow, this->denseBlockInfo.idxCol, BlockYTY<Scalar, StorageIndex>(Y, T, this->denseBlockInfo.idxRow, this->denseBlockInfo.idxCol, 0)));

      // Update the trailing columns of the matrix block
      this->updateMat(this->denseBlockInfo.idxCol, m_pmatDense.cols(), m_pmatDense, this->m_blocksYT.size() - 1);

      // Add solved columns to R
      // m_nonzeroPivots is telling us where is the current diagonal position    
      // Don't forget to add the upper overlap (anything above the current diagonal element is already processed, but is part of R
      for (typename MatrixType::StorageIndex bc = 0; bc < this->denseBlockInfo.numCols; bc++) {
        m_R.startVec(this->m_nonzeroPivots + bc);
        for (typename MatrixType::StorageIndex br = 0; br < this->m_nonzeroPivots; br++) {
          m_R.insertBack(br, this->m_nonzeroPivots + bc) = this->m_pmatDense(br, this->denseBlockInfo.idxCol + houseqr.colsPermutation().indices()(bc));
        }
        for (typename MatrixType::StorageIndex br = 0; br <= bc; br++) {
          m_R.insertBack(this->m_nonzeroPivots + br, this->m_nonzeroPivots + bc) = houseqr.matrixQR()(br, bc);
        }
      }

      // Add nonzero pivots from this block
      this->m_nonzeroPivots += houseqr.nonzeroPivots();
    }

  protected:    
    MatrixType  m_pmat;				              // Sparse version of input matrix - used for ordering and search purposes
    DenseMatrixType  m_pmatDense;	          // Dense version of the input matrix - used for factorization (much faster than using sparse)

    PermutationType m_houseColPerm;
    std::vector<Index> m_nnzColPermIdxs;
    std::vector<Index> m_zeroColPermIdxs;
  };

} // end namespace QRKit

#endif
