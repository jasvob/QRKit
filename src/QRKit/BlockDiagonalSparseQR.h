// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
// Copyright (C) 2016 Sergio Garrido Jurado <i52gajus@uco.es>
// Copyright (C) 2012-2013 Desire Nuentsa <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2012-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef QRKIT_BLOCK_DIAGONAL_SPARSE_QR_H
#define QRKIT_BLOCK_DIAGONAL_SPARSE_QR_H

#include "SparseQRUtils.h"
#include "SparseQROrdering.h"
#include "SparseBlockDiagonal.h"

namespace QRKit {

/**
  * \ingroup SparseQR_Module
  * \class BlockDiagonalSparseQR
  * \brief QR factorization of block-diagonal matrix
  *
  * Performs QR factorization of a block-diagonal matrix and creates matrices Q and R accordingly.
  * Template parameter _QFormat specifies the ordering of Q and R:
  *  MatrixQFormat::FullQ - orthogonal Q, corresponding R is upper triangular (default)
  *  MatrixQFormat::BlockDiagonalQ - orthogonal block diagonal Q, corresponding R is not upper triangular
  *  There exists a permutation that allows to convert between FulLQ and BlockDiagonalQ.
  *
  * \implsparsesolverconcept
  *
  */
template<typename _BlockQRSolver, int _QFormat = 0>
class BlockDiagonalSparseQR : public SparseSolverBase<BlockDiagonalSparseQR<_BlockQRSolver,_QFormat> >
{
  protected:
    typedef SparseSolverBase<BlockDiagonalSparseQR<_BlockQRSolver,_QFormat> > Base;
    using Base::m_isInitialized;
  public:
    using Base::_solve_impl;
    typedef _BlockQRSolver BlockQRSolver;
    typedef typename BlockQRSolver::MatrixType BlockMatrixType;
    typedef SparseBlockDiagonal<BlockMatrixType> MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef typename MatrixType::Index Index;
    typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;
    typedef Matrix<Scalar, Dynamic, 1> ScalarVector;

    typedef SparseMatrix<Scalar, RowMajor, StorageIndex> MatrixQType;
    typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixRType;
    typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

  enum MatrixQFormat {
    FullQ = 0,
    BlockDiagonalQ = 1
  };
  /*
  * Stores information about a dense block in a block sparse matrix.
  * Holds the position of the block (row index, column index) and its size (number of rows, number of columns).
  */
  typedef typename SparseQRUtils::BlockBandedMatrixInfo<StorageIndex, 3> BlockBandedMatrixInfo;

    enum {
      ColsAtCompileTime = Dynamic,//MatrixType::ColsAtCompileTime,
      MaxColsAtCompileTime = Dynamic//MatrixType::MaxColsAtCompileTime
    };

  public:
    BlockDiagonalSparseQR() : m_analysisIsok(false), m_factorizationIsok(false)
    { }

    /** Construct a QR factorization of the matrix \a mat.
      *
      * \sa compute()
      */
    explicit BlockDiagonalSparseQR(const MatrixType& mat) : m_analysisIsok(false), m_factorizationIsok(false)
    {
      compute(mat);
    }

    /** Computes the QR factorization of the sparse matrix \a mat.
      *
      *
      * If input pattern analysis has been successfully performed before, it won't be run again by default.
      * forcePatternAnalysis - if true, forces reruning pattern analysis of the input matrix
      * \sa analyzePattern(), factorize()
      */
    void compute(const MatrixType& mat, const PermutationType &rowPerm = PermutationType(), bool forcePatternAlaysis = false)
    {
      analyzePattern(mat, rowPerm);

      // Reset variables before the factorization
      m_isInitialized = false;
      m_factorizationIsok = false;
      factorize(mat);
    }
    void analyzePattern(const MatrixType& mat, const PermutationType &rowPerm = PermutationType());
    void factorize(const MatrixType& mat);

    /** \returns the number of rows of the represented matrix.
      */
    inline Index rows() const { return m_R.rows(); }

    /** \returns the number of columns of the represented matrix.
      */
    inline Index cols() const { return m_R.cols();}

    /** \returns a const reference to the \b sparse upper triangular matrix R of the QR factorization.
    *
    * The structure of R is typically very sparse and contains upper triangular blocks on the diagonal.
    * A typical pattern of R might look like:
    *
    * QFormat == FullQ
    *
    * X X X
    *   X X
    *     X
    *       X X X
    *         X X
    *           X
    *             X X X
    *               X X
    *                 X
    *                   X X X
    *                     X X
    *                       X
    *
    * QFormat == BlockDiagonalQ 
    * \warning Note that forming block diagonal Q, matrix R will not be upper triangular anymore (it will be upper triangular only up to a row permutation)
    *
    * X X X
    *   X X
    *     X
    *       
    *
    *
    *       X X X
    *         X X
    *           X
    *             X X X
    *               X X
    *                 X
    *
    *
    *                   X X X
    *                     X X
    *                       X
    *
    */
    const MatrixRType& matrixR() const { return m_R; }

    /** \returns the number of non linearly dependent columns.
    * \note Will be always equal to number of columns (full rank) in this case because the solver is not rank-revealing.
    */
    Index rank() const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      return m_nonzeropivots;
    }

    /** \returns a SparseMatrix<Scalar> holding the explicit representation of the matrix Q.
    * The common usage of this function is to apply it to a dense matrix or vector
    * \code
    * VectorXd B1, B2;
    * // Initialize B1
    * B2 = matrixQ() * B1;
    * \endcode
    *
    * To get a plain SparseMatrix representation of Q:
    * \code
    * SparseMatrix<double> Q;
    * Q = SparseQR<SparseMatrix<double> >(A).matrixQ();
    * \endcode
    *
    * Internally, matrix Q is composed explicitly and stored as SparseMatrix for block diagonal matrices. The structure of the output Q
    * is known a-priori and is known to be extremely sparse, and it is therefore not expensive to form it explicitly.
    *
    * The typical structure of the matrix Q will contain two shifted diagonals. The first diagonal is the "economy" matrix Q and the second is the orthogonal complement. It will look something like this:
    *
    * QFormat == FullQ
    *  X X                                      X X X X X
    *  X X                                      X X X X X
    *  X X                                      X X X X X
    *  X X                                      X X X X X
    *  X X                                      X X X X X
    *  X X                                      X X X X X
    *  X X                                      X X X X X
    *      X X                                            X X X X X
    *      X X                                            X X X X X
    *      X X                                            X X X X X
    *      X X                                            X X X X X
    *      X X                                            X X X X X
    *      X X                                            X X X X X
    *      X X                                            X X X X X
    *          X X                                                  X X X X X
    *          X X                                                  X X X X X
    *          X X                                                  X X X X X
    *          X X                                                  X X X X X
    *          X X                                                  X X X X X
    *          X X                                                  X X X X X
    *          X X                                                  X X X X X
    *
    * QFormat == BlockDiagonalQ 
    * \warning Note that forming block diagonal Q, matrix R will not be upper triangular anymore (it will be upper triangular only up to a row permutation)
    *
    *  X X X X X X X
    *  X X X X X X X
    *  X X X X X X X
    *  X X X X X X X
    *  X X X X X X X
    *  X X X X X X X
    *  X X X X X X X
    *                X X X X X X X
    *                X X X X X X X
    *                X X X X X X X
    *                X X X X X X X
    *                X X X X X X X
    *                X X X X X X X
    *                X X X X X X X
    *                              X X X X X X X
    *                              X X X X X X X
    *                              X X X X X X X
    *                              X X X X X X X
    *                              X X X X X X X
    *                              X X X X X X X
    *                              X X X X X X X
    *
    */
    MatrixQType matrixQ() const { 
      return m_Q; 
    }

    /** \returns a const reference to the column permutation P that was applied to A such that A*P = Q*R
    * Right now, this method is not rank revealing and column permutation is always identity.
    */
    const PermutationType& colsPermutation() const
    {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_outputPerm_c;
    }

    /** \returns a const reference to the row permutation P that was applied to A such that P*A = Q*R
    * Permutation that reorders the rows of the input matrix so that it has some block banded structure.
    */
    const PermutationType& rowsPermutation() const {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_rowPerm;
    }
    
    /** \internal */
    template<typename Rhs, typename Dest>
    bool _solve_impl(const MatrixBase<Rhs> &B, MatrixBase<Dest> &dest) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");

      Index rank = this->rank();

      // Compute Q^T * b;
      typename Dest::PlainObject y = this->matrixQ().transpose() * B;
      typename Dest::PlainObject b = y;

      // Solve with the triangular matrix R
      y.resize((std::max<Index>)(cols(),y.rows()),y.cols());
      y.topRows(rank) = this->matrixR().topLeftCorner(rank, rank).template triangularView<Upper>().solve(b.topRows(rank));
      y.bottomRows(y.rows()-rank).setZero();

      // Apply the column permutation
      if (colsPermutation().size())  dest = colsPermutation() * y.topRows(cols());
      else                  dest = y.topRows(cols());

      m_info = Success;
      return true;
    }

    /** \returns the solution X of \f$ A X = B \f$ using the current decomposition of A.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const Solve<BlockDiagonalSparseQR, Rhs> solve(const MatrixBase<Rhs>& B) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
      return Solve<BlockDiagonalSparseQR, Rhs>(*this, B.derived());
    }
    template<typename Rhs>
    inline const Solve<BlockDiagonalSparseQR, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
      return Solve<BlockDiagonalSparseQR, Rhs>(*this, B.derived());
    }

    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was successful,
      *          \c NumericalIssue if the QR factorization reports a numerical problem
      *          \c InvalidInput if the input matrix is invalid
      *
      * \sa iparm()
      */
    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_info;
    }

  protected:
    mutable ComputationInfo m_info;

    MatrixRType m_R;                // The triangular factor matrix
    MatrixQType m_Q;                // The orthogonal reflectors
    ScalarVector m_hcoeffs;         // The Householder coefficients

    PermutationType m_outputPerm_c; // The final column permutation
    PermutationType m_rowPerm;		  // Row permutation matrix, always identity as solver does not perform row permutations

    Index m_nonzeropivots;          // Number of non zero pivots found

    bool m_analysisIsok;
    bool m_factorizationIsok;
    Index m_numNonZeroQ;

    /*
    * Structures filled during sparse matrix pattern analysis.
    */
    BlockBandedMatrixInfo m_blockInfo;
};

template<typename _BlockQRSolver, int _QFormat>
struct SparseQRUtils::HasRowsPermutation<BlockDiagonalSparseQR<_BlockQRSolver, _QFormat>> {
  static const bool value = true;
};

/** \brief Preprocessing step of a QR factorization
*
* \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
*
* In this step, matrix pattern analysis is performed based on the type of the block QR solver.
*
* (a) BlockQRSolver::RowsAtCompileTime != Dynamic && BlockQRSolver::ColsAtCompileTime != Dynamic
* In this case, it is assumed that the matrix has a known block diagonal structure with
* constant block size and no overlaps.
*
* This step assumes that the user knows the structure beforehand and can specify it
* in form of input parameters.
* If called, the block diagonal pattern is automatically computed from the user input
* and there is no need to call analyzePattern later on, and some unnecessary analysis
* can be saved.
*
* An example of such block diagonal structure for a matrix could be:
* (in this example we could assume having blocks 7x2)
*
*  X X
*  X X
*  X X
*  X X
*  X X
*  X X
*  X X
*      X X
*      X X
*      X X
*      X X
*      X X
*      X X
*      X X
*          X X
*          X X
*          X X
*          X X
*          X X
*          X X
*          X X
*
* (b) BlockQRSolver::RowsAtCompileTime == Dynamic && BlockQRSolver::ColsAtCompileTime == Dynamic
* In this case, row-reordering permutation of A is computed and matrix banded structure is analyzed.
* This is neccessary preprocessing step before the matrix factorization is carried out.
*
* This step assumes there is some sort of banded structure in the matrix.
*
* \note In this step it is assumed that there is no empty row in the matrix \a mat.
*/
template <typename BlockQRSolver, int QFormat>
void BlockDiagonalSparseQR<BlockQRSolver, QFormat>::analyzePattern(const MatrixType& mat, const PermutationType &rowPerm) {
  // Save possible row permutation
  if (rowPerm.rows() == 0) {
    // If there's no row permutation given, set it to identity
    m_rowPerm.setIdentity(mat.rows());
  } else {
    // Otherwise use the given row permutation
    m_rowPerm = rowPerm;
  }
  // No need to analyze the matrix, it's already block diagonal
  m_R.resize(mat.rows(), mat.cols());

  m_analysisIsok = true;
}

/** \brief Performs the numerical QR factorization of the input matrix
  *
  * The function SparseQR::analyzePattern(const MatrixType&) must have been called beforehand with
  * a matrix having the same sparsity pattern than \a mat.
  *
  * \param mat The sparse column-major matrix
  */
template <typename BlockQRSolver, int QFormat>
void BlockDiagonalSparseQR<BlockQRSolver, QFormat>::factorize(const MatrixType& mat) {
  // Initialize the matrix for column permutations (will be filled during the factorization)
  m_outputPerm_c.setIdentity(mat.cols());

  Index numBlocks = mat.size();
  Index rank = 0;

  // Q is rows x rows, R is rows x cols
  BlockMatrixType block = mat[0];
  std::vector<Eigen::Triplet<Scalar> > tripletsR(numBlocks * block.rows() * block.cols());
  m_Q.resize(mat.rows(), mat.rows());
  m_Q.reserve(this->m_blockInfo.nonZeroQEstimate);

  Index m1 = 0;
  Index N_start = mat.cols();
  Index base_row = 0;
  Index base_col = 0;
  for (int i = 0; i < numBlocks; i++) {
    // Copy current block
    block = mat[i];

    // Perform QR
    BlockQRSolver blockSolver;
    blockSolver.compute(block);
    //rank += blockSolver.rank();
    rank += blockSolver.cols();	
    // FixMe: jasvob - only inner block is rank revealing for now. 
    // The zero columns of each block are not shifted to the end.
    // Therefore the method as a whole is not and ...
    // ... we cannot treat the result by viewing only first "rank" columns

    typename BlockQRSolver::MatrixQType Qi = blockSolver.matrixQ();
    typename BlockQRSolver::MatrixRType Ri = blockSolver.matrixR();

    // Assemble into final Q
    if (block.rows() >= block.cols()) {
      // each rectangular Qi is partitioned into [U N] where U is rxc and N is rx(r-c)
      // All the Us are gathered in the leftmost nc columns of Q, all Ns to the right

      // Q
      if(QFormat == MatrixQFormat::FullQ) {
        Index curr_m1 = (block.rows() - block.cols());
        for (Index j = 0; j < block.rows(); j++) {
          assert(base_row + j < m_Q.rows());
          m_Q.startVec(base_row + j);
          // Us
          for (Index k = 0; k < block.cols(); k++) {
            assert(base_col + k < m_Q.cols());
            m_Q.insertBack(base_row + j, base_col + k) = Qi.coeff(j, k);
          }
          // Ns
          for (Index k = 0; k < curr_m1; k++) {
            assert(N_start + m1 + k < m_Q.cols());
            m_Q.insertBack(base_row + j, N_start + m1 + k) = Qi.coeff(j, block.cols() + k);
          }
        }
        m1 += curr_m1;

        // R
        // Only the top cxc of R is nonzero, so c rows at a time
        for (Index j = 0; j < block.cols(); j++) {
          for (Index k = j; k < block.cols(); k++) {
            tripletsR.emplace_back(base_col + j, base_col + k, Ri.coeff(j, k));
          }
        }
      } else if(QFormat == MatrixQFormat::BlockDiagonalQ) {
        // Q diag
        Index curr_m1 = (block.rows() - block.cols());
        for (Index j = 0; j < block.rows(); j++) {
          assert(base_row + j < m_Q.rows());
          m_Q.startVec(base_row + j);
          // Us
          for (Index k = 0; k < block.rows(); k++) {
            assert(base_col + k < m_Q.cols());
            m_Q.insertBack(base_row + j, base_row + k) = Qi.coeff(j, k);
          }
        }
        m1 += curr_m1;

        // R
        // Only the top cxc of R is nonzero, so c rows at a time
        for (Index j = 0; j < block.cols(); j++) {
          for (Index k = j; k < block.cols(); k++) {
              tripletsR.emplace_back(base_row + j, base_col + k, Ri.coeff(j, k));
          }
        }
      } else {
        // Non-existing format requested?
        eigen_assert(false && "Block Diagonal QR decomposition ... non-existing format of matrix Q requested!");
        m_info = InvalidInput;
        return;
      }

    }
    else {
      // Just concatenate everything -- it's upper triangular anyway (although not rank-revealing... xxfixme with colperm?)
      // xx and indeed for landscape, don't even need to compute QR after we've done the leftmost #rows columns

      eigen_assert(false && "Block Diagonal QR decomposition ... case mat.cols() > mat.rows() not implemented!");
      m_info = InvalidInput;
      return;
    }

    // fill cols permutation
    for (Index j = 0; j < block.cols(); j++) {
      m_outputPerm_c.indices()(base_col + j, 0) = base_col + blockSolver.colsPermutation().indices()(j, 0);
    }

    // Update base row and base col
    base_row += block.rows();
    base_col += block.cols();
  }
  // Check last bi and if it doesn't reach until the last row of input matrix, it means there's a 0 block below
  // That will mean zeros in R and so no change needed
  // It will mean identity in Q -> add it
  for (int i = base_row; i < mat.rows(); i++) {
    m_Q.startVec(i);
    m_Q.insertBack(i, i) = Scalar(1.0);
  }

  // Now build Q and R from Qs and Rs of each block
  m_Q.finalize();

  m_R.resize(mat.rows(), mat.cols());
  m_R.setZero();
  m_R.setFromTriplets(tripletsR.begin(), tripletsR.end());
  m_R.makeCompressed();

  m_nonzeropivots = rank;
  m_isInitialized = true;
  m_info = Success;
  m_factorizationIsok = true;
}


} // end namespace QRKit

#endif
