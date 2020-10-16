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

#ifndef EIGEN_BLOCK_ANGULAR_SPARSE_QR_H
#define EIGEN_BLOCK_ANGULAR_SPARSE_QR_H

#include <algorithm>
#include <ctime>
#include "BlockMatrix1x2.h"

namespace Eigen {
  template <typename LeftSolver, typename RightSolver > class BlockAngularSparseQR;
  template<typename SparseQRType> struct BlockAngularSparseQRMatrixQReturnType;
  template<typename SparseQRType> struct BlockAngularSparseQRMatrixQTransposeReturnType;
  template<typename SparseQRType, typename Derived> struct BlockAngularSparseQR_QProduct;

  namespace internal {

    // traits<SparseQRMatrixQ[Transpose]>
    template <typename SparseQRType> struct traits<BlockAngularSparseQRMatrixQReturnType<SparseQRType> >
    {
      typedef typename SparseQRType::MatrixType ReturnType;
      typedef typename ReturnType::StorageIndex StorageIndex;
      typedef typename ReturnType::StorageKind StorageKind;
      enum {
        RowsAtCompileTime = Dynamic,
        ColsAtCompileTime = Dynamic
      };
    };

    template <typename SparseQRType> struct traits<BlockAngularSparseQRMatrixQTransposeReturnType<SparseQRType> >
    {
      typedef typename SparseQRType::MatrixType ReturnType;
    };

    template <typename SparseQRType, typename Derived> struct traits<BlockAngularSparseQR_QProduct<SparseQRType, Derived> >
    {
      typedef typename Derived::PlainObject ReturnType;
    };
  } // End namespace internal


  /**
  * \ingroup SparseQR_Module
  * \class BlockAngularSparseQR
  * \brief QR factorization of block matrix, specifying subblock solvers
  *
  * This implementation is restricted to 1x2 block structure, factorizing
  * matrix A = [A1 A2].
  * and makes the assumption that A1 is easy to factorize, 
  * and that dense matrices the shape of A2 are hard.
  *
  * \tparam _BlockQRSolverLeft The type of the QR solver which will factorize A1
  * \tparam _BlockQRSolverRight The type of the QR solver which will factorize Q1'*A2
  *
  * \implsparsesolverconcept
  *
  */

  template<typename _BlockQRSolverLeft, typename _BlockQRSolverRight>
  struct SparseQRUtils::HasRowsPermutation<BlockAngularSparseQR<_BlockQRSolverLeft, _BlockQRSolverRight>> {
    static const bool value = true;
  };

  template<typename _BlockQRSolverLeft, typename _BlockQRSolverRight>
  class BlockAngularSparseQR : public SparseSolverBase<BlockAngularSparseQR<_BlockQRSolverLeft, _BlockQRSolverRight> >
  {
  protected:
    typedef BlockAngularSparseQR<_BlockQRSolverLeft, _BlockQRSolverRight> this_t;
    typedef SparseSolverBase<BlockAngularSparseQR<_BlockQRSolverLeft, _BlockQRSolverRight> > Base;
    using Base::m_isInitialized;
  public:
    using Base::_solve_impl;
    typedef _BlockQRSolverLeft BlockQRSolverLeft;
    typedef _BlockQRSolverRight BlockQRSolverRight;
    typedef typename BlockQRSolverLeft::MatrixType LeftBlockMatrixType;
    typedef typename BlockQRSolverRight::MatrixType RightBlockMatrixType;
    typedef typename BlockQRSolverLeft::MatrixQType LeftBlockMatrixQType;
    typedef typename BlockQRSolverRight::MatrixQType RightBlockMatrixQType;
    typedef typename LeftBlockMatrixType::Scalar Scalar;
    typedef typename LeftBlockMatrixType::RealScalar RealScalar;
    typedef typename LeftBlockMatrixType::StorageIndex StorageIndex;
    typedef typename LeftBlockMatrixType::Index Index;
    typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;
    typedef Matrix<Scalar, Dynamic, 1> DenseVectorType;
    typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrixType;

    typedef BlockAngularSparseQRMatrixQReturnType<BlockAngularSparseQR> MatrixQType;
    typedef SparseMatrix<Scalar> MatrixType;
    typedef MatrixType MatrixRType;
    typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationMatrixType;
    typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

    enum {
      ColsAtCompileTime = Dynamic,
      MaxColsAtCompileTime = Dynamic
    };

  public:
    BlockAngularSparseQR() : m_analysisIsok(false), m_lastError(""), m_isQSorted(false), m_leftCols(1)
    { }

    /** Construct a QR factorization of the matrix \a mat.
    *
    * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
    *
    * \sa compute()
    */
    explicit BlockAngularSparseQR(const BlockMatrix1x2<LeftBlockMatrixType, RightBlockMatrixType>& mat) : m_analysisIsok(false), m_lastError(""), m_isQSorted(false), m_leftCols(1)
    {
      compute(mat);
    }

    /** Computes the QR factorization of the sparse matrix \a mat.
    *
    * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
    *
    * \sa analyzePattern(), factorize()
    */
    void compute(const BlockMatrix1x2<LeftBlockMatrixType, RightBlockMatrixType>& mat)
    {
      analyzePattern(mat);
      factorize(mat);
    }
    void analyzePattern(const BlockMatrix1x2<LeftBlockMatrixType, RightBlockMatrixType>& mat);
    void factorize(const BlockMatrix1x2<LeftBlockMatrixType, RightBlockMatrixType>& mat);

    /** \returns the number of rows of the represented matrix.
    */
    inline Index rows() const { return m_R.rows(); }

    /** \returns the number of columns of the represented matrix.
    */
    inline Index cols() const { return m_R.cols(); }

    /** \returns a const reference to the \b sparse upper triangular matrix R of the QR factorization.
    * \warning The entries of the returned matrix are not sorted. This means that using it in algorithms
    *          expecting sorted entries will fail. This include random coefficient accesses (SpaseMatrix::coeff()),
    *          and coefficient-wise operations. Matrix products and triangular solves are fine though.
    *
    * To sort the entries, you can assign it to a row-major matrix, and if a column-major matrix
    * is required, you can copy it again:
    * \code
    * SparseMatrix<double>          R  = qr.matrixR();  // column-major, not sorted!
    * SparseMatrix<double,RowMajor> Rr = qr.matrixR();  // row-major, sorted
    * SparseMatrix<double>          Rc = Rr;            // column-major, sorted
    * \endcode
    */
    const MatrixRType& matrixR() const { return m_R; }

    /** \returns the number of non linearly dependent columns as determined by the pivoting threshold.
    *
    * \sa setPivotThreshold()
    */
    Index rank() const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      return m_nonzeropivots;
    }

    /** \returns the matrix Q
    */
    BlockAngularSparseQRMatrixQReturnType<BlockAngularSparseQR> matrixQ() const
    {
      return BlockAngularSparseQRMatrixQReturnType<BlockAngularSparseQR>(*this); // xxawf pass pointer not ref
    }

    /** \returns a const reference to the column permutation P that was applied to A such that A*P = Q*R
    * It is the combination of the fill-in reducing permutation and numerical column pivoting.
    */
    const PermutationType& colsPermutation() const
    {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_outputPerm_c;
    }

    const PermutationType& rowsPermutation() const {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return this->m_rowPerm;
    }

    /** \returns A string describing the type of error.
    * This method is provided to ease debugging, not to handle errors.
    */
    std::string lastErrorMessage() const { return m_lastError; }

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
      y.resize((std::max<Index>)(cols(), y.rows()), y.cols());
      y.topRows(rank) = this->matrixR().topLeftCorner(rank, rank).template triangularView<Upper>().solve(b.topRows(rank));
      y.bottomRows(y.rows() - rank).setZero();

      // Apply the column permutation
      if (colsPermutation().size() > 0)
        dest = colsPermutation() * y.topRows(cols());
      else
        dest = y.topRows(cols());

      m_info = Success;
      return true;
    }

    /** Sets the threshold that is used to determine linearly dependent columns during the factorization.
    *
    * In practice, if during the factorization the norm of the column that has to be eliminated is below
    * this threshold, then the entire column is treated as zero, and it is moved at the end.
    */
    void setPivotThreshold(const RealScalar& threshold)
    {
      // No pivoting ...
    }

    /** \returns the solution X of \f$ A X = B \f$ using the current decomposition of A.
    *
    * \sa compute()
    */
    template<typename Rhs>
    inline const Solve<BlockAngularSparseQR, Rhs> solve(const MatrixBase<Rhs>& B) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
      return Solve<BlockAngularSparseQR, Rhs>(*this, B.derived());
    }
    template<typename Rhs>
    inline const Solve<BlockAngularSparseQR, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
      return Solve<BlockAngularSparseQR, Rhs>(*this, B.derived());
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
    
    BlockQRSolverLeft& getLeftSolver() { return m_leftSolver; }
    BlockQRSolverRight& getRightSolver() { return m_rightSolver; }

    int leftBlockRows() const {
      return this->m_leftRows;
    }
    int leftBlockCols() const {
      return this->m_leftCols;
    }

    /*********************** Expression templates for composition of the matrix R ****************************/
    /*********************************************************************************************************/
    template <typename RightBlockSolverType, typename StorageIndex, typename SrcType>
    void makeR(const int m1, const int m2, const int n1, const int n2, const SrcType &R2, const SrcType &J2, const SparseMatrix<Scalar, ColMajor, StorageIndex> &R1, const RightBlockSolverType &rightSolver, SparseMatrix<Scalar, ColMajor, StorageIndex> &Rout) {
      {
        Rout.resize(n1 + n2, m1 + m2);
        Rout.reserve(R1.nonZeros() + n1 * m2 + ((m2 - 1) * m2) / 2);
        // Columns of R1
        for (int c = 0; c < m1; c++) {
          Rout.startVec(c);
          for (typename SparseMatrix<Scalar, ColMajor, StorageIndex>::InnerIterator colIt(R1, c); colIt; ++colIt) {
            Rout.insertBack(colIt.index(), c) = colIt.value();
          }
        }
        // Columns of J2top combined with R2 + the desired column permutation on J2top
        for (int c = 0; c < m2; c++) {
          Rout.startVec(m1 + c);
          for (int r = 0; r < m1; r++) {
            Rout.insertBack(r, m1 + c) = J2(r, rightSolver.colsPermutation().indices()(c));
          }
          for (int r = m1; r <= m1 + c; r++) {
            Rout.insertBack(r, m1 + c) = R2(r - m1, c);
          }
        }
        Rout.finalize();
      }
    }
    template <typename RightBlockSolverType, typename StorageIndex>
    void makeR(const int m1, const int m2, const int n1, const int n2, const SparseMatrix<Scalar, ColMajor, StorageIndex> &R2, const SparseMatrix<Scalar, ColMajor, StorageIndex> &J2, const SparseMatrix<Scalar, ColMajor, StorageIndex> &R1, const RightBlockSolverType &rightSolver, SparseMatrix<Scalar, ColMajor, StorageIndex> &Rout) {
      {
        DenseMatrixType J2top = J2.topRows(m1);

        Rout.resize(n1 + n2, m1 + m2);
        Rout.reserve(R1.nonZeros() + n1 * m2 + R2.nonZeros());
        // Columns of R1
        for (int c = 0; c < m1; c++) {
          Rout.startVec(c);
          for (typename SparseMatrix<Scalar, ColMajor, StorageIndex>::InnerIterator colIt(R1, c); colIt; ++colIt) {
            Rout.insertBack(colIt.index(), c) = colIt.value();
          }
        }
        // Columns of J2top combined with R2 + the desired column permutation on J2top
        for (int c = 0; c < m2; c++) {
          Rout.startVec(m1 + c);
          for (int r = 0; r < m1; r++) {
            Rout.insertBack(r, m1 + c) = J2top(r, rightSolver.colsPermutation().indices()(c));
          }
          for (typename SparseMatrix<Scalar, ColMajor, StorageIndex>::InnerIterator colIt(R2, c); colIt; ++colIt) {
            Rout.insertBack(m1 + colIt.index(), m1 + c) = colIt.value();
          }
        }
        Rout.finalize();
      }
    }

    /*********************** Expressions evaluating the row permutations on demand  ****************************/
    /***********************************************************************************************************/
    template<typename SolverType, typename MatrixType, bool Inverse = false>
    static void applyRowPermutation(const SolverType &slvr, MatrixType &mat) {
      SolverType::PermutationType perm = SparseQRUtils::rowsPermutation<SolverType, typename SolverType::PermutationType>(slvr);
      if (Inverse) {
        mat = perm.inverse() * mat;
      } else {
        mat = perm * mat;
      }
    }

    template<typename SolverType, typename MatrixType>
    static void copyRowPermutation(const SolverType &slvr, MatrixType &rowPerm, const int offset, const int length) {
      SolverType::PermutationType perm = SparseQRUtils::rowsPermutation<SolverType, typename SolverType::PermutationType>(slvr);
      for (Index j = offset; j < offset + length; j++) {
        rowPerm.indices()(j, 0) = perm.indices()(j - offset, 0);
      }
    }

    /*********************** Expression templates for solving of the right block *****************************/
    /*********************************************************************************************************/
    template <typename RightBlockSolver, typename LeftBlockSolver, typename StorageIndex, typename MatType>
    void solveRightBlock(const int m1, const int m2, const int n1, const int n2, MatType &J2, const DenseMatrixType &mat, RightBlockSolver &rightSolver, LeftBlockSolver &leftSolver) {
      MatType J2toprows = mat.topRows(n1);
      applyRowPermutation<LeftBlockSolver, MatType>(leftSolver, J2toprows);

      J2.topRows(n1).noalias() = leftSolver.matrixQ().transpose() * J2toprows;
      J2.bottomRows(n2) = mat.bottomRows(n2);

      rightSolver.compute(J2.bottomRows(n1 + n2 - m1));
    }

    template <typename RightBlockSolver, typename LeftBlockSolver, typename StorageIndex, typename MatType>
    void solveRightBlock(const int m1, const int m2, const int n1, const int n2, MatType &J2, const SparseMatrix<Scalar, RowMajor, StorageIndex> &mat, RightBlockSolver &rightSolver, LeftBlockSolver &leftSolver) {
      MatType J2toprows = mat.topRows(n1).toDense();
      applyRowPermutation<LeftBlockSolver, MatType>(leftSolver, J2toprows);

      J2.topRows(n1).noalias() = leftSolver.matrixQ().transpose() * J2toprows;
      J2.bottomRows(n2) = mat.bottomRows(n2);

      rightSolver.compute(J2.bottomRows(n1 + n2 - m1));
    }

    template <typename RightBlockSolver, typename LeftBlockSolver, typename StorageIndex>
    void solveRightBlock(const int m1, const int m2, const int n1, const int n2, SparseMatrix<Scalar, ColMajor, StorageIndex> &J2, const SparseMatrix<Scalar, ColMajor, StorageIndex> &mat, RightBlockSolver &rightSolver, LeftBlockSolver &leftSolver) {
      J2 = mat;
      SparseMatrix<Scalar, RowMajor, StorageIndex> rmJ2(J2);
      SparseMatrix<Scalar, RowMajor, StorageIndex> rmJ2TopRows = rmJ2.topRows(n1);

      applyRowPermutation<LeftBlockSolver, SparseMatrix<Scalar, RowMajor, StorageIndex> >(leftSolver, rmJ2TopRows);

      SparseMatrix<Scalar, ColMajor, StorageIndex> J2TopRows = leftSolver.matrixQ().transpose() * rmJ2TopRows;
      rmJ2.topRows(n1) = J2TopRows;
      J2 = SparseMatrix<Scalar, ColMajor, StorageIndex>(rmJ2);

      SparseMatrix<Scalar> J2bot = J2.bottomRows(n1 + n2 - m1);

      rightSolver.compute(J2bot);
    }
    /*********************************************************************************************************/

  protected:
    bool m_analysisIsok;
    bool m_factorizationIsok;
    mutable ComputationInfo m_info;
    std::string m_lastError;

    MatrixRType m_R;                  // The triangular factor matrix
    PermutationType m_outputPerm_c;   // The final column permutation
    PermutationType m_rowPerm;		    // The final row permutation
    Index m_nonzeropivots;            // Number of non zero pivots found
    bool m_isQSorted;                 // whether Q is sorted or not

    Index m_leftCols;                 // Cols of first block
    Index m_leftRows;				          // Rows of the first block
                                      // Every row below the first block is treated as a part of already upper triangular block)
    BlockQRSolverLeft m_leftSolver;
    BlockQRSolverRight m_rightSolver;

    template <typename, typename > friend struct BlockAngularSparseQR_QProduct;
  };

  /** \brief Preprocessing step of a QR factorization
  *
  * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
  *
  * In this step, the fill-reducing permutation is computed and applied to the columns of A
  * and the column elimination tree is computed as well. Only the sparsity pattern of \a mat is exploited.
  *
  * \note In this step it is assumed that there is no empty row in the matrix \a mat.
  */
  template <typename BlockQRSolverLeft, typename BlockQRSolverRight>
  void BlockAngularSparseQR<BlockQRSolverLeft, BlockQRSolverRight>::analyzePattern(const BlockMatrix1x2<LeftBlockMatrixType, RightBlockMatrixType>& mat)
  {
    // The left block should be the bigger one, don't check for sparsity now
    eigen_assert(mat.leftBlock().cols() > mat.rightBlock().cols())
    // The blocks should have the same number of rows
    eigen_assert(mat.leftBlock().rows() == mat.rightBlock().rows());

    StorageIndex n = mat.cols();
    m_outputPerm_c.resize(n);
    m_outputPerm_c.indices().setLinSpaced(n, 0, StorageIndex(n - 1));

    StorageIndex m = mat.rows();
    m_rowPerm.resize(m);
    m_rowPerm.indices().setLinSpaced(m, 0, StorageIndex(m - 1));

    // Estimate the left block size based on the template arguments
    m_leftRows = mat.leftBlock().rows();
    m_leftCols = mat.leftBlock().cols();
  }

  /** \brief Performs the numerical QR factorization of the input matrix
  *
  * The function SparseQR::analyzePattern(const MatrixType&) must have been called beforehand with
  * a matrix having the same sparsity pattern than \a mat.
  *
  * \param mat The sparse column-major matrix
  */
  template <typename BlockQRSolverLeft, typename BlockQRSolverRight>
  void BlockAngularSparseQR<BlockQRSolverLeft, BlockQRSolverRight>::factorize(const BlockMatrix1x2<LeftBlockMatrixType, RightBlockMatrixType>& mat)
  {
    typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
    typedef typename MatrixType::Index Index;
    Index m1 = m_leftCols;
    Index m2 = mat.rightBlock().cols();
    Index n1 = m_leftRows;
    Index n2 = mat.rightBlock().rows() - m_leftRows;

    /// mat = | J1 J2 |
    /// J1 has m1 cols 

    /// Compute QR for simple (e.g. block diagonal) matrix J1
    m_leftSolver.compute(mat.leftBlock());
    eigen_assert(m_leftSolver.info() == Success);

    typename BlockQRSolverLeft::MatrixRType R1 = m_leftSolver.matrixR();

    /// A = Q^t * J2
    /// n x m2
    
    /// A = | Atop |      m1 x m2
    ///     | Abot |    n-m1 x m2
    // Upper part of J2 is already in the upper triangle - no need to solve
    RightBlockMatrixType J2(n1 + n2, m2);
    J2.setZero();
    solveRightBlock<BlockQRSolverRight, BlockQRSolverLeft, StorageIndex>(m1, m2, n1, n2, J2, mat.rightBlock(), m_rightSolver, m_leftSolver);
    eigen_assert(m_rightSolver.info() == Success);

    typename BlockQRSolverRight::MatrixRType R2 = m_rightSolver.matrixR();

    /// Compute final Q and R

    /// R Matrix
    /// R = | head(R1,m1) Atop*P2  |      m1 rows
    ///     | 0           R2       |
    makeR<BlockQRSolverRight, StorageIndex>(m1, m2, n1, n2, R2, J2, R1, m_rightSolver, m_R);

    // Fill cols permutation
    for (Index j = 0; j < m1; j++) {
      m_outputPerm_c.indices()(j, 0) = m_leftSolver.colsPermutation().indices()(j, 0);
    }
    for (Index j = m1; j < mat.cols(); j++) {
      m_outputPerm_c.indices()(j, 0) = Index(m1 + m_rightSolver.colsPermutation().indices()(j - m1, 0));
    }

    // fill rows permutation
    // Top block will use row permutation from the left solver
    copyRowPermutation<BlockQRSolverLeft>(m_leftSolver, m_rowPerm, 0, n1);
    copyRowPermutation<BlockQRSolverRight>(m_rightSolver, m_rowPerm, n1, n2);

    m_nonzeropivots = m_leftSolver.rank() + m_rightSolver.rank();
    m_isInitialized = true;
    m_info = Success;

  }

  template <typename SparseQRType, typename Derived>
  struct BlockAngularSparseQR_QProduct : ReturnByValue<BlockAngularSparseQR_QProduct<SparseQRType, Derived> >
  {
    typedef typename SparseQRType::MatrixType MatrixType;
    typedef typename SparseQRType::Scalar Scalar;

    // Get the references 
    BlockAngularSparseQR_QProduct(const SparseQRType& qr, const Derived& other, bool transpose) :
      m_qr(qr), m_other(other), m_transpose(transpose) {}

    inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }

    inline Index cols() const { return m_other.cols(); }

    // Assign to a vector
    template<typename DesType>
    void evalTo(DesType& res) const
    {
      Index n = m_qr.rows();
      Index m1 = m_qr.m_leftCols;
      Index n1 = m_qr.m_leftRows;

      eigen_assert(n == m_other.rows() && "Non conforming object sizes");

      if (m_transpose)
      {
        /// Q' Matrix
        /// Q = | I 0   | * Q1'    | m1xm1    0              | * n x n 
        ///     | 0 Q2' |          |     0   (n-m1)x(n-m1)   |           

        /// Q v = | I 0   | * Q1' * v   = | I 0   | * [ Q1tv1 ]  = [ Q1tv1       ]
        ///       | 0 Q2' |               | 0 Q2' |   [ Q1tv2 ]    [ Q2' * Q1tv2 ]    

        res = m_other;
        // jasvob FixMe: The multipliation has to be split on 3 lines like this in order for the Eigen type inference to work well.
        MatrixType otherTopRows = m_other.topRows(n1);
        MatrixType resTopRows = m_qr.m_leftSolver.matrixQ().transpose() * otherTopRows;
        res.topRows(n1) = resTopRows;
        MatrixType resBottomRows = res.bottomRows(n - m1);
        applyRowPermutation<SparseQRType::BlockQRSolverRight, MatrixType>(m_qr.m_rightSolver, resBottomRows);
        MatrixType Q2v2 = m_qr.m_rightSolver.matrixQ().transpose() * resBottomRows;
        res.bottomRows(n - m1) = Q2v2;
      }
      else
      {
        /// Q Matrix 
        /// Q = Q1 * | I 0  |     n x n * | m1xm1    0            |
        ///          | 0 Q2 |             |     0   (n-m1)x(n-m1) |

        /// Q v = Q1 * | I 0  | * | v1 | =  Q1 * | v1      | 
        ///            | 0 Q2 |   | v2 |         | Q2 * v2 | 

        res = m_other;
        MatrixType resBottomRows = res.bottomRows(n - m1);
        MatrixType Q2v2 = m_qr.m_rightSolver.matrixQ() * resBottomRows;
        applyRowPermutation<SparseQRType::BlockQRSolverRight, MatrixType, true>(m_qr.m_rightSolver, Q2v2);
        res.bottomRows(n - m1) = Q2v2;
        res.topRows(n1) = m_qr.m_leftSolver.matrixQ() * res.topRows(n1);
      }
    }

    const SparseQRType& m_qr;
    const Derived& m_other;
    bool m_transpose;
  };

  template <typename SparseQRType>
  struct BlockAngularSparseQR_QProduct<SparseQRType, typename SparseQRType::DenseVectorType> : ReturnByValue<BlockAngularSparseQR_QProduct<SparseQRType, typename SparseQRType::DenseVectorType> >
  {
    typedef typename SparseQRType::MatrixType MatrixType;
    typedef typename SparseQRType::Scalar Scalar;
    typedef typename SparseQRType::DenseVectorType DenseVectorType;

    // Get the references 
    BlockAngularSparseQR_QProduct(const SparseQRType& qr, const DenseVectorType& other, bool transpose) :
      m_qr(qr), m_other(other), m_transpose(transpose) {}

    inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }

    inline Index cols() const { return m_other.cols(); }

    // Assign to a vector
    template<typename DesType>
    void evalTo(DesType& res) const
    {
      Index n = m_qr.rows();
      Index m1 = m_qr.m_leftCols;
      Index n1 = m_qr.m_leftRows;

      eigen_assert(n == m_other.rows() && "Non conforming object sizes");

      if (m_transpose)
      {
        /// Q' Matrix
        /// Q = | I 0   | * Q1'    | m1xm1    0              | * n x n 
        ///     | 0 Q2' |          |     0   (n-m1)x(n-m1)   |           

        /// Q v = | I 0   | * Q1' * v   = | I 0   | * [ Q1tv1 ]  = [ Q1tv1       ]
        ///       | 0 Q2' |               | 0 Q2' |   [ Q1tv2 ]    [ Q2' * Q1tv2 ]    

        res = m_other;
        // jasvob FixMe: The multipliation has to be split on 3 lines like this in order for the Eigen type inference to work well. 
        DenseVectorType otherTopRows = m_other.topRows(n1);
        DenseVectorType resTopRows = m_qr.m_leftSolver.matrixQ().transpose() * otherTopRows;
        res.topRows(n1) = resTopRows;
        DenseVectorType resBottomRows = res.bottomRows(n - m1);
        SparseQRType::applyRowPermutation<SparseQRType::BlockQRSolverRight, DenseVectorType>(m_qr.m_rightSolver, resBottomRows);
        DenseVectorType Q2v2 = m_qr.m_rightSolver.matrixQ().transpose() * resBottomRows;
        res.bottomRows(n - m1) = Q2v2;
      }
      else
      {
        /// Q Matrix 
        /// Q = Q1 * | I 0  |     n x n * | m1xm1    0            |
        ///          | 0 Q2 |             |     0   (n-m1)x(n-m1) |

        /// Q v = Q1 * | I 0  | * | v1 | =  Q1 * | v1      | 
        ///            | 0 Q2 |   | v2 |         | Q2 * v2 | 

        res = m_other;
        DenseVectorType resBottomRows = res.bottomRows(n - m1);
        DenseVectorType Q2v2 = m_qr.m_rightSolver.matrixQ() * resBottomRows;
        SparseQRType::applyRowPermutation<SparseQRType::BlockQRSolverRight, DenseVectorType, true>(m_qr.m_rightSolver, Q2v2); 
        res.bottomRows(n - m1) = Q2v2;
        DenseVectorType topRows = res.topRows(n1);
        topRows = m_qr.m_leftSolver.matrixQ() * topRows;
        res.topRows(n1) = topRows;
      }
    }

    const SparseQRType& m_qr;
    const DenseVectorType& m_other;
    bool m_transpose;
  };

  template<typename SparseQRType>
  struct BlockAngularSparseQRMatrixQReturnType : public EigenBase<BlockAngularSparseQRMatrixQReturnType<SparseQRType> >
  {
    typedef typename SparseQRType::Scalar Scalar;
    typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
    enum {
      RowsAtCompileTime = Dynamic,
      ColsAtCompileTime = Dynamic
    };
    explicit BlockAngularSparseQRMatrixQReturnType(const SparseQRType& qr) : m_qr(qr) {}
    template<typename Derived>
    BlockAngularSparseQR_QProduct<SparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
    {
      return BlockAngularSparseQR_QProduct<SparseQRType, Derived>(m_qr, other.derived(), false);
    }
    template<typename _Scalar, int _Options, typename _Index>
    BlockAngularSparseQR_QProduct<SparseQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
    {
      return BlockAngularSparseQR_QProduct<SparseQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, false);
    }
    BlockAngularSparseQRMatrixQTransposeReturnType<SparseQRType> adjoint() const
    {
      return BlockAngularSparseQRMatrixQTransposeReturnType<SparseQRType>(m_qr);
    }
    inline Index rows() const { return m_qr.rows(); }
    inline Index cols() const { return m_qr.rows(); }
    // To use for operations with the transpose of Q
    BlockAngularSparseQRMatrixQTransposeReturnType<SparseQRType> transpose() const
    {
      return BlockAngularSparseQRMatrixQTransposeReturnType<SparseQRType>(m_qr);
    }

    const SparseQRType& m_qr;
  };

  template<typename SparseQRType>
  struct BlockAngularSparseQRMatrixQTransposeReturnType
  {
    explicit BlockAngularSparseQRMatrixQTransposeReturnType(const SparseQRType& qr) : m_qr(qr) {}
    template<typename Derived>
    BlockAngularSparseQR_QProduct<SparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
    {
      return BlockAngularSparseQR_QProduct<SparseQRType, Derived>(m_qr, other.derived(), true);
    }
    template<typename _Scalar, int _Options, typename _Index>
    BlockAngularSparseQR_QProduct<SparseQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
    {
      return BlockAngularSparseQR_QProduct<SparseQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, true);
    }
    const SparseQRType& m_qr;
  };

  namespace internal {

    template<typename SparseQRType>
    struct evaluator_traits<BlockAngularSparseQRMatrixQReturnType<SparseQRType> >
    {
      typedef typename SparseQRType::MatrixType MatrixType;
      typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
      typedef SparseShape Shape;
    };

    template< typename DstXprType, typename SparseQRType>
    struct Assignment<DstXprType, BlockAngularSparseQRMatrixQReturnType<SparseQRType>, internal::assign_op<typename DstXprType::Scalar, typename BlockAngularSparseQRMatrixQReturnType<SparseQRType>::Scalar>, Sparse2Sparse>
    {
      typedef BlockAngularSparseQRMatrixQReturnType<SparseQRType> SrcXprType;
      typedef typename DstXprType::Scalar Scalar;
      typedef typename DstXprType::StorageIndex StorageIndex;
      static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, typename SrcXprType::Scalar> &/*func*/)
      {
        typename DstXprType::PlainObject idMat(src.m_qr.rows(), src.m_qr.rows());
        idMat.setIdentity();
        dst = BlockAngularSparseQR_QProduct<SparseQRType, DstXprType>(src.m_qr, idMat, false);
      }
    };

    template< typename DstXprType, typename SparseQRType>
    struct Assignment<DstXprType, BlockAngularSparseQRMatrixQReturnType<SparseQRType>, internal::assign_op<typename DstXprType::Scalar, typename BlockAngularSparseQRMatrixQReturnType<SparseQRType>::Scalar>, Sparse2Dense>
    {
      typedef BlockAngularSparseQRMatrixQReturnType<SparseQRType> SrcXprType;
      typedef typename DstXprType::Scalar Scalar;
      typedef typename DstXprType::StorageIndex StorageIndex;
      static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, typename SrcXprType::Scalar> &/*func*/)
      {
        dst = src.m_qr.matrixQ() * DstXprType::Identity(src.m_qr.rows(), src.m_qr.rows());
      }
    };

  } // end namespace internal

} // end namespace Eigen

#endif