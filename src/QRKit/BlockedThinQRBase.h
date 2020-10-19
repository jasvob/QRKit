// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef QRKIT_BLOCKED_THIN_QR_BASE_H
#define QRKIT_BLOCKED_THIN_QR_BASE_H

#include <ctime>
#include <typeinfo>
#include <shared_mutex>
#include <Eigen/Householder>
#include "SparseBlockYTY.h"
#include "SparseQRUtils.h"
#include "SparseQROrdering.h"

namespace QRKit {

  template<typename MatrixType, typename DenseBlockQR, typename MatrixRType, int SuggestedBlockCols> class BlockedThinQRBase;
  template<typename BlockedThinQRBaseType> struct BlockedThinQRBaseMatrixQReturnType;
  template<typename BlockedThinQRBaseType> struct BlockedThinQRBaseMatrixQTransposeReturnType;
  template<typename BlockedThinQRBaseType, typename Derived> struct BlockedThinQRBase_QProduct;
  namespace internal {

    // traits<BlockedThinQRBaseMatrixQ[Transpose]>
    template <typename BlockedThinQRBaseType> struct traits<BlockedThinQRBaseMatrixQReturnType<BlockedThinQRBaseType> >
    {
      typedef typename BlockedThinQRBaseType::MatrixType ReturnType;
      typedef typename ReturnType::StorageIndex StorageIndex;
      typedef typename ReturnType::StorageKind StorageKind;
      enum {
        RowsAtCompileTime = Dynamic,
        ColsAtCompileTime = Dynamic
      };
    };

    template <typename BlockedThinQRBaseType> struct traits<BlockedThinQRBaseMatrixQTransposeReturnType<BlockedThinQRBaseType> >
    {
      typedef typename BlockedThinQRBaseType::MatrixType ReturnType;
    };

    template <typename BlockedThinQRBaseType, typename Derived> struct traits<BlockedThinQRBase_QProduct<BlockedThinQRBaseType, Derived> >
    {
      typedef typename Derived::PlainObject ReturnType;
    };

    // BlockedThinQRBase_traits
    template <typename T> struct BlockedThinQRBase_traits {  };
    template <class T, int Rows, int Cols, int Options> struct BlockedThinQRBase_traits<Matrix<T, Rows, Cols, Options>> {
      typedef Matrix<T, Rows, 1, Options> Vector;
    };
    template <class Scalar, int Options, typename Index> struct BlockedThinQRBase_traits<SparseMatrix<Scalar, Options, Index>> {
      typedef SparseVector<Scalar, Options> Vector;
    };
  } // End namespace internal

    /**
    * \ingroup BlockedThinQRBase_Module
    * \class BlockedThinQRBase
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
    * \implsparsesolverconcept
    *
    * \warning The input sparse matrix A must be in compressed mode (see SparseMatrix::makeCompressed()).
    *
    */

  template<typename _MatrixType, typename _BlockQRSolver, typename _MatrixRType, int _SuggestedBlockCols>
  struct SparseQRUtils::HasRowsPermutation<BlockedThinQRBase<_MatrixType, _BlockQRSolver, _MatrixRType, _SuggestedBlockCols>> {
    static const bool value = true;
  };

  template<typename _MatrixType, typename _DenseBlockQR, typename _MatrixRType, int _SuggestedBlockCols = 2>
  class BlockedThinQRBase : public SparseSolverBase<BlockedThinQRBase<_MatrixType, _DenseBlockQR, _MatrixRType, _SuggestedBlockCols> >
  {
  protected:
    typedef SparseSolverBase<BlockedThinQRBase<_MatrixType, _DenseBlockQR, _MatrixRType, _SuggestedBlockCols> > Base;
    using Base::m_isInitialized;
  public:
    using Base::_solve_impl;
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;
    typedef Matrix<Scalar, Dynamic, 1> DenseVectorType;
    typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrixType;

    typedef BlockedThinQRBaseMatrixQReturnType<BlockedThinQRBase> MatrixQType;
    typedef _MatrixRType MatrixRType;
    typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

    typedef _DenseBlockQR DenseBlockQR;

    /*
    * Stores information about a dense block in a block sparse matrix.
    * Holds the position of the block (row index, column index) and its size (number of rows, number of columns).
    */
    typedef typename SparseQRUtils::BlockBandedMatrixInfo<StorageIndex, _SuggestedBlockCols> BlockBandedMatrixInfo;

    enum {
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

  public:
    BlockedThinQRBase() : m_analysisIsok(false)
    { }

    /** Construct a QR factorization of the matrix \a mat.
    *
    * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
    *
    * \sa compute()
    */
    explicit BlockedThinQRBase(const MatrixType& mat) : m_analysisIsok(false)
    {
      compute(mat);
    }

    /** Computes the QR factorization of the sparse matrix \a mat.
    *
    * \warning The matrix \a mat must be in compressed mode (see SparseMatrix::makeCompressed()).
    *
    * If input pattern analysis has been successfully performed before, it won't be run again by default.
    * forcePatternAnalysis - if true, forces reruning pattern analysis of the input matrix
    * \sa analyzePattern(), factorize()
    */
    virtual void compute(const MatrixType& mat) = 0;
    virtual void analyzePattern(const MatrixType& mat, bool rowPerm = true, bool colPerm = true) = 0;
    virtual void updateBlockInfo(const Index solvedCols, const MatrixType& mat, const Index newPivots, const Index blockCols = -1) = 0;
    virtual void factorize(DenseMatrixType& mat) = 0;

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
      return m_nonzeroPivots;
    }

    /** \returns an expression of the matrix Q as products of sparse Householder reflectors.
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
    * Q = BlockedThinQRBase<SparseMatrix<double> >(A).matrixQ();
    * \endcode
    * Internally, this call simply performs a sparse product between the matrix Q
    * and a sparse identity matrix. However, due to the fact that the sparse
    * reflectors are stored unsorted, two transpositions are needed to sort
    * them before performing the product.
    */
    BlockedThinQRBaseMatrixQReturnType<BlockedThinQRBase> matrixQ() const
    {
      return BlockedThinQRBaseMatrixQReturnType<BlockedThinQRBase>(*this);
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
        
    /** \internal */
    template<typename Rhs, typename Dest>
    bool _solve_impl(const MatrixBase<Rhs> &B, MatrixBase<Dest> &dest) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "BlockedThinQRBase::solve() : invalid number of rows in the right hand side matrix");

      Index rank = this->rank();

      // Compute Q^T * b;
      typename Dest::PlainObjecty = this->matrixQ().transpose() * B;
      typename Dest::PlainObjectb = y;

      // Solve with the triangular matrix R
      y.resize((std::max<Index>)(cols(), y.rows()), y.cols());
      y.topRows(rank) = this->matrixR().topLeftCorner(rank, rank).template triangularView<Upper>().solve(b.topRows(rank));
      y.bottomRows(y.rows() - rank).setZero();

      dest = y.topRows(cols());

      m_info = Success;
      return true;
    }
    
    /** \returns the solution X of \f$ A X = B \f$ using the current decomposition of A.
    *
    * \sa compute()
    */
    template<typename Rhs>
    inline const Solve<BlockedThinQRBase, Rhs> solve(const MatrixBase<Rhs>& B) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "BlockedThinQRBase::solve() : invalid number of rows in the right hand side matrix");
      return Solve<BlockedThinQRBase, Rhs>(*this, B.derived());
    }
    template<typename Rhs>
    inline const Solve<BlockedThinQRBase, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "BlockedThinQRBase::solve() : invalid number of rows in the right hand side matrix");
      return Solve<BlockedThinQRBase, Rhs>(*this, B.derived());
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
    typedef SparseBlockYTY<Scalar, StorageIndex> SparseBlockYTYType;
    
    mutable ComputationInfo m_info;
      
    MatrixRType m_R;              // The triangular factor matrix
    SparseBlockYTYType m_blocksYT;    // Sparse block matrix storage holding the dense YTY blocks of the blocked representation of Householder reflectors.

    PermutationType m_outputPerm_c;   // The final column permutation (for compatibility here, set to identity)
    PermutationType m_rowPerm;

    Index m_nonzeroPivots;            // Number of non zero pivots found

    bool m_analysisIsok;
    bool m_factorizationIsok;

    /*
    * Structures filled during sparse matrix pattern analysis.
    */
    typename BlockBandedMatrixInfo::MatrixBlockInfo denseBlockInfo;

    template <typename, typename > friend struct BlockedThinQRBase_QProduct;

    void updateMat(const Index &fromIdx, const Index &toIdx, DenseMatrixType &mat, const Index &blockK = -1);
    void computeBlockedRepresentation(const DenseBlockQR &slvr, DenseMatrixType &Y, DenseMatrixType &T);
  };

  template <typename MatrixType, typename DenseBlockQR, typename MatrixRType, int SuggestedBlockCols>
  void BlockedThinQRBase<MatrixType, DenseBlockQR, MatrixRType, SuggestedBlockCols>::updateMat(const Index &fromIdx, const Index &toIdx, DenseMatrixType &mat, const Index &blockK) {
    // Now update the unsolved rest of m_pmat
    Index blockRows = this->m_blocksYT[blockK].value.rows();

    // loop over all items
    #pragma omp parallel for
    for (int j = fromIdx; j < toIdx; j++) {
        mat.middleRows(this->denseBlockInfo.idxRow, blockRows).col(j).noalias()
          += (this->m_blocksYT[blockK].value.Y() * (this->m_blocksYT[blockK].value.T().transpose() * (this->m_blocksYT[blockK].value.Y().transpose() * mat.middleRows(this->denseBlockInfo.idxRow, blockRows).col(j))));
    }
  }

  template <typename MatrixType, typename DenseBlockQR, typename MatrixRType, int SuggestedBlockCols>
  void BlockedThinQRBase<MatrixType, DenseBlockQR, MatrixRType, SuggestedBlockCols>::computeBlockedRepresentation(const DenseBlockQR &slvr, DenseMatrixType &Y, DenseMatrixType &T) {
    Index numRows = this->denseBlockInfo.numRows;
    Index numCols = this->denseBlockInfo.numCols;

    T = DenseMatrixType::Zero(numCols, numCols);
    Y = DenseMatrixType::Identity(numRows, numCols);
    for (int bc = 0; bc < numCols; bc++) {
      Y.col(bc).segment(bc + 1, numRows - bc - 1) = slvr.householderQ().essentialVector(bc);
    }
    Eigen::internal::make_block_householder_triangular_factor<DenseMatrixType, DenseMatrixType, DenseVectorType>(T, Y, slvr.hCoeffs());
    T = -T;
  }

  /*
  * General Householder product evaluation performing Q * A or Q.T * A.
  * The general version is assuming that A is sparse and that the output will be sparse as well.
  * Offers single-threaded and multi-threaded implementation.
  * The choice of implementation depends on a template parameter of the BlockedThinQRBase class.
  * The single-threaded implementation cannot work in-place. It is implemented this way for performance related reasons.
  */
  template <typename BlockedThinQRBaseType, typename Derived>
  struct BlockedThinQRBase_QProduct : ReturnByValue<BlockedThinQRBase_QProduct<BlockedThinQRBaseType, Derived> >
  {
    typedef typename SparseMatrix<Scalar, ColMajor, Index> MatrixType;
    typedef typename BlockedThinQRBaseType::MatrixType DenseMatrixType;
    typedef typename BlockedThinQRBaseType::DenseVectorType DenseVectorType;
    typedef typename BlockedThinQRBaseType::Scalar Scalar;

    typedef typename internal::BlockedThinQRBase_traits<MatrixType>::Vector SparseVector;

    typedef std::vector<std::vector<std::pair<typename MatrixType::Index, Scalar>>> ResValsVector;

    // Get the references 
    BlockedThinQRBase_QProduct(const BlockedThinQRBaseType& qr, const Derived& other, bool transpose) :
      m_qr(qr), m_other(other), m_transpose(transpose) {}
    inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }
    inline Index cols() const { return m_other.cols(); }

    // Assign to a vector
    template<typename DesType>
    void evalTo(DesType& res) const
    {
      Index m = m_qr.rows();
      Index n = m_qr.cols();

      Eigen::DynamicSparseMatrix<typename Derived::Scalar, 0, typename Derived::Index> tmp(m_other.rows(), m_other.cols());
      Index numNonZeros = 0;

      #pragma omp parallel for
      // loop over all items
      for (int j = 0; j<m_other.cols(); j++)
      {
        DenseVectorType resColJd = m_other.col(j);

        if (m_transpose) {
          resColJd.noalias() = m_qr.m_blocksYT.sequenceYTY().transpose() * resColJd;
        }
        else {
          resColJd.noalias() = m_qr.m_blocksYT.sequenceYTY() * resColJd;
        }

        // Write the result back to j-th column of res
        SparseVector resColJ = resColJd.sparseView();
        for (SparseVector::InnerIterator it(resColJ); it; ++it) {
          tmp.coeffRef(it.row(), j) = it.value();
        }
      }
      res = tmp;

      // Don't forget to call finalize
      res.finalize();
    }

    const BlockedThinQRBaseType& m_qr;
    const Derived& m_other;
    bool m_transpose;
  };

  /*
  * Specialization of the Householder product evaluation performing Q * A or Q.T * A
  * for the case when A and the output are dense vectors.=
  * Offers only single-threaded implementation as the overhead of multithreading would not bring any speedup for a dense vector (A is single column).
  */

  template <typename BlockedThinQRBaseType>
  struct BlockedThinQRBase_QProduct<BlockedThinQRBaseType, typename BlockedThinQRBaseType::DenseVectorType> : ReturnByValue<BlockedThinQRBase_QProduct<BlockedThinQRBaseType, typename BlockedThinQRBaseType::DenseVectorType> >
  {
    typedef typename BlockedThinQRBaseType::MatrixType MatrixType;
    typedef typename BlockedThinQRBaseType::Scalar Scalar;
    typedef typename BlockedThinQRBaseType::DenseVectorType DenseVectorType;

    // Get the references 
    BlockedThinQRBase_QProduct(const BlockedThinQRBaseType& qr, const DenseVectorType& other, bool transpose) :
      m_qr(qr), m_other(other), m_transpose(transpose) {}
    inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }
    inline Index cols() const { return m_other.cols(); }

    // Assign to a vector
    template<typename DesType>
    void evalTo(DesType& res) const
    {
      Index m = m_qr.rows();
      Index n = m_qr.cols();
      res = m_other;

      //Compute res = Q' * other (other is vector - only one column => no iterations of j)
      if (m_transpose) {
        res.noalias() = m_qr.m_blocksYT.sequenceYTY().transpose() * res;
      } else {
        res.noalias() = m_qr.m_blocksYT.sequenceYTY() * res;
      }
    }

    const BlockedThinQRBaseType& m_qr;
    const DenseVectorType& m_other;
    bool m_transpose;
  };

  template<typename BlockedThinQRBaseType>
  struct BlockedThinQRBaseMatrixQReturnType : public EigenBase<BlockedThinQRBaseMatrixQReturnType<BlockedThinQRBaseType> >
  {
    typedef typename BlockedThinQRBaseType::Scalar Scalar;
    typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
    enum {
      RowsAtCompileTime = Dynamic,
      ColsAtCompileTime = Dynamic
    };
    explicit BlockedThinQRBaseMatrixQReturnType(const BlockedThinQRBaseType& qr) : m_qr(qr) {}
    /*BlockedThinQRBase_QProduct<BlockedThinQRBaseType, Derived> operator*(const MatrixBase<Derived>& other)
    {
    return BlockedThinQRBase_QProduct<BlockedThinQRBaseType,Derived>(m_qr,other.derived(),false);
    }*/
    template<typename Derived>
    BlockedThinQRBase_QProduct<BlockedThinQRBaseType, Derived> operator*(const MatrixBase<Derived>& other)
    {
      return BlockedThinQRBase_QProduct<BlockedThinQRBaseType, Derived>(m_qr, other.derived(), false);
    }
    template<typename _Scalar, int _Options, typename _Index>
    BlockedThinQRBase_QProduct<BlockedThinQRBaseType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
    {
      return BlockedThinQRBase_QProduct<BlockedThinQRBaseType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, false);
    }
    BlockedThinQRBaseMatrixQTransposeReturnType<BlockedThinQRBaseType> adjoint() const
    {
      return BlockedThinQRBaseMatrixQTransposeReturnType<BlockedThinQRBaseType>(m_qr);
    }
    inline Index rows() const { return m_qr.rows(); }
    inline Index cols() const { return m_qr.rows(); }
    // To use for operations with the transpose of Q
    BlockedThinQRBaseMatrixQTransposeReturnType<BlockedThinQRBaseType> transpose() const
    {
      return BlockedThinQRBaseMatrixQTransposeReturnType<BlockedThinQRBaseType>(m_qr);
    }

    const BlockedThinQRBaseType& m_qr;
  };

  template<typename BlockedThinQRBaseType>
  struct BlockedThinQRBaseMatrixQTransposeReturnType
  {
    explicit BlockedThinQRBaseMatrixQTransposeReturnType(const BlockedThinQRBaseType& qr) : m_qr(qr) {}
    template<typename Derived>
    BlockedThinQRBase_QProduct<BlockedThinQRBaseType, Derived> operator*(const MatrixBase<Derived>& other)
    {
      return BlockedThinQRBase_QProduct<BlockedThinQRBaseType, Derived>(m_qr, other.derived(), true);
    }
    template<typename _Scalar, int _Options, typename _Index>
    BlockedThinQRBase_QProduct<BlockedThinQRBaseType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
    {
      return BlockedThinQRBase_QProduct<BlockedThinQRBaseType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, true);
    }
    const BlockedThinQRBaseType& m_qr;
  };

  namespace internal {

    template<typename BlockedThinQRBaseType>
    struct evaluator_traits<BlockedThinQRBaseMatrixQReturnType<BlockedThinQRBaseType> >
    {
      typedef typename BlockedThinQRBaseType::MatrixType MatrixType;
      typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
      typedef SparseShape Shape;
    };

    template< typename DstXprType, typename BlockedThinQRBaseType>
    struct Assignment<DstXprType, BlockedThinQRBaseMatrixQReturnType<BlockedThinQRBaseType>, internal::assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Sparse2Sparse>
    {
      typedef BlockedThinQRBaseMatrixQReturnType<BlockedThinQRBaseType> SrcXprType;
      typedef typename DstXprType::Scalar Scalar;
      typedef typename DstXprType::StorageIndex StorageIndex;
      static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, Scalar> &/*func*/)
      {
        typename DstXprType::PlainObject idMat(src.m_qr.rows(), src.m_qr.rows());
        idMat.setIdentity();
        dst = BlockedThinQRBase_QProduct<BlockedThinQRBaseType, DstXprType>(src.m_qr, idMat, false);
      }
    };

    template< typename DstXprType, typename BlockedThinQRBaseType>
    struct Assignment<DstXprType, BlockedThinQRBaseMatrixQReturnType<BlockedThinQRBaseType>, internal::assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Sparse2Dense>
    {
      typedef BlockedThinQRBaseMatrixQReturnType<BlockedThinQRBaseType> SrcXprType;
      typedef typename DstXprType::Scalar Scalar;
      typedef typename DstXprType::StorageIndex StorageIndex;
      static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, Scalar> &/*func*/)
      {
        dst = src.m_qr.matrixQ() * DstXprType::Identity(src.m_qr.rows(), src.m_qr.rows());
      }
    };

  } // end namespace internal

} // end namespace QRKit

#endif
