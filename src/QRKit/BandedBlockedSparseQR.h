// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef QRKIT_BANDED_BLOCKED_SPARSE_QR_H
#define QRKIT_BANDED_BLOCKED_SPARSE_QR_H

#include <ctime>
#include <typeinfo>
#include <shared_mutex>
#include <Eigen/Householder>
#include <unsupported/Eigen/SparseExtra>
#include "SparseBlockYTY.h"
#include "SparseQRUtils.h"
#include "SparseQROrdering.h"

namespace QRKit {

  template<typename MatrixType, typename BlockQRSolver, int BlockOverlap, int SuggestedBlockCols> class BandedBlockedSparseQR;
  template<typename BandedBlockedSparseQRType> struct BandedBlockedSparseQRMatrixQReturnType;
  template<typename BandedBlockedSparseQRType> struct BandedBlockedSparseQRMatrixQTransposeReturnType;
  template<typename BandedBlockedSparseQRType, typename Derived> struct BandedBlockedSparseQR_QProduct;
  namespace internal {

    // traits<BandedBlockedSparseQRMatrixQ[Transpose]>
    template <typename BandedBlockedSparseQRType> 
    struct traits<BandedBlockedSparseQRMatrixQReturnType<BandedBlockedSparseQRType> >
    {
      typedef typename BandedBlockedSparseQRType::MatrixType ReturnType;
      typedef typename ReturnType::StorageIndex StorageIndex;
      typedef typename ReturnType::StorageKind StorageKind;
      enum {
        RowsAtCompileTime = Dynamic,
        ColsAtCompileTime = Dynamic
      };
    };

    template <typename BandedBlockedSparseQRType> 
    struct traits<BandedBlockedSparseQRMatrixQTransposeReturnType<BandedBlockedSparseQRType> >
    {
      typedef typename BandedBlockedSparseQRType::MatrixType ReturnType;
    };

    template <typename BandedBlockedSparseQRType, typename Derived> 
    struct traits<BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, Derived> >
    {
      typedef typename Derived::PlainObject ReturnType;
    };

    // BandedBlockedSparseQR_traits
    template <typename T> struct BandedBlockedSparseQR_traits {  };
    template <class T, int Rows, int Cols, int Options> 
    struct BandedBlockedSparseQR_traits<Matrix<T, Rows, Cols, Options>> {
      typedef Matrix<T, Rows, 1, Options> Vector;
    };
    template <class Scalar, int Options, typename Index> 
    struct BandedBlockedSparseQR_traits<SparseMatrix<Scalar, Options, Index>> {
      typedef SparseVector<Scalar, Options> Vector;
    };
  } // End namespace internal

    /**
    * \ingroup BandedBlockedSparseQR_Module
    * \class BandedBlockedSparseQR
    * \brief Sparse Householder QR Factorization for banded matrices
    *
    * An example of block banded structure for matrix with overlapping blocks could be:
    * (in this example we could assume having blocks 7x4 with column overlap of 2)
    *
    *  X X
    *  X X
    *  X X
    *  X X
    *  X X
    *  X X
    *  X X X X
    *      X X
    *      X X
    *      X X
    *      X X
    *      X X
    *      X X
    *      X X X X
    *          X X
    *          X X
    *          X X
    *          X X
    *          X X
    *          X X
    *          X X X X
    *
    * \note This implementation is not rank revealing and uses Eigen::HouseholderQR for solving the dense blocks.
    *
    * Q is the orthogonal matrix represented as products of Householder reflectors.
    * Use matrixQ() to get an expression and matrixQ().transpose() to get the transpose.
    * You can then apply it to a vector.
    *
    * R is the sparse triangular matrix. Since the current implementation is not rank-revealing,
    * the diagonal of R can potentially contain zeros.
    *
    * \tparam _MatrixType The type of the sparse matrix A, must be a column-major SparseMatrix<>
    *
    * \implsparsesolverconcept
    *
    */

  template<typename _MatrixType, typename _BlockQRSolver, int _BlockOverlap, int _SuggestedBlockCols>
  struct SparseQRUtils::HasRowsPermutation<BandedBlockedSparseQR<_MatrixType, _BlockQRSolver, _BlockOverlap, _SuggestedBlockCols>> {
    static const bool value = true;
  };

  template<typename _MatrixType, typename _BlockQRSolver, int _BlockOverlap = Dynamic, int _SuggestedBlockCols = 2>
  class BandedBlockedSparseQR : public SparseSolverBase<BandedBlockedSparseQR<_MatrixType, _BlockQRSolver, _BlockOverlap, _SuggestedBlockCols> >
  {
  protected:
    typedef SparseSolverBase<BandedBlockedSparseQR<_MatrixType, _BlockQRSolver, _BlockOverlap, _SuggestedBlockCols> > Base;
    using Base::m_isInitialized;
  public:
    using Base::_solve_impl;
    typedef _MatrixType MatrixType;
    typedef _BlockQRSolver BlockQRSolver;
    typedef typename BlockQRSolver::MatrixType BlockMatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef Matrix<StorageIndex, Dynamic, 1> IndexVector;
    typedef Matrix<Scalar, Dynamic, 1> DenseVectorType;
    typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrixType;

    typedef BandedBlockedSparseQRMatrixQReturnType<BandedBlockedSparseQR> MatrixQType;
    typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixRType;
    typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

    typedef typename SparseQRUtils::BlockBandedMatrixInfo<StorageIndex, _SuggestedBlockCols> BlockBandedMatrixInfo;

    enum {
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

  public:
    BandedBlockedSparseQR() : m_analysisIsok(false)
    { }

    /** Construct a QR factorization of the matrix \a mat.
    *
    * \sa compute()
    */
    explicit BandedBlockedSparseQR(const MatrixType& mat) : m_analysisIsok(false)
    {
      compute(mat);
    }

    /** Computes the QR factorization of the sparse matrix \a mat which is expected to contain some blocked banded structure.
    *
    * If input pattern analysis has been successfully performed before, it won't be run again by default.
    * forcePatternAnalysis - if true, forces reruning pattern analysis of the input matrix
    * \sa analyzePattern(), factorize()
    */
    void compute(const MatrixType& mat, bool forcePatternAlaysis = false)
    {
      // If successful analysis was performed before
      if (!m_analysisIsok || forcePatternAlaysis) {
        analyzePattern(mat);
      }

      // Reset variables before the factorization
      m_isInitialized = false;
      m_factorizationIsok = false;
      m_blocksYT.clear();
      factorize(mat);
    }
    void analyzePattern(const MatrixType& mat);
    void factorize(const MatrixType& mat);

    /** \returns the number of rows of the represented matrix.
    */
    inline Index rows() const { return m_pmat.rows(); }

    /** \returns the number of columns of the represented matrix.
    */
    inline Index cols() const { return m_pmat.cols(); }

    /** \returns a const reference to the \b sparse upper triangular matrix R of the QR factorization.
    *
    * The structure of R is typically very sparse and contains upper triangular blocks on the diagonal.
    * A typical pattern of R might look like:
    *
    *
    * X X X X
    *   X X X
    *     X X X X
    *       X X X
    *         X X X X
    *           X X X
    *             X X X X
    *               X X X
    *                 X X X X
    *                   X X X
    *                     X X X X
    *                       X X X X
    *                         X X X
    *                           X X
    *                             X
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

    /** \returns an expression of the matrix Q as products of sparse Householder reflectors.
    * The common usage of this function is to apply it to a dense matrix or vector
    * \code
    * VectorXd B1;
    * // Initialize B1
    * VectorXd B2 = matrixQ() * B1;
    * \endcode
    *
    * Internally, the householder vectors are stored in blocked representation YTY and the blocks
    * are ordered. Evaluation of Q * v epxressed as sequence of products (Y * (T * (Y.transpose() * v))),
    * where v is a dense vector or similarly for the case Q * A, where A is a matrix.
    *
    * The matrix Q has about 20-50% nonzeros, but is practically never required to be expressed explicitly.
    * Instead, it is stored as series of small dense blocks YT, which are aligned w.r.t. to the real position in
    * the input sparse matrix.
    * For a 7x4 blocks, each T is going to be 4 x 4 upper triangular:
    * X X X X
    *   X X X
    *     X X
    *       X
    *
    * and corresponding Y is going to be 7 x 4 with the following structure:
    *  X
    *  X X
    *  X X X
    *  X X X X
    *  X X X X
    *  X X X X
    *  X X X X
    *
    * To get a plain SparseMatrix representation of Q:
    * \code
    * SparseMatrix<double> I(A.rows(), A.rows());
    * I.setIdentity();
    * SparseMatrix<double> Q(A.rows(), A.rows());
    * Q = BandedBlockedSparseQR<SparseMatrix<double> >(A).matrixQ() * I;
    * \endcode
    *
    */
    BandedBlockedSparseQRMatrixQReturnType<BandedBlockedSparseQR> matrixQ() const
    {
      return BandedBlockedSparseQRMatrixQReturnType<BandedBlockedSparseQR>(*this);
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
      return this->m_rowPerm;
    }

    /** \internal */
    template<typename Rhs, typename Dest>
    bool _solve_impl(const MatrixBase<Rhs> &B, MatrixBase<Dest> &dest) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "BandedBlockedSparseQR::solve() : invalid number of rows in the right hand side matrix");

      Index rank = this->rank();

      // Compute Q^T * b;
      typename Dest::PlainObject y = this->matrixQ().transpose() * B;
      typename Dest::PlainObject b = y;

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
    inline const Solve<BandedBlockedSparseQR, Rhs> solve(const MatrixBase<Rhs>& B) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "BandedBlockedSparseQR::solve() : invalid number of rows in the right hand side matrix");
      return Solve<BandedBlockedSparseQR, Rhs>(*this, B.derived());
    }
    template<typename Rhs>
    inline const Solve<BandedBlockedSparseQR, Rhs> solve(const SparseMatrixBase<Rhs>& B) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "BandedBlockedSparseQR::solve() : invalid number of rows in the right hand side matrix");
      return Solve<BandedBlockedSparseQR, Rhs>(*this, B.derived());
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
    typedef SparseMatrix<Scalar, ColMajor, StorageIndex> MatrixQStorageType;
    typedef SparseBlockYTY<Scalar, StorageIndex> SparseBlockYTYType;

    bool m_analysisIsok;
    bool m_factorizationIsok;
    mutable ComputationInfo m_info;
    MatrixQStorageType m_pmat;            // Temporary matrix
    MatrixRType m_R;                      // The triangular factor matrix
    SparseBlockYTYType m_blocksYT;		        // Sparse block matrix storage holding the dense YTY blocks of the blocked representation of Householder reflectors.
    PermutationType m_outputPerm_c;       // The final column permutation (for compatibility here, set to identity)
    PermutationType m_rowPerm;
    Index m_nonzeropivots;                // Number of non zero pivots found

                                          /*
                                          * Structures filled during sparse matrix pattern analysis.
                                          */
    BlockBandedMatrixInfo m_blockInfo;

    template <typename, typename > friend struct BandedBlockedSparseQR_QProduct;
  };

  /** \brief Preprocessing step of a QR factorization
  *
  * In this step, matrix pattern analysis is performed based on the type of the block QR solver.
  *
  * (a) BlockQRSolver::RowsAtCompileTime != Dynamic && BlockQRSolver::ColsAtCompileTime != Dynamic
  * In this case, it is assumed that the matrix has a known block banded structure with
  * constant block size and column overlaps.
  *
  * This step assumes that the user knows the structure beforehand and can specify it
  * in form of input parameters.
  * If called, the block banded pattern is automatically computed from the user input
  * and there is no need to call analyzePattern later on, and some unnecessary analysis
  * can be saved.
  *
  * (b) BlockQRSolver::RowsAtCompileTime == Dynamic && BlockQRSolver::ColsAtCompileTime == Dynamic
  * In this case, row-reordering permutation of A is computed and matrix banded structure is analyzed.
  * This is neccessary preprocessing step before the matrix factorization is carried out.
  *
  * This step assumes there is some sort of banded structure in the matrix.
  *
  * \note In this step it is assumed that there is no empty row in the matrix \a mat.
  */
  template <typename MatrixType, typename BlockQRSolver, int BlockOverlap, int SuggestedBlockCols>
  void BandedBlockedSparseQR<MatrixType, BlockQRSolver, BlockOverlap, SuggestedBlockCols>::analyzePattern(const MatrixType& mat)
  {
    typedef SparseMatrix<Scalar, RowMajor, MatrixType::StorageIndex> RowMajorMatrixType;

    /* If the BlockQRSolver solver is specified with fixed-size Matrix, it means that
    the input matrix pattern will be constant - generate it
    */
    if (BlockQRSolver::RowsAtCompileTime != Dynamic && BlockQRSolver::ColsAtCompileTime != Dynamic && BlockOverlap != Dynamic) {
      // In case we know the pattern, rows are already sorted, no permutation needed
      this->m_rowPerm.resize(mat.rows());
      this->m_rowPerm.setIdentity();
      // Same situation goes for columns
      this->m_outputPerm_c.resize(mat.cols());
      this->m_outputPerm_c.setIdentity();

      // Set the block map based on block paramters passed ot this method	
      this->m_blockInfo.fromBlockBandedPattern(mat.rows(), mat.cols(), BlockQRSolver::RowsAtCompileTime, BlockQRSolver::ColsAtCompileTime, BlockOverlap);
    }
    else {
      /* Otherwise, we need to do generic analysis of the input matrix
      */
      // Create column permutation (according to the number of nonzeros in columns
      this->m_outputPerm_c.resize(mat.cols());
      this->m_outputPerm_c.setIdentity();

      // Looking for as-banded-as-possible structure in the matrix
      AsBandedAsPossibleOrdering<StorageIndex> abapOrdering;
      RowMajorMatrixType rmMat(mat);
      abapOrdering(rmMat, this->m_rowPerm);

      // Permute if permutation found
      if (abapOrdering.hasPermutation) {
        rmMat = this->m_rowPerm * rmMat;
      }
      // Compute matrix block structure
      this->m_blockInfo(rmMat);
    }

    // Finalize analysis
    m_R.resize(mat.rows(), mat.cols());

    m_analysisIsok = true;
  }

  /** \brief Performs the numerical QR factorization of the input matrix
  *
  * The function BandedBlockedSparseQR::analyzePattern(const MatrixType&) must have been called beforehand with
  * a matrix having the same sparsity pattern as \a mat.
  *
  * \param mat The sparse column-major matrix
  */
  template <typename MatrixType, typename BlockQRSolver, int BlockOverlap, int SuggestedBlockCols>
  void BandedBlockedSparseQR<MatrixType, BlockQRSolver, BlockOverlap, SuggestedBlockCols>::factorize(const MatrixType& mat)
  {
    // Permute the input matrix using the precomputed row permutation
    m_pmat = (this->m_rowPerm * mat);

    // Triplet array for the matrix R
    QRKit::TripletArray<Scalar, typename MatrixType::Index> Rvals(2 * mat.nonZeros());

    // Dense QR solver used for each dense block 
    // jasvob ToDo: Template solver over block size
    Eigen::HouseholderQR<DenseMatrixType> houseqr;
    Index numBlocks = this->m_blockInfo.blockOrder.size();

    // Prepare the first block
    typename BlockBandedMatrixInfo::MatrixBlockInfo bi = this->m_blockInfo.blockMap.at(this->m_blockInfo.blockOrder.at(0));
    DenseMatrixType Ji = m_pmat.block(bi.idxRow, bi.idxCol, bi.numRows, bi.numCols);
    Index activeRows = bi.numRows;
    Index numZeros = 0;

    // Process all blocks
    for (Index i = 0; i < numBlocks; i++) {
      // Current block info
      bi = this->m_blockInfo.blockMap.at(this->m_blockInfo.blockOrder.at(i));

      // 1) Solve the current dense block using dense Householder QR
      houseqr.compute(Ji);

      // 2) Create matrices T and Y
      DenseMatrixType Y = DenseMatrixType::Identity(activeRows, bi.numCols);
      DenseMatrixType T = DenseMatrixType::Zero(bi.numCols, bi.numCols);
      for (int bc = 0; bc < bi.numCols; bc++) {
        Y.col(bc).segment(bc + 1, activeRows - bc - 1) = houseqr.householderQ().essentialVector(bc);
      }
      Eigen::internal::make_block_householder_triangular_factor<DenseMatrixType, DenseMatrixType, DenseVectorType>(T, Y, houseqr.hCoeffs());
      T = -T;

      // Save current Y and T. The block YTY contains a main diagonal and subdiagonal part separated by (numZeros) zero rows.
      Index diagIdx = bi.idxCol;
      m_blocksYT.insert(typename SparseBlockYTYType::Element(diagIdx, diagIdx, BlockYTY<Scalar, StorageIndex>(Y, T, diagIdx, diagIdx, numZeros)));

      // 3) Get the R part of the dense QR decomposition 
      MatrixXd V = houseqr.matrixQR().template triangularView<Upper>();
      // Update sparse R with the rows solved in this step
      int solvedRows = (i == numBlocks - 1) ? bi.numRows : this->m_blockInfo.blockMap.at(this->m_blockInfo.blockOrder.at(i + 1)).idxCol - bi.idxCol;
      for (typename MatrixType::StorageIndex br = 0; br < solvedRows; br++) {
        for (typename MatrixType::StorageIndex bc = 0; bc < bi.numCols; bc++) {
          Rvals.add_if_nonzero(diagIdx + br, bi.idxCol + bc, V(br, bc));
        }
      }

      // 4) If this is not the last block, proceed to the next block
      if (i < numBlocks - 1) {
        typename BlockBandedMatrixInfo::MatrixBlockInfo biNext = this->m_blockInfo.blockMap.at(this->m_blockInfo.blockOrder.at(i + 1));
        Index blockOverlap = (bi.idxCol + bi.numCols) - biNext.idxCol;
        Index colIncrement = bi.numCols - blockOverlap;
        activeRows = bi.numRows + biNext.numRows - colIncrement;
        numZeros = (biNext.idxRow + biNext.numRows) - activeRows - biNext.idxCol;
        numZeros = (numZeros < 0) ? 0 : numZeros;

        typename MatrixType::StorageIndex numCols = (biNext.numCols >= blockOverlap) ? biNext.numCols : blockOverlap;
        Ji = m_pmat.block(bi.idxRow + colIncrement, biNext.idxCol, activeRows, numCols).toDense();
        if (blockOverlap > 0) {
          Ji.block(0, 0, activeRows - biNext.numRows, blockOverlap) = V.block(colIncrement, colIncrement, activeRows - biNext.numRows, blockOverlap);
        }
      }
    }

    // 5) Finalize the R matrix and set factorization-related flags
    m_R.setFromTriplets(Rvals.begin(), Rvals.end());
    m_R.makeCompressed();

    m_nonzeropivots = m_R.cols();	// Assuming all cols are nonzero

    m_isInitialized = true;
    m_factorizationIsok = true;
    m_info = Success;
  }

  /*
  * General Householder product evaluation performing Q * A or Q^T * A.
  * Householder vectors are represented in compressed blocked form YT.
  * The general version is assuming that A is sparse and that the output will be sparse as well.
  * Offers single-threaded and multi-threaded implementation.
  * The choice of implementation depends on a template parameter of the BandedBlockedSparseQR class.
  * The single-threaded implementation cannot work in-place. It is implemented this way for performance related reasons.
  */
  template <typename BandedBlockedSparseQRType, typename Derived>
  struct BandedBlockedSparseQR_QProduct : ReturnByValue<BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, Derived> >
  {
    typedef typename BandedBlockedSparseQRType::MatrixType MatrixType;
    typedef typename BandedBlockedSparseQRType::Scalar Scalar;
    typedef typename BandedBlockedSparseQRType::DenseVectorType DenseVectorType;

    typedef typename internal::BandedBlockedSparseQR_traits<MatrixType>::Vector SparseVector;

    typedef std::vector<std::vector<std::pair<typename MatrixType::Index, Scalar>>> ResValsVector;

    // Get the references 
    BandedBlockedSparseQR_QProduct(const BandedBlockedSparseQRType& qr, const Derived& other, bool transpose) :
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

      // loop over all items
      #pragma omp parallel for
      for (int j = 0; j < m_other.cols(); j++)
      {
        DenseVectorType resColJd = m_other.col(j).toDense();

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

    const BandedBlockedSparseQRType& m_qr;
    const Derived& m_other;
    bool m_transpose;
  };

  /*
  * Specialization of the Householder product evaluation performing Q * A or Q.T * A
  * for the case when A and the output are dense matrices.
  */
  template <typename BandedBlockedSparseQRType>
  struct BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, typename BandedBlockedSparseQRType::DenseMatrixType> : ReturnByValue<BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, typename BandedBlockedSparseQRType::DenseMatrixType> >
  {
    typedef typename BandedBlockedSparseQRType::MatrixType MatrixType;
    typedef typename BandedBlockedSparseQRType::Scalar Scalar;
    typedef typename BandedBlockedSparseQRType::DenseVectorType DenseVectorType;
    typedef typename BandedBlockedSparseQRType::DenseMatrixType DenseMatrixType;

    typedef typename internal::BandedBlockedSparseQR_traits<MatrixType>::Vector SparseVector;

    // Get the references 
    BandedBlockedSparseQR_QProduct(const BandedBlockedSparseQRType& qr, const DenseMatrixType& other, bool transpose) :
      m_qr(qr), m_other(other), m_transpose(transpose) {}
    inline Index rows() const { return m_transpose ? m_qr.rows() : m_qr.cols(); }
    inline Index cols() const { return m_other.cols(); }

    // Assign to a vector
    template<typename DesType>
    void evalTo(DesType& res) const
    {
      Index m = m_qr.rows();
      Index n = m_qr.cols();

      res = DenseMatrixType::Zero(m_other.rows(), m_other.cols());
      #pragma omp parallel for
      for (int j = 0; j < m_other.cols(); j++)
      {
        DenseVectorType resColJd = m_other.col(j);

        if (m_transpose) {
          resColJd.noalias() = m_qr.m_blocksYT.sequenceYTY().transpose() * resColJd;
        }
        else {
          resColJd.noalias() = m_qr.m_blocksYT.sequenceYTY() * resColJd;
        }
        // Write the result back to j-th column of res
        res.col(j) = resColJd;
      }
    }

    const BandedBlockedSparseQRType& m_qr;
    const DenseMatrixType& m_other;
    bool m_transpose;
  };

  /*
  * Specialization of the Householder product evaluation performing Q * A or Q.T * A
  * for the case when A and the output are dense vectors.=
  * Offers only single-threaded implementation as the overhead of multithreading would not bring any speedup for a dense vector (A is single column).
  */
  template <typename BandedBlockedSparseQRType>
  struct BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, typename BandedBlockedSparseQRType::DenseVectorType> : ReturnByValue<BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, typename BandedBlockedSparseQRType::DenseVectorType> >
  {
    typedef typename BandedBlockedSparseQRType::MatrixType MatrixType;
    typedef typename BandedBlockedSparseQRType::Scalar Scalar;
    typedef typename BandedBlockedSparseQRType::DenseVectorType DenseVectorType;

    // Get the references 
    BandedBlockedSparseQR_QProduct(const BandedBlockedSparseQRType& qr, const DenseVectorType& other, bool transpose) :
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
      if (m_transpose)
      {
        //Compute res = Q' * other (other is vector - only one column => no iterations of j)
        res = m_qr.m_blocksYT.sequenceYTY().transpose() * res;
      }
      else
      {
        // Compute res = Q * other (other is vector - only one column => no iterations of j)
        res = m_qr.m_blocksYT.sequenceYTY() * res;
      }
    }

    const BandedBlockedSparseQRType& m_qr;
    const DenseVectorType& m_other;
    bool m_transpose;
  };

  template<typename BandedBlockedSparseQRType>
  struct BandedBlockedSparseQRMatrixQReturnType : public EigenBase<BandedBlockedSparseQRMatrixQReturnType<BandedBlockedSparseQRType> >
  {
    typedef typename BandedBlockedSparseQRType::Scalar Scalar;
    typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;
    enum {
      RowsAtCompileTime = Dynamic,
      ColsAtCompileTime = Dynamic
    };
    explicit BandedBlockedSparseQRMatrixQReturnType(const BandedBlockedSparseQRType& qr) : m_qr(qr) {}
    template<typename Derived>
    BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
    {
      return BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, Derived>(m_qr, other.derived(), false);
    }
    template<typename _Scalar, int _Options, typename _Index>
    BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
    {
      return BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, false);
    }
    BandedBlockedSparseQRMatrixQTransposeReturnType<BandedBlockedSparseQRType> adjoint() const
    {
      return BandedBlockedSparseQRMatrixQTransposeReturnType<BandedBlockedSparseQRType>(m_qr);
    }
    inline Index rows() const { return m_qr.rows(); }
    inline Index cols() const { return m_qr.rows(); }
    // To use for operations with the transpose of Q
    BandedBlockedSparseQRMatrixQTransposeReturnType<BandedBlockedSparseQRType> transpose() const
    {
      return BandedBlockedSparseQRMatrixQTransposeReturnType<BandedBlockedSparseQRType>(m_qr);
    }

    const BandedBlockedSparseQRType& m_qr;
  };

  template<typename BandedBlockedSparseQRType>
  struct BandedBlockedSparseQRMatrixQTransposeReturnType
  {
    explicit BandedBlockedSparseQRMatrixQTransposeReturnType(const BandedBlockedSparseQRType& qr) : m_qr(qr) {}
    template<typename Derived>
    BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
    {
      return BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, Derived>(m_qr, other.derived(), true);
    }
    template<typename _Scalar, int _Options, typename _Index>
    BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
    {
      return BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, SparseMatrix<_Scalar, _Options, _Index>>(m_qr, other, true);
    }
    const BandedBlockedSparseQRType& m_qr;
  };

  namespace internal {

    template<typename BandedBlockedSparseQRType>
    struct evaluator_traits<BandedBlockedSparseQRMatrixQReturnType<BandedBlockedSparseQRType> >
    {
      typedef typename BandedBlockedSparseQRType::MatrixType MatrixType;
      typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
      typedef SparseShape Shape;
    };

    template< typename DstXprType, typename BandedBlockedSparseQRType>
    struct Assignment<DstXprType, BandedBlockedSparseQRMatrixQReturnType<BandedBlockedSparseQRType>, internal::assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Sparse2Sparse>
    {
      typedef BandedBlockedSparseQRMatrixQReturnType<BandedBlockedSparseQRType> SrcXprType;
      typedef typename DstXprType::Scalar Scalar;
      typedef typename DstXprType::StorageIndex StorageIndex;
      static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar, Scalar> &/*func*/)
      {
        typename DstXprType::PlainObject idMat(src.m_qr.rows(), src.m_qr.rows());
        idMat.setIdentity();
        dst = BandedBlockedSparseQR_QProduct<BandedBlockedSparseQRType, DstXprType>(src.m_qr, idMat, false);
      }
    };

    template< typename DstXprType, typename BandedBlockedSparseQRType>
    struct Assignment<DstXprType, BandedBlockedSparseQRMatrixQReturnType<BandedBlockedSparseQRType>, internal::assign_op<typename DstXprType::Scalar, typename DstXprType::Scalar>, Sparse2Dense>
    {
      typedef BandedBlockedSparseQRMatrixQReturnType<BandedBlockedSparseQRType> SrcXprType;
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