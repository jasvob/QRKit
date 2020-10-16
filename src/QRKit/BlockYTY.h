// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef BLOCK_YTY_H
#define BLOCK_YTY_H

#include <Eigen/Eigen>

/*
* A dense block of the compressed WY representation (YTY) of the Householder product.
* Stores matrices Y (m x n) and T (n x n) and number of zeros between main diagonal and subdiagonal parts of the block YTY.
* Provides overloaded multiplication operator (*) allowing to easily perform the multiplication with a dense vector (Y * (T * (Y' * v)))
*/
namespace QRKit {
  template <typename ScalarType, typename IndexType> class BlockYTY;
  template<typename BlockYTYType> struct BlockYTYProductReturnType;
  template<typename BlockYTYType> struct BlockYTYProductTransposeReturnType;
  template<typename BlockYTYType, typename Derived> struct BlockYTY_VecProduct;

  namespace internal {

    // traits<BlockYTYProduct[Transpose]>
    template <typename BlockYTYType> struct traits<BlockYTYProductReturnType<BlockYTYType> >
    {
      typedef typename BlockYTYType::MatrixType ReturnType;
      typedef typename ReturnType::StorageIndex StorageIndex;
      typedef typename ReturnType::StorageKind StorageKind;
      enum {
        RowsAtCompileTime = Dynamic,
        ColsAtCompileTime = Dynamic
      };
    };

    template <typename BlockYTYType> struct traits<BlockYTYProductTransposeReturnType<BlockYTYType> >
    {
      typedef typename BlockYTYType::MatrixType ReturnType;
    };

    template <typename BlockYTYType, typename Derived> struct traits<BlockYTY_VecProduct<BlockYTYType, Derived> >
    {
      typedef typename Derived::PlainObject ReturnType;
    };

    // BlockYTY_traits
    template <typename T> struct BlockYTY_traits {  };

  } // End namespace internal

  template <typename BlockYTY>
  class BlockYTYTranspose {
  public:
    typedef typename BlockYTY::VectorType VectorType;

    BlockYTYTranspose() : yty(NULL) {
    }

    BlockYTYTranspose(BlockYTY *blk) : yty(blk) {
    }

    VectorType operator*(const VectorType &other) const {
      return (yty->Y() * (yty->T().transpose() * (yty->Y().transpose() * other)));
    }

  private:
    BlockYTY *yty;
  };

  template <typename ScalarType, typename IndexType>
  class BlockYTY {
  public:
    typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
    typedef ScalarType Scalar;
    typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> VectorType;

    BlockYTY() {
    }

    BlockYTY(const MatrixType &Y, const MatrixType &T, const IndexType row, const IndexType col, const IndexType numZeros)
      : matY(Y), matT(T), ridx(row), cidx(col), nzrs(numZeros) {

    }

    const MatrixType& Y() const {
      return this->matY;
    }

    const MatrixType& T() const {
      return this->matT;
    }

    IndexType rowIndex() const {
      return this->ridx;
    }
    IndexType colIndex() const {
      return this->cidx;
    }

    IndexType rows() const {
      return this->matY.rows();
    }
    IndexType cols() const {
      return this->matY.cols();
    }

    IndexType numZeros() const {
      return this->nzrs;
    }

    BlockYTYTranspose<BlockYTY<ScalarType, IndexType>> transpose() const {
      return BlockYTYTranspose<BlockYTY<ScalarType, IndexType>>(this);
    }
    
    VectorType operator*(const VectorType &other) const {
      return (this->matY * (this->matT * (this->matY.transpose() * other)));
    }

    BlockYTYProductReturnType<BlockYTY> yty() const {
      return BlockYTYProductReturnType<BlockYTY>(*this);
    }

  protected:
    MatrixType matY;
    MatrixType matT;

    IndexType ridx;
    IndexType cidx;
    IndexType nzrs;
    
    template <typename, typename > friend struct BlockYTY_VecProduct;
  };

  /************************************ Expression templates for the evaluation of single YTY product ***************************************/
  template <typename BlockYTYType, typename Derived>
  struct BlockYTY_VecProduct : Eigen::ReturnByValue<BlockYTY_VecProduct<BlockYTYType, Derived> >
  {
    // Get the references 
    BlockYTY_VecProduct(const BlockYTYType& yty, const Derived& other, bool transpose) :
      m_yty(yty), m_other(other), m_transpose(transpose) {}
    inline Index rows() const { return m_other.rows(); }
    inline Index cols() const { return m_other.cols(); }

    // Assign to a vector
    template<typename DesType>
    void evalTo(DesType& res) const
    {
      res = m_other;

      Derived segmentVec;
      SparseQRUtils::SegmentDescriptors segDescs = {
        { m_yty.rowIndex(), m_yty.cols() },
        { m_yty.rowIndex() + m_yty.cols() + m_yty.numZeros(), m_yty.rows() - m_yty.cols() }
      };
      SparseQRUtils::getVectorSegments<Derived, 2>(segmentVec, res, segDescs, m_yty.rows());
      if (m_transpose) {
        // Non-aliasing expr -> noalias()
        segmentVec.noalias() += (m_yty.matY * (m_yty.matT.transpose() * (m_yty.matY.transpose() * segmentVec)));
      }
      else {
        // Non-aliasing expr -> noalias()
        segmentVec.noalias() += (m_yty.matY * (m_yty.matT * (m_yty.matY.transpose() * segmentVec)));
      }
      SparseQRUtils::setVectorSegments<Derived, 2>(res, segmentVec, segDescs);
    }

    const BlockYTYType& m_yty;
    const Derived& m_other;
    bool m_transpose;
  };

  template <typename BlockYTYType>
  struct BlockYTYProductReturnType : public EigenBase<BlockYTYProductReturnType<BlockYTYType> >
  {
    typedef typename BlockYTYType::Scalar Scalar;
    typedef typename BlockYTYType::MatrixType DenseMatrix;
    enum {
      RowsAtCompileTime = Dynamic,
      ColsAtCompileTime = Dynamic
    };
    explicit BlockYTYProductReturnType(const BlockYTYType& yty) : m_yty(yty) {}

    template<typename Derived>
    BlockYTY_VecProduct<BlockYTYType, Derived> operator*(const MatrixBase<Derived>& other)
    {
      return BlockYTY_VecProduct<BlockYTYType, Derived>(m_yty, other.derived(), false);
    }
    template<typename _Scalar, int _Options, typename _Index>
    BlockYTY_VecProduct<BlockYTYType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
    {
      return BlockYTY_VecProduct<BlockYTYType, SparseMatrix<_Scalar, _Options, _Index>>(m_yty, other, false);
    }
    BlockYTYProductTransposeReturnType<BlockYTYType> adjoint() const
    {
      return BlockYTYProductTransposeReturnType<BlockYTYType>(m_yty);
    }
    BlockYTYProductTransposeReturnType<BlockYTYType> transpose() const
    {
      return BlockYTYProductTransposeReturnType<BlockYTYType>(m_yty);
    }

    const BlockYTYType& m_yty;
  };

  template<typename BlockYTYType>
  struct BlockYTYProductTransposeReturnType
  {
    explicit BlockYTYProductTransposeReturnType(const BlockYTYType& yty) : m_yty(yty) {}
    template<typename Derived>
    BlockYTY_VecProduct<BlockYTYType, Derived> operator*(const MatrixBase<Derived>& other)
    {
      return BlockYTY_VecProduct<BlockYTYType, Derived>(m_yty, other.derived(), true);
    }
    template<typename _Scalar, int _Options, typename _Index>
    BlockYTY_VecProduct<BlockYTYType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
    {
      return BlockYTY_VecProduct<BlockYTYType, SparseMatrix<_Scalar, _Options, _Index>>(m_yty, other, true);
    }
    const BlockYTYType& m_yty;
  };

  namespace internal {
    template<typename BlockYTYType>
    struct evaluator_traits<BlockYTYProductReturnType<BlockYTYType> >
    {
      typedef typename BlockYTYType::MatrixType MatrixType;
      typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
      typedef SparseShape Shape;
    };
  }
  /************************************************************************************************************************************/
}

#endif

