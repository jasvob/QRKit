// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SPARSE_BLOCK_YTY_H
#define SPARSE_BLOCK_YTY_H

#include <Eigen/Core>
#include "SparseBlockCOO.h"
#include "BlockYTY.h"

namespace QRKit {
  template <typename ValueType, typename IndexType> class SparseBlocksYTY;
  template<typename SparseBlockYTYType> struct SparseBlockYTYProductReturnType;
  template<typename SparseBlockYTYType> struct SparseBlockYTYProductTransposeReturnType;
  template<typename SparseBlockYTYType, typename Derived> struct SparseBlockYTY_VecProduct;

  namespace internal {

    // Eigen::internal::traits<BlockYTYProduct[Transpose]>
    template <typename SparseBlockYTYType> struct Eigen::internal::traits<SparseBlockYTYProductReturnType<SparseBlockYTYType> >
    {
      typedef typename SparseBlockYTYType::MatrixType ReturnType;
      typedef typename ReturnType::StorageIndex StorageIndex;
      typedef typename ReturnType::StorageKind StorageKind;
      enum {
        RowsAtCompileTime = Dynamic,
        ColsAtCompileTime = Dynamic
      };
    };

    template <typename SparseBlockYTYType> struct Eigen::internal::traits<SparseBlockYTYProductTransposeReturnType<SparseBlockYTYType> >
    {
      typedef typename SparseBlockYTYType::MatrixType ReturnType;
    };

    template <typename SparseBlockYTYType, typename Derived> struct Eigen::internal::traits<SparseBlockYTY_VecProduct<SparseBlockYTYType, Derived> >
    {
      typedef typename Derived::PlainObject ReturnType;
    };

    // BlockYTY_traits
    template <typename T> struct SparseBlockYTY_traits {  };

  } // End namespace internal

  /*
  * SparseBlockCOO specialization for sparse matrix storing YT blocks.
  * Each element holds block position (row index, column index) and the values in the block stored in ValueType.
  * ValueType is a template type and can generally represent any datatype, both default and user defined.
  *
  * This specialization is important for good performance of the YTY product evaluation. 
  * Each call to yty() expression of BlockYTY class includes some additional time-consuming processing. 
  * yty() is therefore not suitable for being used inside for loops. For such reasons, this specialization 
  * of SparseBlockCOO for storing BlockYTY elements provides an expression sequenceYTY(), which internally
  * handles series of YTY multiplications in much more efficient way.
  *
  */
  template <typename ScalarType, typename IndexType>
  class SparseBlockYTY : public SparseBlockCOO<BlockYTY<ScalarType, IndexType>, IndexType> {
  public:
    typedef typename BlockYTY<ScalarType, IndexType>::MatrixType MatrixType;

    SparseBlockYTY()
      : SparseBlockCOO(0, 0) {
    }

    SparseBlockYTY(const IndexType &rows, const IndexType &cols)
      : SparseBlockCOO(rows, cols) {
    }

    SparseBlockYTYProductReturnType<SparseBlockYTY> sequenceYTY() const {
      return SparseBlockYTYProductReturnType<SparseBlockYTY>(*this);
    }

  protected:
    template <typename, typename > friend struct SparseBlockYTY_VecProduct;
  };

  /************************************ Expression templates for the evaluation of single YTY product ***************************************/
  template <typename SparseBlockYTYType, typename Derived>
  struct SparseBlockYTY_VecProduct : Eigen::ReturnByValue<SparseBlockYTY_VecProduct<SparseBlockYTYType, Derived> >
  {
    // Get the references 
    SparseBlockYTY_VecProduct(const SparseBlockYTYType& blocksYT, const Derived& other, bool transpose) :
      m_blocksYT(blocksYT), m_other(other), m_transpose(transpose) {}
    inline Index rows() const { return m_other.rows(); }
    inline Index cols() const { return m_other.cols(); }

    // Assign to a vector
    template<typename DesType>
    void evalTo(DesType& res) const
    {
      res = m_other;

      /*
      * While expressing product of multiple YTY blocks using for loop, it is much more efficient to write the YTY expressions
      * explicitly rather than using yty() expression.
      */
      Derived segmentVec;
      if (m_transpose) {
        for (Index k = 0; k < m_blocksYT.size(); k++) {
          SparseQRUtils::SegmentDescriptors segDescs = {
            { m_blocksYT[k].row, m_blocksYT[k].value.cols() },
            { m_blocksYT[k].row + m_blocksYT[k].value.cols() + m_blocksYT[k].value.numZeros(), m_blocksYT[k].value.rows() - m_blocksYT[k].value.cols() }
          };
          SparseQRUtils::getVectorSegments<Derived, 2>(segmentVec, res, segDescs, m_blocksYT[k].value.rows());

          // Non-aliasing expr -> noalias()
          segmentVec.noalias() += (m_blocksYT[k].value.Y() * (m_blocksYT[k].value.T().transpose() * (m_blocksYT[k].value.Y().transpose() * segmentVec)));

          SparseQRUtils::setVectorSegments<Derived, 2>(res, segmentVec, segDescs);
        }
      } else {
        for (Index k = m_blocksYT.size() - 1; k >= 0; k--) {
          SparseQRUtils::SegmentDescriptors segDescs = {
            { m_blocksYT[k].row, m_blocksYT[k].value.cols() },
            { m_blocksYT[k].row + m_blocksYT[k].value.cols() + m_blocksYT[k].value.numZeros(), m_blocksYT[k].value.rows() - m_blocksYT[k].value.cols() }
          };
          SparseQRUtils::getVectorSegments<Derived, 2>(segmentVec, res, segDescs, m_blocksYT[k].value.rows());

          // Non-aliasing expr -> noalias()
          segmentVec.noalias() += (m_blocksYT[k].value.Y() * (m_blocksYT[k].value.T() * (m_blocksYT[k].value.Y().transpose() * segmentVec)));

          SparseQRUtils::setVectorSegments<Derived, 2>(res, segmentVec, segDescs);
        }
      }
    }

    const SparseBlockYTYType& m_blocksYT;
    const Derived& m_other;
    bool m_transpose;
  };

  template <typename SparseBlockYTYType>
  struct SparseBlockYTYProductReturnType : public EigenBase<SparseBlockYTYProductReturnType<SparseBlockYTYType> >
  {
    enum {
      RowsAtCompileTime = Dynamic,
      ColsAtCompileTime = Dynamic
    };
    explicit SparseBlockYTYProductReturnType(const SparseBlockYTYType& blocksYT) : m_blocksYT(blocksYT) {}

    template<typename Derived>
    SparseBlockYTY_VecProduct<SparseBlockYTYType, Derived> operator*(const MatrixBase<Derived>& other)
    {
      return SparseBlockYTY_VecProduct<SparseBlockYTYType, Derived>(m_blocksYT, other.derived(), false);
    }
    template<typename _Scalar, int _Options, typename _Index>
    SparseBlockYTY_VecProduct<SparseBlockYTYType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
    {
      return SparseBlockYTY_VecProduct<SparseBlockYTYType, SparseMatrix<_Scalar, _Options, _Index>>(m_blocksYT, other, false);
    }
    SparseBlockYTYProductTransposeReturnType<SparseBlockYTYType> adjoint() const
    {
      return SparseBlockYTYProductTransposeReturnType<SparseBlockYTYType>(m_blocksYT);
    }
    SparseBlockYTYProductTransposeReturnType<SparseBlockYTYType> transpose() const
    {
      return SparseBlockYTYProductTransposeReturnType<SparseBlockYTYType>(m_blocksYT);
    }

    const SparseBlockYTYType& m_blocksYT;
  };

  template<typename SparseBlockYTYType>
  struct SparseBlockYTYProductTransposeReturnType
  {
    explicit SparseBlockYTYProductTransposeReturnType(const SparseBlockYTYType& blocksYT) : m_blocksYT(blocksYT) {}
    template<typename Derived>
    SparseBlockYTY_VecProduct<SparseBlockYTYType, Derived> operator*(const MatrixBase<Derived>& other)
    {
      return SparseBlockYTY_VecProduct<SparseBlockYTYType, Derived>(m_blocksYT, other.derived(), true);
    }
    template<typename _Scalar, int _Options, typename _Index>
    SparseBlockYTY_VecProduct<SparseBlockYTYType, SparseMatrix<_Scalar, _Options, _Index>> operator*(const SparseMatrix<_Scalar, _Options, _Index>& other)
    {
      return SparseBlockYTY_VecProduct<SparseBlockYTYType, SparseMatrix<_Scalar, _Options, _Index>>(m_blocksYT, other, true);
    }
    const SparseBlockYTYType& m_blocksYT;
  };

  namespace internal {
    template<typename SparseBlockYTYType>
    struct Eigen::internal::evaluator_traits<SparseBlockYTYProductReturnType<SparseBlockYTYType> >
    {
      typedef typename SparseBlockYTYType::MatrixType MatrixType;
      typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
      typedef SparseShape Shape;
    };
  }
  /************************************************************************************************************************************/
}

#endif

