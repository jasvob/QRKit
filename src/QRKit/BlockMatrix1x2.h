// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef BLOCK_MATRIX_1X2_H
#define BLOCK_MATRIX_1X2_H

namespace QRKit {
  /*
  * Container for storing 2x1 block matrices.
  * Such matrices are composed from two subblocks of arbitrary types specified as template arguments.
  * 
  * An example would be block angular matrix, which typically contains a very 
  * sparse left block and dense (or almost dense) right block. An example:
  *
  *  XXX            XXX
  *    XXX          XXX
  *      XXX        XXX
  *        XXX      XXX
  *          XXX    XXX
  *            XXX  XXX
  *              XXXXXX
  *
  */
  template <typename LeftBlockMatrixType, typename RightBlockMatrixType>
  class BlockMatrix1x2 {
  public:
    BlockMatrix1x2(LeftBlockMatrixType &leftBlock, RightBlockMatrixType &rightBlock)
      : leftBlockNat(leftBlock), rightBlockMat(rightBlock) {

      eigen_assert(leftBlockNat.rows() == rightBlockMat.rows());
    }

    RightBlockMatrixType& rightBlock() {
      return this->rightBlockMat;
    }

    LeftBlockMatrixType& leftBlock() {
      return this->leftBlockNat;
    }

    const RightBlockMatrixType& rightBlock() const {
      return this->rightBlockMat;
    }

    const LeftBlockMatrixType& leftBlock() const {
      return this->leftBlockNat;
    }

    int rows() const {
      return this->leftBlockNat.rows();
    }

    int cols() const {
      return this->leftBlockNat.cols() + this->rightBlockMat.cols();
    }

  protected:
    RightBlockMatrixType &rightBlockMat;
    LeftBlockMatrixType &leftBlockNat;
  };
}

#endif

