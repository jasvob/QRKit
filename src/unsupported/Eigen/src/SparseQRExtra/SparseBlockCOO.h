// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SPARSE_BLOCK_COO_H
#define SPARSE_BLOCK_COO_H

#include <Eigen/Eigen>

namespace Eigen {
  /*
  * Storage type for general sparse matrix with block structure.
  * Each element holds block position (row index, column index) and the values in the block stored in ValueType.
  * ValueType is a template type and can generally represent any datatype, both default and user defined.
  *
  */
  template <typename ValueType, typename IndexType>
  class SparseBlockCOO {
  public:
    /*
    * Building block of the SparseBlockCOO matrix.
    * Parameters 'row' and 'col' are the indices of the top left corner of the block.
    * For example:
    *   11111
    *   11111
    *   1113322
    *      2222
    * is {Element(0, 0, MatrixXd::Ones(3, 5)), Element(2, 3, MatrixXd::Ones(2, 4))}
    *
    */
    struct Element {
      IndexType row;
      IndexType col;

      ValueType value;

      Element()
        : row(0), col(0) {
      }

      Element(const IndexType row, const IndexType col, const ValueType &val)
        : row(row), col(col), value(val) {
      }
    };
    typedef std::vector<Element> ElementsVec;

    SparseBlockCOO()
      : nRows(0), nCols(0) {
    }

    SparseBlockCOO(const IndexType &rows, const IndexType &cols)
      : nRows(rows), nCols(cols) {
    }

    void insert(const Element &elem) {
      this->elems.push_back(elem);
    }

    IndexType size() const {
      return this->elems.size();
    }

    void clear() {
      this->elems.clear();
    }

    Element& operator[](IndexType i) {
      return this->elems[i];
    }
    const Element& operator[](IndexType i) const {
      return this->elems[i];
    }

    IndexType rows() const {
      return this->nRows;
    }
    IndexType cols() const {
      return this->nCols;
    }

  protected:
    ElementsVec elems;

    IndexType nRows;
    IndexType nCols;
  };
}

#endif

