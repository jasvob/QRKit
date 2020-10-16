// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SPARSE_QR_ORDERING_H
#define SPARSE_QR_ORDERING_H

#include <Eigen/Eigen>
#include "SparseQRUtils.h"

namespace Eigen {
  namespace SparseQROrdering {
    
    template <typename StorageIndex>
    class ColumnDensity {
    public:
      typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

      /* 
       * Compute the permutation vector from a column-major sparse matrix. 
       * Create column permutation (according to the number of nonzeros in columns.
       */
      template <typename MatrixType>
      void operator()(const MatrixType &mat, PermutationType &perm) {
        eigen_assert(!mat.IsRowMajor && "Input matrix has to be ColMajor!");

        typedef SparseQRUtils::VectorCount<MatrixType::StorageIndex> MatrixColCount;
        
        std::vector<MatrixColCount> colNnzs;
        // Record number of nonzeros in columns
        for (Index c = 0; c < mat.cols(); c++) {
          colNnzs.push_back(MatrixColCount(c, mat.col(c).nonZeros()));
        }
        // Sort the column according to the number of nonzeros in an ascending order
        std::stable_sort(colNnzs.begin(), colNnzs.end());
        // Create permutation matrix out of the sorted vector
        Eigen::Matrix<MatrixType::StorageIndex, Dynamic, 1> colpermIndices(colNnzs.size());
        for (Index c = 0; c < colNnzs.size(); c++) {
          colpermIndices(colNnzs[c].origIdx) = c;
        }
        perm = PermutationType(colpermIndices);
      }
    };

    template <typename StorageIndex>
    class AsBandedAsPossible {
    public:
      typedef PermutationMatrix<Dynamic, Dynamic, StorageIndex> PermutationType;

      bool hasPermutation;
      
      /*
       * Compute the permutation vector from a row-major sparse matrix.
       * Look for as banded as possible structure in the matrix. 
       * Row band-widths are detected and rows are sorted by the band start index.
       * In addition, it records auxiliary information about the banded structure of the permuted matrix.
       */
      template <typename MatrixType>
      void operator()(const MatrixType &mat, PermutationType &perm) {
        eigen_assert(mat.IsRowMajor && "Input matrix has to be RowMajor!");

        typedef SparseQRUtils::RowRange<MatrixType::StorageIndex> MatrixRowRange;

        // 1) Compute and store band information for each row in the matrix
        std::vector<MatrixRowRange> rowRanges;
        // Compute band information for each row
        for (MatrixType::StorageIndex j = 0; j < mat.rows(); j++) {
          MatrixType::InnerIterator rowIt(mat, j);
          typename MatrixType::StorageIndex startIdx = mat.cols();
          if (rowIt) {  // Necessary from the nature of the while loop below 
            startIdx = rowIt.index();
          }
          MatrixType::StorageIndex endIdx = startIdx;
          while (++rowIt) { endIdx = rowIt.index(); }	// FixMe: Is there a better way?
          rowRanges.push_back(MatrixRowRange(j, startIdx, endIdx));
        }

        // 2) Sort the rows to form as-banded-as-possible matrix
        // Set an indicator whether row sorting is needed
        this->hasPermutation = !std::is_sorted(rowRanges.begin(), rowRanges.end(), [](const MatrixRowRange &lhs, const MatrixRowRange &rhs) {
          return (lhs.start < rhs.start);
        });
        // Perform the actual row sorting if needed
        if (this->hasPermutation) {
          std::stable_sort(rowRanges.begin(), rowRanges.end(), [](const MatrixRowRange &lhs, const MatrixRowRange &rhs) {
            return (lhs.start < rhs.start);
            /*if (lhs.start < rhs.start) {
              return true;
            }
            else if (lhs.start == rhs.start) {
              if (lhs.end < rhs.end) {
                return true;
              }
              else {
                return lhs.origIdx < rhs.origIdx;
              }
            }
            else {
              return false;
            }*/
          });
        }

        // And record the estimated block structure
        Eigen::Matrix<MatrixType::StorageIndex, Dynamic, 1> permIndices(rowRanges.size());
        MatrixType::StorageIndex rowIdx = 0;
        for (auto it = rowRanges.begin(); it != rowRanges.end(); ++it, rowIdx++) {
          permIndices(it->origIdx) = rowIdx;
        }
        // Create row permutation matrix that achieves the desired row reordering
        perm = PermutationType(permIndices);
      }
    };

  }
}

#endif

