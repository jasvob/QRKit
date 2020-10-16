// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SPARSE_QR_UTILS_H
#define SPARSE_QR_UTILS_H

#include <Eigen/Eigen>

namespace Eigen {
  namespace SparseQRUtils {

    /*********************** Sparse QR solver traits class  ****************************
    * Necessary because not all the solvers have the rowsPermutation() method.
    ***********************************************************************************/
    template <typename QRSolver> struct HasRowsPermutation {
      static const bool value = false;
    };

    template <typename SolverType, typename PermType>
    typename std::enable_if<HasRowsPermutation<SolverType>::value, PermType>::type rowsPermutation(const SolverType &slvr) {
      return slvr.rowsPermutation();
    }
    template <typename SolverType, typename PermType>
    typename std::enable_if<!HasRowsPermutation<SolverType>::value, PermType>::type rowsPermutation(const SolverType &slvr) {
      PermType perm;
      perm.setIdentity(slvr.rows());
      return perm;
    }

    /*
    * Templated functions allowing to copy multiple segments from a vector type.
    * Starting positions and lengths of the segments are specificed as std::pair, as:
    *   .first  ... starting index of the segment
    *   .second ... length of the segment
    * The definition of the segments is then passed as vector of pairs.
    */
    typedef std::pair<int, int> SegmentDesc;
    typedef std::vector<SegmentDesc> SegmentDescriptors;
    // Create new dense vector from a segments of another dense vector
    template <typename VectorType, int NumSegments = 2>
    void getVectorSegments(VectorType &dst, const VectorType &src, const SegmentDescriptors &segments, int totalLength = -1) {
      // Total length (sum of lengths of the segments) can be provided as a template parameters
      if (totalLength > 0) {
        dst = VectorType(totalLength);
      }
      // If total length is not provided, compute it from the vector of segments automatically (presumably slower)
      else {
        int length = 0;
        for (auto seg = segments.begin(); seg != segments.end(); ++seg) {
          length += seg->second;
        }
        dst = VectorType(length);
      }

      // Copy the segments of the source vector into the destination vector
      if (NumSegments == 2) {
        dst.segment(0, segments[0].second) = src.segment(segments[0].first, segments[0].second);
        dst.segment(segments[0].second, segments[1].second) = src.segment(segments[1].first, segments[1].second);
      }
      else {
        int currPos = 0;
        for (auto seg = segments.begin(); seg != segments.end(); ++seg) {
          dst.segment(currPos, seg->second) = src.segment(seg->first, seg->second);
          currPos += seg->second;
        }
      }
    }
    // Set segments of a dense vector from another dense vector
    template <typename VectorType, int NumSegments = 2>
    void setVectorSegments(VectorType &dst, const VectorType &src, const SegmentDescriptors &segments) {
      if (NumSegments == 2) {
        dst.segment(segments[0].first, segments[0].second) = src.segment(0, segments[0].second);
        dst.segment(segments[1].first, segments[1].second) = src.segment(segments[0].second, segments[1].second);
      }
      else {
        int currPos = 0;
        for (auto seg = segments.begin(); seg != segments.end(); ++seg) {
          dst.segment(seg->first, seg->second) = src.segment(currPos, seg->second);
          currPos += seg->second;
        }
      }
    }

    /*
    * Stores information about a dense block in a block sparse matrix.
    * Holds the position of the block (row index, column index) and its size (number of rows, number of columns).
    */
    template <typename IndexType>
    struct BlockInfo {
      typedef std::map<IndexType, BlockInfo<IndexType>> Map;
      typedef std::vector<IndexType> MapOrder;

      IndexType idxRow;
      IndexType idxCol;
      IndexType numRows;
      IndexType numCols;

      BlockInfo()
        : idxRow(0), idxCol(0), numRows(0), numCols(0) {
      }

      BlockInfo(const IndexType &rowIdx, const IndexType &colIdx, const IndexType &nr, const IndexType &nc)
        : idxRow(rowIdx), idxCol(colIdx), numRows(nr), numCols(nc) {
      }

      BlockInfo(const IndexType &diagIdx, const IndexType &nr, const IndexType &nc)
        : idxRow(diagIdx), idxCol(diagIdx), numRows(nr), numCols(nc) {
      }
    };
    /*
    * Overloaded << operator for the BlockInfo struct to easily output block position and size.
    */
    template <typename IndexType>
    std::ostream& operator<<(std::ostream& os, const BlockInfo<IndexType> &bi) {
      std::cout << "[" << bi.idxRow << ", " << bi.idxCol << "] = " << bi.numRows << ", " << bi.numCols;
      return os;
    }

    /*
    * Helper structure holding band information for a single row.
    * Stores original row index (before any row reordering was performed),
    * index of the first nonzero (start) and last nonzero(end) in the band and the band length (length).
    */
    template <typename IndexType>
    struct RowRange {
      IndexType origIdx;
      IndexType start;
      IndexType end;
      IndexType length;

      RowRange() : start(0), end(0), length(0) {
      }

      RowRange(const IndexType &origIdx, const IndexType &start, const IndexType &end)
        : origIdx(origIdx), start(start), end(end) {
        this->length = this->end - this->start + 1;
      }
    };

    /*
    * Helper structure holding nonzero count for a vector in a matrix.
    * Stores original row/column index (before any row reordering was performed),
    * and nthe number of nonzero elements (numNnz).
    */
    template <typename IndexType>
    struct VectorCount {
      IndexType origIdx;
      IndexType numNnz;

      VectorCount() : origIdx(0), numNnz(0) {
      }

      VectorCount(const IndexType &origIdx, const IndexType &numNnz)
        : origIdx(origIdx), numNnz(numNnz) {
      }

      bool operator<(const VectorCount &right) const {
        return this->numNnz < right.numNnz;
      }
    };

    /*
    * Stores information about block bands of the input row-major matrix.
    * SuggestedBlockCols is telling the algorithm how many columns per block is ideally desired.
    * The output block structure does not have to produce block with SuggestedBlockCols column,
    * but aims to get at least close to it. SuggestedBlockCols is a lower bound on the block width. 
    * The algorithm seeks only valid blocks: nrows >= ncols. If an invalid block is constructed, the encapsulating
    * factorization algorithm will not work properly.
    * If valid block (nrows >= ncols) cannot be constructed with SuggestedBlockCols column, it is beign extended
    * by the smallest number of columns that assure (nrows >= ncols).
    */
    template <typename StorageIndex, int SuggestedBlockCols = 2>
    struct BlockBandedMatrixInfo {
      typedef SparseQRUtils::BlockInfo<StorageIndex> MatrixBlockInfo;
      typename MatrixBlockInfo::Map blockMap;		// Sparse matrix block information
      typename MatrixBlockInfo::MapOrder blockOrder; // Sparse matrix block order
      StorageIndex nonZeroQEstimate;		// Estimate of number of nonzero elements in Q matrix formed by QR decomposition of the analyzed matrix

      template <typename MatrixType>
      void operator()(const MatrixType &mat) {
        eigen_assert(mat.IsRowMajor && "Input matrix has to be RowMajor!");

        typedef SparseQRUtils::RowRange<MatrixType::StorageIndex> MatrixRowRange;
        typedef std::map<typename MatrixType::StorageIndex, typename MatrixType::StorageIndex> BlockBandSize;
        // 1) Compute and store band information for each start index that has nonzero in column i
        BlockBandSize bandWidths, bandHeights;
        std::vector<MatrixRowRange> rowRanges;
        for (typename MatrixType::StorageIndex j = 0; j < mat.rows(); j++) {
          typename MatrixType::InnerIterator rowIt(mat, j);
          typename MatrixType::StorageIndex startIdx = mat.cols();
          if (rowIt) {  // Necessary from the nature of the while loop below 
            startIdx = rowIt.index();
          }
          typename MatrixType::StorageIndex endIdx = startIdx;
          while (++rowIt) { endIdx = rowIt.index(); }	// FixMe: Is there a better way?
          rowRanges.push_back(MatrixRowRange(j, startIdx, endIdx));

          typename MatrixType::StorageIndex bw = endIdx - startIdx + 1;
          if (bandWidths.find(startIdx) == bandWidths.end()) {
            bandWidths.insert(std::make_pair(startIdx, bw));
          }
          else {
            if (bandWidths.at(startIdx) < bw) {
              bandWidths.at(startIdx) = bw;
            }
          }

          if (bandHeights.find(startIdx) == bandHeights.end()) {
            bandHeights.insert(std::make_pair(startIdx, 1));
          }
          else {
            bandHeights.at(startIdx) += 1;
          }
        }

        // 2) Search for banded blocks (blocks of row sharing same/similar band)		
        typename MatrixType::StorageIndex maxColStep = 0;
        for (typename MatrixType::StorageIndex j = 0; j < rowRanges.size() - 1; j++) {
          if ((rowRanges.at(j + 1).start - rowRanges.at(j).start) > maxColStep) {
            maxColStep = (rowRanges.at(j + 1).start - rowRanges.at(j).start);
          }
        }

        // And record the estimated block structure
        this->nonZeroQEstimate = 0;
        this->blockMap.clear();
        this->blockOrder.clear();
        typename MatrixType::StorageIndex rowIdx = 0;
        for (auto it = rowRanges.begin(); it != rowRanges.end(); ++it, rowIdx++) {
          // std::find is terribly slow for large arrays
          // assuming m_blockOrder is ordered, we can use binary_search
          // is m_blockOrder always ordered? can we always use binary_search???
          if (!std::binary_search(this->blockOrder.begin(), this->blockOrder.end(), it->start)) {
            // If start is out of bounds, it means that this is a 0 block - ignore it
            if (it->start < mat.cols()) {
              this->blockOrder.push_back(it->start);
              this->blockMap.insert(std::make_pair(it->start, MatrixBlockInfo(rowIdx, it->start, bandHeights.at(it->start), bandWidths.at(it->start))));
              this->nonZeroQEstimate += this->blockMap.at(it->start).numRows * this->blockMap.at(it->start).numRows;
            }
          }
        }

        // 3) Go through the estimated block structure
        // And merge several blocks together if needed/possible in order to form reasonably big banded blocks
        this->mergeBlocks<StorageIndex>(this->blockOrder, this->blockMap, maxColStep);
      }

      void fromBlockDiagonalPattern(const StorageIndex matRows, const StorageIndex matCols,
        const StorageIndex blockRows, const StorageIndex blockCols) {

        // 1) Set the block map based on block paramters passed ot this method	
        typename StorageIndex numBlocks = matCols / blockCols;
        this->blockMap.clear();
        this->blockOrder.clear();
        this->nonZeroQEstimate = 0;
        typename StorageIndex rowIdx = 0;
        typename StorageIndex colIdx = 0;
        for (int i = 0; i < numBlocks; i++) {
          rowIdx = i * blockRows;
          colIdx = i * blockCols;
          this->blockOrder.push_back(colIdx);
          this->blockMap.insert(std::make_pair(colIdx, MatrixBlockInfo(rowIdx, colIdx, blockRows, blockCols)));
          this->nonZeroQEstimate += blockRows * blockRows;
        }
      }

      void fromBlockBandedPattern(const StorageIndex matRows, const StorageIndex matCols,
        const StorageIndex blockRows, const StorageIndex blockCols, const StorageIndex blockOverlap) {

        // 1) Set the block map based on block paramters passed ot this method	
        StorageIndex maxColStep = blockCols - blockOverlap;
        StorageIndex numBlocks = matCols / maxColStep;
        this->blockMap.clear();
        this->blockOrder.clear();
        this->nonZeroQEstimate = 0;
        StorageIndex rowIdx = 0;
        StorageIndex colIdx = 0;
        for (int i = 0; i < numBlocks; i++) {
          rowIdx = i * blockRows;
          colIdx = i * maxColStep;
          this->blockOrder.push_back(colIdx);
          if (i < numBlocks - 1) {
            this->blockMap.insert(std::make_pair(colIdx, MatrixBlockInfo(rowIdx, colIdx, blockRows, blockCols)));
          }
          else {
            // Last block need to be treated separately (only block overlap - we're at matrix bound)
            this->blockMap.insert(std::make_pair(colIdx, MatrixBlockInfo(rowIdx, colIdx, blockRows, blockCols - blockOverlap)));
          }
          this->nonZeroQEstimate += blockRows * blockRows;
        }

        // 2) Go through the estimated block structure
        // And merge several blocks together if needed/possible in order to form reasonably big banded blocks
        this->mergeBlocks<StorageIndex>(this->blockOrder, this->blockMap, maxColStep);
      }

    private:
      /*
      * Going through a block map and looking for a possibility to merge several blocks together in order to obtain the desired block structure.
      */
      template <typename StorageIndex>
      void mergeBlocks(typename MatrixBlockInfo::MapOrder &blOrder, typename MatrixBlockInfo::Map &blMap, const int maxColStep) {
        typename MatrixBlockInfo::Map newBlockMap;
        typename MatrixBlockInfo::MapOrder newBlockOrder;
        MatrixBlockInfo firstBlock;
        int currRows, currCols;

        /*
        * Analyze parameters of each block.
        * If the existing block already has the required shape, leave it as is.
        * Otherwise start merging it with the next blocks until the conditions are met.
        */
        auto it = blOrder.begin();
        for (; it != blOrder.end(); ++it) {
          MatrixBlockInfo currBlock = blMap.at(*it);

          /*
          * If there are already some new block recorded and the current block is column-wise contained in the last block,
          * just merge the current block with the last block
          */
          if (!newBlockOrder.empty()) {
            MatrixBlockInfo lastBlock = newBlockMap[newBlockOrder.back()];
            if (currBlock.idxCol + currBlock.numCols <= lastBlock.idxCol + lastBlock.numCols) {
              int numRows = lastBlock.numRows + currBlock.numRows;
              int numCols = lastBlock.numCols;
              newBlockMap[newBlockOrder.back()] = MatrixBlockInfo(lastBlock.idxRow, lastBlock.idxCol, numRows, numCols);

              continue;
            }
          }
          /*
          * Otherwise, proceed with the typical block processing
          */
          if (firstBlock.numRows == 0) {  // If the first block has 0 rows, it is not set, the current block will be the first
            firstBlock = blMap.at(*it);
            currRows = currBlock.numRows;
            currCols = currBlock.numCols;
          }
          else {  // Otherwise we already have first block, estimate new merged block size
            currRows = currBlock.idxRow + currBlock.numRows - firstBlock.idxRow;
            currCols = currBlock.idxCol + currBlock.numCols - firstBlock.idxCol;
          }

          /*
          * If the conditions for a new block are met, create it
          * 1) Each block has to have a portrait shape
          * 2) Each block should be at least maxColStep columns wide
          * 3) If requested by user, template parameter SuggestedBlockCols says the desired approximate amount of columns per block
          */
          if (currRows > currCols && currCols >= maxColStep && currCols >= SuggestedBlockCols) {
            newBlockOrder.push_back(firstBlock.idxCol);
            newBlockMap.insert(std::make_pair(firstBlock.idxCol, MatrixBlockInfo(firstBlock.idxRow, firstBlock.idxCol, currRows, currCols)));

            // Reset first block
            firstBlock = MatrixBlockInfo();
          }
        }

        /*
        * If merging of the last block was not done and we are already at the end, merge the remainder with the last block in the new array.
        */
        if (firstBlock.numRows != 0) {
          if (currRows > currCols && currCols >= maxColStep && currCols >= SuggestedBlockCols) {
            newBlockOrder.push_back(firstBlock.idxCol);
            newBlockMap.insert(std::make_pair(firstBlock.idxCol, MatrixBlockInfo(firstBlock.idxRow, firstBlock.idxCol, currRows, currCols)));
          }
          else {
            MatrixBlockInfo lastBlock = newBlockMap[newBlockOrder.back()];
            int numRows = lastBlock.numRows + currRows;
            int numCols = firstBlock.idxCol + currCols - lastBlock.idxCol;
            newBlockMap[newBlockOrder.back()] = MatrixBlockInfo(lastBlock.idxRow, lastBlock.idxCol, numRows, numCols);
          }
        }

        // Save the final banded block structure
        blOrder = newBlockOrder;
        blMap = newBlockMap;
      }
    };


    /*
    * General parallel for loop.
    * Expects functor which loops through some subset of the range, bi..ei
    * This means it can be efficient even when the workload inside the loop is small.
    * Number of threads is passed as a paramter:
    *  nthread = 0 ... use multithreading with std::thread::hardware_concurrency() threads
    *  nthread = 1 ... do not use multithreading, evaluate as normal function call
    *  nthread >= 2 ... use multithreading with nthread threads
    */
    template <class Functor>
    void parallel_for(const int bi, const int ei, Functor &f, size_t nthreads = 0) {

      if (nthreads == 1) {
        /********************************* ST *****************************/
        f(bi, ei);
      }
      else {
        if (nthreads == 0)
          nthreads = std::thread::hardware_concurrency();
        
        /********************************* MT *****************************/
        const size_t nloop = ei - bi;
        {
          std::vector<std::thread> threads(nthreads);
          for (int t = 0; t < nthreads; t++) {
            threads[t] = std::thread(f, bi + t*nloop / nthreads, bi + ((t + 1) == nthreads ? nloop : (t + 1)*nloop / nthreads));
          }
          std::for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join(); });
        }
      }
    }

  }
}

#endif
