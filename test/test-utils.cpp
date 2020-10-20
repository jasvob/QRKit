// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Jan Svoboda <jan.svoboda@nnaisense.com>
// Copyright (C) 2020 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


// import basic and product tests for deprectaed DynamicSparseMatrix
#define EIGEN_NO_DEPRECATED_WARNING

#include <iostream>
#include <iomanip>
#include <ctime>
#include <future>
#include <random>

#include "test.h"

#include <QRKit/QRKit>
#include <Eigen/SparseCore>

using namespace Eigen;
using namespace QRKit;

typedef double Scalar;

typedef SparseMatrix<Scalar, ColMajor, int> JacobianType;
typedef SparseMatrix<Scalar, RowMajor, int> JacobianTypeRowMajor;
typedef Matrix<Scalar, Dynamic, 1> DenseVectorType;
typedef Eigen::SparseVector<Scalar> SparseVec;

/*
* Generate block diagonal sparse matrix with overlapping diagonal blocks.
*/
void generate_overlapping_block_diagonal_matrix(const Eigen::Index numParams, const Eigen::Index numResiduals, JacobianType &spJ, bool permuteRows = true) {
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.5, 5.0);

  int stride = 7;
  std::vector<Eigen::Triplet<Scalar, typename JacobianType::Index> > jvals(stride * numParams);
  for (int i = 0; i < numParams; i++) {
    for (int j = i * 2; j < (i * 2) + 2 && j < numParams; j++) {
      jvals.emplace_back(i * stride, j, dist(gen));
      jvals.emplace_back(i * stride + 1, j, dist(gen));
      jvals.emplace_back(i * stride + 2, j, dist(gen));
      jvals.emplace_back(i * stride + 3, j, dist(gen));
      jvals.emplace_back(i * stride + 4, j, dist(gen));
      jvals.emplace_back(i * stride + 5, j, dist(gen));
      jvals.emplace_back(i * stride + 6, j, dist(gen));
      if (j < numParams - 2) {
        jvals.emplace_back(i * stride + 6, j + 2, dist(gen));
      }
    }
  }

  spJ.resize(numResiduals, numParams);
  spJ.setZero();
  spJ.setFromTriplets(jvals.begin(), jvals.end());
  spJ.makeCompressed();

  // Permute Jacobian rows (we want to see how our QR handles a general matrix)	
  if (permuteRows) {
    PermutationMatrix<Dynamic, Dynamic, JacobianType::StorageIndex> perm(spJ.rows());
    perm.setIdentity();
    std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
    spJ = perm * spJ;
  }
}

/*
* Generate block diagonal sparse matrix.
*/
void generate_block_diagonal_matrix(const Eigen::Index numParams, const Eigen::Index numResiduals, JacobianType &spJ, bool permuteRows = true) {
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.5, 5.0);

  int stride = 7;
  std::vector<Eigen::Triplet<Scalar, typename JacobianType::Index> > jvals(stride * numParams);
  for (int i = 0; i < numParams; i++) {
    for (int j = i * 2; j < (i * 2) + 2 && j < numParams; j++) {
      jvals.emplace_back(i * stride, j, dist(gen));
      jvals.emplace_back(i * stride + 1, j, dist(gen));
      jvals.emplace_back(i * stride + 2, j, dist(gen));
      jvals.emplace_back(i * stride + 3, j, dist(gen));
      jvals.emplace_back(i * stride + 4, j, dist(gen));
      jvals.emplace_back(i * stride + 5, j, dist(gen));
      jvals.emplace_back(i * stride + 6, j, dist(gen));
    }
  }

  spJ.resize(numResiduals, numParams);
  spJ.setZero();
  spJ.setFromTriplets(jvals.begin(), jvals.end());
  spJ.makeCompressed();

  // Permute Jacobian rows (we want to see how our QR handles a general matrix)	
  if (permuteRows) {
    PermutationMatrix<Dynamic, Dynamic, JacobianType::StorageIndex> perm(spJ.rows());
    perm.setIdentity();
    std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
    spJ = perm * spJ;
  }
}

/*
* Generate block angular sparse matrix with overlapping diagonal blocks.
*/
void generate_block_angular_matrix(const Eigen::Index numParams, const Eigen::Index numAngularParams, const Eigen::Index numResiduals, JacobianType &spJ) {
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.5, 5.0);

  int stride = 7;
  std::vector<Eigen::Triplet<Scalar, typename JacobianType::Index> > jvals(stride * numParams + numResiduals * numAngularParams);
  for (int i = 0; i < numParams; i++) {
    for (int j = i * 2; j < (i * 2) + 2 && j < numParams; j++) {
      jvals.emplace_back(i * stride, j, dist(gen));
      jvals.emplace_back(i * stride + 1, j, dist(gen));
      jvals.emplace_back(i * stride + 2, j, dist(gen));
      jvals.emplace_back(i * stride + 3, j, dist(gen));
      jvals.emplace_back(i * stride + 4, j, dist(gen));
      jvals.emplace_back(i * stride + 5, j, dist(gen));
      jvals.emplace_back(i * stride + 6, j, dist(gen));
      if (j < numParams - 2) {
        jvals.emplace_back(i * stride + 6, j + 2, dist(gen));
      }
    }
  }
  for (int i = 0; i < numResiduals; i++) {
    for (int j = 0; j < numAngularParams; j++) {
      jvals.emplace_back(i, numParams + j, dist(gen));
    }
  }

  spJ.resize(numResiduals, numParams + numAngularParams);
  spJ.setZero();
  spJ.setFromTriplets(jvals.begin(), jvals.end());
  spJ.makeCompressed();
}


void rowpermADiagLambda(const JacobianType &A, const Scalar lambda, JacobianType &outA) {
  const Index nParams = A.cols();
  const Index nResiduals = A.rows();

  // Rowpermute the diagonal lambdas into A
  // Always place lambda below the last element of each column
  PermutationMatrix<Dynamic, Dynamic, JacobianType::StorageIndex> rowPerm(nResiduals + nParams);
  Index currRow = 0;
  for (Index c = 0; c < nParams; c++) {
    JacobianType::InnerIterator colIt(A, c);
    Index lastNnzIdx = 0;
    if (colIt) { // Necessary from the nature of the while loop below 
      lastNnzIdx = colIt.index();
    }
    while (++colIt) { lastNnzIdx = colIt.index(); }

    // Don't permute the nnz elements in the column
    while (currRow <= lastNnzIdx + c) {
      rowPerm.indices()(currRow - c) = currRow;
      currRow++;
    }
    // Put current diagonal element on this position
    rowPerm.indices()(nResiduals + c) = currRow;
    currRow++;
  }

  // Create concatenation of the Jacobian with the diagonal matrix of lambdas
  JacobianTypeRowMajor I(nParams, nParams);
  I.setIdentity();

  JacobianTypeRowMajor Arm(nResiduals + nParams, nParams);
  Arm.reserve(A.nonZeros() + nParams);
  Arm.topRows(nResiduals) = A;
  Arm.bottomRows(nParams) = I * std::sqrt(lambda);
  outA = rowPerm * Arm;
}

bool test_blockdiag_permuted(const JacobianType &mat) {
  bool test_res = true;
  
  // Looking for as-banded-as-possible structure in the matrix
  PermutationMatrix<Dynamic, Dynamic, JacobianType::StorageIndex> permMat;
  SparseQROrdering::AsBandedAsPossible<JacobianType::StorageIndex> abapOrdering;
  JacobianTypeRowMajor rmMat(mat);
  abapOrdering(rmMat, permMat);

  // Permute if permutation found
  if (abapOrdering.hasPermutation) {
    rmMat = permMat * rmMat;
  }

  SparseQRUtils::BlockBandedMatrixInfo<JacobianType::StorageIndex> bInfo;
  bInfo(rmMat);

  test_res &= VERIFY_IS_EQUAL(bInfo.blockOrder.size(), 256);  // Expecting 256 blocks
  for (int i = 0; i < bInfo.blockOrder.size(); i++) {
    SparseQRUtils::BlockBandedMatrixInfo<JacobianType::StorageIndex>::MatrixBlockInfo bi = bInfo.blockMap[bInfo.blockOrder[i]];
    test_res &= VERIFY_IS_EQUAL(bi.idxRow, i * 7);
    test_res &= VERIFY_IS_EQUAL(bi.idxCol, i * 2);
    test_res &= VERIFY_IS_EQUAL(bi.numRows, 7);
    test_res &= VERIFY_IS_EQUAL(bi.numCols, 2);
  }

  return test_res;
}

bool test_overlapping_permuted(const JacobianType &mat) {
  bool test_res = true;

  // Looking for as-banded-as-possible structure in the matrix
  PermutationMatrix<Dynamic, Dynamic, JacobianType::StorageIndex> permMat;
  SparseQROrdering::AsBandedAsPossible<JacobianType::StorageIndex> abapOrdering;
  JacobianTypeRowMajor rmMat(mat);
  abapOrdering(rmMat, permMat);

  // Permute if permutation found
  if (abapOrdering.hasPermutation) {
    rmMat = permMat * rmMat;
  }

  SparseQRUtils::BlockBandedMatrixInfo<JacobianType::StorageIndex> bInfo;
  bInfo(rmMat);

  test_res &= VERIFY_IS_EQUAL(bInfo.blockOrder.size(), 255);  // Expecting 256 blocks
  for (int i = 0; i < bInfo.blockOrder.size(); i++) {
    SparseQRUtils::BlockBandedMatrixInfo<JacobianType::StorageIndex>::MatrixBlockInfo bi = bInfo.blockMap[bInfo.blockOrder[i]];
    if (i < bInfo.blockOrder.size() - 1) {  // Expecting block 7x4
      test_res &= VERIFY_IS_EQUAL(bi.idxRow, i * 7);
      test_res &= VERIFY_IS_EQUAL(bi.idxCol, i * 2);
      test_res &= VERIFY_IS_EQUAL(bi.numRows, 7);
      test_res &= VERIFY_IS_EQUAL(bi.numCols, 4);
    } else {  // Expecting last block 14x4
      test_res &= VERIFY_IS_EQUAL(bi.idxRow, i * 7);
      test_res &= VERIFY_IS_EQUAL(bi.idxCol, i * 2);
      test_res &= VERIFY_IS_EQUAL(bi.numRows, 14);
      test_res &= VERIFY_IS_EQUAL(bi.numCols, 4);
    }
  }

  // For debugging purposes
  //std::cout << bInfo.blockOrder.size() << std::endl;
  //auto it = bInfo.blockOrder.begin();
  //for (it; it != bInfo.blockOrder.end(); ++it) {
  //  std::cout << "[" << bInfo.blockMap[*it].idxRow << ", " << bInfo.blockMap[*it].idxCol << "] = " << bInfo.blockMap[*it].numRows << " x " << bInfo.blockMap[*it].numCols << std::endl;
  //}

  return test_res;
}

bool test_blockdiag_vertperm_diag(const JacobianType &mat) {
  bool test_res = true;

  JacobianType matDiag;
  rowpermADiagLambda(mat, 1e-3, matDiag);

  JacobianTypeRowMajor rmMat(matDiag);
  SparseQRUtils::BlockBandedMatrixInfo<JacobianType::StorageIndex> bInfo;
  bInfo(rmMat);

  test_res &= VERIFY_IS_EQUAL(bInfo.blockOrder.size(), 256);  // Expecting 256 blocks
  for (int i = 0; i < bInfo.blockOrder.size(); i++) {
    SparseQRUtils::BlockBandedMatrixInfo<JacobianType::StorageIndex>::MatrixBlockInfo bi = bInfo.blockMap[bInfo.blockOrder[i]];
    test_res &= VERIFY_IS_EQUAL(bi.idxRow, i * 9);
    test_res &= VERIFY_IS_EQUAL(bi.idxCol, i * 2);
    test_res &= VERIFY_IS_EQUAL(bi.numRows, 9);
    test_res &= VERIFY_IS_EQUAL(bi.numCols, 2);
  }

  return test_res;
}

bool test_parallel_for(const JacobianType &mat) {
  typedef std::vector<std::vector<std::pair<typename JacobianType::Index, Scalar>>> ResValsVector;

  bool test_res = true;

  for(int i = 2; i < 5; i++) {
    /********************************************************************************/
    // Do it sequentially
    ResValsVector resVals(mat.cols());
    Index numNonZeros = 0;
    SparseQRUtils::parallel_for(0, mat.cols(), [&](const int bi, const int ei) {
      // loop over all items
      for (int j = bi; j < ei; j++)
      {
        DenseVectorType resColJd = mat.col(j).toDense();

        resColJd = 5.0 * resColJd;
     
        // Write the result back to j-th column of res
        SparseVec resColJ = resColJd.sparseView();
        numNonZeros += resColJ.nonZeros();
        resVals[j].reserve(resColJ.nonZeros());
        for (SparseVec::InnerIterator it(resColJ); it; ++it) {
          resVals[j].push_back(std::make_pair(it.row(), it.value()));
        }
      }
    }, 0);
    // Form the output
    JacobianType res(mat.rows(), mat.cols());
    res.reserve(numNonZeros);
    for (int j = 0; j < resVals.size(); j++) {
      res.startVec(j);
      for (auto it = resVals[j].begin(); it != resVals[j].end(); ++it) {
        res.insertBack(it->first, j) = it->second;
      }
    }
    // Don't forget to call finalize
    res.finalize();

    /********************************************************************************/
    // Do it parallelly
    resVals.clear();
    resVals = ResValsVector(mat.cols());
    numNonZeros = 0;
    SparseQRUtils::parallel_for(0, mat.cols(), [&](const int bi, const int ei) {
      // loop over all items
      for (int j = bi; j < ei; j++)
      {
        DenseVectorType resColJd = mat.col(j).toDense();

        resColJd = 5.0 * resColJd;

        // Write the result back to j-th column of res
        SparseVec resColJ = resColJd.sparseView();
        numNonZeros += resColJ.nonZeros();
        resVals[j].reserve(resColJ.nonZeros());
        for (SparseVec::InnerIterator it(resColJ); it; ++it) {
          resVals[j].push_back(std::make_pair(it.row(), it.value()));
        }
      }
    }, i);
    // Form the output
    JacobianType res2(mat.rows(), mat.cols());
    res2.reserve(numNonZeros);
    for (int j = 0; j < resVals.size(); j++) {
      res2.startVec(j);
      for (auto it = resVals[j].begin(); it != resVals[j].end(); ++it) {
        res2.insertBack(it->first, j) = it->second;
      }
    }
    // Don't forget to call finalize
    res2.finalize();

    /********************************************************************************/
    // Check that result of sequential equals result of parallel
    test_res &= VERIFY_IS_APPROX((res - res2).cwiseAbs().sum(), 0.0);
  }

  return test_res;
}

int main(int argc, char *argv[]) {
    /*
    * Set-up the problem to be solved
    */
    // Problem size
    Eigen::Index numVars = 256;
    Eigen::Index numParams = numVars * 2;
    Eigen::Index numResiduals = numVars * 3 + numVars + numVars * 3;

    // Generate the 7x2 block diagonal pattern, permute and try to find the ordering back
    JacobianType spJ;
    generate_block_diagonal_matrix(numParams, numResiduals, spJ, true);
    RUN_TEST(test_blockdiag_permuted(spJ), 0);

    // Generate the 7x4 overlapping pattern, permute and try to find the ordering back
    generate_overlapping_block_diagonal_matrix(numParams, numResiduals, spJ, true);
    RUN_TEST(test_overlapping_permuted(spJ), 1);

    // Generate the 7x2 block diagonal pattern, and vertically concatenate it with a diagonal matrix
    // and rowpermute to form blocks 9x2
    generate_block_diagonal_matrix(numParams, numResiduals, spJ, false);
    RUN_TEST(test_blockdiag_vertperm_diag(spJ), 2);

    // Generate a random sparse matrix - for example block banded matrix (doesn't really matter)
    // and process it using parallel for
    generate_overlapping_block_diagonal_matrix(numParams, numResiduals, spJ, true);
    RUN_TEST(test_parallel_for(spJ), 3);

    return 0;
}
