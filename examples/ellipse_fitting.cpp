// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Jan Svoboda <jan.svoboda@usi.ch>
// Copyright (C) 2016 Andrew Fitzgibbon <awf@microsoft.com>
// Copyright (C) 2016 Sergio Garrido Jurado <i52gajus@uco.es>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


// import basic and product tests for deprecated DynamicSparseMatrix
#define EIGEN_NO_DEPRECATED_WARNING

#include <iostream>
#include <iomanip>
#include <ctime>
#include <future>
#include <random>

#include "main.h"
//#include "sparse_basic.cpp"
//#include "sparse_product.cpp"

#include <Eigen/Eigen>
#include <Eigen/SparseCore>
#include <QRKit/QRKit>
#include <Eigen/LevenbergMarquardt>

using namespace Eigen;
using namespace QRKit;

// Eigen's better DenseQR:
typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > DenseQR;

typedef int IndexType;

typedef double Scalar;

typedef SparseMatrix<Scalar, ColMajor, IndexType> JacobianType;
typedef Matrix<Scalar, Dynamic, Dynamic> DenseMatrixType;

template <typename _Scalar>
struct EllipseFitting : SparseFunctor<_Scalar, IndexType>
{
  // Class data: 2xN matrix with each column a 2D point
  Matrix2Xd ellipsePoints;

  // Number of parameters in the model, to which will be added
  // one latent variable per point.
  static const int nParamsModel = 5;

  // Constructor initializes points, and tells the base class how many parameters there are in total
  EllipseFitting(const Matrix2Xd& points) :
    SparseFunctor<_Scalar, IndexType>(nParamsModel + points.cols(), points.cols() * 2),
    ellipsePoints(points)
  {
  }

  // Functor functions
  int operator()(const InputType& uv, ValueType& fvec) const 
  {
    // Ellipse parameters are the last 5 entries
    auto params = uv.tail(nParamsModel);
    double a = params[0];
    double b = params[1];
    double x0 = params[2];
    double y0 = params[3];
    double r = params[4];

    // Correspondences (t values) are the first N
    for (int i = 0; i < ellipsePoints.cols(); i++) {
      double t = uv(i);
      double x = a*cos(t)*cos(r) - b*sin(t)*sin(r) + x0;
      double y = a*cos(t)*sin(r) + b*sin(t)*cos(r) + y0;
      fvec(2 * i + 0) = ellipsePoints(0, i) - x;
      fvec(2 * i + 1) = ellipsePoints(1, i) - y;
    }

    return 0;
  }

  // Functor jacobian
  int df(const InputType& uv, JacobianType& fjac)
  {
    // X_i - (a*cos(t_i) + x0)
    // Y_i - (b*sin(t_i) + y0)
    int npoints = ellipsePoints.cols();
    auto params = uv.tail(nParamsModel);
    double a = params[0];
    double b = params[1];
    double r = params[4];

    TripletArray<JacobianType::Scalar, IndexType> triplets(npoints * 2 * 5); // npoints * rows_per_point * nonzeros_per_row
    for (int i = 0; i < npoints; i++) {
      double t = uv(i);
      triplets.add(2 * i, i, +a*cos(r)*sin(t) + b*sin(r)*cos(t));
      triplets.add(2 * i, npoints + 0, -cos(t)*cos(r));
      triplets.add(2 * i, npoints + 1, +sin(t)*sin(r));
      triplets.add(2 * i, npoints + 2, -1);
      triplets.add(2 * i, npoints + 4, +a*cos(t)*sin(r) + b*sin(t)*cos(r));

      triplets.add(2 * i + 1, i, +a*sin(r)*sin(t) - b*cos(r)*cos(t));
      triplets.add(2 * i + 1, npoints + 0, -cos(t)*sin(r));
      triplets.add(2 * i + 1, npoints + 1, -sin(t)*cos(r));
      triplets.add(2 * i + 1, npoints + 3, -1);
      triplets.add(2 * i + 1, npoints + 4, -a*cos(t)*cos(r) + b*sin(t)*sin(r));
    }

    fjac.setFromTriplets(triplets.begin(), triplets.end());
    return 0;
  }
};

template <typename _Scalar>
struct SparseBlockDiagonalQR_EllipseFitting : public EllipseFitting<_Scalar> {
  // QR for J1 subblocks is 2x1
  typedef Matrix<Scalar, 2, 1> DenseMatrix2x1;
  typedef ColPivHouseholderQR<DenseMatrix2x1> DenseQRSolver2x1;
  // QR for J1 is block diagonal
  typedef BlockDiagonalSparseQR<DenseQRSolver2x1> LeftSuperBlockSolver;
  // QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
  typedef DenseQR RightSuperBlockSolver;
  
  class QRSolver : public BlockAngularSparseQR<LeftSuperBlockSolver, RightSuperBlockSolver> {
    const int RightBlockColumns = nParamsModel;
  public:
    // Solver has to know how to treat input general SparseMatrix
    QRSolver(const JacobianType& mat) 
    {
      // Left block is sparse block diagonal matrix
      JacobianType leftBlock = mat.block(0, 0, mat.rows(), mat.cols() - RightBlockColumns);
      SparseBlockDiagonal<DenseMatrix2x1> leftBlockDiag;
      leftBlockDiag.fromBlockDiagonalPattern<JacobianType>(leftBlock, 2, 1);
      // Right block is general DenseMatrix
      DenseMatrixType rightBlock = mat.block(0, mat.cols() - RightBlockColumns, mat.rows(), RightBlockColumns);
      // Input to the block angular solver is sparse block angular matrix
      BlockMatrix1x2<SparseBlockDiagonal<DenseMatrix2x1>, DenseMatrixType> blkAngular(leftBlockDiag, rightBlock);
      this->compute(blkAngular);
    }
  };
  
  SparseBlockDiagonalQR_EllipseFitting(const Matrix2Xd& points) :
    EllipseFitting<_Scalar>(points) {
  }
};

template <typename _Scalar>
struct SparseBlockBandedQR_EllipseFitting : public EllipseFitting<_Scalar> 
{
  typedef Matrix<Scalar, 2, 1> DenseMatrix2x1;
  typedef HouseholderQR<DenseMatrix2x1> BandBlockQRSolver;
  typedef BandedBlockedSparseQR<JacobianType, BandBlockQRSolver, 0, 8, false> BandedBlockedQRSolver;
  // QR for J1 is banded blocked QR
  typedef BandedBlockedQRSolver LeftSuperBlockSolver;
  // QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
  typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic>> RightSuperBlockSolver;
  
  // Define the full QR solver
  class QRSolver : public BlockAngularSparseQR<LeftSuperBlockSolver, RightSuperBlockSolver> {
    const int RightBlockColumns = 5;
  public:
    // Solver has to know how to treat input general SparseMatrix
    QRSolver(const JacobianType& mat)
    {
      // Left block is general SparseMatrix
      JacobianType leftBlock = mat.block(0, 0, mat.rows(), mat.cols() - RightBlockColumns);
      // Right block is general DenseMatrix
      DenseMatrixType rightBlock = mat.block(0, mat.cols() - RightBlockColumns, mat.rows(), RightBlockColumns);
      // Input to the block angular solver is sparse block angular matrix
      BlockMatrix1x2<JacobianType, DenseMatrixType> blkAngular(leftBlock, rightBlock);
      this->compute(blkAngular);
    }
  };

  SparseBlockBandedQR_EllipseFitting(const Matrix2Xd& points) :
    EllipseFitting<_Scalar>(points) {
  }
};

typedef EllipseFitting<Scalar>::InputType ParamsType;

void printParamsHeader() 
{
  std::cout << "a \t";
  std::cout << "b \t";
  std::cout << "x0\t";
  std::cout << "y0\t";
  std::cout << "r \t";
  std::cout << "Duration";
  std::cout << std::endl;
}

void printParams(ParamsType &params, int npoints, double duration = -1.) 
{
  std::cout << params(npoints) << "\t";
  std::cout << params(npoints + 1) << "\t";
  std::cout << params(npoints + 2) << "\t";
  std::cout << params(npoints + 3) << "\t";
  std::cout << params(npoints + 4)*180. / EIGEN_PI << "\t";
  if (duration >= 0) {
    std::cout << duration << "s";
  }
  std::cout << std::endl;
}

void initializeParams(const Matrix2Xd &ellipsePoints, double incr, ParamsType &params) 
{
  int npoints = ellipsePoints.cols(); 
  
  params.resize(EllipseFitting<Scalar>::nParamsModel + npoints);
 
  // TODO: use Eigen's max/min over columns mat.colwise().maxCoeff() 
  double minX, minY, maxX, maxY;
  minX = maxX = ellipsePoints(0, 0);
  minY = maxY = ellipsePoints(1, 0);
  for (int i = 0; i<npoints; i++) {
    minX = min(minX, ellipsePoints(0, i));
    maxX = max(maxX, ellipsePoints(0, i));
    minY = min(minY, ellipsePoints(1, i));
    maxY = max(maxY, ellipsePoints(1, i));
  }
  params(npoints) = 0.5*(maxX - minX);
  params(npoints + 1) = 0.5*(maxY - minY);
  params(npoints + 2) = 0.5*(maxX + minX);
  params(npoints + 3) = 0.5*(maxY + minY);
  params(npoints + 4) = 0;
  for (int i = 0; i<npoints; i++) {
    params(i) = Scalar(i)*incr;
  }
}

void checkParamsAmbiguity(const Matrix2Xd &ellipsePoints, ParamsType &params)
{
  int npoints = ellipsePoints.cols();

  // check parameters ambiguity before test result
  // a should be bigger than b
  if (fabs(params(npoints + 1)) > fabs(params(npoints))) {
    std::swap(params(npoints), params(npoints + 1));
    params(npoints + 4) -= 0.5*EIGEN_PI;
  }
  // a and b should be positive
  if (params(npoints)<0) {
    params(npoints) *= -1.;
    params(npoints + 1) *= -1.;
    params(npoints + 4) += EIGEN_PI;
  }
  // fix rotation angle range
  while (params(npoints + 4) < 0) params(npoints + 4) += 2.*EIGEN_PI;
  while (params(npoints + 4) > EIGEN_PI) params(npoints + 4) -= EIGEN_PI;
}

void test_block_diagonal(const Ellipse &el, const Matrix2Xd &ellipsePoints, ParamsType &params)
{
  Eigen::LevenbergMarquardtSpace::Status info;
  typedef SparseBlockDiagonalQR_EllipseFitting<Scalar>  SparseBlockDiagonalQRFunctor;
  SparseBlockDiagonalQRFunctor functor3(ellipsePoints);
  Eigen::LevenbergMarquardt< SparseBlockDiagonalQRFunctor > lm3(functor3);
  info = lm3.minimize(params);

  checkParamsAmbiguity(ellipsePoints, params);

  VERIFY_IS_APPROX(el.a, params(ellipsePoints.cols()));
  VERIFY_IS_APPROX(el.b, params(ellipsePoints.cols() + 1));
  VERIFY_IS_APPROX(el.x0, params(ellipsePoints.cols() + 2));
  VERIFY_IS_APPROX(el.y0, params(ellipsePoints.cols() + 3));
  VERIFY_IS_APPROX(el.r, params(ellipsePoints.cols() + 4));
}

void test_banded_blocked(const Ellipse &el, const Matrix2Xd &ellipsePoints, ParamsType &params)
{
  Eigen::LevenbergMarquardtSpace::Status info;
  typedef SparseBlockBandedQR_EllipseFitting<Scalar>  SparseBlockBandedQRFunctor;
  SparseBlockBandedQRFunctor functor4(ellipsePoints);
  Eigen::LevenbergMarquardt<SparseBlockBandedQRFunctor> lm4(functor4);
  info = lm4.minimize(params);

  checkParamsAmbiguity(ellipsePoints, params);

  VERIFY_IS_APPROX(el.a, params(ellipsePoints.cols()));
  VERIFY_IS_APPROX(el.b, params(ellipsePoints.cols() + 1));
  VERIFY_IS_APPROX(el.x0, params(ellipsePoints.cols() + 2));
  VERIFY_IS_APPROX(el.y0, params(ellipsePoints.cols() + 3));
  VERIFY_IS_APPROX(el.r, params(ellipsePoints.cols() + 4));
}

// TODO: this looks like it's not used below?
struct Ellipse {
  Ellipse()
    : a(1), b(1), x0(0), y0(0), r(1) {
  }

  Ellipse(const double a, const double b, const double x0, const double y0, const double r)
    : a(a), b(b), x0(x0), y0(y0), r(r) {
  }

  double a;
  double b;
  double x0;
  double y0;
  double r;
};

void test_sparse_qr_extra_lm_fitting()
{
  const int NumSamplePoints = 50000;
  // Create the ellipse parameters and data points
  // ELLIPSE PARAMETERS
  Ellipse el(7.5, 2, 17, 23., 0.23);

  // CREATE DATA SAMPLES
  int npoints = NumSamplePoints;
  Matrix2Xd ellipsePoints;
  ellipsePoints.resize(2, npoints);
  Scalar incr = 1.3*EIGEN_PI / Scalar(npoints);
  for (int i = 0; i<npoints; i++) {
    Scalar t = Scalar(i)*incr;
    ellipsePoints(0, i) = el.x0 + el.a*cos(t)*cos(el.r) - el.b*sin(t)*sin(el.r);
    ellipsePoints(1, i) = el.y0 + el.a*cos(t)*sin(el.r) + el.b*sin(t)*cos(el.r);
  }


  // Test LM with block diagonal solver
  ParamsType params;
  initializeParams(ellipsePoints, incr, params);
  CALL_SUBTEST_1(test_block_diagonal(el, ellipsePoints, params));


  // Test LM with banded blocked solver
  initializeParams(ellipsePoints, incr, params);
  CALL_SUBTEST_2(test_banded_blocked(el, ellipsePoints, params));
}
