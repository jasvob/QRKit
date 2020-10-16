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


// import basic and product tests for deprectaed DynamicSparseMatrix
#define EIGEN_NO_DEPRECATED_WARNING

//#define WITH_SPQR 

#include <iostream>
#include <iomanip>
#include <ctime>
#include <future>
#include <random>

#include "eigen/test/main.h"

#include <Eigen/Eigen>
#include <Eigen/SparseCore>
#include <QRKit/QRKit>
#include <Eigen/LevenbergMarquardt>

#ifdef WITH_SPQR
#include <suitesparse/SuiteSparseQR.hpp>
#endif

#include <Eigen/src/CholmodSupport/CholmodSupport.h>
#include <Eigen/src/SPQRSupport/SuiteSparseQRSupport.h>


using namespace Eigen;
using namespace QRKit;

#ifdef WITH_SPQR
typedef SuiteSparse_long IndexType;
#else
typedef int IndexType;
#endif

typedef SparseMatrix<Scalar, ColMajor, IndexType> JacobianType;
typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

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
  int operator()(const InputType& uv, ValueType& fvec) const {
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
  int df(const InputType& uv, JacobianType& fjac) {
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
struct SparseQR_EllipseFitting : public EllipseFitting<_Scalar> {
  // For generic Jacobian, one might use this Dense QR solver.
  typedef SparseQR<JacobianType, COLAMDOrdering<IndexType> > GeneralQRSolver;
  typedef GeneralQRSolver QRSolver;

  SparseQR_EllipseFitting(const Matrix2Xd& points) :
    EllipseFitting<_Scalar>(points) {
  }

  void initQRSolver(GeneralQRSolver &qr) {
    // set left block size
  }
};

#ifdef WITH_SPQR
template <typename _Scalar>
struct SPQR_EllipseFitting : public EllipseFitting<_Scalar> {
  typedef SPQR<JacobianType> SPQRSolver;
  typedef SPQRSolver QRSolver;

  SPQR_EllipseFitting(const Matrix2Xd& points) :
    EllipseFitting<_Scalar>(points) {
  }

  // And tell the algorithm how to set the QR parameters.
  void initQRSolver(SPQRSolver &qr) {

  }
};
#endif

template <typename _Scalar>
struct SparseBlockDiagonalQR_EllipseFitting : public EllipseFitting<_Scalar> {
  // QR for J1 subblocks is 2x1
  typedef ColPivHouseholderQR<Matrix<Scalar, 2, 1> > DenseQRSolver2x1;
  // QR for J1 is block diagonal
  typedef BlockDiagonalSparseQR<JacobianType, DenseQRSolver2x1> LeftSuperBlockSolver;
  // QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
  typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > RightSuperBlockSolver;
  // QR for J is concatenation of the above.
  typedef BlockAngularSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver, 5> SchurlikeQRSolver;

  typedef SchurlikeQRSolver QRSolver;


  SparseBlockDiagonalQR_EllipseFitting(const Matrix2Xd& points) :
    EllipseFitting<_Scalar>(points) {
  }

  // And tell the algorithm how to set the QR parameters.
  void initQRSolver(SchurlikeQRSolver &qr) {
  }
};

template <typename _Scalar>
struct SparseBlockBandedQR_EllipseFitting : public EllipseFitting<_Scalar> {
  typedef HouseholderQR<Matrix<Scalar, 2, 1>> BandBlockQRSolver;
  typedef BandedBlockedSparseQR<JacobianType, BandBlockQRSolver, 0, 8, false> BandedBlockedQRSolver;
  // QR for J1 is banded blocked QR
  typedef BandedBlockedQRSolver LeftSuperBlockSolver;
  // QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
  typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic>> RightSuperBlockSolver;
  // QR solver for sparse block angular matrix
  typedef BlockAngularSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver, 5> BlockAngularQRSolver;

  typedef BlockAngularQRSolver QRSolver;

  SparseBlockBandedQR_EllipseFitting(const Matrix2Xd& points) :
    EllipseFitting<_Scalar>(points) {
  }

  void initQRSolver(BlockAngularQRSolver &qr) {
  }
};

typedef EllipseFitting<Scalar>::InputType ParamsType;

void printParamsHeader() {
  std::cout << "a \t";
  std::cout << "b \t";
  std::cout << "x0\t";
  std::cout << "y0\t";
  std::cout << "r \t";
  std::cout << "Duration";
  std::cout << std::endl;
}

void printParams(ParamsType &params, int nDataPoints, double duration = -1.) {
  std::cout << params(nDataPoints) << "\t";
  std::cout << params(nDataPoints + 1) << "\t";
  std::cout << params(nDataPoints + 2) << "\t";
  std::cout << params(nDataPoints + 3) << "\t";
  std::cout << params(nDataPoints + 4)*180. / EIGEN_PI << "\t";
  if (duration >= 0) {
    std::cout << duration << "s";
  }
  std::cout << std::endl;
}

void initializeParams(int nDataPoints, const Matrix2Xd &ellipsePoints, double incr, ParamsType &params) {
  params.resize(EllipseFitting<Scalar>::nParamsModel + nDataPoints);
  double minX, minY, maxX, maxY;
  minX = maxX = ellipsePoints(0, 0);
  minY = maxY = ellipsePoints(1, 0);
  for (int i = 0; i<ellipsePoints.cols(); i++) {
    minX = (std::min)(minX, ellipsePoints(0, i));
    maxX = (std::max)(maxX, ellipsePoints(0, i));
    minY = (std::min)(minY, ellipsePoints(1, i));
    maxY = (std::max)(maxY, ellipsePoints(1, i));
  }
  params(ellipsePoints.cols()) = 0.5*(maxX - minX);
  params(ellipsePoints.cols() + 1) = 0.5*(maxY - minY);
  params(ellipsePoints.cols() + 2) = 0.5*(maxX + minX);
  params(ellipsePoints.cols() + 3) = 0.5*(maxY + minY);
  params(ellipsePoints.cols() + 4) = 0;
  for (int i = 0; i<ellipsePoints.cols(); i++) {
    params(i) = Scalar(i)*incr;
  }
}

#define LM_VERBOSE 0

const int NumTests = 8;
const int SparseQR_LimitN = 2000;
const int SPQR_LimitN = 500000; // For some reason it is failing to converge -> ruled out for now...
int NumSamplePoints[NumTests] = { 500, 1000, 2000, 5000, 10000, 50000, 100000, 500000 };

int main() {
  /***************************************************************************/
  std::cout << "################### Ellipse fitting benchmark #######################" << std::endl;
  std::cout << "#####################################################################" << std::endl;
  std::cout << "N - Number of data points" << std::endl;
  std::cout << "Sparse QR - Eigen's Sparse QR" << std::endl;
  std::cout << "SuiteSparse QR - SuiteSparse QR" << std::endl;
  std::cout << "Bl Diag Sp QR - Block Diagonal Sparse QR" << std::endl;
  std::cout << "Sp Band Bl QR - Sparse Banded Blocked QR" << std::endl;
  std::cout << "#####################################################################" << std::endl;
  for (int i = 0; i < NumTests; i++) {
    // Create the ellipse paramteers and data points
    // ELLIPSE PARAMETERS
    double a, b, x0, y0, r;
    a = 7.5;
    b = 2;
    x0 = 17.;
    y0 = 23.;
    r = 0.23;

    std::cout << "N = " << NumSamplePoints[i] << "   \t";
    printParamsHeader();
    std::cout << "=====================================================================" << std::endl;

    // CREATE DATA SAMPLES
    int nDataPoints = NumSamplePoints[i];
    Matrix2Xd ellipsePoints;
    ellipsePoints.resize(2, nDataPoints);
    Scalar incr = 1.3*EIGEN_PI / Scalar(nDataPoints);
    for (int i = 0; i<nDataPoints; i++) {
      Scalar t = Scalar(i)*incr;
      ellipsePoints(0, i) = x0 + a*cos(t)*cos(r) - b*sin(t)*sin(r);
      ellipsePoints(1, i) = y0 + a*cos(t)*sin(r) + b*sin(t)*cos(r);
    }

    // INITIAL PARAMS
    ParamsType params;
    initializeParams(nDataPoints, ellipsePoints, incr, params);

    /***************************************************************************/
    // Run the optimization problem
    std::cout << "Initialization:" << "\t";
    printParams(params, nDataPoints);
    std::cout << "Ground Truth:" << "\t";
    std::cout << a << "\t";
    std::cout << b << "\t";
    std::cout << x0 << "\t";
    std::cout << y0 << "\t";
    std::cout << r*180. / EIGEN_PI << "\t";
    std::cout << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;

    clock_t begin;
    double duration;
    Eigen::LevenbergMarquardtSpace::Status info;
    if (NumSamplePoints[i] <= SparseQR_LimitN) {
      initializeParams(nDataPoints, ellipsePoints, incr, params);
      typedef SparseQR_EllipseFitting<Scalar>  SparseQRFunctor;
      SparseQRFunctor functor1(ellipsePoints);
      Eigen::LevenbergMarquardt< SparseQRFunctor > lm1(functor1);
      lm1.setVerbose(LM_VERBOSE);
      begin = clock();
      info = lm1.minimize(params);
      duration = double(clock() - begin) / CLOCKS_PER_SEC;
      std::cout << "Sparse QR:\t";
      printParams(params, nDataPoints, duration);
    }

#ifdef WITH_SPQR
    if (NumSamplePoints[i] <= SPQR_LimitN) {
      initializeParams(nDataPoints, ellipsePoints, incr, params);
      typedef SPQR_EllipseFitting<Scalar>  SPQRFunctor;
      SPQRFunctor functor2(ellipsePoints);
      Eigen::LevenbergMarquardt< SPQRFunctor > lm2(functor2);
      lm2.setVerbose(LM_VERBOSE);
      begin = clock();
      info = lm2.minimize(params);
      duration = double(clock() - begin) / CLOCKS_PER_SEC;
      std::cout << "SuiteSparse QR:\t";
      printParams(params, nDataPoints, duration);
    }
#endif

    initializeParams(nDataPoints, ellipsePoints, incr, params);
    typedef SparseBlockDiagonalQR_EllipseFitting<Scalar>  SparseBlockDiagonalQRFunctor;
    SparseBlockDiagonalQRFunctor functor3(ellipsePoints);
    Eigen::LevenbergMarquardt< SparseBlockDiagonalQRFunctor > lm3(functor3);
    lm3.setVerbose(LM_VERBOSE);
    begin = clock();
    info = lm3.minimize(params);
    duration = double(clock() - begin) / CLOCKS_PER_SEC;
    std::cout << "Bl Diag Sp QR:\t";
    printParams(params, nDataPoints, duration);

    initializeParams(nDataPoints, ellipsePoints, incr, params);
    typedef SparseBlockBandedQR_EllipseFitting<Scalar>  SparseBlockBandedQRFunctor;
    SparseBlockBandedQRFunctor functor4(ellipsePoints);
    Eigen::LevenbergMarquardt< SparseBlockBandedQRFunctor > lm4(functor4);
    lm4.setVerbose(LM_VERBOSE);
    begin = clock();
    info = lm4.minimize(params);
    duration = double(clock() - begin) / CLOCKS_PER_SEC;
    std::cout << "Sp Band Bl QR:\t";
    printParams(params, nDataPoints, duration);
    std::cout << "#####################################################################" << std::endl;
  }

  return 0;
}

