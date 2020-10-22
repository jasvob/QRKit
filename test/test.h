// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Jan Svoboda <jan.svoboda@nnaisense.com>
// Copyright (C) 2020 Andrew Fitzgibbon <awf@microsoft.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef TEST_H
#define TEST_H

#include <Eigen/Core>

// Running a test
#define RUN_TEST(test_fn, test_num) std::cout << "Running test " << test_num << "..." << std::endl; \
	if(test_fn) { \
		std::cout << "Passed." << std::endl; \
	} else { \
		std::cout << "Failed." << std::endl; \
	}

// Evaluating results of a test
#define VERIFY_IS_APPROX(x, y) (call_verify_is_approx(x, y))
#define VERIFY_IS_EQUAL(x, y) (call_verify_is_equal(x, y))

// Define precisions for different types
template<typename T> inline typename Eigen::NumTraits<T>::Real test_precision() { return Eigen::NumTraits<T>::dummy_precision(); }
template<> inline float test_precision<float>() { return 1e-3f; }
template<> inline double test_precision<double>() { return 1e-6; }
template<> inline long double test_precision<long double>() { return 1e-6l; }
template<> inline float test_precision<std::complex<float> >() { return test_precision<float>(); }
template<> inline double test_precision<std::complex<double> >() { return test_precision<double>(); }
template<> inline long double test_precision<std::complex<long double> >() { return test_precision<long double>(); }

// Verify-is-approx function overloads
inline bool verify_is_approx(const short &x, const short &y) {
	return Eigen::internal::isApprox(x, y, test_precision<short>());
}
inline bool verify_is_approx(const unsigned short &x, const unsigned short &y) {
	return Eigen::internal::isApprox(x, y, test_precision<unsigned short>());
}
inline bool verify_is_approx(const unsigned int &x, const unsigned int &y) {
	return Eigen::internal::isApprox(x, y, test_precision<unsigned int>());
}
inline bool verify_is_approx(const long &x, const long &y) {
	return Eigen::internal::isApprox(x, y, test_precision<long>());
}
inline bool verify_is_approx(const unsigned long& x, const unsigned long& y) {
	return Eigen::internal::isApprox(x, y, test_precision<unsigned long>());
}
inline bool verify_is_approx(const int &x, const int &y) {
	return Eigen::internal::isApprox(x, y, test_precision<int>());
}
inline bool verify_is_approx(const float &x, const float &y) {
	return Eigen::internal::isApprox(x, y, test_precision<float>());
}
inline bool verify_is_approx(const double &x, const double &y) {
	return Eigen::internal::isApprox(x, y, test_precision<double>());
}
template <typename T1, typename T2>
inline bool verify_is_approx(const T1 &x, const T2 &y, typename T1::Scalar* = 0) {
	return x.isApprox(y, test_precision<typename T1::Scalar>());
}

template<typename T1, typename T2>
inline bool call_verify_is_approx(const T1& x, const T2& y) {
	bool ret = verify_is_approx(x, y);
	if (!ret) {
		std::cerr << "Difference too large wrt tolerance!" << std::endl; // << get_test_precision(a) << ", relative error is: " << test_relative_error(a, b) << std::endl;
	}
	return ret;
}


// Verify is equal
template<typename T1, typename T2>
bool call_verify_is_equal(const T1 &actual, const T2 &expected, bool expect_equal = true);
template<typename T1, typename T2>
bool call_verify_is_equal(const T1 &actual, const T2 &expected, bool expect_equal) {
	if ((actual == expected) == expect_equal)
		return true;
	
	// false:
	std::cerr
		<< std::endl << "    actual   = " << actual
		<< std::endl << "    expected " << (expect_equal ? "= " : "!=") << expected << std::endl;

	return false;
}

#endif
