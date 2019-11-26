/*
 *
 * linear_regression.hpp
 * Header file for regression
 *
 */
#pragma once

// C/C++/C++11 headers
#include <tuple>

// dlib headers
#include <dlib/matrix.h>
typedef dlib::matrix<double,0,1> column_vector;

#include "link_function.hpp"
#include "reference.hpp"

std::tuple<float, float> core_acc = regress_kmers(const Reference * r1, 
                                                  const Reference * r2, 
                                                  const column_vector& kmers);;

