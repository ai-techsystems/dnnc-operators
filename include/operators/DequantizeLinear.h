// Copyright 2018 The AITS DNNC Authors.All Rights Reserved.
//
// Licensed to the Apache Software Foundation(ASF) under one
// or more contributor license agreements.See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.See the License for the
// specific language governing permissionsand limitations
// under the License.
//
// This file is part of AITS DNN compiler maintained at
// https://github.com/ai-techsystems/dnnCompiler
//

#pragma once
#include "operators/baseOperator.h"
#include <string>

using namespace Eigen;

namespace dnnc {
template <typename T> class DequantizeLinear : public baseOperator<T> {
public:
  DequantizeLinear(std::string name = "opDequantizeLinear")
      : baseOperator<T>(opDequantizeLinear, name) {}
	    
  tensor<T> 
      compute(tensor<T>& a,tensor<T>& a_scale,tensor<T>& a_zero_point)
	  {
		  if (a_scale.shape() != a_zero_point.shape())
			  throw std::invalid_argument("tensor dimenions not appropriate for DequantizeLinear operator."); 
		  /*
		  
		  // Check this when you can take input both int and float tensor together in compute function

		  if (typeid(a_zero_point)!=typeid(tensor<int>) || typeid(a)!=typeid(tensor<int>))
			  throw std::invalid_argument("tensor types not appropriate for DequantizeLinear operator."); 
		  
		  */
		  tensor<T> result(a.shape()[0], a.shape()[1]); 
		  
		  DNNC_EIGEN_MATRIX(eigenMatrixA, a) ; 
		  DNNC_EIGEN_MATRIX(eigenMatrixA_scale, a_scale) ; 
		  DNNC_EIGEN_MATRIX(eigenMatrixA_zero_point, a_zero_point) ; 

		  Matrix<T, Dynamic, Dynamic> eResult = (eigenMatrixA - eigenMatrixA_zero_point) * eigenMatrixA_scale;
		  
		  result.load( eResult.data() ); 

		  return result;
	  }
};
} // namespace dnnc