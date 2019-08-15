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
template <typename T> class LeakyRelu : public baseOperator<T> {
public:
  LeakyRelu(std::string name = "opLeakyRelu")
      : baseOperator<T>(opLeakyRelu, name) {}
      float alpha = 0.01;
      bool getAttribute(OPATTR attrName,float &obj)
      {
        if (attrName == attr_alpha){
          obj = alpha;
          return true;
        }
        return false;
      }
      void setAttribute(OPATTR attrName,float &obj)
      {
        if (attrName == attr_alpha){
          alpha = obj;
        }
      }
      static bool compare()
      {
        return ( (typeid(T) == typeid(float))||(typeid(T) == typeid(double)) );
      }

      static T Leaky_Relu(T x,float alpha){
        if(x<0)
          return T(alpha*x);
        else
          return x;
      }

      tensor<T> compute(tensor<T>& a)
      {
      if(!compare() )
          throw std::invalid_argument("Constrain input and output types to float tensors.");
      tensor<T> result(a.shape(),a.name());
      //f(x) = alpha * x for x < 0, f(x) = x for x >= 0
      auto c0 = std::bind(Leaky_Relu, std::placeholders::_1, alpha);
      for(size_t i=0;i< a.length();i++)
      {
        result[i]=c0(a[i]);
      }
      return result;
      }
};
} // namespace dnnc
