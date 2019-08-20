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
#include "core/tensor.h"
#include "operators/Add.h"
#include "operators/MatMul.h"
#include "operators/ThresholdedRelu.h"
#include "operators/GlobalAveragePool.h"
#include "operators/GlobalLpPool.h"
#include "operators/GlobalMaxPool.h"
#include "operators/Greater.h"
#include "operators/Hardmax.h"
#include "operators/HardSigmoid.h"
#include "operators/Identity.h"
#include "operators/IsInf.h"
#include "operators/IsNaN.h"
#include "operators/LeakyRelu.h"
#include "operators/InstanceNormalization.h"

using namespace dnnc;

tensor<float> make_tensor(size_t x, size_t y = 0, size_t z = 0, size_t w = 0) {
  return tensor<float>(x, y, z, w);
}

tensor<float> multiply(tensor<float> &a, tensor<float> &b) {
  MatMul<float> op;
  return op.compute(a, b);
}

tensor<float> add(tensor<float> &a, tensor<float> &b) {
  Add<float> op;
  return op.compute(a, b);
}

tensor<float> thresholded_relu(tensor<float> &input) {
  ThresholdedRelu<float> op;
  return op.compute(input);
}

tensor<float> global_average_pool(tensor<float> &input) {
  GlobalAveragePool<float> op;
  return op.compute(input);
}

tensor<float> global_lp_pool(tensor<float> & a,int p=2) {
  GlobalLpPool<float> op("localOpName",p) ;
  return op.compute(a);
}

tensor<float> global_max_pool(tensor<float>& a) {
  GlobalMaxPool<float> op;
  return op.compute(a);
}

tensor<bool> greater(tensor<float>& a, tensor<float> &b) {
  Greater<float> op;
  return op.compute(a,b);
}

tensor<float> hardmax(tensor<float>& a,int axis=0) {
  Hardmax<float> op("localOpName",axis);
  return op.compute(a);
}

tensor<float> hardsigmoid(tensor<float>& a,float alpha = 0.2,float beta = 0.5) {
  HardSigmoid<float> op("localOpName",alpha,beta);
  return op.compute(a);
}

tensor<float> identity(tensor<float>& a) {
  Identity<float> op;
  return op.compute(a);
}

tensor<bool> isinf(tensor<float>&a,int detect_positive=1,int detect_negative=1) {
  IsInf<float> op("localOpName",detect_positive,detect_negative);
  return op.compute(a);
}

tensor<bool> isnan(tensor<float>&a) {
  IsNaN<float> op;
  return op.compute(a);
}

tensor<float> leakyrelu(tensor<float>& a,float alpha = 0.01) {
  LeakyRelu<float> op("localOpName",alpha);
  return op.compute(a);
}

tensor<float> instancenormalization(tensor<float>& input,tensor<float>& scale,tensor<float>& B,float epsilon=1e-5) {
  InstanceNormalization<float> op("localOpName",epsilon);
  return op.compute(input,scale,B);
}
