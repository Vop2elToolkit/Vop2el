/*
 * Project: Vop2el
 *
 * Author: Mohamed Mssaouri
 *
 * Copyright (c) 2024 Mohamed Mssaouri
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace Vop2el
{
//-------------------------------------------------------------------------------------------
struct EssentielMatrixOptimizer
{
  EssentielMatrixOptimizer(Eigen::Vector3d refImagePointCoor,
                        Eigen::Vector3d actImagePointCoor,
                        Eigen::Matrix3d calibrationMatrix) :
                        RefImagePointCoor(refImagePointCoor), ActImagePointCoor(actImagePointCoor), CalibrationMatrix(calibrationMatrix) {}

  template <typename T>
  bool operator()(const T* const x, T* residual) const
  {
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> essentielMatrix(&x[0]);

    Eigen::Matrix<T, 3, 3> calibration = this->CalibrationMatrix.cast<T>();

    Eigen::Matrix<T, 3, 3> fundamentalToActual = (calibration.transpose().inverse() * essentielMatrix * calibration.inverse()).transpose();

    Eigen::Matrix<T, 3, 1> epipolarLineActImage =
          fundamentalToActual * Eigen::Matrix<T, 3, 1>(T(RefImagePointCoor[0]), T(RefImagePointCoor[1]), T(RefImagePointCoor[2]));

    Eigen::Matrix<T, 3, 3> fundamentalToPrevious = (calibration.transpose().inverse() * essentielMatrix * calibration.inverse());

    Eigen::Matrix<T, 3, 1> epipolarLineRefImage =
          fundamentalToPrevious * Eigen::Matrix<T, 3, 1>(T(ActImagePointCoor[0]), T(ActImagePointCoor[1]), T(ActImagePointCoor[2]));

    T refImageResidus = epipolarLineRefImage.dot(RefImagePointCoor) /
                              ceres::sqrt(ceres::pow(epipolarLineRefImage[0], 2) + ceres::pow(epipolarLineRefImage[1], 2));

    T actImageResidus = epipolarLineActImage.dot(ActImagePointCoor) /
                              ceres::sqrt(ceres::pow(epipolarLineActImage[0], 2) + ceres::pow(epipolarLineActImage[1], 2));

    residual[0] = !ceres::isinf(refImageResidus) ? refImageResidus : T(0.0);
    residual[1] = !ceres::isinf(actImageResidus) ? actImageResidus : T(0.0);

    return true;
  }

  Eigen::Vector3d RefImagePointCoor;
  Eigen::Vector3d ActImagePointCoor;
  Eigen::Matrix3d CalibrationMatrix;
};

//-------------------------------------------------------------------------------------------
struct ScaleCostPreviousLeftActualRight
{
  ScaleCostPreviousLeftActualRight(Eigen::Vector3d refImagePointCoor,
                                  Eigen::Vector3d actImagePointCoor,
                                  Eigen::Affine3d transformPreviousActual,
                                  Eigen::Affine3d transformExtrinsicCamera,
                                  Eigen::Matrix3d calibrationMatrix) :
    RefImagePointCoor(refImagePointCoor), ActImagePointCoor(actImagePointCoor),
    TransformPreviousActual(transformPreviousActual), TransformExtrinsicCamera(transformExtrinsicCamera),
    CalibrationMatrix(calibrationMatrix) {}

  template <typename T>
  bool operator()(const T* const x, T* residual) const
  {
    Eigen::Transform<T, 3, Eigen::Affine> transformPreviousLeftActualLeft = this->TransformPreviousActual.cast<T>();
    transformPreviousLeftActualLeft.translation() = x[0] * transformPreviousLeftActualLeft.translation();

    Eigen::Transform<T, 3, Eigen::Affine> transformPreviousLeftActualRight =
                              transformPreviousLeftActualLeft * this->TransformExtrinsicCamera.cast<T>();

    Eigen::Matrix<T, 3, 1> translation = transformPreviousLeftActualRight.translation();
    T skewSymetricTranslationArray[9] = {T(0.0), -translation(2), translation(1),
                                         translation(2), T(0.0), -translation(0),
                                         -translation(1), translation(0), T(0.0)};

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> skewSymetricTranslation(&skewSymetricTranslationArray[0]);
    Eigen::Matrix<T, 3, 3> essentielMatrix = skewSymetricTranslation * transformPreviousLeftActualRight.rotation();

    Eigen::Matrix<T, 3, 3> calibration = this->CalibrationMatrix.cast<T>();

    Eigen::Matrix<T, 3, 3> fundamentalToActual = calibration.transpose().inverse() * essentielMatrix * calibration.inverse();

    Eigen::Matrix<T, 3, 1> epipolarLineActImage = fundamentalToActual.transpose() *
                Eigen::Matrix<T, 3, 1>(T(RefImagePointCoor[0]), T(RefImagePointCoor[1]), T(RefImagePointCoor[2]));

    Eigen::Matrix<T, 3, 1> epipolarLineRefImage = fundamentalToActual *
                Eigen::Matrix<T, 3, 1>(T(ActImagePointCoor[0]), T(ActImagePointCoor[1]), T(ActImagePointCoor[2]));

    T refImageResidus = epipolarLineRefImage.dot(RefImagePointCoor) /
                              ceres::sqrt(ceres::pow(epipolarLineRefImage[0], 2) + ceres::pow(epipolarLineRefImage[1], 2));

    T actImageResidus = epipolarLineActImage.dot(ActImagePointCoor) /
                              ceres::sqrt(ceres::pow(epipolarLineActImage[0], 2) + ceres::pow(epipolarLineActImage[1], 2));

    residual[0] = !ceres::isinf(refImageResidus) ? refImageResidus : T(0.0);
    residual[1] = !ceres::isinf(actImageResidus) ? actImageResidus : T(0.0);

    return true;
  }

  Eigen::Vector3d RefImagePointCoor;
  Eigen::Vector3d ActImagePointCoor;
  Eigen::Affine3d TransformPreviousActual;
  Eigen::Affine3d TransformExtrinsicCamera;
  Eigen::Matrix3d CalibrationMatrix;
};

//-------------------------------------------------------------------------------------------
struct ScaleCostActualLeftPreviousRight
{
  ScaleCostActualLeftPreviousRight(Eigen::Vector3d refImagePointCoor,
                                  Eigen::Vector3d actImagePointCoor,
                                  Eigen::Affine3d transformPreviousActual,
                                  Eigen::Affine3d transformExtrinsicCamera,
                                  Eigen::Matrix3d calibrationMatrix) :
    RefImagePointCoor(refImagePointCoor), ActImagePointCoor(actImagePointCoor),
    TransformPreviousActual(transformPreviousActual), TransformExtrinsicCamera(transformExtrinsicCamera),
    CalibrationMatrix(calibrationMatrix) {}

  template <typename T>
  bool operator()(const T* const x, T* residual) const
  {
    Eigen::Transform<T, 3, Eigen::Affine> transformPreviousLeftActualLeft = this->TransformPreviousActual.cast<T>();
    transformPreviousLeftActualLeft.translation() = x[0] * transformPreviousLeftActualLeft.translation();
    Eigen::Transform<T, 3, Eigen::Affine> transformActualLeftPreviousRight =
                            transformPreviousLeftActualLeft.inverse() * this->TransformExtrinsicCamera.cast<T>();

    Eigen::Matrix<T, 3, 1> translation = transformActualLeftPreviousRight.translation();
    T skewSymetricTranslationArray[9] = {T(0.0), -translation(2), translation(1),
                                         translation(2), T(0.0), -translation(0),
                                         -translation(1), translation(0), T(0.0)};

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> skewSymetricTranslation(&skewSymetricTranslationArray[0]);
    Eigen::Matrix<T, 3, 3> essentielMatrix = skewSymetricTranslation * transformActualLeftPreviousRight.rotation();

    Eigen::Matrix<T, 3, 3> calibration = this->CalibrationMatrix.cast<T>();
    Eigen::Matrix<T, 3, 3> fundamentalToActual = calibration.transpose().inverse() * essentielMatrix * calibration.inverse();

    Eigen::Matrix<T, 3, 1> epipolarLineActImage = fundamentalToActual.transpose() *
                                        Eigen::Matrix<T, 3, 1>(T(RefImagePointCoor[0]), T(RefImagePointCoor[1]), T(RefImagePointCoor[2]));

    Eigen::Matrix<T, 3, 1> epipolarLineRefImage = fundamentalToActual *
                                        Eigen::Matrix<T, 3, 1>(T(ActImagePointCoor[0]), T(ActImagePointCoor[1]), T(ActImagePointCoor[2]));

    T refImageResidus = epipolarLineActImage.dot(ActImagePointCoor) /
                              ceres::sqrt(ceres::pow(epipolarLineActImage[0], 2) + ceres::pow(epipolarLineActImage[1], 2));
    T actImageResidus =  epipolarLineRefImage.dot(RefImagePointCoor) /
                              ceres::sqrt(ceres::pow(epipolarLineRefImage[0], 2) + ceres::pow(epipolarLineRefImage[1], 2));

    residual[0] = !ceres::isinf(refImageResidus) ? refImageResidus : T(0.0);
    residual[1] = !ceres::isinf(actImageResidus) ? actImageResidus : T(0.0);

    return true;
  }

  Eigen::Vector3d RefImagePointCoor;
  Eigen::Vector3d ActImagePointCoor;
  Eigen::Affine3d TransformPreviousActual;
  Eigen::Affine3d TransformExtrinsicCamera;
  Eigen::Matrix3d CalibrationMatrix;
};

//-------------------------------------------------------------------------------------------
struct ScaleCostPreviousRightActualRight
{
  ScaleCostPreviousRightActualRight(Eigen::Vector3d refImagePointCoor,
                                    Eigen::Vector3d actImagePointCoor,
                                    Eigen::Affine3d transformPreviousActual,
                                    Eigen::Affine3d transformExtrinsicCamera,
                                    Eigen::Matrix3d calibrationMatrix) :
    RefImagePointCoor(refImagePointCoor), ActImagePointCoor(actImagePointCoor),
    TransformPreviousActual(transformPreviousActual), TransformExtrinsicCamera(transformExtrinsicCamera),
    CalibrationMatrix(calibrationMatrix) {}

  template <typename T>
  bool operator()(const T* const x, T* residual) const
  {
    Eigen::Transform<T, 3, Eigen::Affine> transformPreviousLeftActualLeft = this->TransformPreviousActual.cast<T>();
    transformPreviousLeftActualLeft.translation() = x[0] * transformPreviousLeftActualLeft.translation();
    Eigen::Transform<T, 3, Eigen::Affine> transformPreviousRightActualRight =
                this->TransformExtrinsicCamera.cast<T>().inverse() * transformPreviousLeftActualLeft * this->TransformExtrinsicCamera.cast<T>();

    Eigen::Matrix<T, 3, 1> translation = transformPreviousRightActualRight.translation();
    T skewSymetricTranslationArray[9] = {T(0.0), -translation(2), translation(1),
                                         translation(2), T(0.0), -translation(0),
                                         -translation(1), translation(0), T(0.0)};

    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> skewSymetricTranslation(&skewSymetricTranslationArray[0]);
    Eigen::Matrix<T, 3, 3> essentielMatrix = skewSymetricTranslation * transformPreviousRightActualRight.rotation();

    Eigen::Matrix<T, 3, 3> calibration = this->CalibrationMatrix.cast<T>();
    Eigen::Matrix<T, 3, 3> fundamentalToActual = calibration.transpose().inverse() * essentielMatrix * calibration.inverse();

    Eigen::Matrix<T, 3, 1> epipolarLineActImage = fundamentalToActual.transpose() *
                                  Eigen::Matrix<T, 3, 1>(T(RefImagePointCoor[0]), T(RefImagePointCoor[1]), T(RefImagePointCoor[2]));

    Eigen::Matrix<T, 3, 1> epipolarLineRefImage = fundamentalToActual *
                                  Eigen::Matrix<T, 3, 1>(T(ActImagePointCoor[0]), T(ActImagePointCoor[1]), T(ActImagePointCoor[2]));

    T refImageResidus =  epipolarLineActImage.dot(ActImagePointCoor) /
                              ceres::sqrt(ceres::pow(epipolarLineActImage[0], 2) + ceres::pow(epipolarLineActImage[1], 2));
    T actImageResidus =  epipolarLineRefImage.dot(RefImagePointCoor) /
                              ceres::sqrt(ceres::pow(epipolarLineRefImage[0], 2) + ceres::pow(epipolarLineRefImage[1], 2));

    residual[0] = !ceres::isinf(refImageResidus) ? refImageResidus : T(0.0);
    residual[1] = !ceres::isinf(actImageResidus) ? actImageResidus : T(0.0);

    return true;
  }

  Eigen::Vector3d RefImagePointCoor;
  Eigen::Vector3d ActImagePointCoor;
  Eigen::Affine3d TransformPreviousActual;
  Eigen::Affine3d TransformExtrinsicCamera;
  Eigen::Matrix3d CalibrationMatrix;
};
}