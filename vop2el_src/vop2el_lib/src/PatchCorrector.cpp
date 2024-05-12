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

#include "PatchCorrector.h"

namespace Vop2el
{
//---------------------------------------------------------------------------------------
PatchCorrector::PatchCorrector(std::shared_ptr<const cv::Mat> referenceImage,
                            std::shared_ptr<const cv::Mat> targetImage,
                            cv::Mat transformFromRefToTarg,
                            cv::Mat calibrationMatrix,
                            cv::Vec3f planeNormal,
                            float distanceToPlane,
                            int halfPatchRows,
                            int halfPatchCols) :
                            ReferenceImage(referenceImage), TargetImage(targetImage), HalfPatchRows(halfPatchRows), HalfPatchCols(halfPatchCols)
{
    if ((transformFromRefToTarg.type() != CV_32F) || (calibrationMatrix.type() != CV_32F))
        throw std::runtime_error("[ERROR] transform and calibration matrices given to patch corrector must be of type CV_32F");

    this->CalibrationMatrix = calibrationMatrix;
    this->PlaneNormalRef = planeNormal;
    this->PlanePointRef = -distanceToPlane * this->PlaneNormalRef;

    cv::Mat planeNormalTarg = cv::Mat(cv::Affine3f(transformFromRefToTarg).inv().rotation()) * this->PlaneNormalRef;
    this->PlaneNormalTarg = cv::Vec3f(planeNormalTarg.at<float>(0), planeNormalTarg.at<float>(1), planeNormalTarg.at<float>(2));
    this->PlanePointTarg = -distanceToPlane * this->PlaneNormalTarg;

    this->ImageCols = referenceImage->cols;
    this->ImageRows = referenceImage->rows;

    this->Transform = transformFromRefToTarg(cv::Range(0, 3), cv::Range::all());

    cv::Mat extrinsicTransformInv = cv::Mat(cv::Affine3f(transformFromRefToTarg).inv().matrix);
    this->TransformInversed = extrinsicTransformInv(cv::Range(0, 3), cv::Range::all());

    this->CalibrationMatrixInversed = this->CalibrationMatrix.inv();

    this->ProjectionTargetImage = this->CalibrationMatrix * this->TransformInversed;
    this->ProjectionReferenceImage = this->CalibrationMatrix * this->Transform;
}

//---------------------------------------------------------------------------------------
void PatchCorrector::ProjectKeyPointOnPlane(const cv::Point2f& keyPoint,
                                        const cv::Vec3f& planNormal,
                                        const cv::Vec3f& planPoint,
                                        cv::Vec3f& keyPointProjectedOnPlane) const
{
    if (!(this->IsKeyPointInImage(keyPoint)))
    {
        keyPointProjectedOnPlane =  cv::Vec3f::all(-1.f);
        return;
    }

    cv::Vec3f pixelicKeyPoint(keyPoint.x, keyPoint.y, 1.f);
    cv::Mat keyPointCameraCoor = this->CalibrationMatrixInversed * pixelicKeyPoint;

    cv::Vec3f lineDirection(keyPointCameraCoor.at<float>(0), keyPointCameraCoor.at<float>(1), keyPointCameraCoor.at<float>(2));
    if (cv::norm(lineDirection) < 1e-6f)
    {
        keyPointProjectedOnPlane =  cv::Vec3f::all(-1.f);
        return;
    }

    lineDirection /= cv::norm(lineDirection);

    float dotPlaneNormalLineDirection = lineDirection.dot(planNormal);
    if (std::abs(dotPlaneNormalLineDirection) < 1e-6f)
    {
        keyPointProjectedOnPlane =  cv::Vec3f::all(-1.f);
        return;
    }

    cv::Vec3f linePoint(0.f, 0.f, 0.f);
    float distPlanePointToIntersection = (planPoint - linePoint).dot(planNormal) / dotPlaneNormalLineDirection;
    keyPointProjectedOnPlane = distPlanePointToIntersection * lineDirection;
}

//---------------------------------------------------------------------------------------
bool PatchCorrector::IsKeyPointPatchInImage(const cv::Point2f& keyPoint) const
{
    if (((keyPoint.x + static_cast<float>(this->HalfPatchCols)) > (static_cast<float>(this->ImageCols) - 1.f)) ||
        ((keyPoint.x - static_cast<float>(this->HalfPatchCols)) < 0.f) ||
        ((keyPoint.y + static_cast<float>(this->HalfPatchRows)) > (static_cast<float>(this->ImageRows) - 1.f)) ||
        ((keyPoint.y - static_cast<float>(this->HalfPatchRows)) < 0.f))
    {
        return false;
    }

    return true;
}

//---------------------------------------------------------------------------------------
bool PatchCorrector::IsKeyPointInImage(const cv::Point2f& keyPoint) const
{
    if ((keyPoint.x > (static_cast<float>(this->ImageCols) - 1.f)) || (keyPoint.x < 0.f) ||
        (keyPoint.y > (static_cast<float>(this->ImageRows) - 1.f)) || (keyPoint.y < 0.f))
    {
        return false;
    }

    return true;
}

//---------------------------------------------------------------------------------------
void PatchCorrector::GetKeyPointByPlaneProjection(const cv::Point2f& originalKeyPoint,
                                                cv::Point2f& reprojectedKeyPoint) const
{
    cv::Vec3f keyPointProjectedOnPlane;
    this->ProjectKeyPointOnPlane(originalKeyPoint, this->PlaneNormalRef, this->PlanePointRef, keyPointProjectedOnPlane);

    if (keyPointProjectedOnPlane[2] < 1e-6f)
    {
        reprojectedKeyPoint.x = -1.f;
        reprojectedKeyPoint.y = -1.f;
        return;
    }

    cv::Vec4f keyPointProjectedOnPlane4D(keyPointProjectedOnPlane[0], keyPointProjectedOnPlane[1], keyPointProjectedOnPlane[2], 1.f);
    cv::Mat keyPointOnRightCamera = this->ProjectionTargetImage * keyPointProjectedOnPlane4D;
    if (keyPointOnRightCamera.at<float>(2) < 1e-6f)
    {
        reprojectedKeyPoint.x = -1.f;
        reprojectedKeyPoint.y = -1.f;
        return;
    }

    keyPointOnRightCamera /= keyPointOnRightCamera.at<float>(2);

    reprojectedKeyPoint.x = keyPointOnRightCamera.at<float>(0);
    reprojectedKeyPoint.y = keyPointOnRightCamera.at<float>(1);

    if (!this->IsKeyPointInImage(reprojectedKeyPoint))
    {
        reprojectedKeyPoint.x = -1.f;
        reprojectedKeyPoint.y = -1.f;
    }
}

//---------------------------------------------------------------------------------------
void PatchCorrector::GetPatchPerspectiveCorrected(const cv::Point2f& keyPoint,
                                                cv::Mat& correctedPatch) const
{
    cv::Point2f projectedKeyPoint(-1.f, -1.f);
    correctedPatch = cv::Mat();

    this->GetKeyPointByPlaneProjection(keyPoint, projectedKeyPoint);

    if (this->IsKeyPointPatchInImage(projectedKeyPoint) && this->IsKeyPointInImage(projectedKeyPoint))
    {
        correctedPatch.create(this->HalfPatchRows * 2 + 1, this->HalfPatchCols * 2 + 1, CV_8U);
        for (int row = -this->HalfPatchRows; row <= this->HalfPatchRows; ++row)
            for (int col = -this->HalfPatchCols; col <= this->HalfPatchCols; ++col)
            {
                cv::Point2f patchElement(projectedKeyPoint.x + static_cast<float>(col), projectedKeyPoint.y + static_cast<float>(row));
                cv::Vec3f keyPointProjectedOnPlane;
                this->ProjectKeyPointOnPlane(patchElement, this->PlaneNormalTarg, this->PlanePointTarg, keyPointProjectedOnPlane);

                if (keyPointProjectedOnPlane[2] < 1e-6f)
                {
                    projectedKeyPoint.x = -1.f;
                    projectedKeyPoint.y = -1.f;
                    correctedPatch = cv::Mat();
                    return;
                }

                cv::Vec4f keyPointProjectedOnPlane4D(keyPointProjectedOnPlane[0], keyPointProjectedOnPlane[1], keyPointProjectedOnPlane[2], 1.f);
                cv::Mat elementReproOnLeftImg = this->ProjectionReferenceImage * keyPointProjectedOnPlane4D;

                if (elementReproOnLeftImg.at<float>(2) < 1e-6f)
                {
                    projectedKeyPoint.x = -1.f;
                    projectedKeyPoint.y = -1.f;
                    correctedPatch = cv::Mat();
                    return;
                }

                elementReproOnLeftImg = elementReproOnLeftImg / elementReproOnLeftImg.at<float>(2);
                cv::Point2f elementReproOnLeftImgPt(elementReproOnLeftImg.at<float>(0), elementReproOnLeftImg.at<float>(1));

                if (!(this->IsKeyPointInImage(elementReproOnLeftImgPt)))
                {
                  projectedKeyPoint.x = -1.f;
                  projectedKeyPoint.y = -1.f;
                  correctedPatch = cv::Mat();
                  return;
                }

                cv::Mat pixelInterpolated;
                cv::getRectSubPix(*(this->ReferenceImage), cv::Size(1,1), elementReproOnLeftImgPt, pixelInterpolated);
                correctedPatch.at<uchar>(row + this->HalfPatchRows, col + this->HalfPatchCols) = pixelInterpolated.at<uchar>(0);
            }
    }
}
}