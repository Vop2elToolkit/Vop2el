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

#include <opencv2/opencv.hpp>

namespace Vop2el
{
class PatchCorrector
{
    public:
        PatchCorrector(std::shared_ptr<const cv::Mat> referenceImage,
                    std::shared_ptr<const cv::Mat> targetImage,
                    cv::Mat transformFromRefToTarg,
                    cv::Mat calibrationMatrix,
                    cv::Vec3f PlaneNormal,
                    float distanceToPlane,
                    int halfPatchRows,
                    int halfPatchCols);
        // Correct perspective of a patch using ground plane
        void GetPatchPerspectiveCorrected(const cv::Point2f& keyPoint,
                                        cv::Mat& correctedPatch) const;
        // Reproject a reference image keypoint on ground plane than project it on target image
        void GetKeyPointByPlaneProjection(const cv::Point2f& originalkeyPoint,
                                        cv::Point2f& reprojectedkeyPoint) const;

    private:
        // Reference image
        std::shared_ptr<const cv::Mat> ReferenceImage;
        // Target image
        std::shared_ptr<const cv::Mat> TargetImage;
        // Transform from reference camera to target camera
        cv::Mat Transform;
        // Inversed transform from reference camera to target camera
        cv::Mat TransformInversed;
        // Calibration matrix
        cv::Mat CalibrationMatrix;
        // Inversed calibration matrix
        cv::Mat CalibrationMatrixInversed;
        // Reference ground plane normal
        cv::Vec3f PlaneNormalRef;
        // Target ground plane normal
        cv::Vec3f PlaneNormalTarg;
        // Reference ground plane point
        cv::Vec3f PlanePointRef;
        // Target ground plane point
        cv::Vec3f PlanePointTarg;
        // Projection of 3d point on reference image
        cv::Mat ProjectionReferenceImage;
        // Projection of 3d point on target image
        cv::Mat ProjectionTargetImage;
        // Total number of columns of a patch is (HalfPatchCols * 2 + 1)
        int HalfPatchCols = 4;
        // Total number of rows of a patch is (HalfPatchRows * 2 + 1)
        int HalfPatchRows = 4;
        // Image number of columns
        int ImageCols;
        // Image number of rows
        int ImageRows;
        // Project a keypoint on ground plane
        void ProjectKeyPointOnPlane(const cv::Point2f& keyPoint,
                                const cv::Vec3f& planNormal,
                                const cv::Vec3f& planPoint,
                                cv::Vec3f& keyPointProjectedOnPlane) const;
        // Check whether the entire patch of a key point fall inside the image
        bool IsKeyPointPatchInImage(const cv::Point2f& keyPoint) const;
        // Check wether a keypoint is inside the image
        bool IsKeyPointInImage(const cv::Point2f& keyPoint) const;
};
}