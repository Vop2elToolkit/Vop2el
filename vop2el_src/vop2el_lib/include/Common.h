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

namespace Common
{
    // Stereo camera parameters
    struct Camera
    {
        cv::Mat CalibrationMatrix; // CV_64F 3x3 calibration matrix
        cv::Mat ExtrinsicRotation; // extrinsic CV_64F 3x3 rotation from left camera to right camera
        cv::Mat ExtrinsicTranslation; // extrinsic CV_64F 3x1 translation from left camera to right camera
        int cols; // images number of columns
        int rows; // images number of rows
    };

    // Compute fundamental matrix from rotation, translation and calibration
    void ComputeFundMatrixFromRotTranCal(const cv::Mat& rotation,
                                        const cv::Mat& translation,
                                        const cv::Mat& calibration,
                                        cv::Mat& fundamental);
};