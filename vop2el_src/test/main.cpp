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

#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

#include <opencv2/opencv.hpp>

#include "Vop2elAlgorithm.h"
#include "Utils.h"

#ifdef RERUN_VISUALIZATION
#include "RerunVisualizer.h"
#endif

int main(int argc, char** argv)
{
    bool haveGroundTruth = false;
    if (argc == 6)
        haveGroundTruth = true;
    else if (argc != 5)
    {
        std::cout << "[ERROR] Number of inputs is invalid, please give the following inputs" << std::endl;
        std::cout << "1- Left images folder path" << std::endl;
        std::cout << "2- Right images folder path" << std::endl;
        std::cout << "3- Parameters ini file path" << std::endl;
        std::cout << "4- Estimated poses file path" << std::endl;
        std::cout << "5- (optional) Ground truth poses if the user want to output distance and rotation angle evaluation metrics " << std::endl;
        return 0;
    }

    // Read parameters from ini file
    std::string iniFile = argv[3];
    Vop2el::Vop2elParameters vop2elParameters;
    Utils::GenerateVop2elParamsFromIniFile(iniFile, vop2elParameters);
    Vop2el::Vop2elAlgorithm algorithm(vop2elParameters);

    // Read ground truth poses if provided
    std::vector<Eigen::Affine3d> gtPosesVector;
    if (argc == 6)
    {
        std::string groundTruthPosesFile = argv[5];
        Utils::ReadKittiGroundTruth(groundTruthPosesFile, gtPosesVector);
    }

    // Get stereo images dirs
    std::string leftImagesDirectory = argv[1];
    std::string rightImagesDirectory = argv[2];

    std::vector<cv::String> leftImagesPath;
    cv::glob(leftImagesDirectory, leftImagesPath, false);
    std::sort(leftImagesPath.begin(), leftImagesPath.end());

    std::vector<cv::String> rightImagesPath;
    cv::glob(rightImagesDirectory, rightImagesPath, false);
    std::sort(rightImagesPath.begin(), rightImagesPath.end());

    if (leftImagesPath.size() != rightImagesPath.size())
        throw std::runtime_error("[ERROR] left and right images sizes must be the same ");

    double totalEstimatedDistance = 0.0;
    double totalGroundTruthDistance = 0.0;
    double totalRadPerMeterError = 0.0;
    int numFramesToProcess = leftImagesPath.size();

#ifdef RERUN_VISUALIZATION
    RerunVisualizer rerunVisualizer;
#endif

    for (int frameIdx = 0; frameIdx < numFramesToProcess; ++frameIdx)
    {
        std::cout << "Processing frame: " << frameIdx << std::endl;

        if (frameIdx == 0)
        {
            Eigen::Affine3d computedRelativeTransform;
            // Process frame
            algorithm.ProcessStereoFrame(leftImagesPath[frameIdx], rightImagesPath[frameIdx], computedRelativeTransform);
#ifdef RERUN_VISUALIZATION
            rerunVisualizer.StreamFrame(leftImagesPath[frameIdx], rightImagesPath[frameIdx], algorithm.GetCurrentAbsPose());
#endif
            continue;
        }

        auto timeStart = std::chrono::high_resolution_clock::now();
        Eigen::Affine3d computedRelativeTransform;

        // Process frame
        algorithm.ProcessStereoFrame(leftImagesPath[frameIdx], rightImagesPath[frameIdx], computedRelativeTransform);
        auto timeEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Time total is: " << std::chrono::duration<double, std::milli>(timeEnd - timeStart).count() << std::endl;
#ifdef RERUN_VISUALIZATION
        rerunVisualizer.StreamFrame(leftImagesPath[frameIdx], rightImagesPath[frameIdx], algorithm.GetCurrentAbsPose());
#endif

        // If have ground truth, compute metrics
        if (haveGroundTruth)
        {
            Eigen::Affine3d previousGtPose = gtPosesVector[frameIdx - 1];
            Eigen::Affine3d actualGtPose = gtPosesVector[frameIdx];
            Eigen::Affine3d gtRelativeTransform = previousGtPose.inverse() * actualGtPose;

            Eigen::Affine3d relativeTransformError = computedRelativeTransform.inverse() * gtRelativeTransform;
            double translationScaleError = relativeTransformError.translation().norm();
            Eigen::AngleAxisd angleAxisErrorRotation(relativeTransformError.rotation());
            double rotationError = angleAxisErrorRotation.angle();

            totalGroundTruthDistance += gtRelativeTransform.translation().norm();
            totalEstimatedDistance += computedRelativeTransform.translation().norm();
            totalRadPerMeterError += std::abs(rotationError) / gtRelativeTransform.translation().norm();
            std::cout << "Distance error is: " << translationScaleError << " (m) | Angle error is: " << rotationError << " (rad)" << std::endl;
        }
        std::cout << "---------------------------------------------------------" << std::endl;
    }

    double distanceTotalError = std::abs(totalEstimatedDistance - totalGroundTruthDistance);
    double angleTotalError = totalRadPerMeterError / static_cast<double>(numFramesToProcess);

    std::cout << "Total estimated distance: " << totalEstimatedDistance << std::endl;
    std::cout << "Total ground truth distance: " << totalGroundTruthDistance << std::endl;

    if (haveGroundTruth)
        std::cout << "Distance total error is: " << distanceTotalError  << " (m) | Angle total error : " << angleTotalError << " (rad)" << std::endl;

    // Get computed poses
    const std::vector<Eigen::Affine3d>& estimatedPoses = algorithm.GetPoses();
    Utils::WritePosesAsTransform(estimatedPoses, argv[4]);

    return 0;
}