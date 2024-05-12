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

#include<string>

#include "ini.h"
#include "Vop2elAlgorithm.h"
#include "Common.h"

namespace Utils
{
    // Generate vop2el parameters structure from ini file parameters
    void GenerateVop2elParamsFromIniFile(const std::string& iniFile,
                                        Vop2el::Vop2elParameters& vop2elParameters);
    // Write poses row major in text file
    void WritePosesAsTransform(const std::vector<Eigen::Affine3d>& poses,
                            std::string textFilePath);
    // Read kitti format ground truth poses from text file
    void ReadKittiGroundTruth(std::string pathToGroundTruth,
                            std::vector<Eigen::Affine3d>& positionsVector);
    // Get stereo camera parameters from text file
    void GetCameraParamsFromTxtFile(const std::string& calibrationFile,
                                    cv::Mat& calibration,
                                    cv::Mat& extrinsicRotation,
                                    cv::Mat& extrinsicTranslation);
};