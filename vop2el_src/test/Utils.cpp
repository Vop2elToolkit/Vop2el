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

#include "Utils.h"

//---------------------------------------------------------------------------------------
void Utils::GetCameraParamsFromTxtFile(const std::string& stereoCameraPath,
                                    cv::Mat& calibration,
                                    cv::Mat& extrinsicRotation,
                                    cv::Mat& extrinsicTranslation)
{
    std::ifstream stereoCameraFile(stereoCameraPath);
    if (!stereoCameraFile.is_open())
        throw std::runtime_error("[ERROR] Could not read camera calibration file : " + stereoCameraPath);

    int lineIdx = 0;
    std::vector<double> calibrationMatrixVector;
    std::vector<double> extrinsicRotationVector;
    std::vector<double> extrinsicTranslationVector;
    std::string calibrationMatrixFileLine;
    while (std::getline(stereoCameraFile, calibrationMatrixFileLine))
    {
        if (lineIdx < 3)
        {
            std::stringstream ss(calibrationMatrixFileLine);
            for (int lineElementIdx = 0; lineElementIdx < 3; ++lineElementIdx)
            {
                std::string nextElement;
                std::getline(ss, nextElement, ' ');

                calibrationMatrixVector.push_back(std::stod(nextElement));
            }
        }
        else if (lineIdx < 6)
        {
            std::stringstream ss(calibrationMatrixFileLine);
            for (int lineElementIdx = 0; lineElementIdx < 4; ++lineElementIdx)
            {
                std::string nextElement;
                std::getline(ss, nextElement, ' ');
                if (lineElementIdx == 3)
                {
                    extrinsicTranslationVector.push_back(std::stod(nextElement));
                    continue;
                }
                extrinsicRotationVector.push_back(std::stod(nextElement));
            }
        }
        ++lineIdx;
    }

    calibration = cv::Mat(3, 3, CV_64F, calibrationMatrixVector.data()).clone();
    extrinsicRotation = cv::Mat(3, 3, CV_64F, extrinsicRotationVector.data()).clone();
    extrinsicTranslation = cv::Mat(3, 1, CV_64F, extrinsicTranslationVector.data()).clone();
}

//---------------------------------------------------------------------------------------
void Utils::GenerateVop2elParamsFromIniFile(const std::string& iniFile,
                                            Vop2el::Vop2elParameters& vop2elParameters)
{
    mINI::INIFile parametersFile(iniFile);
    mINI::INIStructure ini;

    if (!(parametersFile.read(ini)))
        throw std::runtime_error("[ERROR] Could not read ini parameters file: " + iniFile);

    std::string stereoCameraPath = ini["stereo_camera_parameters"]["stereo_camera_parameters"];
    Utils::GetCameraParamsFromTxtFile(stereoCameraPath, vop2elParameters.CameraParams.CalibrationMatrix,
                                    vop2elParameters.CameraParams.ExtrinsicRotation, vop2elParameters.CameraParams.ExtrinsicTranslation);

    vop2elParameters.CameraParams.cols = std::stoi(ini["stereo_camera_parameters"]["image_cols"]);
    vop2elParameters.CameraParams.rows = std::stoi(ini["stereo_camera_parameters"]["image_rows"]);

    vop2elParameters.OfWindowSize = {std::stoi(ini["optical_flow_parameters"]["of_window_cols"]),
                                    std::stoi(ini["optical_flow_parameters"]["of_window_rows"])};
    vop2elParameters.OfPyramidLevel = std::stoi(ini["optical_flow_parameters"]["of_pyramid_level"]);
    vop2elParameters.OfEigenTreshold = std::stod(ini["optical_flow_parameters"]["of_eigen_treshold"]);
    vop2elParameters.OfForwardBackwardTreshold = std::stof(ini["optical_flow_parameters"]["of_forward_backward_treshold"]);
    vop2elParameters.OfCriteria = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                            std::stoi(ini["optical_flow_parameters"]["of_criteria_max_count"]),
                                            std::stod(ini["optical_flow_parameters"]["of_criteria_epsilon"]));

    vop2elParameters.CostFunctionsMaxNumIterations = std::stoi(ini["cost_functions_parameters"]["cost_functions_max_num_iterations"]);
    std::string useTukeyEstimator = ini["cost_functions_parameters"]["use_tukey_estimator"];
    if (useTukeyEstimator == "true" || useTukeyEstimator == "True")
        vop2elParameters.TukeyParameter = std::stod(ini["cost_functions_parameters"]["tukey_parameter"]);
    else
        vop2elParameters.UseTukeyEstimator = false;

    vop2elParameters.Vop2elMatcherParams.MaxNumberOfMatches = std::stof(ini["vop2el_matcher_parameters"]["max_number_matches"]);
    vop2elParameters.Vop2elMatcherParams.NccTreshold = std::stof(ini["vop2el_matcher_parameters"]["ncc_treshold"]);
    vop2elParameters.Vop2elMatcherParams.EpipolarLineSearchInterval = std::stoi(ini["vop2el_matcher_parameters"]["epipolar_line_search_interval"]);
    vop2elParameters.Vop2elMatcherParams.MaxStereoPointsToProcess = std::stoi(ini["vop2el_matcher_parameters"]["max_stereo_points_to_process"]);
    vop2elParameters.Vop2elMatcherParams.HalfPatchRows = std::stoi(ini["vop2el_matcher_parameters"]["half_patch_rows"]);
    vop2elParameters.Vop2elMatcherParams.HalfPatchCols = std::stoi(ini["vop2el_matcher_parameters"]["half_patch_cols"]);
    vop2elParameters.Vop2elMatcherParams.HalfVerticalSearch = std::stoi(ini["vop2el_matcher_parameters"]["half_vertical_search"]);
    vop2elParameters.Vop2elMatcherParams.HalfHorizontalSearch = std::stoi(ini["vop2el_matcher_parameters"]["half_horizontal_search"]);
    vop2elParameters.Vop2elMatcherParams.MaxThresh = std::stof(ini["vop2el_matcher_parameters"]["max_thresh"]);

    vop2elParameters.StereoImagesHandlerParams.NumFramesCapacity = std::stoi(ini["stereo_images_handler_parameters"]["num_frames_capacity"]);
    vop2elParameters.StereoImagesHandlerParams.BinWidth = std::stoi(ini["stereo_images_handler_parameters"]["bin_width"]);
    vop2elParameters.StereoImagesHandlerParams.BinHeight = std::stoi(ini["stereo_images_handler_parameters"]["bin_height"]);
    vop2elParameters.StereoImagesHandlerParams.MaxNumberOfKeyPointsPerBin =
                                                std::stoi(ini["stereo_images_handler_parameters"]["max_key_points_per_bin"]);

    std::string arePlaneParamsAvailable = ini["ground_plane_parameters"]["use_ground_plane_correction"];
    vop2elParameters.PlaneNormal.reset();
    vop2elParameters.PlaneDistance.reset();
    if (arePlaneParamsAvailable == "true" || arePlaneParamsAvailable == "True")
    {
        vop2elParameters.PlaneNormal.reset(new cv::Vec3f(std::stof(ini["ground_plane_parameters"]["plane_normal_x"]),
                                                        std::stof(ini["ground_plane_parameters"]["plane_normal_y"]),
                                                        std::stof(ini["ground_plane_parameters"]["plane_normal_z"])));
        vop2elParameters.PlaneDistance.reset(new float(std::stof(ini["ground_plane_parameters"]["camera_ground_plane_distance"])));
    }
}

//---------------------------------------------------------------------------------------
void Utils::WritePosesAsTransform(const std::vector<Eigen::Affine3d>& poses,
                                std::string textFilePath)
{
    std::ofstream file(textFilePath);
    if (file.is_open())
    {
        for (const auto& pose : poses)
        {
            std::string poseAsString;

            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 4; ++col)
                {
                    if (row == 2 && col == 3)
                    {
                        poseAsString = poseAsString + std::to_string(pose(row, col));
                        continue;
                    }

                    poseAsString = poseAsString + std::to_string(pose(row, col)) + " ";
                }
            }
            file << poseAsString << std::endl;
        }
    }
    else
        throw std::runtime_error("[ERROR] Cannot write poses in: " + textFilePath);
}

//---------------------------------------------------------------------------------------
void Utils::ReadKittiGroundTruth(std::string pathToGroundTruth,
                        std::vector<Eigen::Affine3d>& positionsVector)
{
    std::ifstream file(pathToGroundTruth);
    if (!(file.is_open()))
        throw std::runtime_error("[ERROR] Cannot read ground truth file: " + pathToGroundTruth);

    std::string line;
    while (std::getline(file, line))
    {
        std::vector<double> values;

        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ' '))
        {
            double value = std::stod(token);
            values.push_back(value);
        }

        Eigen::Affine3d pose = Eigen::Affine3d::Identity();
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                pose(row, col) = values[4 * row + col];

        positionsVector.push_back(pose);
    }

    file.close();
}