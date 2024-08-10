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
#include <ceres/ceres.h>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>

#include "Vop2elMatcher.h"
#include "StereoImagesHandler.h"
#include "Common.h"

namespace Eigen
{
    using Matrix3dRowMajor = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
}

namespace Vop2el
{
struct Vop2elParameters
{
    cv::Size OfWindowSize{31, 31}; // optical flow window size
    int OfPyramidLevel = 3; // optical flow pyramid level
    double OfEigenTreshold =  0.001; // optical flow eigen treshold
    float OfForwardBackwardTreshold = 2.f; // optical flow forward/backward error treshold to consider a match valid
    cv::TermCriteria OfCriteria{cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50, 0.05}; // optical flow term criteria
    int CostFunctionsMaxNumIterations = 500; // non linear cost function maximum number of iterations
    bool UseTukeyEstimator = true; // if true, use Tukey instead of square loss function
    double TukeyParameter = 1.0; // the point at which the Tukey loss transitions from quadratic to linear behavior
    std::shared_ptr<cv::Vec3f> PlaneNormal; // ground plane normal vector
    std::shared_ptr<float> PlaneDistance; // distance between camera and ground plane on ground plane normal
    Common::Camera CameraParams; // stereo camera parameters
    Vop2el::StereoImagesHandlerParameters StereoImagesHandlerParams; // stereo images handler
    Vop2el::Vop2elMatcherParameters Vop2elMatcherParams; // matcher parameters
};

class Vop2elAlgorithm
{
    public:
        Vop2elAlgorithm(const Vop2el::Vop2elParameters& vop2elParams) :
        Vop2elParams(vop2elParams), FramesHandler(vop2elParams.StereoImagesHandlerParams) {}

        // Compute relative/absolute transform between previous frame and actual frame
        void ProcessStereoFrame(const std::string& leftImage,
                                const std::string& rightImage,
                                Eigen::Affine3d& relativeTransform);
        // Get absolute poses
        const std::vector<Eigen::Affine3d>& GetPoses() const { return this->AbsolutePoses; }
        // Get current absolute pose
        Eigen::Affine3d GetCurrentAbsPose() const { return this->AbsolutePoses.back(); }

    private:
        // Frames handler
        Vop2el::StereoImagesHandler FramesHandler;
        // Vop2el parameters
        Vop2el::Vop2elParameters Vop2elParams;
        // actual frame index
        int FrameIndex = 0;
        // Relative poses
        std::vector<Eigen::Affine3d> RelativePoses;
        // Absolute poses
        std::vector<Eigen::Affine3d> AbsolutePoses;

        // Compute matches using vop2el matcher
        void ComputeMatchesUsingVop2elMatcher(const Vop2el::StereoImagesPairWithKeyPoints& imagesWithKeyPoints,
                                            const Eigen::Affine3d& previousActualTransform,
                                            std::vector<Vop2el::Match>& matches,
                                            int& NumberFixedKeyPoints) const;
        // Compute essentiel matrix and return relative transform (no scale) and inliers
        void ComputeScalessRelativeTransform(const std::vector<Vop2el::Match>& matches,
                                            Eigen::Affine3d& relativeTransform,
                                            std::vector<Vop2el::Match>& inliers) const;
        // Optimize essentiel matrix using point to epipolar line constraint (non linear)
        void OptimizeEssentielMatrix(const cv::Mat& essentielMatrix,
                                    const std::vector<cv::Point2f>& prevPoints,
                                    const std::vector<cv::Point2f>& actPoints,
                                    const std::vector<bool>& maskInliers,
                                    cv::Mat& optimizedEssentielMatrix) const;
        // Compute scale using point to epipolar line constraint (non linear)
        double ComputeScale(const Eigen::Affine3d& transformPreviousActual,
                            const std::vector<Vop2el::Match>& matches) const;
        // Verify whether the rotation is logic because when the movement is very small the SVD
        // decomposition could give a rotation with an error of +/- M_PI
        void CheckRotation(Eigen::Matrix3dRowMajor& rotationFromDecomposition) const;
        // Estimate initial relative transform using optical flow
        void EstimateInitScalessRelativeTransform(Eigen::Affine3d& transformPreviousActual) const;
        // Estimate intial matches using Lucas Kanade optical flow
        void EstimateInitMatchesUsingOF(std::shared_ptr<const cv::Mat> refImage,
                                        std::shared_ptr<const cv::Mat> targImage,
                                        std::shared_ptr<const std::vector<cv::Point2f>> refImageKeyPoints,
                                        std::vector<cv::Point2f>& refMatchesKeyPoints,
                                        std::vector<cv::Point2f>& targMatchesKeyPoints) const;
        // Estimate intial scale using optical flow matches
        double EstimateInitScale(const Eigen::Affine3d& transformPreviousActual) const;
        // Estimate intial scaled relative transform
        void EstimateInitScaledRelativeTransform(Eigen::Affine3d& initialRelativeTransform) const;
        // Extrapolate relative/absolute pose when number of valid matches computed by the matcher is insufficient
        void ProcessNumMatchesInsufficient();
};
}