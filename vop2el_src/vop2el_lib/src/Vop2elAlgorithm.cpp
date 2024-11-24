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

#include <thread>

#include "Vop2elAlgorithm.h"
#include "Common.h"
#include "Vop2elCostStructs.h"

namespace Vop2el
{
//-------------------------------------------------------------------------------------------
void Vop2elAlgorithm::EstimateInitMatchesUsingOF(std::shared_ptr<const cv::Mat> refImage,
                                                std::shared_ptr<const cv::Mat> tarImage,
                                                std::shared_ptr<const std::vector<cv::Point2f>> refImageKeyPoints,
                                                std::vector<cv::Point2f>& refMatchesKeyPoints,
                                                std::vector<cv::Point2f>& tarMatchesKeyPoints) const
{
    refMatchesKeyPoints.clear();
    tarMatchesKeyPoints.clear();
    std::vector<uchar> statusForward;
    std::vector<float> errorsForward;
    std::vector<cv::Point2f> actualKeyPoints;
    cv::calcOpticalFlowPyrLK(*refImage, *tarImage, *refImageKeyPoints, actualKeyPoints, statusForward, errorsForward,
                            this->Vop2elParams.OfWindowSize, this->Vop2elParams.OfPyramidLevel, this->Vop2elParams.OfCriteria,
                            0, this->Vop2elParams.OfEigenTreshold);

    std::vector<uchar> statusBackward;
    std::vector<float> errorsBackward;
    std::vector<cv::Point2f> refKeyPointsBack;
    cv::calcOpticalFlowPyrLK(*tarImage, *refImage, actualKeyPoints, refKeyPointsBack, statusBackward, errorsBackward,
                            this->Vop2elParams.OfWindowSize, this->Vop2elParams.OfPyramidLevel, this->Vop2elParams.OfCriteria,
                            0, this->Vop2elParams.OfEigenTreshold);

    for (int keyPointIdx = 0; keyPointIdx < refImageKeyPoints->size(); ++keyPointIdx)
    {
        cv::Vec2f forwardVect(actualKeyPoints[keyPointIdx] - (*refImageKeyPoints)[keyPointIdx]);
        cv::Vec2f backwardVect(refKeyPointsBack[keyPointIdx] - actualKeyPoints[keyPointIdx]);
        if ((cv::norm(forwardVect + backwardVect) < this->Vop2elParams.OfForwardBackwardTreshold))
        {
            refMatchesKeyPoints.push_back(((*refImageKeyPoints)[keyPointIdx]));
            tarMatchesKeyPoints.push_back(actualKeyPoints[keyPointIdx]);
        }
    }
}

//-------------------------------------------------------------------------------------------
void Vop2elAlgorithm::EstimateInitScalessRelativeTransform(Eigen::Affine3d& transformPreviousActual) const
{
    std::vector<cv::Point2f> validPreviousLeftKeyPoints;
    std::vector<cv::Point2f> validActualLeftKeyPoints;

    std::shared_ptr<const cv::Mat> previousLeftImage = this->FramesHandler.GetLeftImage(-2);
    std::shared_ptr<const cv::Mat> actualLeftImage = this->FramesHandler.GetLeftImage(-1);
    std::shared_ptr<const std::vector<cv::Point2f>> previousLeftKeyPoints = this->FramesHandler.GetLeftImageKeyPoints(-2);

    this->EstimateInitMatchesUsingOF(previousLeftImage, actualLeftImage, previousLeftKeyPoints,
                                    validPreviousLeftKeyPoints, validActualLeftKeyPoints);

    cv::Mat maskInliers;
    cv::Mat essentialMatrix = cv::findEssentialMat(validActualLeftKeyPoints, validPreviousLeftKeyPoints,
                                                 this->Vop2elParams.CameraParams.CalibrationMatrix, cv::RANSAC, 0.999, 1 , maskInliers);

    cv::Mat relativeRotation, relativeTranslation;
    cv::recoverPose(essentialMatrix, validActualLeftKeyPoints, validPreviousLeftKeyPoints,
                    this->Vop2elParams.CameraParams.CalibrationMatrix, relativeRotation, relativeTranslation);

    Eigen::Matrix3dRowMajor rotationAfterOptimization(reinterpret_cast<double*>(relativeRotation.data));
    this->CheckRotation(rotationAfterOptimization);
    transformPreviousActual.linear() = rotationAfterOptimization;
    transformPreviousActual.translation() = Eigen::Vector3d(reinterpret_cast<double*>(relativeTranslation.data));
}

//-------------------------------------------------------------------------------------------
double Vop2elAlgorithm::EstimateInitScale(const Eigen::Affine3d& transformPreviousActual) const
{
    std::shared_ptr<const cv::Mat> previousLeftImage = this->FramesHandler.GetLeftImage(-2);
    std::shared_ptr<const cv::Mat> actualLeftImage = this->FramesHandler.GetLeftImage(-1);
    std::shared_ptr<const cv::Mat> previousRightImage = this->FramesHandler.GetRightImage(-2);
    std::shared_ptr<const cv::Mat> actualRightImage = this->FramesHandler.GetRightImage(-1);

    std::shared_ptr<const std::vector<cv::Point2f>> previousLeftKeyPoints = this->FramesHandler.GetLeftImageKeyPoints(-2);
    std::shared_ptr<const std::vector<cv::Point2f>> actualLeftKeyPoints = this->FramesHandler.GetLeftImageKeyPoints(-1);
    std::shared_ptr<const std::vector<cv::Point2f>> previousRightKeyPoints = this->FramesHandler.GetRightImageKeyPoints(-2);

    std::vector<cv::Point2f> refPreviousLeftActualRight;
    std::vector<cv::Point2f> tarPreviousLeftActualRight;
    this->EstimateInitMatchesUsingOF(previousLeftImage, actualRightImage, previousLeftKeyPoints,
                            refPreviousLeftActualRight, tarPreviousLeftActualRight);

    std::vector<cv::Point2f> refActualLeftPreviousRight;
    std::vector<cv::Point2f> tarActualLeftPreviousRight;
    this->EstimateInitMatchesUsingOF(actualLeftImage, previousRightImage, actualLeftKeyPoints,
                            refActualLeftPreviousRight, tarActualLeftPreviousRight);

    std::vector<cv::Point2f> refPreviousRightActualRight;
    std::vector<cv::Point2f> tarPreviousRightActualRight;
    this->EstimateInitMatchesUsingOF(previousRightImage, actualRightImage, previousRightKeyPoints,
                            refPreviousRightActualRight, tarPreviousRightActualRight);

    Eigen::Affine3d extrinsicTransformCamera =  Eigen::Affine3d::Identity();
    extrinsicTransformCamera.translation() = Eigen::Vector3d(reinterpret_cast<double*>(this->Vop2elParams.CameraParams.ExtrinsicTranslation.data));
    Eigen::Matrix3dRowMajor calibrationMatrix(reinterpret_cast<double*>(this->Vop2elParams.CameraParams.CalibrationMatrix.data));

    ceres::LossFunction* initScaleLossFunction = nullptr;
    if (this->Vop2elParams.UseTukeyEstimator)
       initScaleLossFunction = new ceres::TukeyLoss(this->Vop2elParams.TukeyParameter);

    double scale = 1.0;
    ceres::Problem problem;
    for (int keyPointIdx = 0; keyPointIdx < refPreviousLeftActualRight.size(); ++keyPointIdx)
    {
        cv::Point2d prevLeftPoint(refPreviousLeftActualRight[keyPointIdx]);
        cv::Point2d actRightPoint(tarPreviousLeftActualRight[keyPointIdx]);
        Eigen::Vector3d prevLeft(prevLeftPoint.x, prevLeftPoint.y, 1.0);
        Eigen::Vector3d actRight(actRightPoint.x, actRightPoint.y, 1.0);

        ceres::CostFunction* costPreviousLeftActualRight = new ceres::AutoDiffCostFunction<Vop2el::ScaleCostPreviousLeftActualRight, 2, 1>(
            new Vop2el::ScaleCostPreviousLeftActualRight(prevLeft, actRight, transformPreviousActual, extrinsicTransformCamera, calibrationMatrix));
        problem.AddResidualBlock(costPreviousLeftActualRight, initScaleLossFunction, &scale);
    }

    for (int keyPointIdx = 0; keyPointIdx < refActualLeftPreviousRight.size(); ++keyPointIdx)
    {
        cv::Point2d actLeftPoint(refActualLeftPreviousRight[keyPointIdx]);
        cv::Point2d prevRightPoint(tarActualLeftPreviousRight[keyPointIdx]);
        Eigen::Vector3d actLeft(actLeftPoint.x, actLeftPoint.y, 1.0);
        Eigen::Vector3d prevRight(prevRightPoint.x, prevRightPoint.y, 1.0);

        ceres::CostFunction* costActualLeftPreviousRight = new ceres::AutoDiffCostFunction<Vop2el::ScaleCostActualLeftPreviousRight, 2, 1>(
            new Vop2el::ScaleCostActualLeftPreviousRight(actLeft, prevRight, transformPreviousActual, extrinsicTransformCamera, calibrationMatrix));
        problem.AddResidualBlock(costActualLeftPreviousRight, initScaleLossFunction, &scale);
    }

    for (int keyPointIdx = 0; keyPointIdx < refPreviousRightActualRight.size(); ++keyPointIdx)
    {
        cv::Point2d prevRightPoint(refPreviousRightActualRight[keyPointIdx]);
        cv::Point2d actRightPoint(tarPreviousRightActualRight[keyPointIdx]);
        Eigen::Vector3d prevRight(prevRightPoint.x, prevRightPoint.y, 1.0);
        Eigen::Vector3d actRight(actRightPoint.x, actRightPoint.y, 1.0);

        ceres::CostFunction* costPreviousRightActualRight = new ceres::AutoDiffCostFunction<Vop2el::ScaleCostPreviousRightActualRight, 2, 1>(
            new Vop2el::ScaleCostPreviousRightActualRight(prevRight, actRight, transformPreviousActual, extrinsicTransformCamera, calibrationMatrix));
        problem.AddResidualBlock(costPreviousRightActualRight, initScaleLossFunction, &scale);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = std::thread::hardware_concurrency();
    ceres::Solver::Summary summary;
    options.max_num_iterations = this->Vop2elParams.CostFunctionsMaxNumIterations;
    ceres::Solve(options, &problem, &summary);

    return scale;
}

//-------------------------------------------------------------------------------------------
void Vop2elAlgorithm::CheckRotation(Eigen::Matrix3dRowMajor& rotationFromDecomposition) const
{
    Eigen::AngleAxisd axisAngle(rotationFromDecomposition);

    if (std::abs(axisAngle.angle() - M_PI) < 0.1)
    {
        axisAngle.angle() =  axisAngle.angle() - M_PI;
        rotationFromDecomposition = axisAngle.toRotationMatrix();
    }
}

//-------------------------------------------------------------------------------------------
void Vop2elAlgorithm::OptimizeEssentielMatrix(const cv::Mat& essentielMatrix,
                                            const std::vector<cv::Point2f>& prevPoints,
                                            const std::vector<cv::Point2f>& actPoints,
                                            const std::vector<bool>& maskInliers,
                                            cv::Mat& optimizedEssentielMatrix) const
{
    if ((prevPoints.size() != maskInliers.size()) || (actPoints.size() != maskInliers.size()))
        std::runtime_error("[ERROR] prevPoints, actPoints and maskInliers must have the same size");

    optimizedEssentielMatrix = essentielMatrix.clone();
    Eigen::Matrix3dRowMajor calibrationMatrixEigen(reinterpret_cast<const double*>(this->Vop2elParams.CameraParams.CalibrationMatrix.data));

    ceres::LossFunction* essentielMatLossFunction = nullptr;
    if (this->Vop2elParams.UseTukeyEstimator)
       essentielMatLossFunction = new ceres::TukeyLoss(this->Vop2elParams.TukeyParameter);

    ceres::Problem problem;
    for (int keyPointIdx = 0; keyPointIdx < maskInliers.size(); ++keyPointIdx)
    {
        if (maskInliers[keyPointIdx])
        {
            Eigen::Vector3d prevPoint(static_cast<double>(prevPoints[keyPointIdx].x), static_cast<double>(prevPoints[keyPointIdx].y), 1.0);
            Eigen::Vector3d actPoint(static_cast<double>(actPoints[keyPointIdx].x), static_cast<double>(actPoints[keyPointIdx].y), 1.0);

            ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<Vop2el::EssentielMatrixOptimizer, 2, 9>(
                new Vop2el::EssentielMatrixOptimizer(prevPoint, actPoint, calibrationMatrixEigen));
            problem.AddResidualBlock(costFunction, essentielMatLossFunction, reinterpret_cast<double*>(optimizedEssentielMatrix.data));
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = std::thread::hardware_concurrency();
    options.max_num_iterations = this->Vop2elParams.CostFunctionsMaxNumIterations;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

//-------------------------------------------------------------------------------------------
void Vop2elAlgorithm::ComputeScalessRelativeTransform(const std::vector<Vop2el::Match>& matches,
                                                    Eigen::Affine3d& relativeTransform,
                                                    std::vector<Vop2el::Match>& inliers) const
{
    inliers.clear();

    std::vector<cv::Point2f> prevPoints, actPoints;
    for (const auto& match : matches)
    {
        prevPoints.push_back(match.PreviousLeft);
        actPoints.push_back(match.ActualLeft);
    }

    cv::Mat maskInliers;
    cv::Mat essentielMatrix;
    essentielMatrix = cv::findEssentialMat(actPoints, prevPoints, this->Vop2elParams.CameraParams.CalibrationMatrix,
                                        cv::RANSAC, 0.999, 1, 1000, maskInliers);

    std::vector<bool> keyPointsInliersStatus(matches.size(), false);
    for (int keyPointIdx = 0; keyPointIdx < maskInliers.rows; ++keyPointIdx)
    {
        for (int keyPointItrIdx = 0; keyPointItrIdx <  maskInliers.cols; ++keyPointItrIdx)
            if (maskInliers.at<unsigned char>(keyPointIdx, keyPointItrIdx) == 1)
            {
                keyPointsInliersStatus[keyPointIdx] = true;
                inliers.push_back(matches[keyPointIdx]);
                break;
            }
    }

    cv::Mat optimizedEssentielMatrix;
    this->OptimizeEssentielMatrix(essentielMatrix, prevPoints, actPoints, keyPointsInliersStatus, optimizedEssentielMatrix);

    cv::Mat finalRotation, finalTranslation;
    cv::Mat inliersMask = (maskInliers.col(maskInliers.cols - 1)).clone();
    cv::recoverPose(optimizedEssentielMatrix, actPoints, prevPoints, this->Vop2elParams.CameraParams.CalibrationMatrix,
                    finalRotation, finalTranslation, inliersMask);

    Eigen::Matrix3dRowMajor rotationAfterOptimization(reinterpret_cast<double*>(finalRotation.data));
    this->CheckRotation(rotationAfterOptimization);
    relativeTransform.linear() = rotationAfterOptimization;
    relativeTransform.translation() = Eigen::Vector3d(reinterpret_cast<double*>(finalTranslation.data));
}

//-------------------------------------------------------------------------------------------
double Vop2elAlgorithm::ComputeScale(const Eigen::Affine3d& transformPreviousActual,
                                    const std::vector<Vop2el::Match>& matches) const
{
    double scale = 1;

    if (!this->RelativePoses.empty())
        scale = this->RelativePoses.back().translation().norm();

    Eigen::Affine3d extrinsicTransformCamera =  Eigen::Affine3d::Identity();
    extrinsicTransformCamera.translation() = Eigen::Vector3d(reinterpret_cast<double*>(this->Vop2elParams.CameraParams.ExtrinsicTranslation.data));
    Eigen::Matrix3dRowMajor calibrationMatrix(reinterpret_cast<double*>(this->Vop2elParams.CameraParams.CalibrationMatrix.data));

    ceres::LossFunction* scaleLossFunction = nullptr;
    if (this->Vop2elParams.UseTukeyEstimator)
       scaleLossFunction = new ceres::TukeyLoss(this->Vop2elParams.TukeyParameter);

    ceres::Problem problem;
    for (const auto& match : matches)
    {
        Eigen::Vector3d prevLeft(static_cast<double>(match.PreviousLeft.x), static_cast<double>(match.PreviousLeft.y), 1.0);
        Eigen::Vector3d actLeft(static_cast<double>(match.ActualLeft.x), static_cast<double>(match.ActualLeft.y), 1.0);
        Eigen::Vector3d prevRight(static_cast<double>(match.PreviousRight.x), static_cast<double>(match.PreviousRight.y), 1.0);
        Eigen::Vector3d actRight(static_cast<double>(match.ActualRight.x), static_cast<double>(match.ActualRight.y), 1.0);

        ceres::CostFunction* costPreviousLeftActualRight = new ceres::AutoDiffCostFunction<Vop2el::ScaleCostPreviousLeftActualRight, 2, 1>(
            new Vop2el::ScaleCostPreviousLeftActualRight(prevLeft, actRight, transformPreviousActual, extrinsicTransformCamera, calibrationMatrix));
        problem.AddResidualBlock(costPreviousLeftActualRight, scaleLossFunction, &scale);

        ceres::CostFunction* costActualLeftPreviousRight = new ceres::AutoDiffCostFunction<Vop2el::ScaleCostActualLeftPreviousRight, 2, 1>(
            new Vop2el::ScaleCostActualLeftPreviousRight(actLeft, prevRight, transformPreviousActual, extrinsicTransformCamera, calibrationMatrix));
        problem.AddResidualBlock(costActualLeftPreviousRight, scaleLossFunction, &scale);

        ceres::CostFunction* costPreviousRightActualRight = new ceres::AutoDiffCostFunction<Vop2el::ScaleCostPreviousRightActualRight, 2, 1>(
            new Vop2el::ScaleCostPreviousRightActualRight(prevRight, actRight, transformPreviousActual, extrinsicTransformCamera, calibrationMatrix));
        problem.AddResidualBlock(costPreviousRightActualRight, scaleLossFunction, &scale);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = std::thread::hardware_concurrency();
    ceres::Solver::Summary summary;
    options.max_num_iterations = this->Vop2elParams.CostFunctionsMaxNumIterations;
    ceres::Solve(options, &problem, &summary);

    return scale;
}

//-------------------------------------------------------------------------------------------
void Vop2elAlgorithm::ComputeMatchesUsingVop2elMatcher(const Vop2el::StereoImagesPairWithKeyPoints& imagesWithKeyPoints,
                                                    const Eigen::Affine3d& previousActualTransform,
                                                    std::vector<Vop2el::Match>& matches,
                                                    int& NumberFixedKeyPoints) const
{
    matches.clear();

    cv::Mat cvPreviousActualTransform(4, 4, CV_64F);
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            cvPreviousActualTransform.at<double>(row, col) = previousActualTransform(row, col);

    cv::Affine3d cvAffinePreviousActual(cvPreviousActualTransform);
    Vop2el::Vop2elMatcher matcher(imagesWithKeyPoints, this->Vop2elParams.Vop2elMatcherParams, this->Vop2elParams.CameraParams, cvAffinePreviousActual);
    if (this->Vop2elParams.PlaneNormal && this->Vop2elParams.PlaneDistance)
        matcher.SetPlaneParameters(*(this->Vop2elParams.PlaneNormal), *(this->Vop2elParams.PlaneDistance));

    matcher.GetMatches(matches);
    NumberFixedKeyPoints = matcher.GetNumberFixedKeyPoints();
}

//-------------------------------------------------------------------------------------------
void Vop2elAlgorithm::EstimateInitScaledRelativeTransform(Eigen::Affine3d& initialRelativeTransform) const
{
    initialRelativeTransform = Eigen::Affine3d::Identity();
    this->EstimateInitScalessRelativeTransform(initialRelativeTransform);

    double initialScale = 0;
    if (this->RelativePoses.size() == 0)
        initialScale = this->EstimateInitScale(initialRelativeTransform);
    else
        initialScale = this->RelativePoses.back().translation().norm();

    initialRelativeTransform.translation() = initialScale * initialRelativeTransform.translation();
}

//-------------------------------------------------------------------------------------------
void Vop2elAlgorithm::ProcessNumMatchesInsufficient()
{
    std::cerr << "[WARNING] The number of computed matches is insufficient. We will extrapolate." << std::endl;
    this->RelativePoses.push_back(this->RelativePoses.back());
    Eigen::Affine3d absolutePose = this->AbsolutePoses.back() * this->RelativePoses.back();
    this->AbsolutePoses.push_back(absolutePose);
}

//-------------------------------------------------------------------------------------------
void Vop2elAlgorithm::ProcessStereoFrame(const std::string& leftImage,
                                        const std::string& rightImage,
                                        Eigen::Affine3d& relativeTransform)
{
    if (this->FrameIndex < 2)
        this->FramesHandler.AddStereoPair(leftImage, rightImage);
    else
        this->FramesHandler.AddStereoPair(leftImage, rightImage, true, false);

    if (this->FrameIndex == 0)
    {
       this->AbsolutePoses.push_back(Eigen::Affine3d::Identity());
       ++this->FrameIndex;
    }
    else
    {
        Vop2el::StereoImagesPairWithKeyPoints StereoImagesPairWithKeyPoints{
                                                this->FramesHandler.GetLeftImage(-2), this->FramesHandler.GetRightImage(-2),
                                                this->FramesHandler.GetLeftImage(-1), this->FramesHandler.GetRightImage(-1),
                                                this->FramesHandler.GetLeftImageKeyPoints(-1)};

        Eigen::Affine3d initialRelativeTransform;
        this->EstimateInitScaledRelativeTransform(initialRelativeTransform);

        std::vector<Vop2el::Match> matches;
        int NumberFixedKeyPoints = 0;
        this->ComputeMatchesUsingVop2elMatcher(StereoImagesPairWithKeyPoints, initialRelativeTransform, matches, NumberFixedKeyPoints);

        std::cout << "Number of matches: " <<  matches.size() << std::endl;

        if (matches.size() < 10)
        {
            this->ProcessNumMatchesInsufficient();
            return;
        }

        double ratioOfFixedKeyPoints = static_cast<double>(NumberFixedKeyPoints) / static_cast<double>(matches.size());
        if (ratioOfFixedKeyPoints > 0.6)
        {
            this->RelativePoses.push_back(Eigen::Affine3d::Identity());
            this->AbsolutePoses.push_back(this->AbsolutePoses.back());
            relativeTransform = Eigen::Affine3d::Identity();
            ++this->FrameIndex;
            return;
        }

        Eigen::Affine3d scalessTransform = Eigen::Affine3d::Identity();
        std::vector<Vop2el::Match> inliers;
        this->ComputeScalessRelativeTransform(matches, scalessTransform, inliers);

        double scale = this->ComputeScale(scalessTransform, inliers);

        Eigen::Affine3d scaledTransform = scalessTransform;
        scaledTransform.translation() = scale * scalessTransform.translation();
        this->RelativePoses.push_back(scaledTransform);
        relativeTransform = scaledTransform;

        Eigen::Affine3d absolutePose = Eigen::Affine3d::Identity();
        absolutePose = this->AbsolutePoses.back() * scaledTransform;
        this->AbsolutePoses.push_back(absolutePose);

        ++this->FrameIndex;
    }
}
}