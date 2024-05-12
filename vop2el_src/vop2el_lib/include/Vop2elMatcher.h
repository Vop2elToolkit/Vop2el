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

#include <shared_mutex>
#include <omp.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "Common.h"
#include "PatchCorrector.h"

namespace Vop2el
{
struct StereoImagesPairWithKeyPoints
{
    std::shared_ptr<const cv::Mat> PreviousLeftImage; // previous left image
    std::shared_ptr<const cv::Mat> PreviousRightImage; // previous right image
    std::shared_ptr<const cv::Mat> ActualLeftImage; // actual left image
    std::shared_ptr<const cv::Mat> ActualRightImage; // actual right image

    std::shared_ptr<const std::vector<cv::Point2f>> ActualLeftKeyPoints; // actual left keypoints vector
};

struct Match
{
    cv::Point2f ActualLeft; // keypoint match in actual left image
    cv::Point2f ActualRight; // keypoint match in actual right image
    cv::Point2f PreviousLeft; // keypoint match in previous left image
    cv::Point2f PreviousRight; // keypoint match in previous right image
};

struct Vop2elMatcherParameters
{
    float NccTreshold = 0.7f; // normalized cross-correlation score above which the match is considered valid
    int EpipolarLineSearchInterval = 100; // interval of search on the epipolar line, depends on the speed of the robot
    int MaxStereoPointsToProcess = 10; // maximum number of stereo keypoints candidates to search their matches in previous frame
    int HalfPatchRows = 4; // total number of rows of a patch is (HalfPatchRows * 2 + 1)
    int HalfPatchCols = 4; // total number of columns of a patch is (HalfPatchCols * 2 + 1)
    int HalfVerticalSearch = 4; // total number of rows of the narrow region to search in is (HalfVerticalSearch * 2 + 1)
    int HalfHorizontalSearch = 4; // total number of columns of the narrow region to search in is (HalfHorizontalSearch * 2 + 1)
    float MaxThresh = 0.25f; // treshold first/last stereo point ncc score to consider a left image keypoint not ambiguous
};

class Vop2elMatcher
{
    public:
        Vop2elMatcher(const Vop2el::StereoImagesPairWithKeyPoints& stereoImagesPairWithKeyPoints,
                const Vop2el::Vop2elMatcherParameters& matcherParams,
                const Common::Camera& cameraParams,
                const cv::Affine3d& transformPreviousActual);
        // Compute and get matches
        void GetMatches(std::vector<Vop2el::Match>& matches);
        // Get number of keypoints with a very small optical flow
        int GetNumberFixedKeyPoints(){ return this->NumFixedKeyPoints; }
        // Set ground plane normal, and distance to camera on that normal
        void SetPlaneParameters(const cv::Vec3f& normal, float distance);

    private:
        // Previous frame, actual frame and keypoints of actual left image
        const Vop2el::StereoImagesPairWithKeyPoints PairWithKeyPoints;
        // Transform from previous left camera to actual left camera
        cv::Affine3d PreviousActualTransform;
        // Transform from left camera to right camera
        cv::Affine3d ExtrinsicTransformAffine;
        // Projection of a 3D point on actual left image
        cv::Mat ProjectionActualLeft;
        // Projection of a 3D point on actual right image
        cv::Mat ProjectionActualRight;
        // Projection of a 3D point on previous left image
        cv::Mat ProjectionPreviousLeft;
        // Projection of a 3D point on previous right image
        cv::Mat ProjectionPreviousRight;
        // Stereo camera parameters
        Common::Camera CameraParams;
        // Matcher parameters
        Vop2el::Vop2elMatcherParameters Vop2elMatcherParams;
        // Number of keypoints with a very small optical flow
        int NumFixedKeyPoints = 0;
        // Patch corrector from actual left to actual right image
        std::unique_ptr<Vop2el::PatchCorrector> CorrectorActualLeftActualRight;
        // Patch corrector from actual right to previous left image
        std::unique_ptr<Vop2el::PatchCorrector> CorrectorActualRightPreviousLeft;
        // Patch corrector from actual right to previous right image
        std::unique_ptr<Vop2el::PatchCorrector> CorrectorActualRightPreviousRight;
        // Patch corrector from actual left to previous left image
        std::unique_ptr<Vop2el::PatchCorrector> CorrectorActualLeftPreviousLeft;
        // Patch corrector from actual left to previous right image
        std::unique_ptr<Vop2el::PatchCorrector> CorrectorActualLeftPreviousRight;
        // Patch with its ncc score and type
        struct PatchWithScore;
        // Patch type that could be either original or corrected using ground plane
        enum struct PatchType {ORIGINAL, CORRECTED};

        // Compute normalized cross-correlation of a patch from reference image
        // on the epipolar line of target image
        void ComputeNccOnEpipolarLine(const cv::Mat& referencePatch,
                                    const cv::Mat& targetImage,
                                    PatchType patchType,
                                    const cv::Point2f& KeyPoint,
                                    const cv::Vec3f& epipolarLine,
                                    std::vector<PatchWithScore>& matches) const;
        // Check whether the difference between the highest and the lowest ncc score
        // is bigger than a threshold which means that the keypoint is not ambigious
        bool IsNumberOfCandidateValid(const std::vector<PatchWithScore>& matches) const;
        // Search matches in narrow region in an image
        void SearchMatchesPreviousFrame(const cv::Mat& referencePatch,
                                        const cv::Mat& targetImage,
                                        const cv::Point2f& pixTargetImage,
                                        std::pair<cv::Point2f, float>& optimalMatch) const;
        // Compute the match that have the highest sum of ncc scores: actual left image,
        // previous left image and previous right image
        int GetBestMatch(const std::vector<PatchWithScore>& matches,
                         const std::vector<std::pair<cv::Point2f, float>>& bestMatchesPreviousLeft,
                         const std::vector<std::pair<cv::Point2f, float>>& bestMatchesPreviousRight) const;
        // Check whether the entire patch of a key point fall inside the image
        bool IsKeyPointPatchInImage(const cv::Point2f& keyPoint) const;
        // Compute matches candidates on the actual right image for original left
        // image keypoint patch and for the corrected patch if ground plane parameters are given
        void GetStereoCandidatesMatches(const cv::Mat& referencePatch,
                                        const cv::Point2f& keyPoint,
                                        const cv::Vec3f& epipolarLine,
                                        std::vector<PatchWithScore>& matches) const;
        // Compute matches in previous frame by triangulation/projection, and reprojection/projection
        // on ground plane if the ground plane parameters are given
        void GetMatchesPreviousFrame(const cv::Mat& referencePatch,
                                    const cv::Point2f& keyPoint,
                                    const std::vector<PatchWithScore>& matches,
                                    std::vector<std::pair<cv::Point2f, float>>& bestKeyPointPreviousLeft,
                                    std::vector<std::pair<cv::Point2f, float>>& bestKeyPointPreviousRight) const;
        // Compute matches in previous frame by triangulation/projection
        void GetMatchPreviousFrameOriginal(const cv::Mat& referencePatch,
                                        const cv::Point2f& keyPoint,
                                        const PatchWithScore& match,
                                        std::pair<cv::Point2f, float>& bestKeyPointPreviousLeft,
                                        std::pair<cv::Point2f, float>& bestKeyPointPreviousRight) const;
        // Compute matches in previous frame by reprojection
        // on ground plane then projection on previous frame
        void GetMatchPreviousFrameCorrected(const cv::Mat& correctedPatchPreviousLeft,
                                            const cv::Mat& correctedPatchPreviousRight,
                                            const cv::Point2f& keyPoint,
                                            const PatchWithScore& match,
                                            std::pair<cv::Point2f, float>& bestKeyPointPreviousLeft,
                                            std::pair<cv::Point2f, float>& bestKeyPointPreviousRight) const;
        // Check whether a keypoint is in the image
        bool IsKeyPointInImage(const cv::Point2f& keyPoint) const;
        // Check whether a patch variance is equal to zero, we must check whether the patches are flat before
        // cv::matchTemplate because of this opencv bug https://github.com/opencv/opencv/issues/5688
        bool IsPatchVarianceZero(const cv::Mat& patch) const;
        // Keep matches candidates that have ncc score higher than a treshold
        void KeepValidStereoCandidates(const std::vector<PatchWithScore>& matches,
                                    std::vector<PatchWithScore>& onlyGoodMatches) const;
};
}