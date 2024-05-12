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
struct StereoImagesHandlerParameters
{
    int NumFramesCapacity = 2; // capacity of the stereo images handler, reset once exceeded
    int BinWidth = 50; // GFTT bin width
    int BinHeight = 50; // GFTT bin height
    int MaxNumberOfKeyPointsPerBin = 3; // maximum number of keypoints to have in a single BinWidth x BinHeight bin
};

class StereoImagesHandler
{
    public:
        StereoImagesHandler(Vop2el::StereoImagesHandlerParameters stereoImagesHandlerParams) :
                            StereoImagesHandlerParams(stereoImagesHandlerParams) {};

        // Add a stereo pair
        void AddStereoPair(const std::string& leftImagePath,
                        const std::string& rightImagePath,
                        bool computeLeftImageKeyPoints = true,
                        bool computeRightImageKeyPoints = true);
        // Get left image from given index
        std::shared_ptr<const cv::Mat> GetLeftImage(int imageIndex) const;
        // Get right image from given index
        std::shared_ptr<const cv::Mat> GetRightImage(int imageIndex) const ;
        // Get left image keypoints from given index
        std::shared_ptr<const std::vector<cv::Point2f>> GetLeftImageKeyPoints(int imageIndex) const;
        // Get right image keypoints from given index
        std::shared_ptr<const std::vector<cv::Point2f>> GetRightImageKeyPoints(int imageIndex) const;
        // Get number of left images
        int GetSize() const { return this->LeftImages.size(); }

    private:
        // Frames count
        int FramesCount = 0;
        // Number of left images that have associated keyPoints
        int LeftKeyPointsCount = 0;
        // Number of right images that have associated keyPoints
        int RightKeyPointsCount = 0;
        // Vector of raw left images
        std::vector<std::shared_ptr<cv::Mat>> LeftImages;
        // Vector of raw right images
        std::vector<std::shared_ptr<cv::Mat>> RightImages;
        // Vector of left images keyPoints
        std::vector<std::shared_ptr<std::vector<cv::Point2f>>> LeftPrepImagesKeyPoints;
        // Vector of right images keyPoints
        std::vector<std::shared_ptr<std::vector<cv::Point2f>>> RightPrepImagesKeyPoints;
        // Stereo images handler parameters
        Vop2el::StereoImagesHandlerParameters StereoImagesHandlerParams;
        // Compute good keyPoints to track of an image
        void ComputeGFTTKeyPoints(std::shared_ptr<const cv::Mat> image,
                                std::shared_ptr<std::vector<cv::Point2f>>& keyPoints);
        // Reset stereo images handler
        void Reset();
        // Check wether image index is valid
        bool IsFrameIndexValid(int index) const;
};
}