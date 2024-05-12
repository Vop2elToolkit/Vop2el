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

#include "StereoImagesHandler.h"

namespace Vop2el
{
//---------------------------------------------------------------------------------------
void StereoImagesHandler::AddStereoPair(const std::string& leftImagePath,
                                        const std::string& rightImagePath,
                                        bool computeLeftImageKeyPoints,
                                        bool computeRightImageKeyPoints)
{
    if (this->FramesCount > this->StereoImagesHandlerParams.NumFramesCapacity)
        this->Reset();

    this->LeftImages.push_back(std::make_shared<cv::Mat>(cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE)));
    this->RightImages.push_back(std::make_shared<cv::Mat>(cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE)));

    if (computeLeftImageKeyPoints)
    {
        std::shared_ptr<std::vector<cv::Point2f>> leftImageKeyPoints;
        this->ComputeGFTTKeyPoints(this->LeftImages.back(), leftImageKeyPoints);
        this->LeftPrepImagesKeyPoints.emplace_back(leftImageKeyPoints);
        ++this->LeftKeyPointsCount;
    }

    if (computeRightImageKeyPoints)
    {
        std::shared_ptr<std::vector<cv::Point2f>> rightImageKeyPoints;
        this->ComputeGFTTKeyPoints(this->RightImages.back(), rightImageKeyPoints);
        this->RightPrepImagesKeyPoints.emplace_back(rightImageKeyPoints);
        ++this->RightKeyPointsCount;
    }

    ++this->FramesCount;
}

//---------------------------------------------------------------------------------------
void StereoImagesHandler::ComputeGFTTKeyPoints(std::shared_ptr<const cv::Mat> image,
                                            std::shared_ptr<std::vector<cv::Point2f>>& keyPoints)
{
    keyPoints = std::make_shared<std::vector<cv::Point2f>>();
    cv::Ptr<cv::GFTTDetector> gFTTDetector = cv::GFTTDetector::create();
    std::vector<cv::KeyPoint> binKeyPoints;

    for (int row = image->rows - 1; row > 0; row -= this->StereoImagesHandlerParams.BinHeight)
    {
        int binHeight = (row - this->StereoImagesHandlerParams.BinHeight) <= 0 ? row : this->StereoImagesHandlerParams.BinHeight;

        for (int col = 0; col < image->cols; col += this->StereoImagesHandlerParams.BinWidth)
        {
            int binWidth = (col + this->StereoImagesHandlerParams.BinWidth) >= image->cols ?
                                                        image->cols % col : this->StereoImagesHandlerParams.BinWidth;

            cv::Rect binRegion(col, row - binHeight, binWidth, binHeight);
            cv::Mat bin = (*image)(binRegion);

            gFTTDetector->setMaxFeatures(this->StereoImagesHandlerParams.MaxNumberOfKeyPointsPerBin);
            gFTTDetector->setMinDistance(1);
            gFTTDetector->detect(bin, binKeyPoints);
            for (auto& keyPoint : binKeyPoints)
                keyPoints->push_back(cv::Point2f(keyPoint.pt.x + col, keyPoint.pt.y + row - binHeight));
        }
    }
}

 //---------------------------------------------------------------------------------------
std::shared_ptr<const cv::Mat> StereoImagesHandler::GetLeftImage(int imageIndex) const
{
    if(!this->IsFrameIndexValid(imageIndex))
        throw std::runtime_error("Left image index is out of range.");

    if (imageIndex >= 0)
        return this->LeftImages[imageIndex];
    else
        return this->LeftImages[this->LeftImages.size() + imageIndex];
}

//---------------------------------------------------------------------------------------
std::shared_ptr<const cv::Mat> StereoImagesHandler::GetRightImage(int imageIndex) const
{
    if(!this->IsFrameIndexValid(imageIndex))
        throw std::runtime_error("Right image index is out of range.");

    if (imageIndex >= 0)
        return this->RightImages[imageIndex];
    else
        return this->RightImages[this->RightImages.size() + imageIndex];
}

//---------------------------------------------------------------------------------------
std::shared_ptr<const std::vector<cv::Point2f>> StereoImagesHandler::GetLeftImageKeyPoints(int imageIndex) const
{
    if(!this->IsFrameIndexValid(imageIndex))
        throw std::runtime_error("Left image index is out of range.");

    if (imageIndex >= 0)
        return this->LeftPrepImagesKeyPoints[imageIndex];
    else
        return this->LeftPrepImagesKeyPoints[this->LeftPrepImagesKeyPoints.size() + imageIndex];
}

//---------------------------------------------------------------------------------------
std::shared_ptr<const std::vector<cv::Point2f>> StereoImagesHandler::GetRightImageKeyPoints(int imageIndex) const
{
    if(!this->IsFrameIndexValid(imageIndex))
        throw std::runtime_error("Right image index is out of range.");

    if (imageIndex >= 0)
        return this->RightPrepImagesKeyPoints[imageIndex];
    else
        return this->RightPrepImagesKeyPoints[this->RightPrepImagesKeyPoints.size() + imageIndex];
}

//---------------------------------------------------------------------------------------
bool StereoImagesHandler::IsFrameIndexValid(int index) const
{
    if (index >= 0)
    {
        if (index > (this->StereoImagesHandlerParams.NumFramesCapacity - 1) ||  index > (this->FramesCount - 1))
            return false;
        else
            return true;
    }
    else
    {
        if (index < -this->StereoImagesHandlerParams.NumFramesCapacity || index < -this->FramesCount)
            return false;
        else
            return true;
    }
}

//---------------------------------------------------------------------------------------
void StereoImagesHandler::Reset()
{
    this->LeftImages.erase(this->LeftImages.begin(), this->LeftImages.end() - this->StereoImagesHandlerParams.NumFramesCapacity);
    this->RightImages.erase(this->RightImages.begin(), this->RightImages.end() - this->StereoImagesHandlerParams.NumFramesCapacity);

    if (this->LeftPrepImagesKeyPoints.size() > this->StereoImagesHandlerParams.NumFramesCapacity)
        this->LeftPrepImagesKeyPoints.erase(this->LeftPrepImagesKeyPoints.begin(),
                                            this->LeftPrepImagesKeyPoints.end() - this->StereoImagesHandlerParams.NumFramesCapacity);

    if (this->RightPrepImagesKeyPoints.size() > this->StereoImagesHandlerParams.NumFramesCapacity)
        this->RightPrepImagesKeyPoints.erase(this->RightPrepImagesKeyPoints.begin(),
                                             this->RightPrepImagesKeyPoints.end() - this->StereoImagesHandlerParams.NumFramesCapacity);

    this->FramesCount = this->StereoImagesHandlerParams.NumFramesCapacity;
}
}