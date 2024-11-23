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

#include "Vop2elMatcher.h"

namespace Vop2el
{
struct Vop2elMatcher::PatchWithScore
{
    cv::Point2f KeyPoint;
    float Score;
    PatchType Type;
};

//---------------------------------------------------------------------------------------
Vop2elMatcher::Vop2elMatcher(const Vop2el::StereoImagesPairWithKeyPoints& stereoImagesPairWithKeyPoints,
                            const Vop2el::Vop2elMatcherParameters& matcherParams,
                            const Common::Camera& cameraParams,
                            const cv::Affine3d& transformPreviousActual) :
                            PairWithKeyPoints(stereoImagesPairWithKeyPoints), Vop2elMatcherParams(matcherParams), CameraParams(cameraParams)
{
    cv::Affine3d extrinsicTransformAffine;
    extrinsicTransformAffine.linear(this->CameraParams.ExtrinsicRotation);
    extrinsicTransformAffine.translation(this->CameraParams.ExtrinsicTranslation);
    this->ExtrinsicTransformAffine = extrinsicTransformAffine;

    this->ProjectionActualLeft = this->CameraParams.CalibrationMatrix * cv::Mat::eye(3, 4, CV_64F);

    cv::Mat extrinsicTransformInv4x3(cv::Mat(extrinsicTransformAffine.inv().matrix)(cv::Range(0, 3), cv::Range::all()));
    this->ProjectionActualRight = this->CameraParams.CalibrationMatrix * extrinsicTransformInv4x3;

    cv::Mat transformPreviousActual4x3(cv::Mat(transformPreviousActual.matrix)(cv::Range(0, 3), cv::Range::all()));
    this->ProjectionPreviousLeft = this->CameraParams.CalibrationMatrix * transformPreviousActual4x3;

    cv::Mat transformPrevRightActLeft(cv::Mat((extrinsicTransformAffine.inv() * transformPreviousActual).matrix)(cv::Range(0, 3), cv::Range::all()));
    this->ProjectionPreviousRight = this->CameraParams.CalibrationMatrix * transformPrevRightActLeft;

    this->PreviousActualTransform = transformPreviousActual;
}

//---------------------------------------------------------------------------------------
void Vop2elMatcher::SetPlaneParameters(const cv::Vec3f& normal, float distance)
{
    if (normal.rows != 3 || normal.cols != 1)
        throw std::runtime_error("Normal vector must be of size [1 x 3]");

    cv::Affine3d extrinsicAffine = cv::Affine3d(this->CameraParams.ExtrinsicRotation, this->CameraParams.ExtrinsicTranslation);
    cv::Mat extrinsicTransform = cv::Mat(extrinsicAffine.matrix);
    cv::Mat extrinsicTransformInv = cv::Mat(extrinsicAffine.inv().matrix);

    cv::Mat extrinsicTransformFloat;
    extrinsicTransform.convertTo(extrinsicTransformFloat, CV_32F);
    cv::Mat essentielMatrixFloat;
    this->CameraParams.CalibrationMatrix.convertTo(essentielMatrixFloat, CV_32F);

    this->CorrectorActualLeftActualRight = std::make_unique<PatchCorrector>(
        this->PairWithKeyPoints.ActualLeftImage, this->PairWithKeyPoints.ActualRightImage, extrinsicTransformFloat,
        essentielMatrixFloat, normal, distance, this->Vop2elMatcherParams.HalfPatchRows, this->Vop2elMatcherParams.HalfPatchCols);

    cv::Mat actualRightPreviousLeftTransform = extrinsicTransformInv * this->PreviousActualTransform.inv().matrix;
    actualRightPreviousLeftTransform.convertTo(actualRightPreviousLeftTransform, CV_32F);
    this->CorrectorActualRightPreviousLeft = std::make_unique<PatchCorrector>(
        this->PairWithKeyPoints.ActualRightImage, this->PairWithKeyPoints.PreviousLeftImage, actualRightPreviousLeftTransform,
        essentielMatrixFloat, normal, distance, this->Vop2elMatcherParams.HalfPatchRows, this->Vop2elMatcherParams.HalfPatchCols);

    cv::Mat actualRightPreviousRightTransform = extrinsicTransformInv * this->PreviousActualTransform.inv().matrix * extrinsicTransform;
    actualRightPreviousRightTransform.convertTo(actualRightPreviousRightTransform, CV_32F);
    this->CorrectorActualRightPreviousRight = std::make_unique<PatchCorrector>(
        this->PairWithKeyPoints.ActualRightImage, this->PairWithKeyPoints.PreviousRightImage, actualRightPreviousRightTransform,
        essentielMatrixFloat, normal, distance, this->Vop2elMatcherParams.HalfPatchRows, this->Vop2elMatcherParams.HalfPatchCols);

    cv::Mat actualLeftPreviousLeftTransform = cv::Mat(this->PreviousActualTransform.inv().matrix);
    actualLeftPreviousLeftTransform.convertTo(actualLeftPreviousLeftTransform, CV_32F);
    this->CorrectorActualLeftPreviousLeft = std::make_unique<PatchCorrector>(
        this->PairWithKeyPoints.ActualLeftImage, this->PairWithKeyPoints.PreviousLeftImage, actualLeftPreviousLeftTransform,
        essentielMatrixFloat, normal, distance, this->Vop2elMatcherParams.HalfPatchRows, this->Vop2elMatcherParams.HalfPatchCols);

    cv::Mat actualLeftPreviousRightTransform = this->PreviousActualTransform.inv().matrix  * extrinsicTransform;
    actualLeftPreviousRightTransform.convertTo(actualLeftPreviousRightTransform, CV_32F);
    this->CorrectorActualLeftPreviousRight = std::make_unique<PatchCorrector>(
        this->PairWithKeyPoints.ActualLeftImage, this->PairWithKeyPoints.PreviousRightImage, actualLeftPreviousRightTransform,
        essentielMatrixFloat, normal, distance, this->Vop2elMatcherParams.HalfPatchRows, this->Vop2elMatcherParams.HalfPatchCols);
}

//---------------------------------------------------------------------------------------
bool Vop2elMatcher::IsPatchVarianceZero(const cv::Mat& patch) const
{
    double min = 0;
    double* minPtr = &min;
    double max = 0;
    double* maxPtr = &max;
    cv::minMaxLoc(patch, minPtr, maxPtr);

    if (std::abs((*minPtr) - (*maxPtr)) < 1e-6)
        return true;

    return false;
}

//---------------------------------------------------------------------------------------
bool Vop2elMatcher::IsKeyPointPatchInImage(const cv::Point2f& keyPoint) const
{
    if (((keyPoint.x + static_cast<float>(this->Vop2elMatcherParams.HalfPatchCols)) > (static_cast<float>(this->CameraParams.cols) - 1.f)) ||
        ((keyPoint.x - static_cast<float>(this->Vop2elMatcherParams.HalfPatchCols)) < 0.f) ||
        ((keyPoint.y + static_cast<float>(this->Vop2elMatcherParams.HalfPatchRows)) > (static_cast<float>(this->CameraParams.rows) - 1.f)) ||
        ((keyPoint.y - static_cast<float>(this->Vop2elMatcherParams.HalfPatchRows)) < 0.f))
    {
        return false;
    }

    return true;
}

//---------------------------------------------------------------------------------------
bool Vop2elMatcher::IsKeyPointInImage(const cv::Point2f& keyPoint) const
{
    if ((keyPoint.x > (static_cast<float>(this->CameraParams.cols) - 1.f)) || (keyPoint.x < 0.f) ||
        (keyPoint.y > (static_cast<float>(this->CameraParams.rows) - 1.f)) || (keyPoint.y < 0.f))
    {
        return false;
    }

    return true;
}


//---------------------------------------------------------------------------------------
void Vop2elMatcher::ComputeCandidatesOnEpipolarLine(const cv::Mat& targetImage,
                                                    const cv::Point2f& keyPoint,
                                                    const cv::Vec3f& epipolarLine,
                                                    std::vector<PatchWithScore>& matches,
                                                    cv::Mat& candidatePatches) const
{
    matches.clear();
    candidatePatches.release();
    if (std::abs(epipolarLine[1]) < 1e-6f)
        return;

    float keyPointMinusSearchInterval = keyPoint.x - static_cast<float>(this->Vop2elMatcherParams.EpipolarLineSearchInterval);
    float HalfPatchColsFloat = static_cast<float>(this->Vop2elMatcherParams.HalfPatchCols);
    float startColumn = keyPointMinusSearchInterval < HalfPatchColsFloat ? HalfPatchColsFloat : keyPointMinusSearchInterval;

    float keyPointPlusSearchInterval = keyPoint.x + static_cast<float>(this->Vop2elMatcherParams.EpipolarLineSearchInterval);
    float imageColsMinusHalfPatchCols = static_cast<float>(this->CameraParams.cols - 1 - this->Vop2elMatcherParams.HalfPatchCols);
    float endColumn = (keyPointPlusSearchInterval > imageColsMinusHalfPatchCols) ? imageColsMinusHalfPatchCols : keyPointPlusSearchInterval;

    int startColumnInt = static_cast<int>(startColumn);
    int endColumnInt = static_cast<int>(endColumn);

    cv::Size patchSize(this->Vop2elMatcherParams.HalfPatchCols * 2 + 1, this->Vop2elMatcherParams.HalfPatchRows * 2 + 1);
    for (int keyPointColumn = startColumnInt; keyPointColumn < endColumnInt; ++keyPointColumn)
    {
        float keyPointColumnFloat = static_cast<float>(keyPointColumn);
        float keyPointRow = (-epipolarLine[0] * keyPointColumnFloat - epipolarLine[2]) / epipolarLine[1];

        cv::Point2f patchCenter(keyPointColumnFloat, keyPointRow);
        if (!(this->IsKeyPointPatchInImage(patchCenter)))
            continue;

        cv::Mat candidatePatch;
        cv::getRectSubPix(targetImage, patchSize, patchCenter, candidatePatch);

        if (this->IsPatchVarianceZero(candidatePatch))
            continue;

        if (candidatePatches.empty())
            candidatePatches = candidatePatch;
        else
        {
            cv::Mat tempPatches;
            cv::hconcat(candidatePatches, candidatePatch, tempPatches);
            candidatePatches = tempPatches;
        }
        PatchWithScore validMatch;
        validMatch.KeyPoint = cv::Point2f(keyPointColumnFloat, keyPointRow);
        matches.emplace_back(validMatch);
    }
}

//---------------------------------------------------------------------------------------
void Vop2elMatcher::ComputeNccOnEpipolarLine(const cv::Mat& referencePatch,
                                            PatchType patchType,
                                            const cv::Mat& candidatePatches,
                                            std::vector<PatchWithScore>& matches) const
{
    if (!candidatePatches.empty())
    {
        cv::Size patchSize(this->Vop2elMatcherParams.HalfPatchCols * 2 + 1, this->Vop2elMatcherParams.HalfPatchRows * 2 + 1);
        cv::Mat nVCCScore;
        cv::matchTemplate(referencePatch, candidatePatches, nVCCScore, cv::TM_CCOEFF_NORMED);
        for (int matchIdx = 0; matchIdx < matches.size(); ++matchIdx)
        {
            matches[matchIdx].Score = nVCCScore.at<float>(patchSize.width * matchIdx);
            matches[matchIdx].Type = patchType;
        }
    }
}

//---------------------------------------------------------------------------------------
bool Vop2elMatcher::IsNumberOfCandidateValid(const std::vector<PatchWithScore>& matches) const
{
    if ((matches.size() == 0) || (matches[0].Score < this->Vop2elMatcherParams.NccTreshold))
        return false;

    float firstElement = matches[0].Score;
    int maxOfCandidates = (matches.size() >= this->Vop2elMatcherParams.MaxStereoPointsToProcess) ?
                                    this->Vop2elMatcherParams.MaxStereoPointsToProcess : matches.size();

    if (maxOfCandidates < this->Vop2elMatcherParams.MaxStereoPointsToProcess)
        return true;

    std::vector<float> differenceFirstXElements(maxOfCandidates);
    std::transform(matches.begin(), matches.begin() + maxOfCandidates, differenceFirstXElements.begin(),
    [firstElement](const PatchWithScore& element)
    {
        return firstElement - element.Score;
    });

    auto maxDiff = std::max_element(differenceFirstXElements.begin(), differenceFirstXElements.end());

    if (*maxDiff < this->Vop2elMatcherParams.MaxThresh)
        return false;

    return true;
}

//---------------------------------------------------------------------------------------
void Vop2elMatcher::KeepValidStereoCandidates(const std::vector<PatchWithScore>& matches,
                                            std::vector<PatchWithScore>& onlyGoodMatches) const
{
    onlyGoodMatches.clear();

    for (const auto& match : matches)
    {
        if (match.Score > this->Vop2elMatcherParams.NccTreshold)
            onlyGoodMatches.push_back(match);
    }
}

//---------------------------------------------------------------------------------------
void Vop2elMatcher::SearchMatchesPreviousFrame(const cv::Mat& referencePatch,
                                            const cv::Mat& targetImage,
                                            const cv::Point2f& pixTargetImage,
                                            std::pair<cv::Point2f, float>& optimalMatch) const
{
    if (!(this->IsKeyPointInImage(pixTargetImage)))
    {
        optimalMatch = std::pair<cv::Point2f, float>(cv::Point2f(-1.f, -1.f), std::numeric_limits<float>::lowest());
        return;
    }

    cv::Size patchSize(this->Vop2elMatcherParams.HalfPatchCols * 2 + 1, this->Vop2elMatcherParams.HalfPatchRows * 2 + 1);
    cv::Mat candidatePatches;
    std::vector<std::pair<cv::Point2f, float>> nVCCScoresAndKeyPoints;
    for (int rowIdx = -this->Vop2elMatcherParams.HalfVerticalSearch; rowIdx <= this->Vop2elMatcherParams.HalfVerticalSearch; ++rowIdx)
    {
        for (int colIdx = -this->Vop2elMatcherParams.HalfHorizontalSearch; colIdx <= this->Vop2elMatcherParams.HalfHorizontalSearch; ++colIdx)
        {
            cv::Point2f keyPointCenter(static_cast<float>(colIdx) + pixTargetImage.x, static_cast<float>(rowIdx) + pixTargetImage.y);
            if (!(this->IsKeyPointPatchInImage(keyPointCenter)))
                continue;

            cv::Mat candidatePatch;
            cv::getRectSubPix(targetImage, patchSize, keyPointCenter, candidatePatch);

            if (this->IsPatchVarianceZero(candidatePatch))
                continue;

            if (candidatePatches.empty())
                candidatePatches = candidatePatch;
            else
            {
                cv::Mat tempPatches;
                cv::hconcat(candidatePatches, candidatePatch, tempPatches);
                candidatePatches = tempPatches;
            }
            nVCCScoresAndKeyPoints.emplace_back(std::pair(keyPointCenter, 0.f));
        }
    }

    if (nVCCScoresAndKeyPoints.size() == 0)
    {
        optimalMatch = std::pair<cv::Point2f, float>(cv::Point2f(-1.f, -1.f), std::numeric_limits<float>::lowest());
        return;
    }

    cv::Mat nVCCScores;
    cv::matchTemplate(candidatePatches, referencePatch, nVCCScores, cv::TM_CCOEFF_NORMED);
    for (int nVCCScoreAndKeyPointIdx = 0; nVCCScoreAndKeyPointIdx < nVCCScoresAndKeyPoints.size(); ++nVCCScoreAndKeyPointIdx)
        nVCCScoresAndKeyPoints[nVCCScoreAndKeyPointIdx].second = nVCCScores.at<float>(patchSize.width * nVCCScoreAndKeyPointIdx);

    auto optimalMatchItr = std::max_element(nVCCScoresAndKeyPoints.begin(), nVCCScoresAndKeyPoints.end(),
    [](const std::pair<cv::Point2f, float>& firstScore, const std::pair<cv::Point2f, float>& secondScore)
    {
        return (firstScore.second < secondScore.second);
    });
    optimalMatch = *optimalMatchItr;
}

//---------------------------------------------------------------------------------------
int Vop2elMatcher::GetBestMatch(const std::vector<PatchWithScore>& matches,
                              const std::vector<std::pair<cv::Point2f, float>>& bestMatchesPreviousLeft,
                              const std::vector<std::pair<cv::Point2f, float>>& bestMatchesPreviousRight) const
{
    if ((bestMatchesPreviousLeft.size() != bestMatchesPreviousRight.size()) || (matches.size() == 0))
        return -1;

    std::vector<float> totalNccs;
    for (int pointIdx = 0; pointIdx < bestMatchesPreviousLeft.size(); ++pointIdx)
    {
        if ((bestMatchesPreviousLeft[pointIdx].second > this->Vop2elMatcherParams.NccTreshold) &&
            (bestMatchesPreviousRight[pointIdx].second > this->Vop2elMatcherParams.NccTreshold))
        {
            float totalNccElement = matches[pointIdx].Score +
                                    bestMatchesPreviousLeft[pointIdx].second +
                                    bestMatchesPreviousRight[pointIdx].second;
            totalNccs.push_back(totalNccElement);
        }
        else
            totalNccs.push_back(std::numeric_limits<float>::lowest());
    }

    auto optimalMatchItr = std::max_element(totalNccs.begin(), totalNccs.end());

    if ((*optimalMatchItr) < -1.f)
        return -1;
    return std::distance(totalNccs.begin(), optimalMatchItr);
}

//---------------------------------------------------------------------------------------
void Vop2elMatcher::GetStereoCandidatesMatches(const cv::Mat& referencePatch,
                                            const cv::Point2f& keyPoint,
                                            const cv::Vec3f& epipolarLine,
                                            std::vector<PatchWithScore>& matches) const
{
    matches.clear();
    cv::Mat candidatePatches;
    this->ComputeCandidatesOnEpipolarLine(*this->PairWithKeyPoints.ActualRightImage, keyPoint, epipolarLine, matches, candidatePatches);
    this->ComputeNccOnEpipolarLine(referencePatch, PatchType::ORIGINAL, candidatePatches, matches);

    if (this->CorrectorActualLeftActualRight)
    {
        cv::Mat correctedPatch;
        this->CorrectorActualLeftActualRight->GetPatchPerspectiveCorrected(keyPoint, correctedPatch);
        if (correctedPatch.size() != cv::Size(0, 0) && !(this->IsPatchVarianceZero(correctedPatch)))
        {
            std::vector<PatchWithScore> matchesPerspectiveCorrected = matches;
            this->ComputeNccOnEpipolarLine(correctedPatch, PatchType::CORRECTED, candidatePatches, matchesPerspectiveCorrected);
            matches.insert(matches.end(), matchesPerspectiveCorrected.begin(), matchesPerspectiveCorrected.end());
        }
    }

    std::sort(matches.begin(), matches.end(), [](const PatchWithScore& firstElement, const PatchWithScore& secondElement)
    {
        return (firstElement.Score > secondElement.Score);
    });
}

//---------------------------------------------------------------------------------------
void Vop2elMatcher::GetMatchPreviousFrameOriginal(const cv::Mat& referencePatch,
                                                const cv::Point2f& keyPoint,
                                                const PatchWithScore& match,
                                                std::pair<cv::Point2f, float>& bestKeyPointPreviousLeft,
                                                std::pair<cv::Point2f, float>& bestKeyPointPreviousRight) const
{
    cv::Mat keyPoint2dLeftImage = (cv::Mat_<double>(2, 1) << keyPoint.x, keyPoint.y);
    cv::Mat keyPoint2dRightImage = (cv::Mat_<double>(2, 1) << match.KeyPoint.x, match.KeyPoint.y);

    cv::Mat points4D;
    cv::triangulatePoints(this->ProjectionActualLeft, this->ProjectionActualRight, keyPoint2dLeftImage, keyPoint2dRightImage, points4D);
    cv::Mat triangProjectedPreviousLeft = this->ProjectionPreviousLeft * points4D;
    triangProjectedPreviousLeft = triangProjectedPreviousLeft / triangProjectedPreviousLeft.at<double>(2);

    if (triangProjectedPreviousLeft.at<double>(2) < 1e-6)
    {
        bestKeyPointPreviousLeft = std::make_pair<cv::Point2f, float>(cv::Point2f(-1.f, -1.f), std::numeric_limits<float>::lowest());
        bestKeyPointPreviousRight = std::make_pair<cv::Point2f, float>(cv::Point2f(-1.f, -1.f), std::numeric_limits<float>::lowest());
    }

    cv::Point2f triangProjectedPreviousLeftPoint(static_cast<float>(triangProjectedPreviousLeft.at<double>(0)),
                                                static_cast<float>(triangProjectedPreviousLeft.at<double>(1)));

    cv::Mat triangProjectedPreviousRight = this->ProjectionPreviousRight * points4D;
    triangProjectedPreviousRight = triangProjectedPreviousRight / triangProjectedPreviousRight.at<double>(2);

    if (triangProjectedPreviousRight.at<double>(2) < 1e-6)
    {
        bestKeyPointPreviousLeft = std::make_pair<cv::Point2f, float>(cv::Point2f(-1.f, -1.f), std::numeric_limits<float>::lowest());
        bestKeyPointPreviousRight = std::make_pair<cv::Point2f, float>(cv::Point2f(-1.f, -1.f), std::numeric_limits<float>::lowest());
    }

    cv::Point2f triangProjectedPreviousRightPoint(static_cast<float>(triangProjectedPreviousRight.at<double>(0)),
                                                static_cast<float>(triangProjectedPreviousRight.at<double>(1)));
    std::pair<cv::Point2f, float> candidateMatchPreviousLeft;
    this->SearchMatchesPreviousFrame(referencePatch, *this->PairWithKeyPoints.PreviousLeftImage,
                                    triangProjectedPreviousLeftPoint, candidateMatchPreviousLeft);

    bestKeyPointPreviousLeft = std::move(candidateMatchPreviousLeft);

    std::pair<cv::Point2f, float> candidateMatchPreviousRight;
    this->SearchMatchesPreviousFrame(referencePatch, *this->PairWithKeyPoints.PreviousRightImage,
                                    triangProjectedPreviousRightPoint, candidateMatchPreviousRight);

    bestKeyPointPreviousRight = std::move(candidateMatchPreviousRight);

}

//---------------------------------------------------------------------------------------
void Vop2elMatcher::GetMatchPreviousFrameCorrected(const cv::Mat& correctedPatchPreviousLeft,
                                                const cv::Mat& correctedPatchPreviousRight,
                                                const cv::Point2f& keyPoint,
                                                const PatchWithScore& match,
                                                std::pair<cv::Point2f, float>& bestKeyPointPreviousLeft,
                                                std::pair<cv::Point2f, float>& bestKeyPointPreviousRight) const
{
    cv::Point2f keyPointActual(match.KeyPoint.x, match.KeyPoint.y);

    cv::Point2f reprojectedKeyPointPreviousLeft;
    this->CorrectorActualRightPreviousLeft->GetKeyPointByPlaneProjection(keyPointActual, reprojectedKeyPointPreviousLeft);

    std::pair<cv::Point2f, float> optimalMatchPreviousLeft;
    this->SearchMatchesPreviousFrame(correctedPatchPreviousLeft, *this->PairWithKeyPoints.PreviousLeftImage,
                                    reprojectedKeyPointPreviousLeft, optimalMatchPreviousLeft);
    bestKeyPointPreviousLeft = std::move(optimalMatchPreviousLeft);

    cv::Point2f reprojectedKeyPointPreviousRight;
    this->CorrectorActualRightPreviousRight->GetKeyPointByPlaneProjection(keyPointActual, reprojectedKeyPointPreviousRight);


    std::pair<cv::Point2f, float> optimalMatchPreviousRight;
    this->SearchMatchesPreviousFrame(correctedPatchPreviousRight, *this->PairWithKeyPoints.PreviousRightImage,
                                    reprojectedKeyPointPreviousRight, optimalMatchPreviousRight);
    bestKeyPointPreviousRight = std::move(optimalMatchPreviousRight);
}

//---------------------------------------------------------------------------------------
void Vop2elMatcher::GetMatchesPreviousFrame(const cv::Mat& referencePatch,
                                            const cv::Point2f& keyPoint,
                                            const std::vector<PatchWithScore>& matches,
                                            std::vector<std::pair<cv::Point2f, float>>& bestKeyPointPreviousLeft,
                                            std::vector<std::pair<cv::Point2f, float>>& bestKeyPointPreviousRight) const
{

    cv::Mat correctedPatchPreviousLeft;
    if (this->CorrectorActualLeftPreviousLeft)
        this->CorrectorActualLeftPreviousLeft->GetPatchPerspectiveCorrected(keyPoint, correctedPatchPreviousLeft);
    bool isCorrectedPatchPreviousLeftValid = !(this->IsPatchVarianceZero(correctedPatchPreviousLeft)) &&
                                                                (correctedPatchPreviousLeft.size() != cv::Size(0, 0));

    cv::Mat correctedPatchPreviousRight;
    if (this->CorrectorActualLeftPreviousRight)
        this->CorrectorActualLeftPreviousRight->GetPatchPerspectiveCorrected(keyPoint, correctedPatchPreviousRight);
    bool isCorrectedPatchPreviousRightValid = !(this->IsPatchVarianceZero(correctedPatchPreviousRight)) &&
                                                                 (correctedPatchPreviousRight.size() != cv::Size(0, 0));

    bestKeyPointPreviousLeft.clear();
    bestKeyPointPreviousRight.clear();
    bestKeyPointPreviousLeft.resize(matches.size());
    bestKeyPointPreviousRight.resize(matches.size());

    for (int matchIdx = 0; matchIdx < matches.size(); ++matchIdx)
    {
        if (matches[matchIdx].Type == PatchType::ORIGINAL)
        {
            this->GetMatchPreviousFrameOriginal(referencePatch, keyPoint, matches[matchIdx],
                                    bestKeyPointPreviousLeft[matchIdx], bestKeyPointPreviousRight[matchIdx]);
        }
        else if (matches[matchIdx].Type == PatchType::CORRECTED)
        {
            if (isCorrectedPatchPreviousLeftValid && isCorrectedPatchPreviousRightValid)
            {
                this->GetMatchPreviousFrameCorrected(correctedPatchPreviousLeft, correctedPatchPreviousRight,
                                                    keyPoint, matches[matchIdx],
                                                    bestKeyPointPreviousLeft[matchIdx], bestKeyPointPreviousRight[matchIdx]);
            }
            else
            {
                bestKeyPointPreviousLeft[matchIdx] =
                            std::make_pair<cv::Point2f, float>(cv::Point2f(-1.f, -1.f), std::numeric_limits<float>::lowest());
                bestKeyPointPreviousRight[matchIdx] =
                            std::make_pair<cv::Point2f, float>(cv::Point2f(-1.f, -1.f), std::numeric_limits<float>::lowest());
            }
        }
    }
}

//---------------------------------------------------------------------------------------
void Vop2elMatcher::GetMatches(std::vector<Vop2el::Match>& matches)
{
    matches.clear();
    std::vector<std::pair<Vop2el::Match, int>> matchesAndIdx;
    std::shared_mutex mtutexNewValidMatch;
    std::shared_mutex mutexCorrectedPatches;

    cv::Mat fundamental;
    Common::ComputeFundMatrixFromRotTranCal(this->CameraParams.ExtrinsicRotation, this->CameraParams.ExtrinsicTranslation,
                                            this->CameraParams.CalibrationMatrix, fundamental);

    int numCorrected = 0;
    bool doContinue = true;

    #pragma omp parallel for schedule(dynamic)
    for (int keyPointIdx = 0; keyPointIdx < this->PairWithKeyPoints.ActualLeftKeyPoints->size(); ++keyPointIdx)
    {
        if (doContinue)
        {
            // Check wether keyPoint patch is inside the image
            cv::Point2f keyPoint = this->PairWithKeyPoints.ActualLeftKeyPoints->at(keyPointIdx);

            if (!(this->IsKeyPointPatchInImage(keyPoint)))
            continue;

            cv::Size patchSize(this->Vop2elMatcherParams.HalfPatchCols * 2 + 1, this->Vop2elMatcherParams.HalfPatchRows * 2 + 1);
            cv::Mat referencePatch;
            cv::getRectSubPix((*this->PairWithKeyPoints.ActualLeftImage), patchSize, keyPoint, referencePatch);

            // Compute epipolar line of the keyPoint in the actual right image and get potential candidates matches
            double keyPointarr[3] = {keyPoint.x, keyPoint.y, 1};
            cv::Mat epipolarLine = fundamental * cv::Mat(3, 1, CV_64F, keyPointarr);

            cv::Mat epipolarLineFloat;
            epipolarLine.convertTo(epipolarLineFloat, CV_32F);
            cv::Vec3f epipolarLineVec(epipolarLineFloat.at<float>(0), epipolarLineFloat.at<float>(1), epipolarLineFloat.at<float>(2));

            std::vector<PatchWithScore> matches;
            this->GetStereoCandidatesMatches(referencePatch, keyPoint, epipolarLineVec, matches);

            // If the number of candidates is valid, do the following
            if (this->IsNumberOfCandidateValid(matches))
            {
                std::vector<PatchWithScore> onlyGoodMatches;
                this->KeepValidStereoCandidates(matches, onlyGoodMatches);

                // Find best match in previous left and right image
                std::vector<std::pair<cv::Point2f, float>> bestKeyPointPreviousLeft;
                std::vector<std::pair<cv::Point2f, float>> bestKeyPointPreviousRight;
                this->GetMatchesPreviousFrame(referencePatch, keyPoint, onlyGoodMatches, bestKeyPointPreviousLeft, bestKeyPointPreviousRight);

                // Do the following if best match exist
                int idxBestKeyPoint = this->GetBestMatch(onlyGoodMatches, bestKeyPointPreviousLeft, bestKeyPointPreviousRight);

                if (idxBestKeyPoint != -1)
                {
                    if (matches[idxBestKeyPoint].Type == PatchType::CORRECTED)
                    {
                        std::unique_lock<std::shared_mutex> lock(mutexCorrectedPatches);
                        numCorrected++;
                    }

                    Vop2el::Match match{keyPoint, matches[idxBestKeyPoint].KeyPoint, bestKeyPointPreviousLeft[idxBestKeyPoint].first,
                                bestKeyPointPreviousRight[idxBestKeyPoint].first};
                    {
                        std::unique_lock<std::shared_mutex> lock(mtutexNewValidMatch);

                        matchesAndIdx.emplace_back(std::make_pair<>(match, keyPointIdx));

                        float keyPointMovement = cv::norm(match.PreviousLeft - match.ActualLeft);
                        if (keyPointMovement <= 1.f)
                            ++this->NumFixedKeyPoints;

                        if (matchesAndIdx.size() > this->Vop2elMatcherParams.MaxNumberOfMatches)
                            doContinue = false;
                    }
                }
            }
        }
    }

    std::sort(matchesAndIdx.begin(), matchesAndIdx.end(),
    [](std::pair<Vop2el::Match, int>& firstMatch, std::pair<Vop2el::Match, int>& secondMatch)
    {
        return (firstMatch.second > secondMatch.second);
    });

    matches.clear();
    for (auto correspondency : matchesAndIdx)
        matches.push_back(correspondency.first);

    std::cout << "Number of corrected key points using patch corrector is: " << numCorrected << std::endl;
}
}