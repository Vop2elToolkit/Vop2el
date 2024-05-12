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

#include "Common.h"

//---------------------------------------------------------------------------------------
void Common::ComputeFundMatrixFromRotTranCal(const cv::Mat& rotation,
                                            const cv::Mat& translation,
                                            const cv::Mat& calibration,
                                            cv::Mat& fundamental)
{
    if ((rotation.type() != CV_64F) || (translation.type() != CV_64F) || (calibration.type() != CV_64F))
        throw std::runtime_error("[ERROR] All inputs types of function ComputeFundMatrixFromRotTranCal must be double");

    double skewSymetricArray[9] = {0.0, -translation.at<double>(2), translation.at<double>(1),
                                   translation.at<double>(2), 0.0, -translation.at<double>(0),
                                   -translation.at<double>(1), translation.at<double>(0), 0.0};

    cv::Mat skewSymetricMatrix(3, 3, CV_64F, skewSymetricArray);
    cv::Mat EssentielMatrix = skewSymetricMatrix * rotation;
    fundamental = calibration.t().inv() * EssentielMatrix * calibration.inv();
}