#include "RerunVisualizer.h"

//---------------------------------------------------------------------------------------
RerunVisualizer::RerunVisualizer()
{
    this->Recording = std::make_unique<rerun::RecordingStream>("Vop2el");
    this->Recording->spawn().exit_on_failure();
    sleep(3);
}

//---------------------------------------------------------------------------------------
void RerunVisualizer::StreamFrame(const std::string& leftImagePath,
                            const std::string& rightImagePath,
                            const Eigen::Affine3d& transform)
{
    // Process and stream left image
    cv::Mat leftImage = cv::imread(leftImagePath);
    cv::Mat rgbLeftImage;
    cv::cvtColor(leftImage, rgbLeftImage, cv::COLOR_RGB2BGR);
    cv::Mat rgbLeftImageResized;
    cv::resize(rgbLeftImage, rgbLeftImageResized, this->StreamCvImageSize);
    this->Recording->log("left_image", rerun::Image(this->StreamRerunImageShape, reinterpret_cast<const uint8_t*>(rgbLeftImageResized.data)));

    // Process and stream right image
    cv::Mat rightImage = cv::imread(rightImagePath);
    cv::Mat rgbRightImage;
    cv::cvtColor(rightImage, rgbRightImage, cv::COLOR_RGB2BGR);
    cv::Mat rgbRightImageResized;
    cv::resize(rgbRightImage, rgbRightImageResized, this->StreamCvImageSize);
    this->Recording->log("right_image", rerun::Image(this->StreamRerunImageShape, reinterpret_cast<const uint8_t*>(rgbRightImageResized.data)));

    // Process and stream trajectory
    Eigen::Affine3f transformFloat = transform.cast<float>();
    this->Trajectory.emplace_back(transformFloat.translation().data());
    rerun::LineStrip3D trajectoryAsLines(this->Trajectory);
    this->Recording->log("trajectory", rerun::LineStrips3D(trajectoryAsLines));
    this->Recording->set_time_sequence("currentframe_id", this->FrameIdx);
    ++this->FrameIdx;
}