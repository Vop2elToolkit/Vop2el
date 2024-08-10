#pragma once

#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <rerun.hpp>
#include <rerun/demo_utils.hpp>

class RerunVisualizer
{
    public:
        RerunVisualizer();
        
        // Stream stereo image + trajectory
        void StreamFrame(const std::string& leftImagePath,
                    const std::string& rightImagePath,
                    const Eigen::Affine3d& transform);
    
    private:
        // Rerun stream
        std::unique_ptr<rerun::RecordingStream> Recording;
        // Trajectory
        std::vector<rerun::datatypes::Vec3D> Trajectory;
        // Current frame idx       
        int FrameIdx = 0;
        // Size of the image that will be streamed to rerun viewer (OpenCV format)
        const cv::Size StreamCvImageSize{413, 125};
        // Size of the image that will be streamed to rerun viewer (Rerun format)
        const rerun::Collection<rerun::TensorDimension> StreamRerunImageShape{125, 413, 3};
};