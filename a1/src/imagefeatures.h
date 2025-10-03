#ifndef IMAGEFEATURES_H
#define IMAGEFEATURES_H 

#include <opencv2/core.hpp>
#include <array>
#include <Eigen/Core>

cv::Mat detectAndDrawHarris(const cv::Mat & img, int maxNumFeatures);
cv::Mat detectAndDrawShiAndTomasi(const cv::Mat & img, int maxNumFeatures);
cv::Mat detectAndDrawFAST(const cv::Mat & img, int maxNumFeatures);
cv::Mat detectAndDrawArUco(const cv::Mat & img, int maxNumFeatures);


struct ArucoDetections
{
    std::vector<int> ids;                                // tag IDs
    std::vector<std::array<cv::Point2f,4>> corners;      // TL,TR,BR,BL per tag
    cv::Mat annotated;                                   // input image with markers drawn
};

ArucoDetections detectArUcoPOSE(
    const cv::Mat& imgBGR,
    int dictionary,
    bool doCornerRefine,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    float tagSizeMeters,
    std::vector<cv::Vec3d>* outRvecs = nullptr,
    std::vector<cv::Vec3d>* outTvecs = nullptr,
    std::vector<double>* outMeanReprojErr = nullptr,
    double reprojErrThreshPx = 4.0,
    bool drawRejected = false
);

#endif