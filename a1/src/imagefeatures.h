#ifndef IMAGEFEATURES_H
#define IMAGEFEATURES_H 

#include <opencv2/core.hpp>

cv::Mat detectAndDrawHarris(const cv::Mat & img, int maxNumFeatures);
cv::Mat detectAndDrawShiAndTomasi(const cv::Mat & img, int maxNumFeatures);
cv::Mat detectAndDrawFAST(const cv::Mat & img, int maxNumFeatures);
cv::Mat detectAndDrawArUco(const cv::Mat & img, int maxNumFeatures);

#include <array>
#include <Eigen/Core>

struct ArucoDetections
{
    std::vector<int> ids;                                // tag IDs
    std::vector<std::array<cv::Point2f,4>> corners;      // TL,TR,BR,BL per tag
    cv::Mat annotated;                                   // input image with markers drawn
};

// Choose a dictionary that matches your printed tags (Lab-2 typically used one of these):
//   cv::aruco::DICT_6X6_250   or   cv::aruco::DICT_4X4_50
ArucoDetections detectArucoLab2(const cv::Mat & imgBGR,
                                int dictionary = cv::aruco::DICT_6X6_250,
                                bool doCornerRefine = true);

// Helpful utility: build the 2x(4*N) measurement matrix from corners
Eigen::Matrix<double,2,Eigen::Dynamic>
buildYFromAruco(const std::vector<std::array<cv::Point2f,4>> & corners);

#endif