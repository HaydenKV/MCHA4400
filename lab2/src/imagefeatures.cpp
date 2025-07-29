#include <string>  
#include <print>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include "imagefeatures.h"

cv::Mat detectAndDrawHarris(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO

    return imgout;
}

cv::Mat detectAndDrawShiAndTomasi(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO

    return imgout;
}

cv::Mat detectAndDrawFAST(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO

    return imgout;
}

cv::Mat detectAndDrawArUco(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO

    return imgout;
}
