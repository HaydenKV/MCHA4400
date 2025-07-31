#include <string>  
#include <print>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include "imagefeatures.h"

cv::Mat detectAndDrawHarris(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Compute Harris response
    cv::Mat harrisResponse;
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    cv::cornerHarris(gray, harrisResponse, blockSize, apertureSize, k);

    // Normalize Harris response to 0–255 range
    cv::Mat harrisNorm;
    cv::normalize(harrisResponse, harrisNorm, 0, 255, cv::NORM_MINMAX);
    harrisNorm.convertTo(harrisNorm, CV_32F);

    // Define structure to hold corner data
    struct Corner {
        cv::Point pt;
        float score;
    };
    std::vector<Corner> corners;

    // Threshold and collect high-response points
    float threshold = 180.0f;
    for (int y = 0; y < harrisNorm.rows; ++y) {
        for (int x = 0; x < harrisNorm.cols; ++x) {
            float response = harrisNorm.at<float>(y, x);
            if (response > threshold) {
                corners.push_back({cv::Point(x, y), response});
            }
        }
    }

    // Print metadata for Task 1e
    std::println("Using harris feature detector");
    std::println("Image width: {}", img.cols);
    std::println("Image height: {}", img.rows);
    std::println("Features requested: {}", maxNumFeatures);
    std::println("Features detected: {}", corners.size());

    // Sort corners by descending score
    std::sort(corners.begin(), corners.end(), [](const Corner& a, const Corner& b) {
        return a.score > b.score;
    });

    // Limit to top N features
    int count = std::min(maxNumFeatures, static_cast<int>(corners.size()));

    // Draw all corners as small orange circles
    for (const auto& c : corners) {
        cv::circle(imgout, c.pt, 2, cv::Scalar(0, 165, 255), -1); // Orange filled circles
    }

    // Print and annotate top N corners
    for (int i = 0; i < count; ++i) {
        const auto& c = corners[i];

        std::println("idx: {} at point: ({},{}) Harris Score: {}", i, c.pt.x, c.pt.y, c.score);

        // Draw red circle
        cv::circle(imgout, c.pt, 5, cv::Scalar(0, 0, 255), 1);

        // Draw green index label
        cv::putText(imgout, std::to_string(i), c.pt + cv::Point(5, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

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
