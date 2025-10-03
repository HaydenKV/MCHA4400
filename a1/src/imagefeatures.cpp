#include <string>  
#include <print>
#include <vector>
#include <array>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Core>
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

    // --- Normalization removed ---
    // cv::Mat harrisNorm;
    // cv::normalize(harrisResponse, harrisNorm, 0, 255, cv::NORM_MINMAX);
    // harrisNorm.convertTo(harrisNorm, CV_32F);

    // Use raw response instead
    cv::Mat harrisNorm = harrisResponse.clone();

    // Define structure to hold corner data
    struct Corner {
        cv::Point pt;
        float score;
    };
    std::vector<Corner> corners;

    // Threshold and collect high-response points
    float threshold = 0.001f;
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
        cv::circle(imgout, c.pt, 4, cv::Scalar(0, 165, 255), -1); // Orange filled circles
    }

    // Print and annotate top N corners
    for (int i = 0; i < count; ++i) {
        const auto& c = corners[i];

        std::println("idx: {} at point: ({},{}) Harris Score: {}", i, c.pt.x, c.pt.y, c.score);

        // Draw red circle
        cv::circle(imgout, c.pt, 8, cv::Scalar(0, 0, 255), 1);

        // Draw green index label
        cv::putText(imgout, std::to_string(i), c.pt + cv::Point(5, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    }

    return imgout;
}

cv::Mat detectAndDrawShiAndTomasi(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Compute corner strength using minimum eigenvalue method
    cv::Mat minEigenResponse;
    int blockSize = 2;
    cv::cornerMinEigenVal(gray, minEigenResponse, blockSize);

    // --- Normalization removed ---
    // cv::Mat eigenNorm;
    // cv::normalize(minEigenResponse, eigenNorm, 0, 255, cv::NORM_MINMAX);
    // eigenNorm.convertTo(eigenNorm, CV_32F);

    // Use raw eigenvalue response instead
    cv::Mat eigenNorm = minEigenResponse.clone();

    // Define structure to hold corner data
    struct Corner {
        cv::Point pt;
        float score;
    };
    std::vector<Corner> corners;

    // Threshold response
    float threshold = 0.01f;
    for (int y = 0; y < eigenNorm.rows; ++y) {
        for (int x = 0; x < eigenNorm.cols; ++x) {
            float response = eigenNorm.at<float>(y, x);
            if (response > threshold) {
                corners.push_back({cv::Point(x, y), response});
            }
        }
    }

    // Print info
    std::println("Using Shi-Tomasi feature detector");
    std::println("Image width: {}", img.cols);
    std::println("Image height: {}", img.rows);
    std::println("Features requested: {}", maxNumFeatures);
    std::println("Features detected: {}", corners.size());

    // Sort by score
    std::sort(corners.begin(), corners.end(), [](const Corner& a, const Corner& b) {
        return a.score > b.score;
    });

    int count = std::min(maxNumFeatures, static_cast<int>(corners.size()));

    // Draw all corners (orange)
    for (const auto& c : corners) {
        cv::circle(imgout, c.pt, 3, cv::Scalar(0, 165, 255), -1); // orange filled
    }

    // Draw top N (red + labeled)
    for (int i = 0; i < count; ++i) {
        const auto& c = corners[i];
        std::println("idx: {} at point: ({},{}) Shi Score: {}", i, c.pt.x, c.pt.y, c.score);

        // Red circle for top N
        cv::circle(imgout, c.pt, 6, cv::Scalar(0, 0, 255), 2);

        // Green label
        cv::putText(imgout, std::to_string(i), c.pt + cv::Point(5, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }

    return imgout;
}

cv::Mat detectAndDrawFAST(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Detect keypoints using FAST
    std::vector<cv::KeyPoint> keypoints;
    int threshold = 85;
    bool nonmaxSuppression = true; //10001
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression);
    detector->detect(gray, keypoints);

    std::println("Using FAST feature detector");
    std::println("Image width: {}", img.cols);
    std::println("Image height: {}", img.rows);
    std::println("Features requested: {}", maxNumFeatures);
    std::println("Features detected: {}", keypoints.size());

    // Sort keypoints by response (descending)
    std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        return a.response > b.response;
    });

    int count = std::min(maxNumFeatures, static_cast<int>(keypoints.size()));

    // Draw all keypoints as small orange dots
    for (const auto& kp : keypoints) {
        cv::circle(imgout, kp.pt, 3, cv::Scalar(0, 165, 255), -1);  // Orange filled
    }

    // Draw top N keypoints with larger red circles and index
    for (int i = 0; i < count; ++i) {
        const auto& kp = keypoints[i];
        std::println("idx: {} at point: ({},{}) Score: {}", i, kp.pt.x, kp.pt.y, kp.response);

        // Red circle
        cv::circle(imgout, kp.pt, 6, cv::Scalar(0, 0, 255), 2);

        // Green index label
        cv::putText(imgout, std::to_string(i), kp.pt + cv::Point2f(5, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }

    return imgout;
}

cv::Mat detectAndDrawArUco(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // Get dictionary and wrap in a Ptr (required for detectMarkers)
    cv::Ptr<cv::aruco::Dictionary> dictionary = 
        cv::makePtr<cv::aruco::Dictionary>(
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250)
        );

    // Containers for detected markers
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;

    // Detect ArUco markers
    cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds);

    // Draw the detected markers
    if (!markerIds.empty())
    {
        cv::aruco::drawDetectedMarkers(imgout, markerCorners, markerIds);

        std::println("Using aruco feature detector");
        std::println("Image width: {}", img.cols);
        std::println("Image height: {}", img.rows);
        std::println("Detected {} marker(s)", markerIds.size());

        // Pair marker IDs with their corners and sort
        std::vector<std::pair<int, std::vector<cv::Point2f>>> sortedMarkers;
        for (size_t i = 0; i < markerIds.size(); ++i)
        {
            sortedMarkers.emplace_back(markerIds[i], markerCorners[i]);
        }

        std::sort(sortedMarkers.begin(), sortedMarkers.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        // Print sorted output
        for (const auto& marker : sortedMarkers)
        {
            std::print("ID: {} with corners:", marker.first);
            for (int j = 0; j < 4; ++j)
            {
                cv::Point2f pt = marker.second[j];
                std::print(" ({},{})", static_cast<int>(pt.x), static_cast<int>(pt.y));
            }
            std::print("\n");
        }
    }
    else
    {
        std::println("ArUco: No markers detected.");
    }

    return imgout;
}


ArucoDetections detectArUcoPOSE(
    const cv::Mat& imgBGR,
    int dictionary,
    bool doCornerRefine,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    float tagSizeMeters,
    std::vector<cv::Vec3d>* outRvecs,
    std::vector<cv::Vec3d>* outTvecs,
    std::vector<double>* outMeanReprojErr,
    double reprojErrThreshPx,
    bool drawRejected
)
{
    CV_Assert(!imgBGR.empty());
    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3);
    CV_Assert(tagSizeMeters > 0.f);

    ArucoDetections out;
    out.annotated = imgBGR.clone();

    // --- Dictionary & conservative parameters (loosen only if recall is low) ---
    cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(dictionary);
    cv::aruco::DetectorParameters params;

    params.cornerRefinementMethod = doCornerRefine
        ? cv::aruco::CORNER_REFINE_SUBPIX
        : cv::aruco::CORNER_REFINE_NONE;

    params.minMarkerPerimeterRate = 0.03f;
    params.maxMarkerPerimeterRate = 4.0f;

    params.adaptiveThreshWinSizeMin = 5;
    params.adaptiveThreshWinSizeMax = 35;
    params.adaptiveThreshWinSizeStep = 5;
    params.adaptiveThreshConstant   = 7;

    params.minCornerDistanceRate = 0.05f;
    params.minDistanceToBorder   = 3;
    params.polygonalApproxAccuracyRate = 0.05f;

    params.cornerRefinementWinSize       = 5;
    params.cornerRefinementMaxIterations = 50;
    params.cornerRefinementMinAccuracy   = 0.05;

    params.errorCorrectionRate = 0.5f;

    cv::aruco::ArucoDetector detector(dict, params);

    // --- Detect ---
    std::vector<int> ids_raw;
    std::vector<std::vector<cv::Point2f>> corners_raw, rejected;
    detector.detectMarkers(imgBGR, corners_raw, ids_raw, rejected);

    if (drawRejected && !rejected.empty()) {
        cv::aruco::drawDetectedMarkers(out.annotated, rejected, std::vector<int>(), cv::Scalar(120,120,120));
    }
    if (ids_raw.empty()) {
        return out; // nothing found
    }

    // Visualize detections (even if some will be rejected by PnP gating)
    cv::aruco::drawDetectedMarkers(out.annotated, corners_raw, ids_raw);

    // --- Prepare canonical object points in TL,TR,BR,BL to match OpenCV order ---
    const float L = tagSizeMeters;
    const std::vector<cv::Point3f> objPts = {
        {-L/2.f,  L/2.f, 0.f},   // TL
        { L/2.f,  L/2.f, 0.f},   // TR
        { L/2.f, -L/2.f, 0.f},   // BR
        {-L/2.f, -L/2.f, 0.f}    // BL
    };

    if (outRvecs) outRvecs->clear();
    if (outTvecs) outTvecs->clear();
    if (outMeanReprojErr) outMeanReprojErr->clear();

    out.ids.reserve(ids_raw.size());
    out.corners.reserve(ids_raw.size());
    if (outRvecs) outRvecs->reserve(ids_raw.size());
    if (outTvecs) outTvecs->reserve(ids_raw.size());
    if (outMeanReprojErr) outMeanReprojErr->reserve(ids_raw.size());

    // --- Per-marker pose + optional reprojection gate ---
    for (size_t i = 0; i < ids_raw.size(); ++i) {
        const std::vector<cv::Point2f>& imgPts = corners_raw[i]; // OpenCV order TL,TR,BR,BL

        cv::Vec3d rvec, tvec;
        bool ok = cv::solvePnP(
            objPts, imgPts,
            cameraMatrix, distCoeffs,
            rvec, tvec,
            false, cv::SOLVEPNP_IPPE_SQUARE
        );
        if (!ok) continue;

        // Mean reprojection error (px)
        double meanErr = 0.0;
        {
            std::vector<cv::Point2f> reproj(4);
            cv::projectPoints(objPts, rvec, tvec, cameraMatrix, distCoeffs, reproj);
            for (int c = 0; c < 4; ++c) {
                cv::Point2f d = reproj[c] - imgPts[c];
                meanErr += std::sqrt(d.dot(d));
            }
            meanErr *= 0.25;
        }

        // Optional gate (keeps the solution robust in real video)
        if (reprojErrThreshPx > 0.0 && meanErr > reprojErrThreshPx)
            continue;

        // Keep this marker
        out.ids.push_back(ids_raw[i]);
        std::array<cv::Point2f,4> c{};
        for (int k = 0; k < 4; ++k) c[k] = imgPts[k];
        out.corners.push_back(c);

        if (outRvecs) outRvecs->push_back(rvec);
        if (outTvecs) outTvecs->push_back(tvec);
        if (outMeanReprojErr) outMeanReprojErr->push_back(meanErr);

        // Nice visual confirmation
        cv::drawFrameAxes(out.annotated, cameraMatrix, distCoeffs, rvec, tvec, 0.4f * L, 2);
    }

    return out;
}