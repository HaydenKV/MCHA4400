#include <filesystem>
#include <string>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include <array>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "BufferedVideo.h"
#include "visualNavigation.h"

// === Assignment code you already have ===
#include "Camera.h"
#include "Pose.hpp"

// === Lab 8 visualisation scaffold ===
#include "Plot.h"
#include "SystemSLAMPointLandmarks.h"
#include "SystemSLAMPoseLandmarks.h"
#include "MeasurementSLAMPointBundle.h"
#include "MeasurementSLAMUniqueTagBundle.h"
#include "GaussianInfo.hpp"

#include "imagefeatures.h"
#include "rotation.hpp"

// Helper: convert Rodrigues rvec to rotation matrix
static inline Eigen::Matrix3d rodriguesToRot(const cv::Vec3d& rvec)
{
    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);
    Eigen::Matrix3d R;
    cv::cv2eigen(Rcv, R);
    return R;
}

// Helper: ensure a matrix is upper-triangular (safeguard for Sj)
static inline Eigen::Matrix<double,6,6> makeUpper(const Eigen::Matrix<double,6,6>& M)
{
    Eigen::Matrix<double,6,6> U = M;
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < r; ++c)
            U(r,c) = 0.0;
    return U;
}

// helper: build an empty pose-landmark system (12-state body only)
static SystemSLAMPointLandmarks makeInitialPointSystem()
{
    // From your previous working version (4 seeded points)
    Eigen::VectorXd mu(24);
    mu.setZero();
    mu.segment<3>(12) << 0.0, 0.0, 0.0;
    mu.segment<3>(15) << 1.0, 0.0, 0.0;
    mu.segment<3>(18) << 1.0, 1.0, 0.0;
    mu.segment<3>(21) << 0.0, 1.0, 0.0;
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(24,24) * 1e-3;
    auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);
    return SystemSLAMPointLandmarks(p0);
}

// helper: build a pose-landmark system with ONE pose landmark (12 body + 6 landmark = 18)
static SystemSLAMPoseLandmarks makeInitialPoseSystem()
{
    // Body (12) + one pose-landmark (6) = 18-dim state
    const int nBody = 12;
    const int nLm   = 6;

    Eigen::VectorXd mu(nBody + nLm);
    mu.setZero();

    // Seed the landmark ~1m in front of the camera, no rotation
    mu.segment<3>(nBody + 0) << 0.0, 0.0, 1.0;   // r_n^L/N
    mu.segment<3>(nBody + 3).setZero();          // Theta_nL (Euler)

    // Build sqrt-covariance S (NOT covariance) with realistic sigmas
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nBody + nLm, nBody + nLm);

    // Body uncertainty (keep tight so the camera quadric isn’t huge)
    S.block(0, 0, nBody, nBody).setIdentity();
    S.block(0, 0, nBody, nBody) *= 1e-2;         // sqrt-cov ≈ 0.01 on body state

    // Landmark position: ~0.25 m σ  → sqrt-cov = 0.25
    S.block(nBody + 0, nBody + 0, 3, 3).setIdentity();
    S.block(nBody + 0, nBody + 0, 3, 3) *= 0.25;

    // Landmark orientation: ~15° σ → sqrt-cov = 15° in radians
    const double sigmaEuler = 15.0 * M_PI / 180.0; // ≈ 0.262 rad
    S.block(nBody + 3, nBody + 3, 3, 3).setIdentity();
    S.block(nBody + 3, nBody + 3, 3, 3) *= sigmaEuler;

    auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);
    return SystemSLAMPoseLandmarks(p0);
}

void runVisualNavigationFromVideo(const std::filesystem::path & videoPath,
                                  const std::filesystem::path & cameraPath,
                                  int scenario, int interactive,
                                  const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());

    // ------------------ Output setup ------------------
    std::filesystem::path outputPath;
    const bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // ------------------ Load camera -------------------
    Camera camera;
    {
        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            std::cerr << "File: " << cameraPath << " does not exist or cannot be opened\n";
            std::exit(EXIT_FAILURE);
        }
        fs["camera"] >> camera;
        fs.release();
    }
    camera.printCalibration();

    // ------------------ Open video --------------------
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    // Ensure camera.imageSize is set (Plot uses it)
    {
        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (camera.imageSize.width <= 0 || camera.imageSize.height <= 0)
            camera.imageSize = cv::Size(w, h);
    }

    // ------------------ Plot & export -----------------
    Plot plot(camera);
    const cv::Size plotSize = plot.renderSize();

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        const int codec = cv::VideoWriter::fourcc('m','p','4','v');
        videoOut.open(outputPath.string(), codec, fps, plotSize);
        bufferedVideoWriter.start(videoOut);
    }

    // ------------------ Build system ------------------
    // Scenario 1 uses a pose-landmark system starting with ONLY the 12 body states (no landmarks yet).
    std::unique_ptr<SystemSLAM> systemPtr;
    if (scenario == 1)
    {
        Eigen::VectorXd mu_body(12); 
        mu_body.setZero(); // Start at origin, zero velocity

        Eigen::MatrixXd S_body = Eigen::MatrixXd::Identity(12,12); // tight prior on body

        S_body.block<6,6>(0,0) *= 1e-2; // velocity 
        const double d2r = M_PI / 180.0;
        S_body.block<3,3>(6,6) *= 1e-2;           // Position: 10cm
        S_body.block<3,3>(9,9) *= (0.5 * d2r);   // Orientation: 5°

        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu_body, S_body);
        systemPtr = std::make_unique<SystemSLAMPoseLandmarks>(SystemSLAMPoseLandmarks(p0));
    }
    else
    {
        systemPtr = std::make_unique<SystemSLAMPointLandmarks>(makeInitialPointSystem());
    }
    SystemSLAM& system = *systemPtr;

    // Prime plot with an empty measurement
    {
        Eigen::Matrix<double,2,Eigen::Dynamic> Y0(2,0);
        MeasurementPointBundle m0(0.0, Y0, camera);
        plot.setData(system, m0);
    }

    // Persistent mapping: landmark index -> tag ID
    static std::vector<int> id_by_landmark;
    id_by_landmark.clear();

    // Persistent miss counter (must survive across frames!)
    static std::vector<int> consecutive_misses;
    consecutive_misses.clear();

    const float tagSizeMeters = 0.166f; // set to your printed tag edge length

    // Object points for a square tag in its own frame, TL,TR,BR,BL
    std::vector<cv::Point3f> objPts{
        {-tagSizeMeters/2.f,  tagSizeMeters/2.f, 0.f}, // TL
        { tagSizeMeters/2.f,  tagSizeMeters/2.f, 0.f}, // TR
        { tagSizeMeters/2.f, -tagSizeMeters/2.f, 0.f}, // BR
        {-tagSizeMeters/2.f, -tagSizeMeters/2.f, 0.f}  // BL
    };

    // Helper: reorder detected corners to TL,TR,BR,BL (like OpenCV samples)
    auto reorderTLTRBRBL = [](const std::array<cv::Point2f,4>& in)->std::vector<cv::Point2f>
    {
        std::vector<cv::Point2f> pts(in.begin(), in.end());
        // sort by y to split top vs bottom
        std::sort(pts.begin(), pts.end(),
                  [](const cv::Point2f& a, const cv::Point2f& b){ return a.y < b.y; });
        std::vector<cv::Point2f> top(pts.begin(), pts.begin()+2);
        std::vector<cv::Point2f> bot(pts.begin()+2, pts.end());
        std::sort(top.begin(), top.end(),
                  [](const cv::Point2f& a, const cv::Point2f& b){ return a.x < b.x; }); // TL,TR
        std::sort(bot.begin(), bot.end(),
                  [](const cv::Point2f& a, const cv::Point2f& b){ return a.x < b.x; }); // BL,BR
        // Return TL,TR,BR,BL
        return { top[0], top[1], bot[1], bot[0] };
    };

    int frameIdx = 0;
    while (true)
    {
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty()) break;

        const double t = (fps > 0.0) ? (frameIdx / fps) : frameIdx;

        // ============================================================================
        // SCENARIO 1: ArUco Tag SLAM with 6-DOF Pose Landmarks
        // ============================================================================
        if (scenario == 1)
        {
            // 1) DETECT ARUCO TAGS
            // Returns: ids[], corners[] (4 corners per tag in TL,TR,BR,BL order)
            ArucoDetections dets = detectArucoLab2(imgin, cv::aruco::DICT_6X6_250, /*refine*/true);

            // 2) PER-MARKER PNP SOLVE
            // Solve for tag pose in camera frame using IPPE (best for planar targets)
            std::vector<cv::Vec3d> rvecs, tvecs;
            rvecs.reserve(dets.corners.size());
            tvecs.reserve(dets.corners.size());
            for (const auto& arr : dets.corners)
            {
                // Reorder corners to match objPts (TL,TR,BR,BL)
                const std::vector<cv::Point2f> imgPts = reorderTLTRBRBL(arr);
                cv::Vec3d rvec, tvec;
                bool ok = cv::solvePnP(objPts, imgPts,
                                    camera.cameraMatrix, camera.distCoeffs,
                                    rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
                if (!ok) { rvec = cv::Vec3d(0,0,0); tvec = cv::Vec3d(0,0,0); }
                rvecs.push_back(rvec);
                tvecs.push_back(tvec);
            }

            // 3) INITIALIZE NEW LANDMARKS
            // For any tag ID we haven't seen before, add it to the map
            auto* sysPose = dynamic_cast<SystemSLAMPoseLandmarks*>(&system);
            assert(sysPose && "Scenario 1 expects SystemSLAMPoseLandmarks");

            // Ensure mapping vector is sized to match current landmarks
            if (id_by_landmark.size() < system.numberLandmarks())
                id_by_landmark.resize(system.numberLandmarks(), -1);

            // Get current camera pose from state estimate (mean of Gaussian)
            const Eigen::VectorXd xmean = sysPose->density.mean();
            const Eigen::Vector3d rCNn  = SystemSLAM::cameraPosition(camera, xmean);
            const Eigen::Matrix3d Rnc   = SystemSLAM::cameraOrientation(camera, xmean);

            // Check each detected tag
            for (std::size_t i = 0; i < dets.ids.size(); ++i)
            {
                const int tagId = dets.ids[i];
                
                // Is this tag already in our map?
                const bool known = std::find(id_by_landmark.begin(), id_by_landmark.end(), tagId) 
                                != id_by_landmark.end();
                if (known) continue;  // Skip if already initialized

                // NEW TAG DETECTED - Initialize landmark
                
                // A) Get tag pose in CAMERA frame from PnP
                cv::Mat Rcv;
                cv::Rodrigues(rvecs[i], Rcv); // Convert Rodrigues vector → rotation matrix
                Eigen::Matrix3d Rcj;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        Rcj(r,c) = Rcv.at<double>(r,c);
                const Eigen::Vector3d rCj(tvecs[i][0], tvecs[i][1], tvecs[i][2]);

                // B) Transform tag pose from CAMERA frame to WORLD/NAV frame
                // Rotation: R^n_j = R^n_c * R^c_j
                // Position: r^n_j/N = r^n_C/N + R^n_c * r^c_j/C
                const Eigen::Matrix3d Rnj = Rnc * Rcj;
                const Eigen::Vector3d rnj = rCNn + Rnc * rCj;

                // C) Convert rotation matrix to RPY Euler angles
                // This is required because our state uses Θ^n_j (roll, pitch, yaw)
                const Eigen::Vector3d Thetanj = rot2rpy(Rnj);

                // D) Set initial uncertainty (sqrt-covariance, upper triangular)
                // Position: σ = 0.25 m (reasonable for PnP initialization)
                // Orientation: σ = 15° (converted to radians)
                const double d2r = M_PI / 180.0;
                Eigen::Matrix<double,6,6> Sj = Eigen::Matrix<double,6,6>::Zero();
                Sj.diagonal() << 0.5, 0.5, 0.5,           // position uncertainty
                                (25.0*d2r), (25.0*d2r), (25.0*d2r);  // orientation uncertainty

                // E) Augment state with new landmark [r^n_j/N; Θ^n_j]
                const std::size_t j = sysPose->appendLandmark(rnj, Thetanj, Sj);

                // F) Record tag ID for this landmark index
                if (j >= id_by_landmark.size()) 
                    id_by_landmark.resize(j+1, -1);
                id_by_landmark[j] = tagId;
            }

            // 4) BUILD MEASUREMENT - ALL 4 CORNERS PER TAG
            // 
            // CRITICAL: The assignment spec requires using ALL corner measurements:
            //   log p(y|x) = Σ Σ log p(y_ic | η, m_j) - 4|U| log |Y|
            //              (i,j)∈A c=1..4
            //
            // This means we need Y as a (2 × 4N) matrix, NOT centroids!
            //
            const std::size_t N = dets.ids.size();  // Number of detected tags
            Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, 4*N);  // 4 corners × N tags
            
            for (std::size_t i = 0; i < N; ++i)
            {
                const auto& corners = dets.corners[i];  // 4 corners: TL, TR, BR, BL
                
                // Stack all 4 corner measurements for this tag
                // Column layout: [tag0_TL, tag0_TR, tag0_BR, tag0_BL, tag1_TL, tag1_TR, ...]
                for (int c = 0; c < 4; ++c)
                {
                    Y(0, 4*i + c) = corners[c].x;  // u coordinate (horizontal)
                    Y(1, 4*i + c) = corners[c].y;  // v coordinate (vertical)
                }
            }

            // 5) COMPOSE VISUALIZATION OVERLAY
            // Draw detected tags with axes showing their estimated pose
            cv::Mat overlay = dets.annotated.empty() ? imgin.clone() : dets.annotated.clone();
            for (std::size_t i = 0; i < dets.ids.size(); ++i)
            {
                // Draw RGB axes (X=red, Y=green, Z=blue) at tag location
                cv::drawFrameAxes(overlay,
                                camera.cameraMatrix, camera.distCoeffs,
                                rvecs[i], tvecs[i],
                                tagSizeMeters * 0.4f,  // axis length
                                2);  // line thickness


                // Optional: Draw tag ID as text
                // const auto& corners = dets.corners[i];
                // cv::Point2f center = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25f;
                // cv::putText(overlay, 
                //             std::to_string(dets.ids[i]),
                //             center,
                //             cv::FONT_HERSHEY_SIMPLEX,
                //             0.6,              // font scale
                //             cv::Scalar(255, 255, 0),  // cyan color
                //             2);               // thickness
            }
            system.view() = overlay;

            // 6) TIME UPDATE
            // Propagate state estimate forward to current frame time
            system.predict(t);

            // 7) MEASUREMENT UPDATE
            // Create measurement object with:
            //   - Y: all corner measurements (2×4N)
            //   - camera: calibration parameters
            //   - dets.ids: tag IDs for association
            MeasurementSLAMUniqueTagBundle meas(t, Y, camera, dets.ids);
            meas.setIdByLandmark(id_by_landmark);  // Pass ID mapping for association
            meas.setConsecutiveMisses(consecutive_misses);  // Set miss counter
            meas.process(system);  // Performs: predict → associate → optimize
            //Get updated miss counter back (it was modified during associate())
            consecutive_misses = meas.getConsecutiveMisses();

            // DEBUG COMMENT OUT
            // After measurement update, print diagnostics
            if (frameIdx % 30 == 0) {  // Every 30 frames
                std::cout << "\n=== Frame " << frameIdx << " ===\n";
                std::cout << "  Landmarks: " << system.numberLandmarks() << "\n";
                std::cout << "  Detected tags: " << dets.ids.size() << "\n";
                
                // Count associations
                int nAssoc = 0;
                for (int idx : meas.idxFeatures()) {
                    if (idx >= 0) nAssoc++;
                }
                std::cout << "  Associated: " << nAssoc << "\n";
                
                // Camera position
                const Eigen::VectorXd x = system.density.mean();
                const Eigen::Vector3d camPos = SystemSLAM::cameraPosition(camera, x);
                std::cout << "  Camera pos: [" << camPos.transpose() << "]\n";
            }


            // 8) VISUALIZE
            plot.setData(system, meas);
            plot.render();
            if (doExport) bufferedVideoWriter.write(plot.getFrame());
        }
        else
        {
            // Scenario 2/3: keep your existing point-bundle no-op view
            system.view() = imgin;
            Eigen::Matrix<double,2,Eigen::Dynamic> Ynow(2,0);
            MeasurementPointBundle meas(t, Ynow, camera);
            plot.setData(system, meas);
            plot.render();
            if (doExport) bufferedVideoWriter.write(plot.getFrame());
        }

        if (interactive == 2 || (interactive == 1 && (--nFrames == 0)))
            plot.start();

        ++frameIdx;
    }

    if (doExport) bufferedVideoWriter.stop();
    bufferedVideoReader.stop();
}