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
#include <iomanip>
#include <ctime>

#include "BufferedVideo.h"
#include "visualNavigation.h"
#include "Camera.h"
#include "Pose.hpp"
#include "Plot.h"
#include "SystemSLAMPointLandmarks.h"
#include "SystemSLAMPoseLandmarks.h"
#include "MeasurementSLAMPointBundle.h"
#include "MeasurementSLAMUniqueTagBundle.h"
#include "GaussianInfo.hpp"
#include "imagefeatures.h"
#include "rotation.hpp"

// ============================================================================
// SCENARIO 1 CONSTANTS (Shared across the file)
// ============================================================================
namespace {
    constexpr float  TAG_SIZE_METERS = 0.166f;       // ArUco tag edge length (166mm)
    constexpr double REPROJ_ERR_THRESH_PX = 1.0;     // IPPE reprojection gate (pixels)
    
    // CRITICAL FIX: More realistic initial uncertainties based on typical PnP accuracy
    // At 2-3m distance, solvePnP with IPPE typically achieves:
    // - Position accuracy: ~5-10cm
    // - Orientation accuracy: ~3-5 degrees
    constexpr double INIT_POS_SIGMA = 0.05;          // 10cm position uncertainty (was 40cm!)
    constexpr double INIT_ANG_SIGMA = 3.0 * M_PI / 180.0;  // 5° orientation uncertainty (was 20°!)
    
    // Small offset to avoid perfect initialization (per MCHA4400 lecture slide 21)
    // "Don't initialise landmarks exactly at known solution" to ensure enough
    // quasi-Newton iterations for good Hessian approximation
    constexpr double INIT_POS_OFFSET = 0.02;  // 2cm random offset from PnP solution
}

// Helper: convert Rodrigues rvec to rotation matrix
static inline Eigen::Matrix3d rodriguesToRot(const cv::Vec3d& rvec)
{
    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);
    Eigen::Matrix3d R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r,c) = Rcv.at<double>(r,c);
    return R;
}

void runVisualNavigationFromVideo(
    const std::filesystem::path& videoPath,
    const std::filesystem::path& cameraPath,
    int scenario,
    int interactive,
    const std::filesystem::path& outputDirectory,
    int max_frames)
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

    std::cout << "Total frames in video: " << nFrames << std::endl;
    std::cout << "Video duration (approx): " << (nFrames / fps) << " seconds" << std::endl;

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
    std::unique_ptr<SystemSLAM> systemPtr;
    if (scenario == 1)
    {
        Eigen::VectorXd mu_body(12);
        mu_body.setZero(); // Start at origin, zero velocity

        Eigen::MatrixXd S_body = Eigen::MatrixXd::Identity(12,12); // tight prior on body
        S_body.block<6,6>(0,0) *= 1e-4;               // velocity
        const double d2r = M_PI / 180.0;
        S_body.block<3,3>(6,6) *= 1e-2;               // Position: 1cm
        S_body.block<3,3>(9,9) *= (1.0 * d2r);        // Orientation: 5°

        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu_body, S_body);
        systemPtr = std::make_unique<SystemSLAMPoseLandmarks>(SystemSLAMPoseLandmarks(p0));
    }
    else
    {
        Eigen::VectorXd mu(24);
        mu.setZero();
        mu.segment<3>(12) << 0.0, 0.0, 0.0;
        mu.segment<3>(15) << 1.0, 0.0, 0.0;
        mu.segment<3>(18) << 1.0, 1.0, 0.0;
        mu.segment<3>(21) << 0.0, 1.0, 0.0;
        Eigen::MatrixXd S = Eigen::MatrixXd::Identity(24,24) * 1e-3;
        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);
        systemPtr = std::make_unique<SystemSLAMPointLandmarks>(SystemSLAMPointLandmarks(p0));
    }
    SystemSLAM& system = *systemPtr;

    // Prime plot with an empty measurement
    {
        Eigen::Matrix<double,2,Eigen::Dynamic> Y0(2,0);
        MeasurementPointBundle m0(0.0, Y0, camera);
        plot.setData(system, m0);
    }

    // Persistent mapping structures
    static std::vector<int> id_by_landmark;                 // landmark idx -> tag id
    static std::unordered_map<int, std::size_t> id2lm;      // tag id -> landmark idx
    id_by_landmark.clear();
    id2lm.clear();
    
    // Main loop
    int frameIdx = 0;
    while (true)
    {
        // ArUco detection
        // Landmark initialization
        // Build measurement
        // Time update (via Event::process)
        // Measurement update (via Event::process)
        // Visualization

        // std::cout << "Frame " << frameIdx << " - Start" << std::endl;

        if (max_frames > 0 && frameIdx >= max_frames) break;
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty()) break;

        const double t = (fps > 0.0) ? (frameIdx / fps) : frameIdx;

        if (scenario == 1)
        {
            // ========================================================================
            // STEP 1: DETECT ARUCO TAGS
            // ========================================================================
            // Detect + pose (IPPE) + reprojection gating
            std::vector<cv::Vec3d> rvecs, tvecs;
            std::vector<double> meanErrs;
            ArucoDetections dets = detectArUcoPOSE(
                imgin,
                cv::aruco::DICT_6X6_250,
                /*doCornerRefine*/ true,
                camera.cameraMatrix,
                camera.distCoeffs,
                TAG_SIZE_METERS,
                &rvecs, &tvecs, &meanErrs,
                REPROJ_ERR_THRESH_PX,
                /*drawRejected*/ false
            );

            auto* sysPose = dynamic_cast<SystemSLAMPoseLandmarks*>(&system);
            assert(sysPose && "Scenario 1 expects SystemSLAMPoseLandmarks");

            // Ensure id_by_landmark vector is sized correctly
            if (id_by_landmark.size() < system.numberLandmarks())
                id_by_landmark.resize(system.numberLandmarks(), -1);

            // Current camera pose (mean estimate from filter)
            const Eigen::VectorXd xmean = sysPose->density.mean();
            const Eigen::Vector3d rCNn  = SystemSLAM::cameraPosition(camera, xmean);
            const Eigen::Matrix3d Rnc   = SystemSLAM::cameraOrientation(camera, xmean);

            // ========================================================================
            // STEP 2: INITIALIZE NEW LANDMARKS (only for newly detected tag IDs)
            // ========================================================================
            // PRINCIPLE: Only initialize when we detect a NEW tag ID.
            // The landmark will be IMMEDIATELY associated because IDs are unique.
            //
            // GATE ALIGNMENT: Initialization uses CamDefaults::BorderMarginPx to ensure
            // tags are well-centered in FOV. Association uses NO margin. This is
            // intentional - the margin prevents initialization of tags near image
            // edges that would fail association on subsequent frames.
            
            int nInitialized = 0;
            for (std::size_t i = 0; i < dets.ids.size(); ++i)
            {
                const int tagId = dets.ids[i];
                
                // ================================================================
                // GATE 1: Is this a new tag ID?
                // ================================================================
                if (id2lm.find(tagId) != id2lm.end()) {
                    continue; // Already have a landmark for this tag
                }

                // ================================================================
                // GATE 2: Are all 4 corners inside image bounds with margin?
                // ================================================================
                // Uses CamDefaults::BorderMarginPx margin to ensure:
                // - Tag is well-centered in FOV (not near edges)
                // - Future associations will succeed (association uses NO margin)
                // - Prevents "dead zone" where landmark is initialized but can't associate
                if (!camera.areCornersInside(dets.corners[i], CamDefaults::BorderMarginPx)) {
                    continue;
                }

                // ================================================================
                // GATE 3: Is tag in front of camera?
                // ================================================================
                // Sanity check for PnP solution (Z must be positive)
                if (tvecs[i][2] <= 1e-3) {  // Must be at least 1mm in front
                    continue;
                }

                // ================================================================
                // ALL GATES PASSED → INITIALIZE NEW LANDMARK
                // ================================================================
                
                // Convert IPPE pose (camera→tag in camera frame) to world frame
                const Eigen::Matrix3d Rcj = rodriguesToRot(rvecs[i]);
                const Eigen::Vector3d rCjVec(tvecs[i][0], tvecs[i][1], tvecs[i][2]);
                
                const Eigen::Matrix3d Rnj = Rnc * Rcj;
                Eigen::Vector3d rnj = rCNn + Rnc * rCjVec;
                const Eigen::Vector3d Thetanj = rot2rpy(Rnj);

                // CRITICAL: Add small random offset to avoid perfect initialization
                // Per MCHA4400 lecture slide 21 - ensures enough optimization iterations
                // for quasi-Newton methods to build accurate Hessian approximation
                std::srand(static_cast<unsigned>(std::time(nullptr)) + i);
                rnj(0) += INIT_POS_OFFSET * (2.0 * (std::rand() / (double)RAND_MAX) - 1.0);
                rnj(1) += INIT_POS_OFFSET * (2.0 * (std::rand() / (double)RAND_MAX) - 1.0);
                rnj(2) += INIT_POS_OFFSET * (2.0 * (std::rand() / (double)RAND_MAX) - 1.0);

                // Initial uncertainty (diagonal covariance)
                Eigen::Matrix<double,6,6> Sj = Eigen::Matrix<double,6,6>::Zero();
                Sj(0,0) = INIT_POS_SIGMA;  // x position uncertainty
                Sj(1,1) = INIT_POS_SIGMA;  // y position uncertainty
                Sj(2,2) = INIT_POS_SIGMA;  // z position uncertainty
                Sj(3,3) = INIT_ANG_SIGMA;  // roll uncertainty
                Sj(4,4) = INIT_ANG_SIGMA;  // pitch uncertainty
                Sj(5,5) = INIT_ANG_SIGMA;  // yaw uncertainty

                // Append landmark to SLAM state
                const std::size_t j = sysPose->appendLandmark(rnj, Thetanj, Sj);
                
                // Register this tag ID with the new landmark index
                if (j >= id_by_landmark.size()) id_by_landmark.resize(j+1, -1);
                id_by_landmark[j] = tagId;
                id2lm[tagId] = j;
                
                nInitialized++;
            }

            // ========================================================================
            // STEP 3: BUILD MEASUREMENT MATRIX
            // ========================================================================
            // Pack all detected tag corners into measurement matrix Y (2 × 4N)
            // Corners are in OpenCV order: TL, TR, BR, BL
            const std::size_t N = dets.ids.size();
            Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, 4*N);
            for (std::size_t i = 0; i < N; ++i) {
                const auto& c = dets.corners[i];
                for (int k = 0; k < 4; ++k) {
                    Y(0, 4*i + k) = c[k].x;
                    Y(1, 4*i + k) = c[k].y;
                }
            }

            // GUARD: Verify Y packing is correct (4 columns per detection)
            assert(Y.cols() == 4 * static_cast<int>(N) && 
                   "Y packing error: should have 4 columns per tag detection");

            // ========================================================================
            // STEP 4: TIME UPDATE + MEASUREMENT UPDATE
            // ========================================================================
            
            // Display annotated image
            system.view() = dets.annotated.empty() ? imgin : dets.annotated;

            // Create measurement event
            MeasurementSLAMUniqueTagBundle meas(t, Y, camera, dets.ids);
            meas.setIdByLandmark(id_by_landmark);

            // Process event: time update (propagate) + measurement update (correct)
            // This will automatically associate detected tags with landmarks via ID matching
            meas.process(system);

            // Update persistent ID mapping (in case measurement modified it)
            id_by_landmark = meas.idByLandmark();

            // ========================================================================
            // STEP 5: DIAGNOSTICS (every 10 frames)
            // ========================================================================
            if (frameIdx % 10 == 0) {
                std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
                std::cout << "║ Frame " << std::setw(5) << frameIdx 
                          << " | Time: " << std::fixed << std::setprecision(2) << t << "s"
                          << std::string(30, ' ') << "║\n";
                std::cout << "╠════════════════════════════════════════════════════════════╣\n";
                
                // Detection statistics
                std::cout << "║ DETECTIONS                                                 ║\n";
                std::cout << "║   Total landmarks in map:   " << std::setw(4) << system.numberLandmarks() 
                          << "                                  ║\n";
                std::cout << "║   Tags detected this frame: " << std::setw(4) << dets.ids.size() 
                          << "                                  ║\n";
                std::cout << "║   New landmarks initialized:" << std::setw(4) << nInitialized 
                          << "                                  ║\n";
                
                // Association statistics
                int nAssoc = 0;
                for (int idx : meas.idxFeatures()) if (idx >= 0) ++nAssoc;
                std::cout << "║   Landmarks associated:     " << std::setw(4) << nAssoc 
                          << "                                  ║\n";
                
                std::cout << "╠════════════════════════════════════════════════════════════╣\n";
                
                // Camera state
                const Eigen::VectorXd x = system.density.mean();
                const Eigen::Vector3d camPos = SystemSLAM::cameraPosition(camera, x);
                const Eigen::Vector3d camVel = x.segment<3>(0);  // Body-fixed velocity
                const Eigen::Vector3d camOmega = x.segment<3>(3); // Body-fixed angular velocity
                
                std::cout << "║ CAMERA STATE                                               ║\n";
                std::cout << "║   Position (m):    [" 
                          << std::setw(7) << std::fixed << std::setprecision(3) << camPos(0) << ", "
                          << std::setw(7) << std::fixed << std::setprecision(3) << camPos(1) << ", "
                          << std::setw(7) << std::fixed << std::setprecision(3) << camPos(2) << "]   ║\n";
                std::cout << "║   Velocity (m/s):  [" 
                          << std::setw(7) << std::fixed << std::setprecision(3) << camVel(0) << ", "
                          << std::setw(7) << std::fixed << std::setprecision(3) << camVel(1) << ", "
                          << std::setw(7) << std::fixed << std::setprecision(3) << camVel(2) << "]   ║\n";
                std::cout << "║   Ang vel (rad/s): [" 
                          << std::setw(7) << std::fixed << std::setprecision(3) << camOmega(0) << ", "
                          << std::setw(7) << std::fixed << std::setprecision(3) << camOmega(1) << ", "
                          << std::setw(7) << std::fixed << std::setprecision(3) << camOmega(2) << "]   ║\n";
                
                // Landmark uncertainties (show first 5 for brevity)
                const std::size_t nShow = system.numberLandmarks();
                if (nShow > 0) {
                    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
                    std::cout << "║ LANDMARK UNCERTAINTIES (Show " << nShow << " landmarks)                  ║\n";
                    for (std::size_t i = 0; i < nShow; ++i) {
                        const auto posDen = system.landmarkPositionDensity(i);
                        const Eigen::Vector3d pos_mean = posDen.mean();
                        
                        // FIXED: Extract standard deviations using marginals (same as Plot.cpp)
                        GaussianInfo<double> px = posDen.marginal(Eigen::seqN(0, 1));  // x marginal
                        GaussianInfo<double> py = posDen.marginal(Eigen::seqN(1, 1));  // y marginal
                        GaussianInfo<double> pz = posDen.marginal(Eigen::seqN(2, 1));  // z marginal
                        
                        double std_x = std::abs(px.sqrtCov()(0, 0));  // σ_x
                        double std_y = std::abs(py.sqrtCov()(0, 0));  // σ_y
                        double std_z = std::abs(pz.sqrtCov()(0, 0));  // σ_z
                        
                        std::cout << "║   LM[" << i << "] pos: ["
                                << std::setw(6) << std::fixed << std::setprecision(2) << pos_mean(0) << ","
                                << std::setw(6) << std::fixed << std::setprecision(2) << pos_mean(1) << ","
                                << std::setw(6) << std::fixed << std::setprecision(2) << pos_mean(2) << "]m  ";
                        std::cout << "σ:[" 
                                << std::setw(5) << std::fixed << std::setprecision(3) << std_x << ","
                                << std::setw(5) << std::fixed << std::setprecision(3) << std_y << ","
                                << std::setw(5) << std::fixed << std::setprecision(3) << std_z << "]m ║\n";
                    }
                }
                
                std::cout << "╚════════════════════════════════════════════════════════════╝\n";
            }

            // Visualization
            plot.setData(system, meas);
        }
        // ... [Other scenarios] ...
        else
        {
            // Other scenarios unchanged
            system.view() = imgin;
            Eigen::Matrix<double,2,Eigen::Dynamic> Ynow(2,0);
            MeasurementPointBundle meas(t, Ynow, camera);
            plot.setData(system, meas);
        }

        plot.render();
        if (doExport) bufferedVideoWriter.write(plot.getFrame());

        if (interactive == 2 || (interactive == 1 && (--nFrames == 0)))
            plot.start();

        frameIdx++;
    }
    if (doExport) bufferedVideoWriter.stop();
    bufferedVideoReader.stop();
}