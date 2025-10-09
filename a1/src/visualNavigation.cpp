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
#include <csignal>
#include <atomic>
#include <print>
#include <cstdlib>
#include <numbers> 
#include <algorithm>
#include <cmath>


#include "BufferedVideo.h"
#include "visualNavigation.h"
#include "Camera.h"
#include "Pose.hpp"
#include "Plot.h"
#include "SystemSLAMPointLandmarks.h"
#include "SystemSLAMPoseLandmarks.h"
#include "MeasurementSLAMPointBundle.h"
#include "MeasurementSLAMUniqueTagBundle.h"
#include "MeasurementSLAMDuckBundle.h"
#include "DuckDetectorONNX.h"
#include "GaussianInfo.hpp"
#include "imagefeatures.h"
#include "rotation.hpp"

// ============================================================================
// SCENARIO 1 CONSTANTS (Shared across the file)
// ============================================================================
namespace {
    constexpr float  TAG_SIZE_METERS = 0.166f;       // ArUco tag edge length (166mm)
    constexpr double REPROJ_ERR_THRESH_PX = 3.0;     // IPPE reprojection gate (pixels)
    
    // Typical PnP accuracy at 2–3 m: ≈5–10 cm position, ≈3–5° orientation.
    constexpr double INIT_POS_SIGMA = 0.1;                             // [m]
    constexpr double INIT_ANG_SIGMA = (5.0 * std::numbers::pi / 180.0); // [rad]
    
    // Small jitter avoids perfect init → supports stable Hessian building.
    constexpr double INIT_POS_OFFSET = 0.02;  // [m]
}

// ============================================================================
// Async-signal-safe shutdown flag + handler (handler only flips an atomic)
// ============================================================================
static std::atomic<bool> g_stop{false};
static void onSignal(int) { g_stop.store(true, std::memory_order_relaxed); }

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

/*
runVisualNavigationFromVideo:
Main loop orchestrating model (4)–(5) and measurement updates.

Model (shared across scenarios):
  x = [ν(6), r^n_{B/N}(3), Θ^n_B(3), m...] with   dη/dt = J_K(η)ν,   dm/dt = 0  (4)–(5).

Scenario 1 measurement (unique tags):
  Landmark m_j = [ r^n_{j/N}, Θ^n_j ]^T (6);  3D corners from tag frame via (8)–(9);
  image mapping uses u = π(K,dist, r^c) with association by tag ID; log-likelihood (7).
*/
void runVisualNavigationFromVideo(
    const std::filesystem::path& videoPath,
    const std::filesystem::path& cameraPath,
    int scenario,
    int interactive,
    const std::filesystem::path& outputDirectory,
    int max_frames)
{
    assert(!videoPath.empty());

    // Register Ctrl-C / SIGTERM (sets a flag checked in the loop)
    g_stop.store(false, std::memory_order_relaxed);
    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);

    // ---------- Output setup ----------
    std::filesystem::path outputPath;
    const bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // ---------- Load camera (K, dist, T_bc) ----------
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
    
    // ---------- Open video ----------
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "Total frames in video: " << nFrames << std::endl;
    std::cout << "Video duration (approx): " << (nFrames / fps) << " seconds" << std::endl;

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    // Ensure camera.imageSize is set (used by Plot/image gating)
    {
        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (camera.imageSize.width <= 0 || camera.imageSize.height <= 0)
            camera.imageSize = cv::Size(w, h);
    }

    // ---------- Plot & export ----------
    Plot plot(camera);
    const cv::Size plotSize = plot.renderSize();

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);

    if (doExport) {
        const int codec = cv::VideoWriter::fourcc('m','p','4','v');
        if (!videoOut.open(outputPath.string(), codec, fps, plotSize)) {
            std::cerr << "Failed to open video for writing: " << outputPath << "\n";
        } else {
            bufferedVideoWriter.start(videoOut);
        }
    }

    // ---------- Build system state ----------
    std::unique_ptr<DuckDetectorONNX> duckDetector;
    std::unique_ptr<SystemSLAM> systemPtr;
    if (scenario == 1)
    {
        // Body-only prior (first 12 states): ν(6), r(3), Θ(3)
        Eigen::VectorXd mu_body(12);
        mu_body.setZero();
        mu_body.segment<3>(6) << 0.0, 0.0, -1.6; // r^n_{B/N}
        mu_body.segment<3>(9) << 0.0, 0.0, 0.0; // Θ^n_B

        Eigen::MatrixXd S_body = Eigen::MatrixXd::Identity(12,12);
        S_body.block<3,3>(0,0) *= 1.0;          // v uncertainty
        S_body.block<3,3>(3,3) *= 0.5;         // ω uncertainty
        const double d2r = (1.0 * std::numbers::pi / 180.0);
        S_body.block<3,3>(6,6) *= 0.5;         // r uncertainty (≈1 cm)
        S_body.block<3,3>(9,9) *= (1 * d2r);    // Θ uncertainty (≈1°)

        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu_body, S_body);
        systemPtr = std::make_unique<SystemSLAMPoseLandmarks>(SystemSLAMPoseLandmarks(p0));
    }
    else if (scenario == 2) {
        // ---- Point-landmark SLAM system (we won't update it yet) ----
        Eigen::VectorXd mu_body(12);  mu_body.setZero();        // ν(6), r(3), Θ(3)
        mu_body.segment<3>(6) << 0.0, 0.0, -1.0; // r^n_{B/N}
        mu_body.segment<3>(9) << -M_PI/2.0, -M_PI/2.0, 0.0; // Θ^n_B

        Eigen::MatrixXd S_body = Eigen::MatrixXd::Identity(12,12);
        S_body.block<3,3>(0,0) *= 0.3;          // v uncertainty
        S_body.block<3,3>(6,6) *= 0.3;                         // position sqrt-cov
        const double d2r = (1.0 * std::numbers::pi / 180.0);
        S_body.block<3,3>(9,9) *= (10.0 * d2r);                    // orientation sqrt-cov
        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu_body, S_body);
        systemPtr = std::make_unique<SystemSLAMPointLandmarks>(SystemSLAMPointLandmarks(p0));

        // ---- ONNX Duck detector (same path as Lab 3) ----
        const std::filesystem::path onnx_file = "../src/duck_with_postprocessing.onnx";
        assert(std::filesystem::exists(onnx_file) && "[DuckDetector] ONNX model not found at ../src/duck_with_postprocessing.onnx");
        duckDetector = std::make_unique<DuckDetectorONNX>(onnx_file.string());
        assert(duckDetector && "[DuckDetector] construction failed");
    }
    else
    {
        assert("broken if you're here");
        // Placeholder for other scenarios (unchanged).
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

    // Persistent ID/association (Scenario 1)
    static std::vector<int> id_by_landmark;                 // landmark idx → tag ID
    static std::unordered_map<int, std::size_t> id2lm;      // tag ID → landmark idx
    id_by_landmark.clear();
    id2lm.clear();
    
    // Promotion buffer for Scenario 2: hold unmatched detections for a few frames
    struct PendingDuck {
        cv::Point2f uv;
        double      area;
        int         hits;       // how many times we re-detected it
        int         lastSeen;   // frame index
    };
    std::vector<PendingDuck> pending;


    // ============================== MAIN LOOP ==============================
    int frameIdx = 0;
    while (true)
    {
        // Graceful shutdown (Ctrl-C / SIGTERM)
        if (g_stop.load(std::memory_order_relaxed)) {
            std::cout << "\n[INFO] Caught shutdown signal — finishing and closing video...\n";
            break;
        }

        // Optional frame cap
        if (max_frames > 0 && frameIdx >= max_frames) {
            std::cout << "[INFO] Reached max_frames=" << max_frames << " — stopping.\n";
            break;
        }

        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty()) {
            std::cout << "[INFO] End of video stream — stopping.\n";
            break;
        }

        const double t = (fps > 0.0) ? (frameIdx / fps) : frameIdx;

        if (scenario == 1)
        {
            // --- 1. Detect ArUco Tags ---
            std::vector<cv::Vec3d> rvecs, tvecs;
            ArucoDetections dets = detectArUcoPOSE(
                imgin, cv::aruco::DICT_6X6_250, true,
                camera.cameraMatrix, camera.distCoeffs, TAG_SIZE_METERS,
                &rvecs, &tvecs, nullptr, REPROJ_ERR_THRESH_PX, false);

            auto* sysPose = dynamic_cast<SystemSLAMPoseLandmarks*>(&system);
            assert(sysPose && "Scenario 1 expects SystemSLAMPoseLandmarks");

            // --- 2. Get Current TRUE Camera Pose from SLAM State ---
            // This is the CRITICAL step. We get the body pose T_nb from the state
            // and correctly transform it to the camera pose T_nc using T_bc.
            const Eigen::VectorXd xmean = sysPose->density.mean();
            Pose<double> Tnb;
            Tnb.translationVector = xmean.segment<3>(6);
            Tnb.rotationMatrix = rpy2rot(xmean.segment<3>(9));
            const Pose<double> Tnc = camera.bodyToCamera(Tnb); // Tnc = Tnb * Tbc

            // --- 3. Initialize New Landmarks ---
            if (id_by_landmark.size() < system.numberLandmarks()) {
                id_by_landmark.resize(system.numberLandmarks(), -1);
            }

            for (std::size_t i = 0; i < dets.ids.size(); ++i)
            {
                const int tagId = dets.ids[i];
                if (id2lm.count(tagId)) continue; // Skip if already in map

                // Create the pose of the tag in the camera frame (T_cj)
                Pose<double> Tcj(rodriguesToRot(rvecs[i]), Eigen::Vector3d(tvecs[i][0], tvecs[i][1], tvecs[i][2]));

                // Calculate the world pose of the tag: T_nj = T_nc * T_cj
                Pose<double> Tnj = Tnc * Tcj;
                const Eigen::Vector3d rnj = Tnj.translationVector;
                const Eigen::Vector3d Thetanj = rot2rpy(Tnj.rotationMatrix);

                // Standard initialization with uncertainty
                Eigen::Matrix<double,6,6> Sj = Eigen::Matrix<double,6,6>::Identity();
                Sj.diagonal() << INIT_POS_SIGMA, INIT_POS_SIGMA, INIT_POS_SIGMA, INIT_ANG_SIGMA, INIT_ANG_SIGMA, INIT_ANG_SIGMA;
                
                const std::size_t j = sysPose->appendLandmark(rnj, Thetanj, Sj);

                if (j >= id_by_landmark.size()) id_by_landmark.resize(j+1, -1);
                id_by_landmark[j] = tagId;
                id2lm[tagId] = j;
            }

            // --- 4. Create Measurement and Process ---
            const std::size_t N = dets.ids.size();
            Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, 4*N);
            for (std::size_t i = 0; i < N; ++i) {
                for (int k = 0; k < 4; ++k) {
                    Y(0, 4*i + k) = dets.corners[i][k].x;
                    Y(1, 4*i + k) = dets.corners[i][k].y;
                }
            }
            system.view() = dets.annotated.empty() ? imgin : dets.annotated;

            MeasurementSLAMUniqueTagBundle meas(t, Y, camera, dets.ids);
            meas.setIdByLandmark(id_by_landmark);
            meas.process(system);
            id_by_landmark = meas.idByLandmark();

            // --- 5. Visualization ---
            plot.setData(system, meas);
        }
        else if (scenario == 2)
        {
            auto* sysPts = dynamic_cast<SystemSLAMPointLandmarks*>(&system);
            assert(sysPts && "Scenario 2 expects SystemSLAMPointLandmarks");
            assert(duckDetector && "duckDetector must be initialised in scenario 2");


            // --- Detect (keep your existing detector setup)
            cv::Mat viz = duckDetector->detect(imgin);
            system.view() = viz.empty() ? imgin : viz;

            // --- Pull detections
            const auto& C_raw = duckDetector->last_centroids(); // vector<cv::Point2f>
            const auto& A_raw = duckDetector->last_areas();     // vector<double>
            const int rawN = (int)std::min(C_raw.size(), A_raw.size());

            // --- Early out if nothing and map empty (keeps UI responsive)
            if (rawN == 0 && sysPts->numberLandmarks() == 0) {
                Eigen::Matrix<double,2,Eigen::Dynamic> Y0(2,0);
                MeasurementPointBundle m0(t, Y0, camera);
                plot.setData(system, m0);
                continue;
            }

            // --- Intrinsics/constants (unchanged)
            const double fx = camera.cameraMatrix.at<double>(0,0);
            const double fy = camera.cameraMatrix.at<double>(1,1);
            const double duck_r_m    = 0.054;
            const double pos_sigma_m = 0.25;

            // ===== 1) LIGHT HSV YELLOW GATE (tunable, cheap) =====
            std::vector<cv::Point2f> C; C.reserve(rawN);
            std::vector<double>      A; A.reserve(rawN);

            auto circle_radius_from_area = [](double Ap)->float {
                return std::sqrt(std::max(1.0, Ap) / CV_PI);
            };

            // HSV thresholds (good defaults; tweak per video)
            const int H_MIN = 15, H_MAX = 35;   // yellow-ish
            const int S_MIN = 60, V_MIN = 60;
            const double MIN_YELLOW_RATIO = 0.04; // 4% pixels in circular ROI

            cv::Mat hsv; cv::cvtColor(imgin, hsv, cv::COLOR_BGR2HSV);

            for (int i=0; i<rawN; ++i) {
                const cv::Point2f c = C_raw[i];
                const double area   = std::max(1.0, A_raw[i]);
                const float  r      = std::max<float>(6.f, circle_radius_from_area(area));

                int x0 = (int)std::round(c.x - r), y0 = (int)std::round(c.y - r);
                int w  = (int)std::round(2*r),     h  = (int)std::round(2*r);
                if (w<=0 || h<=0) continue;
                if (x0 < 0) { w += x0; x0 = 0; }
                if (y0 < 0) { h += y0; y0 = 0; }
                if (x0 >= hsv.cols || y0 >= hsv.rows) continue;
                if (x0 + w > hsv.cols) w = hsv.cols - x0;
                if (y0 + h > hsv.rows) h = hsv.rows - y0;
                if (w<=0 || h<=0) continue;

                cv::Rect R(x0,y0,w,h);
                cv::Mat roi = hsv(R);

                int hits=0, tot=0;
                for (int yy=0; yy<roi.rows; ++yy) {
                    const uchar* p = roi.ptr<uchar>(yy);
                    for (int xx=0; xx<roi.cols; ++xx) {
                        const int H = p[3*xx+0], S = p[3*xx+1], V = p[3*xx+2];
                        const float dx = float(R.x+xx) - c.x, dy = float(R.y+yy) - c.y;
                        if (dx*dx + dy*dy > r*r) continue; // circle mask
                        if (S >= S_MIN && V >= V_MIN && H >= H_MIN && H <= H_MAX) ++hits;
                        ++tot;
                    }
                }
                const double ratio = (tot>0) ? double(hits)/double(tot) : 0.0;
                if (ratio >= MIN_YELLOW_RATIO) { C.push_back(c); A.push_back(area); }
            }

            // ===== 2) SIMPLE DE-DUP (prevents double detections on one duck) =====
            {
                const float PROX_SCALE = 0.6f;
                auto circR = [](double Ap)->float { return std::sqrt(std::max(1.0, Ap) / CV_PI); };

                std::vector<char> removed(C.size(), 0);
                for (size_t i=0; i<C.size(); ++i) if (!removed[i]) {
                    const float ri = circR(A[i]);
                    for (size_t j=i+1; j<C.size(); ++j) if (!removed[j]) {
                        const float rj = circR(A[j]);
                        const float rmin = std::max(4.f, std::min(ri, rj));
                        const cv::Point2f d = C[i] - C[j];
                        if (d.dot(d) <= (PROX_SCALE*PROX_SCALE*rmin*rmin)) {
                            const size_t keep = (A[i] >= A[j]) ? i : j;
                            const size_t drop = (keep==i) ? j : i;
                            removed[drop] = 1;
                        }
                    }
                }
                std::vector<cv::Point2f> C2; C2.reserve(C.size());
                std::vector<double>      A2; A2.reserve(A.size());
                for (size_t i=0;i<C.size();++i) if (!removed[i]) { C2.push_back(C[i]); A2.push_back(A[i]); }
                C.swap(C2); A.swap(A2);
            }

            // ===== 3) Build measurement arrays =====
            const int mDet = (int)std::min(C.size(), A.size());
            Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, mDet);
            Eigen::VectorXd Avec(mDet);
            for (int i=0;i<mDet;++i) { Y(0,i) = C[i].x; Y(1,i) = C[i].y; Avec(i) = A[i]; }

            // ===== 4) First-frame bootstrap (if map empty) =====
            if (sysPts->numberLandmarks() == 0 && mDet > 0) {
                (void)sysPts->appendFromDuckDetections(camera, Y, Avec, fx, fy, duck_r_m, pos_sigma_m);
            }

            // ===== 5) Build FOV subset of landmarks =====
            std::vector<std::size_t> idxFOV;
            {
                const Eigen::VectorXd xbar = system.density.mean();
                Pose<double> Tnb;
                Tnb.translationVector = xbar.segment<3>(6);
                Tnb.rotationMatrix    = rpy2rot(xbar.segment<3>(9));
                for (std::size_t j=0; j<sysPts->numberLandmarks(); ++j) {
                    const std::size_t idx = sysPts->landmarkPositionIndex(j);
                    const cv::Vec3d rPNn(xbar(idx+0), xbar(idx+1), xbar(idx+2));
                    if (camera.isWorldWithinFOV(rPNn, Tnb)) idxFOV.push_back(j);
                }
            }

            // ===== 6) Make measurement and SNN-associate on FOV only =====
            MeasurementSLAMDuckBundle meas(
                t, Y, Avec, camera,
                /*duck_radius_m=*/duck_r_m,
                /*sigma_px=*/6.0,      // tune 6–9 if gates too tight/loose
                /*sigma_area=*/800.0   // tune with your video if needed
            );
            meas.associate(system, idxFOV);   // uses your existing SNN under the hood

            // ===== 7) Birth NEW landmarks from UNMATCHED detections (centroid → on-screen) =====
            if (mDet > 0) {
                std::vector<char> used(mDet, false);
                const auto& assoc = meas.idxFeatures(); // global-sized; value is detection index or -1
                for (std::size_t j=0; j<assoc.size(); ++j)
                    if (assoc[j] >= 0 && assoc[j] < mDet) used[(size_t)assoc[j]] = true;

                int mNew = 0; for (int i=0;i<mDet;++i) if (!used[i]) ++mNew;
                if (mNew > 0) {
                    Eigen::Matrix<double,2,Eigen::Dynamic> Ynew(2, mNew);
                    Eigen::VectorXd Anew(mNew);
                    for (int i=0,k=0; i<mDet; ++i) if (!used[i]) {
                        Ynew(0,k) = Y(0,i); Ynew(1,k) = Y(1,i); Anew(k) = Avec(i); ++k;
                    }
                    (void)sysPts->appendFromDuckDetections(camera, Ynew, Anew, fx, fy, duck_r_m, pos_sigma_m);

                    // Rebuild FOV after births and re-associate so new LMs update immediately
                    idxFOV.clear();
                    const Eigen::VectorXd xbar2 = system.density.mean();
                    Pose<double> Tnb2;
                    Tnb2.translationVector = xbar2.segment<3>(6);
                    Tnb2.rotationMatrix    = rpy2rot(xbar2.segment<3>(9));
                    for (std::size_t j=0; j<sysPts->numberLandmarks(); ++j) {
                        const std::size_t idx = sysPts->landmarkPositionIndex(j);
                        const cv::Vec3d rPNn(xbar2(idx+0), xbar2(idx+1), xbar2(idx+2));
                        if (camera.isWorldWithinFOV(rPNn, Tnb2)) idxFOV.push_back(j);
                    }
                    meas.associate(system, idxFOV);
                }
            }

            // ===== 8) SRIF update + plot =====
            meas.process(system);
            plot.setData(system, meas);

            // ===== 9) (Optional) Cull old landmarks via your existing counters =====
            {
                const Eigen::VectorXd xbar = system.density.mean();
                Pose<double> Tnb;
                Tnb.translationVector = xbar.segment<3>(6);
                Tnb.rotationMatrix    = rpy2rot(xbar.segment<3>(9));
                const auto& assoc = meas.idxFeatures();
                for (std::size_t j=0; j<sysPts->numberLandmarks(); ++j) {
                    const std::size_t idx = sysPts->landmarkPositionIndex(j);
                    const cv::Vec3d rPNn(xbar(idx+0), xbar(idx+1), xbar(idx+2));
                    const bool predictedVisible = camera.isWorldWithinFOV(rPNn, Tnb);
                    const bool matched = (j < assoc.size() && assoc[j] >= 0);
                    sysPts->updateFailureCounter(j, predictedVisible && !matched);
                }
            }
            sysPts->cullFailed(10);

        }
        else
        {
            assert("broken if you're here");
            // Other scenarios unchanged in this file
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

    // ---------- Clean shutdown (finalize export and threads) ----------
    if (doExport) {
        bufferedVideoWriter.stop();
        videoOut.release();
    }
    bufferedVideoReader.stop();
}
