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

// Predict current pixel for landmark j (mean only). Returns false if outside image.
static bool predictLmPixel(const SystemSLAMPointLandmarks& sys,
                           const Camera& camera,
                           std::size_t j,
                           cv::Point2d& uv_out)
{
    const Eigen::VectorXd xbar = sys.density.mean();
    if (j >= sys.numberLandmarks()) return false;
    const std::size_t idx = sys.landmarkPositionIndex(j);
    if (idx + 2 >= (std::size_t)xbar.size()) return false;

    Pose<double> Tnb;
    Tnb.translationVector = xbar.segment<3>(6);
    Tnb.rotationMatrix    = rpy2rot(xbar.segment<3>(9));

    const cv::Vec3d rPNn(xbar(idx+0), xbar(idx+1), xbar(idx+2));
    if (!camera.isWorldWithinFOV(rPNn, Tnb)) return false;

    const cv::Vec2d uv = camera.worldToPixel(rPNn, Tnb);
    uv_out = cv::Point2d(uv[0], uv[1]);
    return camera.isPixelInside(Eigen::Vector2d(uv[0], uv[1]));
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

            // ============================ Tunables ============================
            constexpr double DUCK_RADIUS_M    = 0.054;   // physical radius [m]
            constexpr double INIT_POS_SIGMA   = 0.35;    // init 1σ for new LM [m] (looser)
            constexpr int    NMAX_LANDMARKS   = 200;     // bigger map cap
            constexpr int    MAX_NEW_PER_FR   = 6;       // add a few per frame
            constexpr int    NMS_MIN_SEP_PX   = 40;      // avoid clustering seeds
            constexpr double SIGMA_PX         = 2.5;     // pixel noise (strong)
            constexpr double SIGMA_AREA       = 3000;     // area noise (disable area update)
            constexpr double Z_MIN            = 0.60;    // seed depth clamp [m]
            constexpr double Z_MAX            = 3.00;    // seed depth clamp [m]
            constexpr double REPROJ_SEED_THRESH_PX = 25.0; // seed reprojection sanity gate
            constexpr int    CULL_FAIL_THRESH = 6;       // faster cleanup

            auto stateDimOK = [&]()->bool {
                const std::size_t nLM = sysPts->numberLandmarks();
                const std::size_t dim = (std::size_t)system.density.mean().size();
                return (dim == 12u + 3u*nLM);
            };

            // ============================ Detect ============================
            cv::Mat viz = duckDetector->detect(imgin);
            const auto& C   = duckDetector->last_centroids();
            const auto& Apx = duckDetector->last_areas();
            system.view() = viz;

            if (Apx.size() != C.size()) {
                MeasurementPointBundle measEmpty(t, Eigen::Matrix<double,2,Eigen::Dynamic>(2,0), camera);
                plot.setData(system, measEmpty);
                return;
            }

            Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, (int)C.size());
            Eigen::VectorXd Avec((int)C.size());
            for (int i = 0; i < (int)C.size(); ++i) {
                Y(0,i) = C[i].x;  Y(1,i) = C[i].y;
                Avec(i) = std::max(1.0, Apx[i]);
            }

            // ============================ Helpers ============================
            auto buildIdxLandmarks = [&](std::vector<std::size_t>& out){
                out.clear();
                if (!stateDimOK()) return;
                const Eigen::VectorXd xbar = system.density.mean();
                Pose<double> Tnb; Tnb.translationVector = xbar.segment<3>(6);
                Tnb.rotationMatrix    = rpy2rot(xbar.segment<3>(9));
                for (std::size_t j = 0; j < sysPts->numberLandmarks(); ++j) {
                    const std::size_t idx = sysPts->landmarkPositionIndex(j);
                    if (idx + 2 >= (std::size_t)xbar.size()) continue;
                    const cv::Vec3d rPNn(xbar(idx+0), xbar(idx+1), xbar(idx+2));
                    if (camera.isWorldWithinFOV(rPNn, Tnb)) out.push_back(j);
                }
            };

            auto pickSurplusWithNMS = [&](const std::vector<char>& used)->std::vector<int>{
                std::vector<int> surplus;
                for (int i = 0; i < (int)Y.cols(); ++i) if (!used[i]) surplus.push_back(i);
                std::sort(surplus.begin(), surplus.end(),
                        [&](int a, int b){ return Avec(a) > Avec(b); }); // big area first
                std::vector<int> chosen;
                chosen.reserve(MAX_NEW_PER_FR);
                for (int idxFeat : surplus) {
                    bool far = true;
                    for (int kept : chosen) {
                        double dx = Y(0,idxFeat) - Y(0,kept);
                        double dy = Y(1,idxFeat) - Y(1,kept);
                        if (dx*dx + dy*dy < (double)NMS_MIN_SEP_PX*NMS_MIN_SEP_PX) { far = false; break; }
                    }
                    if (!far) { continue; }
                    chosen.push_back(idxFeat);
                    if ((int)chosen.size() >= MAX_NEW_PER_FR) break;
                }
                return chosen;
            };

            // Robust seeding with duplicate-block + anisotropic init
            auto appendFromDetectionsRobust = [&](const std::vector<int>& detIdx, int K){
                if (K <= 0) return 0;

                // --- Tunables just for seeding ---
                constexpr double Z_MIN = 0.40;     // [m]
                constexpr double Z_MAX = 6.00;     // [m]
                constexpr double REPROJ_SEED_THRESH_PX = 25.0; // pixel gate
                constexpr double EXISTING_SEED_BLOCK_PX = 30.0; // don’t birth near an existing LM
                // Anisotropic sqrt-covariance for new LMs (⟂ small, ∥ large)
                constexpr double SIGMA_PERP  = 0.15; // [m] across-ray 1σ
                constexpr double SIGMA_DEPTH = 0.50; // [m] along-ray  1σ

                // --- Camera pose & intrinsics at the mean (fresh) ---
                const Eigen::VectorXd xbar = system.density.mean();
                Pose<double> Tnb;
                Tnb.translationVector = xbar.segment<3>(6);
                Tnb.rotationMatrix    = rpy2rot(xbar.segment<3>(9));
                const Eigen::Matrix3d Rnc = Tnb.rotationMatrix;
                const Eigen::Vector3d rCNn = Tnb.translationVector;
                const double fx = camera.cameraMatrix.at<double>(0,0);
                const double fy = camera.cameraMatrix.at<double>(1,1);

                int added = 0;
                for (int k = 0; k < (int)detIdx.size() && added < K; ++k) {
                    const int iFeat = detIdx[k];
                    // Basic bounds
                    if (iFeat < 0) continue;
                    // NOTE: Y, Avec are captured from outer scope in your Scenario-2 code
                    if (iFeat >= (int)Y.cols()) continue;

                    const double u  = Y(0,iFeat);
                    const double v  = Y(1,iFeat);
                    const double Ai = std::max(1.0, Avec(iFeat));

                    // --- Depth from area with sanity clamp ---
                    const double depth2 = (fx * fy * std::numbers::pi * DUCK_RADIUS_M * DUCK_RADIUS_M) / Ai;
                    if (!(depth2 > 0.0) || !std::isfinite(depth2)) continue;
                    double z = std::sqrt(depth2);
                    if (!std::isfinite(z)) continue;
                    z = std::clamp(z, Z_MIN, Z_MAX);

                    // --- Ray in camera frame → world point ---
                    const cv::Vec3d dirC = camera.pixelToVector(cv::Vec2d(u,v));
                    const Eigen::Vector3d uPCc(dirC[0], dirC[1], dirC[2]);
                    const Eigen::Vector3d rLNn = Rnc * (z * uPCc) + rCNn;

                    // --- Reprojection sanity gate (seed must reproject near (u,v)) ---
                    const cv::Vec2d uv_back = camera.worldToPixel(cv::Vec3d(rLNn.x(), rLNn.y(), rLNn.z()), Tnb);
                    const double du = uv_back[0] - u;
                    const double dv = uv_back[1] - v;
                    if (du*du + dv*dv > REPROJ_SEED_THRESH_PX*REPROJ_SEED_THRESH_PX) {
                        continue;
                    }

                    // --- NEW: Don’t birth if too close to an existing LM’s predicted pixel ---
                    bool nearExisting = false;
                    {
                        const cv::Point2d uv_new(u,v);
                        for (std::size_t j = 0; j < sysPts->numberLandmarks(); ++j) {
                            cv::Point2d uv_j;
                            if (!predictLmPixel(*sysPts, camera, j, uv_j)) continue;
                            const double dx = uv_new.x - uv_j.x;
                            const double dy = uv_new.y - uv_j.y;
                            if (dx*dx + dy*dy < EXISTING_SEED_BLOCK_PX*EXISTING_SEED_BLOCK_PX) {
                                nearExisting = true; break;
                            }
                        }
                    }
                    if (nearExisting) continue;

                    // --- NEW: Anisotropic sqrt-covariance (in camera frame), rotate to world ---
                    Eigen::Matrix3d S_c = Eigen::Matrix3d::Zero();
                    S_c(0,0) = SIGMA_PERP;   // across-ray
                    S_c(1,1) = SIGMA_PERP;   // across-ray
                    S_c(2,2) = SIGMA_DEPTH;  // along-ray
                    // rotate sqrt-cov with Rnc (cols=cam axes in world)
                    Eigen::Matrix3d Spos = Rnc * S_c * Rnc.transpose();

                    sysPts->appendLandmark(rLNn, Spos);
                    ++added;
                }
                return added;
            };


            // ============================ Bootstrap if empty ============================
            if (sysPts->numberLandmarks() == 0) {
                if (Y.cols() == 0) {
                    MeasurementPointBundle measEmpty(t, Eigen::Matrix<double,2,Eigen::Dynamic>(2,0), camera);
                    plot.setData(system, measEmpty);
                    return;
                }
                std::vector<char> used0((int)Y.cols(), 0);
                auto chosen0 = pickSurplusWithNMS(used0);
                int canAdd0 = std::max(0, NMAX_LANDMARKS - (int)sysPts->numberLandmarks());
                appendFromDetectionsRobust(chosen0, std::min((int)chosen0.size(), std::min(MAX_NEW_PER_FR, canAdd0)));
                if (sysPts->numberLandmarks() == 0) {
                    MeasurementPointBundle measEmpty(t, Eigen::Matrix<double,2,Eigen::Dynamic>(2,0), camera);
                    plot.setData(system, measEmpty);
                    return;
                }
            }

            // ============================ Pass 1: associate & cull ============================
            std::vector<std::size_t> idxLandmarks;
            buildIdxLandmarks(idxLandmarks);

            if (!idxLandmarks.empty()) {
                MeasurementSLAMDuckBundle meas1(t, Y, Avec, camera, DUCK_RADIUS_M, SIGMA_PX, SIGMA_AREA);
                meas1.associate(system, idxLandmarks);

                if (stateDimOK()) {
                    const Eigen::VectorXd xbar = system.density.mean();
                    Pose<double> Tnb; Tnb.translationVector = xbar.segment<3>(6);
                    Tnb.rotationMatrix    = rpy2rot(xbar.segment<3>(9));
                    const auto& assoc = meas1.idxFeatures();
                    const std::size_t nLM = sysPts->numberLandmarks();
                    for (std::size_t j = 0; j < nLM; ++j) {
                        const std::size_t idx = sysPts->landmarkPositionIndex(j);
                        if (idx + 2 >= (std::size_t)xbar.size()) continue;
                        const cv::Vec3d rPNn(xbar(idx+0), xbar(idx+1), xbar(idx+2));
                        const bool predictedVisible = camera.isWorldWithinFOV(rPNn, Tnb);
                        const bool matched = (j < assoc.size() && assoc[j] >= 0);
                        sysPts->updateFailureCounter(j, predictedVisible && !matched);
                    }
                }
                sysPts->cullFailed(CULL_FAIL_THRESH);
            }

            // ============================ Bootstrap during run (surplus) ============================
            {
                std::vector<char> used((int)Y.cols(), 0);
                std::vector<std::size_t> idxTmp; buildIdxLandmarks(idxTmp);
                if (!idxTmp.empty()) {
                    MeasurementSLAMDuckBundle measMark(t, Y, Avec, camera, DUCK_RADIUS_M, SIGMA_PX, SIGMA_AREA);
                    measMark.associate(system, idxTmp);
                    const auto& a = measMark.idxFeatures();
                    for (int j = 0; j < (int)a.size(); ++j) {
                        int fi = a[j];
                        if (fi >= 0 && fi < (int)Y.cols()) used[fi] = 1;
                    }
                }
                auto chosen = pickSurplusWithNMS(used);
                int canAdd = std::max(0, NMAX_LANDMARKS - (int)sysPts->numberLandmarks());
                appendFromDetectionsRobust(chosen, std::min((int)chosen.size(), std::min(MAX_NEW_PER_FR, canAdd)));
            }

            // ============================ Final association + update ============================
            std::vector<std::size_t> idxLandmarks2;
            buildIdxLandmarks(idxLandmarks2);
            if (idxLandmarks2.empty()) {
                MeasurementPointBundle measEmpty(t, Eigen::Matrix<double,2,Eigen::Dynamic>(2,0), camera);
                plot.setData(system, measEmpty);
                return;
            }

            MeasurementSLAMDuckBundle meas2(t, Y, Avec, camera, DUCK_RADIUS_M, SIGMA_PX, SIGMA_AREA);
            meas2.associate(system, idxLandmarks2);
            if (stateDimOK()) {
                meas2.process(system);
            }
            plot.setData(system, meas2);
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
