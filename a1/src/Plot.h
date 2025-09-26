#ifndef PLOT_H
#define PLOT_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkImageActor.h>
#include <vtkImageImport.h>
#include <vtkWindowToImageFilter.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkQuadric.h>
#include <vtkSampleFunction.h>
#include <vtkContourFilter.h>
#include <vector>
#include <memory>
#include <Eigen/Core>
#include "Camera.h"

// Forward declarations for remaining VTK classes
class vtkSphereSource;
class vtkUnstructuredGrid;
class vtkDataSetMapper;

/**
 * @brief Renders 3D confidence ellipsoids for SLAM visualization
 * Based on Lab 8 quadric surface implementation
 */
class ConfidenceEllipsoidPlot {
public:
    ConfidenceEllipsoidPlot() : isInitialized(false) {}
    ~ConfidenceEllipsoidPlot() = default;
    
    void updateEllipsoid(const Eigen::Vector3d& center, const Eigen::Matrix3d& covariance, double sigma = 3.0);
    void setColor(double r, double g, double b, double opacity = 0.7);
    vtkActor* getActor();
    bool isValid() const { return isInitialized; }

private:
    void initialize();
    
    vtkSmartPointer<vtkActor> actor;
    vtkSmartPointer<vtkPolyDataMapper> mapper;
    vtkSmartPointer<vtkQuadric> quadric;
    vtkSmartPointer<vtkSampleFunction> sample;
    vtkSmartPointer<vtkContourFilter> contour;
    bool isInitialized;
};

/**
 * @brief Renders camera frustum for 3D visualization
 * Based on Lab 8 frustum implementation
 */
class CameraFrustumPlot {
public:
    explicit CameraFrustumPlot(const Camera& camera) {
        initializeFrustum(camera);
    }
    ~CameraFrustumPlot() = default;
    
    void updatePose(const Eigen::Vector3d& position, const Eigen::Matrix3d& rotation);
    vtkActor* getActor() { return actor; }

private:
    void initializeFrustum(const Camera& camera);
    
    vtkSmartPointer<vtkActor> actor;
    vtkSmartPointer<vtkPolyDataMapper> mapper;
    vtkSmartPointer<vtkPolyData> polyData;
    vtkSmartPointer<vtkPoints> points;
    vtkSmartPointer<vtkCellArray> lines;
};

/**
 * @brief Main SLAM visualization class implementing assignment requirements
 * 
 * Provides dual-pane visualization:
 * - Left pane: Camera image + detected features + confidence ellipses  
 * - Right pane: 3D world view + camera + landmarks + confidence ellipsoids
 * 
 * Based on Lab 8 VTK visualization foundations
 */
class Plot {
public:
    explicit Plot(const Camera& camera);
    ~Plot() = default;
    
    /**
     * @brief Update the camera image with detected features
     * @param image Camera frame
     * @param features Detected feature locations
     * @param featureStatus Status codes (0=tracked, 1=visible_undetected)
     */
    void updateImage(const cv::Mat& image, 
                    const std::vector<cv::Point2f>& features = {},
                    const std::vector<int>& featureStatus = {});
    
    /**
     * @brief Update 3D scene with SLAM state
     * @param cameraPos Camera position in world frame
     * @param cameraRot Camera orientation matrix
     * @param cameraCov Camera position covariance
     * @param landmarkPositions 3D landmark positions
     * @param landmarkCovariances Landmark covariances
     * @param landmarkStatus Landmark status codes (0=tracked, 1=visible_undetected, 2=not_visible)
     */
    void updateScene(const Eigen::Vector3d& cameraPos = Eigen::Vector3d::Zero(),
                    const Eigen::Matrix3d& cameraRot = Eigen::Matrix3d::Identity(),
                    const Eigen::Matrix3d& cameraCov = 0.1 * Eigen::Matrix3d::Identity(),
                    const std::vector<Eigen::Vector3d>& landmarkPositions = {},
                    const std::vector<Eigen::Matrix3d>& landmarkCovariances = {},
                    const std::vector<int>& landmarkStatus = {});
    
    /**
     * @brief Render the visualization
     */
    void render();
    
    /**
     * @brief Get current visualization frame for video export
     */
    cv::Mat getFrame();
    
    /**
     * @brief Handle interactive modes per assignment spec
     * @param interactive 0=none, 1=last frame, 2=all frames
     * @param isLastFrame Whether this is the last frame of video
     */
    void handleInteractivity(int interactive, bool isLastFrame = false);

private:
    void updateImageDisplay(const cv::Mat& image);
    
    // Core camera reference
    const Camera& camera;
    
    // VTK rendering pipeline
    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderer> leftRenderer;   // Image view
    vtkSmartPointer<vtkRenderer> rightRenderer;  // 3D scene
    vtkSmartPointer<vtkRenderWindowInteractor> interactor;
    
    // Image display
    vtkSmartPointer<vtkImageActor> imageActor;
    
    // 3D scene components
    ConfidenceEllipsoidPlot cameraEllipsoid;
    std::unique_ptr<CameraFrustumPlot> cameraFrustum;
    std::vector<ConfidenceEllipsoidPlot> landmarkEllipsoids;
};

#endif // PLOT_H