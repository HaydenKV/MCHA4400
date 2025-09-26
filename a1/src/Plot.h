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
#include <vector>
#include <Eigen/Core>
#include "Camera.h"

// Forward declarations for VTK classes
class vtkPolyDataMapper;
class vtkSphereSource;
class vtkQuadric;
class vtkSampleFunction;
class vtkContourFilter;
class vtkUnstructuredGrid;
class vtkDataSetMapper;
class vtkPoints;
class vtkCellArray;

/**
 * @brief Renders 3D confidence ellipsoids for SLAM visualization
 * Based on Lab 8 quadric surface implementation
 */
class ConfidenceEllipsoidPlot {
public:
    ConfidenceEllipsoidPlot();
    ~ConfidenceEllipsoidPlot();
    
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
    explicit CameraFrustumPlot(const Camera& camera);
    ~CameraFrustumPlot();
    
    void updatePose(const Eigen::Vector3d& position, const Eigen::Matrix3d& rotation);
    vtkActor* getActor();

private:
    void initializeFrustum(const Camera& camera);
    
    vtkSmartPointer<vtkActor> actor;
    vtkSmartPointer<vtkDataSetMapper> mapper;
    vtkSmartPointer<vtkPoints> points;
    vtkSmartPointer<vtkUnstructuredGrid> grid;
    vtkSmartPointer<vtkCellArray> cells;
    Camera camera;
};

/**
 * @brief Main SLAM visualization class implementing assignment requirements
 * 
 * Provides dual-pane visualization:
 * - Left pane: Camera image with features and confidence ellipses
 * - Right pane: 3D scene with camera frustum and landmark ellipsoids
 */
class Plot {
public:
    explicit Plot(const Camera& camera);
    ~Plot();
    
    /**
     * @brief Update the camera image with detected features
     * @param image Input camera frame
     * @param features Detected feature points for overlay
     * @param featureStatus Status of each feature (0=tracked, 1=visible_undetected)
     */
    void updateImage(const cv::Mat& image, 
                    const std::vector<cv::Point2f>& features = {},
                    const std::vector<int>& featureStatus = {});
    
    /**
     * @brief Update 3D scene with SLAM state
     * @param cameraPos Camera position in world coordinates  
     * @param cameraRot Camera rotation matrix
     * @param cameraCov Camera position covariance (3x3)
     * @param landmarkPositions 3D landmark positions
     * @param landmarkCovariances Landmark covariances
     * @param landmarkStatus Status: 0=tracked, 1=visible_undetected, 2=not_visible
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
     * @return Dual-pane image (left: camera view, right: 3D scene)
     */
    cv::Mat getFrame();
    
    /**
     * @brief Handle interactive modes per assignment spec
     * @param interactive 0=none, 1=last_frame, 2=every_frame  
     * @param isLastFrame Whether this is the final video frame
     */
    void handleInteractivity(int interactive, bool isLastFrame = false);

private:
    void setupWindow();
    void setupImagePane();
    void setupScenePane();
    void drawConfidenceEllipses(cv::Mat& image, const std::vector<cv::Point2f>& features, 
                               const std::vector<int>& status);
    void updateLandmarkEllipsoids(const std::vector<Eigen::Vector3d>& positions,
                                 const std::vector<Eigen::Matrix3d>& covariances,
                                 const std::vector<int>& status);
    
    Camera camera;
    
    // VTK rendering components
    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderer> imageRenderer;      // Left pane
    vtkSmartPointer<vtkRenderer> sceneRenderer;      // Right pane
    vtkSmartPointer<vtkRenderWindowInteractor> interactor;
    
    // Image display
    vtkSmartPointer<vtkImageActor> imageActor;
    vtkSmartPointer<vtkImageImport> imageImporter;
    
    // 3D scene elements
    CameraFrustumPlot* cameraFrustum;
    ConfidenceEllipsoidPlot* cameraEllipsoid;
    std::vector<ConfidenceEllipsoidPlot*> landmarkEllipsoids;
    
    // Frame capture for video export
    vtkSmartPointer<vtkWindowToImageFilter> windowFilter;
    
    // Current state
    cv::Mat currentImageWithOverlays;
};

#endif // PLOT_H