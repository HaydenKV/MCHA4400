#ifndef PLOT_H
#define PLOT_H

#include <string>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <vtkActor.h>
#include <vtkActor2D.h>
#include <vtkAxesActor.h>
#include <vtkCellArray.h>
#include <vtkColor.h>
#include <vtkContourFilter.h>
#include <vtkCubeAxesActor.h>
#include <vtkDataSetMapper.h>
#include <vtkImageData.h>
#include <vtkImageMapper.h>
#include <vtkLine.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPyramid.h>
#include <vtkQuadric.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSampleFunction.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>

#include "Camera.h"

// -------------------------------------------------------
// Bounds (from Lab 8)
// -------------------------------------------------------
struct Bounds
{
    Bounds();
    void getVTKBounds(double * bounds) const;
    void setExtremity(Bounds & extremity) const;
    void calculateMaxMinSigmaPoints(const Eigen::Matrix3d & covariance, 
                                   const Eigen::Vector3d & center, 
                                   const double sigma);

    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
};

// -------------------------------------------------------
// QuadricPlot (from Lab 8, adapted for assignment)
// -------------------------------------------------------
struct QuadricPlot
{
    QuadricPlot();
    void update(const Eigen::Vector3d & center, const Eigen::Matrix3d & covariance, double sigma = 3.0);
    vtkActor * getActor() const;
    void setColor(double r, double g, double b, double opacity = 0.7);
    
    Bounds bounds;
    vtkSmartPointer<vtkActor> actor;
    vtkSmartPointer<vtkPolyDataMapper> mapper;
    vtkSmartPointer<vtkContourFilter> contour;
    vtkSmartPointer<vtkSampleFunction> sample;
    vtkSmartPointer<vtkQuadric> quadric;
};

// -------------------------------------------------------
// FrustumPlot (from Lab 8)
// -------------------------------------------------------
struct FrustumPlot
{
    explicit FrustumPlot(const Camera & camera);
    void update(const Eigen::Vector3d & position, const Eigen::Matrix3d & rotation);
    vtkActor * getActor() const;

    vtkSmartPointer<vtkActor> actor;
    vtkSmartPointer<vtkPolyDataMapper> mapper;
    vtkSmartPointer<vtkPolyData> polyData;
    vtkSmartPointer<vtkPoints> points;
    vtkSmartPointer<vtkCellArray> lines;
};

// -------------------------------------------------------
// AxisPlot (from Lab 8)
// -------------------------------------------------------
struct AxisPlot
{
    AxisPlot();
    void init(vtkCamera * camera);
    void update(const Eigen::Vector3d & rCNn, const Eigen::Vector3d & Thetanc);
    vtkProp3D * getActor() const;

    vtkSmartPointer<vtkAxesActor> axesActor;
    vtkSmartPointer<vtkTransform> transform;
    bool isInit;
};

// -------------------------------------------------------
// BasisPlot (from Lab 8)
// -------------------------------------------------------
struct BasisPlot
{
    BasisPlot();
    void update(const Eigen::Vector3d & rCNn, const Eigen::Vector3d & Thetanc);
    vtkProp3D * getActor() const;

    vtkSmartPointer<vtkAxesActor> axesActor;
    vtkSmartPointer<vtkTransform> transform;
    bool isInit;
};

// -------------------------------------------------------
// ImagePlot (from Lab 8, adapted)
// -------------------------------------------------------
struct ImagePlot
{
    ImagePlot();
    void init(double rendererWidth, double rendererHeight);
    void update(const cv::Mat & view);
    vtkActor2D * getActor() const;

    vtkSmartPointer<vtkImageData> viewVTK;
    vtkSmartPointer<vtkActor2D> imageActor2d;
    vtkSmartPointer<vtkImageMapper> imageMapper;
    cv::Mat cvVTKBuffer;
    double width, height;
    bool isInit;
};

// -------------------------------------------------------
// Plot (adapted from Lab 8 for Assignment 1)
// -------------------------------------------------------
struct Plot
{
public:
    explicit Plot(const Camera & camera);
    
    // Assignment 1 interface
    void updateImage(const cv::Mat& image, 
                    const std::vector<cv::Point2f>& features = {},
                    const std::vector<int>& featureStatus = {});
    
    void updateScene(const Eigen::Vector3d& cameraPos = Eigen::Vector3d::Zero(),
                    const Eigen::Matrix3d& cameraRot = Eigen::Matrix3d::Identity(),
                    const Eigen::Matrix3d& cameraCov = 0.1 * Eigen::Matrix3d::Identity(),
                    const std::vector<Eigen::Vector3d>& landmarkPositions = {},
                    const std::vector<Eigen::Matrix3d>& landmarkCovariances = {},
                    const std::vector<int>& landmarkStatus = {});
    
    void render();
    void handleInteractivity(int interactive, bool isLastFrame = false);
    cv::Mat getFrame() const;

private:
    void drawFeaturesOnImage(cv::Mat& image, 
                           const std::vector<cv::Point2f>& features,
                           const std::vector<int>& featureStatus);

    const Camera & camera;
    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderer> threeDimRenderer;
    vtkSmartPointer<vtkRenderer> imageRenderer;
    vtkSmartPointer<vtkRenderWindowInteractor> interactor;
    
    // Lab 8 components adapted for assignment
    QuadricPlot qpCamera;
    std::vector<QuadricPlot> qpLandmarks;
    FrustumPlot fp;
    AxisPlot ap;
    BasisPlot bp;
    ImagePlot ip;
};

#endif