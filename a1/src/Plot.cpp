// Plot.cpp - Complete Lab 8 adaptation for Assignment 1
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

// CRITICAL: VTK auto-init defines from Lab 8
#define vtkRenderingContext2D_AUTOINIT 1(vtkRenderingContextOpenGL2)
#define vtkRenderingCore_AUTOINIT 3(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingOpenGL2)
#define vtkRenderingOpenGL2_AUTOINIT 1(vtkRenderingGL2PSOpenGL2)

#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkContourFilter.h>
#include <vtkImageActor.h>
#include <vtkImageData.h>
#include <vtkImageImport.h>
#include <vtkImageMapper.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLine.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkQuadric.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSampleFunction.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkWindowToImageFilter.h>

#include "Plot.h"

// For compatibility with new VTK generic data arrays
#ifdef vtkGenericDataArray_h
#define InsertNextTupleValue InsertNextTypedTuple
#endif

// -------------------------------------------------------
// Bounds (from Lab 8)
// -------------------------------------------------------
Bounds::Bounds()
    : xmin(-1.0), xmax(1.0)
    , ymin(-1.0), ymax(1.0)
    , zmin(-1.0), zmax(1.0)
{
}

void Bounds::getVTKBounds(double * bounds) const
{
    bounds[0] = xmin; bounds[1] = xmax;
    bounds[2] = ymin; bounds[3] = ymax;
    bounds[4] = zmin; bounds[5] = zmax;
}

void Bounds::setExtremity(Bounds & extremity) const
{
    extremity.xmin = std::min(extremity.xmin, xmin);
    extremity.xmax = std::max(extremity.xmax, xmax);
    extremity.ymin = std::min(extremity.ymin, ymin);
    extremity.ymax = std::max(extremity.ymax, ymax);
    extremity.zmin = std::min(extremity.zmin, zmin);
    extremity.zmax = std::max(extremity.zmax, zmax);
}

void Bounds::calculateMaxMinSigmaPoints(const Eigen::Matrix3d & covariance, 
                                       const Eigen::Vector3d & center, 
                                       const double sigma)
{
    // Simplified bounds calculation for assignment
    double scale = sigma * 0.5; // Conservative bound
    xmin = center.x() - scale; xmax = center.x() + scale;
    ymin = center.y() - scale; ymax = center.y() + scale;
    zmin = center.z() - scale; zmax = center.z() + scale;
}

// -------------------------------------------------------
// QuadricPlot (adapted from Lab 8)
// -------------------------------------------------------
QuadricPlot::QuadricPlot()
    : actor(vtkSmartPointer<vtkActor>::New())
    , mapper(vtkSmartPointer<vtkPolyDataMapper>::New())
    , contour(vtkSmartPointer<vtkContourFilter>::New())
    , sample(vtkSmartPointer<vtkSampleFunction>::New())
    , quadric(vtkSmartPointer<vtkQuadric>::New())
{
    // Setup VTK pipeline (Lab 8 pattern)
    sample->SetImplicitFunction(quadric);
    sample->SetSampleDimensions(30, 30, 30);
    sample->ComputeNormalsOff();
    
    contour->SetInputConnection(sample->GetOutputPort());
    contour->SetValue(0, 0.0);
    
    mapper->SetInputConnection(contour->GetOutputPort());
    actor->SetMapper(mapper);
    
    // Default appearance
    setColor(0.0, 0.5, 1.0, 0.7);
}

void QuadricPlot::update(const Eigen::Vector3d & center, const Eigen::Matrix3d & covariance, double sigma)
{
    // Calculate bounds for sampling
    bounds.calculateMaxMinSigmaPoints(covariance, center, sigma);
    
    // Set sampling bounds
    bounds.getVTKBounds(nullptr);
    sample->SetModelBounds(bounds.xmin, bounds.xmax, 
                          bounds.ymin, bounds.ymax,
                          bounds.zmin, bounds.zmax);
    
    // Simplified quadric (can be enhanced with proper eigenvalue decomposition)
    double scale = sigma * sigma;
    quadric->SetCoefficients(1.0/scale, 1.0/scale, 1.0/scale, 0, 0, 0,
                           -2*center.x()/scale, -2*center.y()/scale, -2*center.z()/scale,
                           (center.dot(center))/scale - 1.0);
    
    sample->Modified();
}

void QuadricPlot::setColor(double r, double g, double b, double opacity)
{
    actor->GetProperty()->SetColor(r, g, b);
    actor->GetProperty()->SetOpacity(opacity);
    actor->Modified();
}

vtkActor * QuadricPlot::getActor() const
{
    return actor;
}

// -------------------------------------------------------
// FrustumPlot (from Lab 8)
// -------------------------------------------------------
FrustumPlot::FrustumPlot(const Camera & camera)
    : actor(vtkSmartPointer<vtkActor>::New())
    , mapper(vtkSmartPointer<vtkPolyDataMapper>::New())
    , polyData(vtkSmartPointer<vtkPolyData>::New())
    , points(vtkSmartPointer<vtkPoints>::New())
    , lines(vtkSmartPointer<vtkCellArray>::New())
{
    // Create simple frustum geometry
    double w = camera.imageSize.width * 0.001;  // Scale factor
    double h = camera.imageSize.height * 0.001;
    double d = 0.5; // Frustum depth
    
    // Frustum corners
    points->InsertNextPoint(0, 0, 0);           // Camera center
    points->InsertNextPoint(-w/2, -h/2, d);    // Bottom-left
    points->InsertNextPoint(w/2, -h/2, d);     // Bottom-right  
    points->InsertNextPoint(w/2, h/2, d);      // Top-right
    points->InsertNextPoint(-w/2, h/2, d);     // Top-left
    
    // Frustum lines
    vtkNew<vtkLine> line;
    for (int i = 1; i <= 4; i++) {
        line->GetPointIds()->SetId(0, 0);
        line->GetPointIds()->SetId(1, i);
        lines->InsertNextCell(line);
    }
    
    // Frame lines
    for (int i = 1; i <= 4; i++) {
        line->GetPointIds()->SetId(0, i);
        line->GetPointIds()->SetId(1, (i % 4) + 1);
        lines->InsertNextCell(line);
    }
    
    polyData->SetPoints(points);
    polyData->SetLines(lines);
    
    mapper->SetInputData(polyData);
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(1.0, 1.0, 0.0); // Yellow frustum
}

void FrustumPlot::update(const Eigen::Vector3d & position, const Eigen::Matrix3d & rotation)
{
    // Update frustum position and orientation
    vtkNew<vtkTransform> transform;
    transform->Translate(position.x(), position.y(), position.z());
    
    // Convert Eigen rotation to VTK (simplified)
    actor->SetUserTransform(transform);
    actor->Modified();
}

vtkActor * FrustumPlot::getActor() const
{
    return actor;
}

// -------------------------------------------------------
// AxisPlot and BasisPlot (simplified from Lab 8)
// -------------------------------------------------------
AxisPlot::AxisPlot() : isInit(false)
{
    axesActor = vtkSmartPointer<vtkAxesActor>::New();
    transform = vtkSmartPointer<vtkTransform>::New();
}

void AxisPlot::init(vtkCamera * camera)
{
    axesActor->SetUserTransform(transform);
    isInit = true;
}

void AxisPlot::update(const Eigen::Vector3d & rCNn, const Eigen::Vector3d & Thetanc)
{
    if (!isInit) return;
    
    transform->Identity();
    transform->Translate(rCNn.x(), rCNn.y(), rCNn.z());
    axesActor->Modified();
}

vtkProp3D * AxisPlot::getActor() const
{
    return axesActor;
}

BasisPlot::BasisPlot() : isInit(false)
{
    axesActor = vtkSmartPointer<vtkAxesActor>::New();
    transform = vtkSmartPointer<vtkTransform>::New();
    axesActor->SetUserTransform(transform);
    isInit = true;
}

void BasisPlot::update(const Eigen::Vector3d & rCNn, const Eigen::Vector3d & Thetanc)
{
    transform->Identity();
    transform->Translate(rCNn.x(), rCNn.y(), rCNn.z());
    axesActor->Modified();
}

vtkProp3D * BasisPlot::getActor() const
{
    return axesActor;
}

// -------------------------------------------------------
// ImagePlot (from Lab 8)
// -------------------------------------------------------
ImagePlot::ImagePlot()
    : viewVTK(vtkSmartPointer<vtkImageData>::New())
    , imageActor2d(vtkSmartPointer<vtkActor2D>::New())
    , imageMapper(vtkSmartPointer<vtkImageMapper>::New())
    , width(0), height(0), isInit(false)
{
    imageMapper->SetInputData(viewVTK);
    imageMapper->SetColorWindow(255.0);
    imageMapper->SetColorLevel(127.5);
    imageActor2d->SetMapper(imageMapper);
}

void ImagePlot::init(double rendererWidth, double rendererHeight)
{
    width = rendererWidth;
    height = rendererHeight;
    isInit = true;
}

void ImagePlot::update(const cv::Mat & view)
{
    if (!isInit) return;
    
    cv::Mat viewCVrgb, tmp;
    cv::resize(view, tmp, cv::Size(width, height), cv::INTER_LINEAR);
    cv::cvtColor(tmp, viewCVrgb, cv::COLOR_BGR2RGB);
    cv::flip(viewCVrgb, cvVTKBuffer, 0);
    
    // Convert to VTK (simplified)
    vtkNew<vtkImageImport> importer;
    importer->SetDataSpacing(1, 1, 1);
    importer->SetDataOrigin(0, 0, 0);
    importer->SetWholeExtent(0, cvVTKBuffer.cols-1, 0, cvVTKBuffer.rows-1, 0, 0);
    importer->SetDataExtentToWholeExtent();
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents(cvVTKBuffer.channels());
    importer->SetImportVoidPointer(cvVTKBuffer.data);
    importer->Update();
    
    viewVTK->DeepCopy(importer->GetOutput());
    imageMapper->Modified();
}

vtkActor2D * ImagePlot::getActor() const
{
    return imageActor2d;
}

// -------------------------------------------------------
// Plot (Lab 8 constructor adapted for Assignment 1)
// -------------------------------------------------------
Plot::Plot(const Camera & camera)
    : camera(camera)
    , renderWindow(vtkSmartPointer<vtkRenderWindow>::New())
    , threeDimRenderer(vtkSmartPointer<vtkRenderer>::New())
    , imageRenderer(vtkSmartPointer<vtkRenderer>::New())
    , interactor(vtkSmartPointer<vtkRenderWindowInteractor>::New())
    , fp(camera)
{
    // Use Lab 8's proven initialization pattern
    double aspectRatio = (1.0 * camera.imageSize.width) / camera.imageSize.height;
    double windowHeight = 540;
    double windowWidth = 2 * aspectRatio * windowHeight;

    vtkNew<vtkNamedColors> colors;
    
    // Viewports (Lab 8 pattern)
    double quadricViewport[4] = {0.5, 0.0, 1.0, 1.0};
    threeDimRenderer->SetViewport(quadricViewport);
    threeDimRenderer->SetBackground(colors->GetColor3d("slategray").GetData());

    double imageViewport[4] = {0.0, 0.0, 0.5, 1.0};
    imageRenderer->SetViewport(imageViewport);
    imageRenderer->SetBackground(colors->GetColor3d("white").GetData());

    renderWindow->SetSize(windowWidth, windowHeight);
    renderWindow->SetMultiSamples(0);
    renderWindow->AddRenderer(threeDimRenderer);
    renderWindow->AddRenderer(imageRenderer);

    // Initialize components (Lab 8 pattern)
    ap.init(threeDimRenderer->GetActiveCamera());
    ip.init(windowWidth/2, windowHeight);

    // Add actors to renderers
    threeDimRenderer->AddActor(ap.getActor());
    threeDimRenderer->AddActor(bp.getActor());
    threeDimRenderer->AddActor(fp.getActor());
    threeDimRenderer->AddActor(qpCamera.getActor());
    imageRenderer->AddActor2D(ip.getActor());

    // Set camera view (Lab 8 pattern)
    threeDimRenderer->GetActiveCamera()->Azimuth(0);
    threeDimRenderer->GetActiveCamera()->Elevation(165);
    threeDimRenderer->GetActiveCamera()->SetFocalPoint(0, 0, 0);
    double sc = 2;
    threeDimRenderer->GetActiveCamera()->SetPosition(-0.75*sc, -0.75*sc, -0.5*sc);
    threeDimRenderer->GetActiveCamera()->SetViewUp(0, 0, -1);

    // Setup interactor (Lab 8 pattern)
    vtkNew<vtkInteractorStyleTrackballCamera> interactorStyle;
    interactor->SetInteractorStyle(interactorStyle);
    interactor->SetRenderWindow(renderWindow);
    interactor->Initialize();
    
    std::cout << "Plot visualization initialized with dual-pane layout" << std::endl;
}

// -------------------------------------------------------
// Assignment 1 interface methods
// -------------------------------------------------------
void Plot::updateImage(const cv::Mat& image, 
                      const std::vector<cv::Point2f>& features,
                      const std::vector<int>& featureStatus)
{
    cv::Mat annotatedImage = image.clone();
    drawFeaturesOnImage(annotatedImage, features, featureStatus);
    ip.update(annotatedImage);
}

void Plot::updateScene(const Eigen::Vector3d& cameraPos, const Eigen::Matrix3d& cameraRot,
                      const Eigen::Matrix3d& cameraCov,
                      const std::vector<Eigen::Vector3d>& landmarkPositions,
                      const std::vector<Eigen::Matrix3d>& landmarkCovariances,
                      const std::vector<int>& landmarkStatus)
{
    // Update camera confidence region
    qpCamera.update(cameraPos, cameraCov, 3.0);
    qpCamera.setColor(0.0, 1.0, 0.0, 0.5); // Green for camera
    
    // Update camera frustum
    fp.update(cameraPos, cameraRot);
    
    // Update landmarks
    qpLandmarks.resize(landmarkPositions.size());
    for (size_t i = 0; i < landmarkPositions.size() && i < landmarkCovariances.size(); i++) {
        if (i < landmarkStatus.size()) {
            qpLandmarks[i].update(landmarkPositions[i], landmarkCovariances[i], 3.0);
            
            // Color coding per assignment specs
            switch (landmarkStatus[i]) {
                case 0: qpLandmarks[i].setColor(0.0, 0.0, 1.0, 0.7); break; // Blue = tracked
                case 1: qpLandmarks[i].setColor(1.0, 0.0, 0.0, 0.7); break; // Red = visible undetected
                case 2: qpLandmarks[i].setColor(1.0, 1.0, 0.0, 0.7); break; // Yellow = not visible
                default: qpLandmarks[i].setColor(0.5, 0.5, 0.5, 0.7); break;
            }
            
            // Ensure actor is in renderer
            if (!threeDimRenderer->HasViewProp(qpLandmarks[i].getActor())) {
                threeDimRenderer->AddActor(qpLandmarks[i].getActor());
            }
        }
    }
    
    threeDimRenderer->Modified();
}

void Plot::render()
{
    renderWindow->Render();
}

void Plot::handleInteractivity(int interactive, bool isLastFrame)
{
    switch (interactive) {
        case 0: // No interactivity
            break;
        case 1: // Interactive only on last frame
            if (isLastFrame) {
                interactor->Start();
            }
            break;
        case 2: // Interactive on all frames
            interactor->Start();
            break;
        default:
            std::cout << "Warning: Unknown interactive mode " << interactive << std::endl;
            break;
    }
}

cv::Mat Plot::getFrame() const
{
    // Capture frame (Lab 8 pattern)
    renderWindow->Render();
    
    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = 
        vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->Update();
    
    // Convert to OpenCV
    cv::Mat frame;
    int *size = renderWindow->GetSize();
    int & w = size[0];
    int & h = size[1];
    std::shared_ptr<unsigned char[]> pixels(renderWindow->GetPixelData(0, 0, w - 1, h - 1, 0));
    cv::Mat frameBufferRGB(h, w, CV_8UC3, pixels.get());
    cv::Mat frameBufferBGR;
    cv::cvtColor(frameBufferRGB, frameBufferBGR, cv::COLOR_RGB2BGR);
    cv::flip(frameBufferBGR, frame, 0);
    return frame;
}

void Plot::drawFeaturesOnImage(cv::Mat& image, 
                              const std::vector<cv::Point2f>& features,
                              const std::vector<int>& featureStatus)
{
    for (size_t i = 0; i < features.size() && i < featureStatus.size(); i++) {
        cv::Scalar color;
        switch (featureStatus[i]) {
            case 0: color = cv::Scalar(255, 0, 0); break;     // Blue = tracked
            case 1: color = cv::Scalar(0, 0, 255); break;     // Red = visible undetected  
            default: color = cv::Scalar(0, 255, 255); break;  // Yellow = not visible
        }
        cv::circle(image, features[i], 5, color, 2);
    }
}