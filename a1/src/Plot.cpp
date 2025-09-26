#include "Plot.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNamedColors.h>
#include <vtkImageData.h>
#include <vtkProperty.h>
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkPyramid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSetMapper.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkQuadric.h>
#include <vtkSampleFunction.h>
#include <vtkContourFilter.h>
#include <vtkCamera.h>

// ===============================================
// ConfidenceEllipsoidPlot Implementation (Lab 8)
// ===============================================

ConfidenceEllipsoidPlot::ConfidenceEllipsoidPlot() : isInitialized(false) {
    initialize();
}

ConfidenceEllipsoidPlot::~ConfidenceEllipsoidPlot() = default;

void ConfidenceEllipsoidPlot::initialize() {
    actor = vtkSmartPointer<vtkActor>::New();
    mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    quadric = vtkSmartPointer<vtkQuadric>::New();
    sample = vtkSmartPointer<vtkSampleFunction>::New();
    contour = vtkSmartPointer<vtkContourFilter>::New();
    
    // Setup VTK pipeline (Lab 8 pattern)
    sample->SetImplicitFunction(quadric);
    sample->SetSampleDimensions(25, 25, 25);
    
    contour->SetInputConnection(sample->GetOutputPort());
    contour->GenerateValues(1, 0.0, 0.0);
    
    mapper->SetInputConnection(contour->GetOutputPort());
    mapper->ScalarVisibilityOff();
    
    actor->SetMapper(mapper);
    
    isInitialized = true;
}

void ConfidenceEllipsoidPlot::updateEllipsoid(const Eigen::Vector3d& center, 
                                             const Eigen::Matrix3d& covariance, double sigma) {
    if (!isInitialized) return;
    
    // Compute quadric coefficients for 3σ ellipsoid: (x-μ)ᵀ Σ⁻¹ (x-μ) = σ²
    // Following Lab 8 quadricSurface implementation
    Eigen::Matrix3d L = covariance.inverse() / (sigma * sigma);
    
    double a0 = L(0,0), a1 = L(1,1), a2 = L(2,2);
    double a3 = 2.0 * L(0,1), a4 = 2.0 * L(1,2), a5 = 2.0 * L(0,2);
    double a6 = -2.0 * (L.row(0).dot(center));
    double a7 = -2.0 * (L.row(1).dot(center)); 
    double a8 = -2.0 * (L.row(2).dot(center));
    double a9 = center.transpose() * L * center - sigma * sigma;
    
    quadric->SetCoefficients(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
    
    // Set sampling bounds based on covariance
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    Eigen::Vector3d eigenvals = solver.eigenvalues();
    Eigen::Vector3d extents = sigma * eigenvals.cwiseSqrt();
    
    double bounds[6] = {
        center.x() - extents.x(), center.x() + extents.x(),
        center.y() - extents.y(), center.y() + extents.y(),
        center.z() - extents.z(), center.z() + extents.z()
    };
    sample->SetModelBounds(bounds);
}

void ConfidenceEllipsoidPlot::setColor(double r, double g, double b, double opacity) {
    actor->GetProperty()->SetColor(r, g, b);
    actor->GetProperty()->SetOpacity(opacity);
}

vtkActor* ConfidenceEllipsoidPlot::getActor() {
    return actor;
}

// ===============================================
// CameraFrustumPlot Implementation (Lab 8)
// ===============================================

CameraFrustumPlot::CameraFrustumPlot(const Camera& cam) : camera(cam) {
    initializeFrustum(cam);
}

CameraFrustumPlot::~CameraFrustumPlot() = default;

void CameraFrustumPlot::initializeFrustum(const Camera& cam) {
    actor = vtkSmartPointer<vtkActor>::New();
    mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    points = vtkSmartPointer<vtkPoints>::New();
    grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    cells = vtkSmartPointer<vtkCellArray>::New();
    
    // Create frustum geometry (Lab 8 pattern)
    double frustumDepth = 0.3;
    
    // Image corners
    std::vector<cv::Point2f> imageCorners = {
        {0.0f, 0.0f},
        {float(cam.imageSize.width-1), 0.0f},
        {float(cam.imageSize.width-1), float(cam.imageSize.height-1)},
        {0.0f, float(cam.imageSize.height-1)}
    };
    
    // Unproject to normalized coordinates
    std::vector<cv::Point2f> normCorners;
    cv::undistortPoints(imageCorners, normCorners, cam.cameraMatrix, cam.distCoeffs);
    
    // Setup frustum points
    points->SetNumberOfPoints(5);
    points->SetPoint(4, 0, 0, 0); // Camera center
    
    for (int i = 0; i < 4; i++) {
        points->SetPoint(i, 
            normCorners[i].x * frustumDepth,
            normCorners[i].y * frustumDepth,
            frustumDepth);
    }
    
    // Setup pyramid connectivity
    auto pyramid = vtkSmartPointer<vtkPyramid>::New();
    for (int i = 0; i < 5; i++) {
        pyramid->GetPointIds()->SetId(i, i);
    }
    
    grid->SetPoints(points);
    grid->InsertNextCell(pyramid->GetCellType(), pyramid->GetPointIds());
    
    mapper->SetInputData(grid);
    actor->SetMapper(mapper);
    
    // Set appearance
    vtkNew<vtkNamedColors> colors;
    actor->GetProperty()->SetColor(colors->GetColor3d("Orange").GetData());
    actor->GetProperty()->SetOpacity(0.3);
    actor->GetProperty()->SetRepresentationToWireframe();
}

void CameraFrustumPlot::updatePose(const Eigen::Vector3d& position, const Eigen::Matrix3d& rotation) {
    // Transform frustum points by camera pose
    Eigen::Matrix<double, 3, 5> localPoints;
    for (int i = 0; i < 5; i++) {
        double p[3];
        points->GetPoint(i, p);
        localPoints.col(i) = Eigen::Vector3d(p[0], p[1], p[2]);
    }
    
    Eigen::Matrix<double, 3, 5> worldPoints = (rotation * localPoints).colwise() + position;
    
    for (int i = 0; i < 5; i++) {
        points->SetPoint(i, worldPoints.col(i).data());
    }
    
    points->Modified();
    grid->Modified();
}

vtkActor* CameraFrustumPlot::getActor() {
    return actor;
}

// ===============================================
// Main Plot Class Implementation
// ===============================================

Plot::Plot(const Camera& cam) : camera(cam) {
    cameraFrustum = nullptr;
    cameraEllipsoid = nullptr;
    
    setupWindow();
    setupImagePane();
    setupScenePane();
    
    // Initialize 3D elements
    cameraFrustum = new CameraFrustumPlot(camera);
    cameraEllipsoid = new ConfidenceEllipsoidPlot();
    
    // Add to scene
    sceneRenderer->AddActor(cameraFrustum->getActor());
    sceneRenderer->AddActor(cameraEllipsoid->getActor());
    
    std::cout << "VTK dual-pane visualization initialized" << std::endl;
}

Plot::~Plot() {
    delete cameraFrustum;
    delete cameraEllipsoid;
    
    for (auto* ellipsoid : landmarkEllipsoids) {
        delete ellipsoid;
    }
    
    if (renderWindow) {
        renderWindow->Finalize();
    }
}

void Plot::setupWindow() {
    // Calculate dual-pane window size
    double aspectRatio = double(camera.imageSize.width) / camera.imageSize.height;
    int windowHeight = 540;
    int windowWidth = int(2 * aspectRatio * windowHeight); // 2x for dual panes
    
    // Create VTK components
    renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    imageRenderer = vtkSmartPointer<vtkRenderer>::New();
    sceneRenderer = vtkSmartPointer<vtkRenderer>::New();
    interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    windowFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
    
    // Setup dual-pane viewports (assignment requirement)
    double imageViewport[4] = {0.0, 0.0, 0.5, 1.0};  // Left half
    double sceneViewport[4] = {0.5, 0.0, 1.0, 1.0};  // Right half
    
    imageRenderer->SetViewport(imageViewport);
    sceneRenderer->SetViewport(sceneViewport);
    
    // Set backgrounds per assignment
    vtkNew<vtkNamedColors> colors;
    imageRenderer->SetBackground(colors->GetColor3d("White").GetData());
    sceneRenderer->SetBackground(colors->GetColor3d("SlateGray").GetData());
    
    // Configure window
    renderWindow->SetSize(windowWidth, windowHeight);
    renderWindow->SetMultiSamples(0);
    renderWindow->AddRenderer(imageRenderer);
    renderWindow->AddRenderer(sceneRenderer);
    renderWindow->SetWindowName("MCHA4400 Assignment 1 - SLAM Visualization");
    
    // Setup interactor
    vtkNew<vtkInteractorStyleTrackballCamera> style;
    interactor->SetInteractorStyle(style);
    interactor->SetRenderWindow(renderWindow);
    interactor->Initialize();
    
    // Setup frame capture
    windowFilter->SetInput(renderWindow);
    windowFilter->ReadFrontBufferOff();
}

void Plot::setupImagePane() {
    // Image display components
    imageActor = vtkSmartPointer<vtkImageActor>::New();
    imageImporter = vtkSmartPointer<vtkImageImport>::New();
    
    imageActor->SetInputData(vtkSmartPointer<vtkImageData>::New());
    imageRenderer->AddActor(imageActor);
    
    // 2D camera for image display
    imageRenderer->GetActiveCamera()->ParallelProjectionOn();
}

void Plot::setupScenePane() {
    // Setup 3D scene camera
    sceneRenderer->GetActiveCamera()->SetPosition(-1.5, -1.5, -1.0);
    sceneRenderer->GetActiveCamera()->SetFocalPoint(0, 0, 0);
    sceneRenderer->GetActiveCamera()->SetViewUp(0, 0, -1);
}

void Plot::updateImage(const cv::Mat& image, const std::vector<cv::Point2f>& features, 
                      const std::vector<int>& featureStatus) {
    // Create image with feature overlays
    currentImageWithOverlays = image.clone();
    drawConfidenceEllipses(currentImageWithOverlays, features, featureStatus);
    
    // Convert to VTK format
    cv::Mat rgbImage;
    if (currentImageWithOverlays.channels() == 3) {
        cv::cvtColor(currentImageWithOverlays, rgbImage, cv::COLOR_BGR2RGB);
    } else if (currentImageWithOverlays.channels() == 1) {
        cv::cvtColor(currentImageWithOverlays, rgbImage, cv::COLOR_GRAY2RGB);
    } else {
        rgbImage = currentImageWithOverlays;
    }
    
    // Flip for VTK coordinate system
    cv::Mat flippedImage;
    cv::flip(rgbImage, flippedImage, 0);
    
    // Update VTK image
    imageImporter->SetDataSpacing(1, 1, 1);
    imageImporter->SetDataOrigin(0, 0, 0);
    imageImporter->SetWholeExtent(0, rgbImage.cols-1, 0, rgbImage.rows-1, 0, 0);
    imageImporter->SetDataExtentToWholeExtent();
    imageImporter->SetDataScalarTypeToUnsignedChar();
    imageImporter->SetNumberOfScalarComponents(3);
    imageImporter->SetImportVoidPointer(flippedImage.data);
    imageImporter->Update();
    
    imageActor->SetInputData(imageImporter->GetOutput());
    
    // Setup 2D view
    imageRenderer->ResetCamera();
    imageRenderer->GetActiveCamera()->SetPosition(rgbImage.cols/2.0, rgbImage.rows/2.0, 1000);
    imageRenderer->GetActiveCamera()->SetFocalPoint(rgbImage.cols/2.0, rgbImage.rows/2.0, 0);
    imageRenderer->GetActiveCamera()->SetViewUp(0, 1, 0);
    imageRenderer->GetActiveCamera()->ParallelProjectionOn();
    double scale = std::max(rgbImage.cols, rgbImage.rows) / 2.0;
    imageRenderer->GetActiveCamera()->SetParallelScale(scale);
}

void Plot::updateScene(const Eigen::Vector3d& cameraPos, const Eigen::Matrix3d& cameraRot,
                      const Eigen::Matrix3d& cameraCov, const std::vector<Eigen::Vector3d>& landmarkPositions,
                      const std::vector<Eigen::Matrix3d>& landmarkCovariances,
                      const std::vector<int>& landmarkStatus) {
    // Update camera frustum and ellipsoid
    if (cameraFrustum) {
        cameraFrustum->updatePose(cameraPos, cameraRot);
    }
    
    if (cameraEllipsoid) {
        cameraEllipsoid->updateEllipsoid(cameraPos, cameraCov);
        cameraEllipsoid->setColor(0.0, 1.0, 0.0, 0.5); // Green camera ellipsoid
    }
    
    // Update landmark ellipsoids
    updateLandmarkEllipsoids(landmarkPositions, landmarkCovariances, landmarkStatus);
}

void Plot::drawConfidenceEllipses(cv::Mat& image, const std::vector<cv::Point2f>& features,
                                 const std::vector<int>& status) {
    // Draw features with assignment color coding
    for (size_t i = 0; i < features.size(); i++) {
        cv::Scalar color;
        int statusCode = (i < status.size()) ? status[i] : 0;
        
        switch (statusCode) {
            case 0: color = cv::Scalar(255, 0, 0); break;   // Blue for tracked
            case 1: color = cv::Scalar(0, 0, 255); break;   // Red for visible undetected
            default: color = cv::Scalar(128, 128, 128); break;
        }
        
        // Draw feature marker
        cv::circle(image, features[i], 5, color, 2);
        
        // TODO: Add confidence ellipse projection from 3D covariance
        // For now, draw simple circles as placeholder
        cv::circle(image, features[i], 15, color, 1);
    }
}

void Plot::updateLandmarkEllipsoids(const std::vector<Eigen::Vector3d>& positions,
                                   const std::vector<Eigen::Matrix3d>& covariances,
                                   const std::vector<int>& status) {
    // Remove old ellipsoids
    for (auto* ellipsoid : landmarkEllipsoids) {
        sceneRenderer->RemoveActor(ellipsoid->getActor());
        delete ellipsoid;
    }
    landmarkEllipsoids.clear();
    
    // Add new ellipsoids
    for (size_t i = 0; i < positions.size(); i++) {
        auto* ellipsoid = new ConfidenceEllipsoidPlot();
        
        Eigen::Matrix3d cov = (i < covariances.size()) ? 
            covariances[i] : 0.1 * Eigen::Matrix3d::Identity();
        
        ellipsoid->updateEllipsoid(positions[i], cov);
        
        // Assignment color coding
        int statusCode = (i < status.size()) ? status[i] : 0;
        switch (statusCode) {
            case 0: ellipsoid->setColor(0.0, 0.0, 1.0, 0.7); break; // Blue tracked
            case 1: ellipsoid->setColor(1.0, 0.0, 0.0, 0.7); break; // Red visible undetected
            case 2: ellipsoid->setColor(1.0, 1.0, 0.0, 0.7); break; // Yellow not visible
            default: ellipsoid->setColor(0.5, 0.5, 0.5, 0.5); break;
        }
        
        sceneRenderer->AddActor(ellipsoid->getActor());
        landmarkEllipsoids.push_back(ellipsoid);
    }
}

void Plot::render() {
    renderWindow->Render();
}

cv::Mat Plot::getFrame() {
    // Capture dual-pane visualization
    windowFilter->Modified();
    windowFilter->Update();
    
    vtkImageData* vtkImage = windowFilter->GetOutput();
    int dims[3];
    vtkImage->GetDimensions(dims);
    
    cv::Mat frame(dims[1], dims[0], CV_8UC3);
    unsigned char* vtkPixels = static_cast<unsigned char*>(vtkImage->GetScalarPointer());
    
    // Convert VTK to OpenCV format
    for (int y = 0; y < dims[1]; y++) {
        for (int x = 0; x < dims[0]; x++) {
            int vtkIndex = ((dims[1] - 1 - y) * dims[0] + x) * 3;
            int cvIndex = (y * dims[0] + x) * 3;
            frame.data[cvIndex + 2] = vtkPixels[vtkIndex + 0]; // R -> B
            frame.data[cvIndex + 1] = vtkPixels[vtkIndex + 1]; // G -> G
            frame.data[cvIndex + 0] = vtkPixels[vtkIndex + 2]; // B -> R
        }
    }
    
    return frame;
}

void Plot::handleInteractivity(int interactive, bool isLastFrame) {
    switch (interactive) {
        case 0: // No interaction
            break;
            
        case 1: // Interactive on last frame only
            if (isLastFrame) {
                std::cout << "Interactive mode: Final frame - explore 3D view with mouse" << std::endl;
                std::cout << "Press 'q' in VTK window to exit" << std::endl;
                interactor->Start();
            }
            break;
            
        case 2: // Interactive on every frame
            std::cout << "Interactive mode: Frame " << " - press 'q' to continue" << std::endl;
            interactor->Start();
            break;
    }
}