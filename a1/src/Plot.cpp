// Plot.cpp - Fixed with proper Eigen includes and ellipsoid computation
#include "Plot.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vtkQuadric.h>
#include <vtkSampleFunction.h>
#include <vtkContourFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkLine.h>
#include <vtkImageImport.h>
#include <vtkImageActor.h>
#include <vtkImageMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkCamera.h>  // ADDED: Fix for vtkCamera incomplete type
#include <iostream>

// CRITICAL FIX: Add proper Eigen includes for eigenvalue decomposition
#include <Eigen/Dense>           // For Matrix operations
#include <Eigen/Eigenvalues>     // For SelfAdjointEigenSolver

// Based on Lab 8 confidence region implementation
void ConfidenceEllipsoidPlot::updateEllipsoid(const Eigen::Vector3d& center, 
                                             const Eigen::Matrix3d& covariance, 
                                             double sigma) {
    if (!isInitialized) {
        initialize();
    }
    
    // Compute eigenvalue decomposition for ellipsoid orientation and scaling
    // This is based on Lab 8 quadric surface computation
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    
    if (solver.info() != Eigen::Success) {
        std::cout << "Warning: Eigenvalue decomposition failed for confidence ellipsoid" << std::endl;
        return;
    }
    
    Eigen::Vector3d eigenvals = solver.eigenvalues();
    Eigen::Matrix3d eigenvecs = solver.eigenvectors();
    
    // Ensure positive eigenvalues (covariance should be positive definite)
    eigenvals = eigenvals.cwiseMax(1e-8);
    
    // Scale by confidence level (sigma)
    eigenvals *= (sigma * sigma);
    
    // Build quadric matrix Q for ellipsoid: (x-c)^T * C^(-1) * (x-c) = sigma^2
    // In homogeneous coordinates: [x y z 1] * Q * [x y z 1]^T = 0
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    
    // Inverse covariance scaled by sigma^2
    Eigen::Matrix3d C_inv = eigenvecs * eigenvals.asDiagonal().inverse() * eigenvecs.transpose();
    
    // Quadric coefficients (based on Lab 8 VTK quadric format)
    Q.block<3,3>(0,0) = C_inv;
    Q.block<3,1>(0,3) = -C_inv * center;
    Q.block<1,3>(3,0) = Q.block<3,1>(0,3).transpose();
    Q(3,3) = center.transpose() * C_inv * center - 1.0;
    
    // Set VTK quadric coefficients: a0x^2 + a1y^2 + a2z^2 + a3xy + a4yz + a5xz + a6x + a7y + a8z + a9 = 0
    double a0 = Q(0,0), a1 = Q(1,1), a2 = Q(2,2);
    double a3 = 2*Q(0,1), a4 = 2*Q(1,2), a5 = 2*Q(0,2);
    double a6 = 2*Q(0,3), a7 = 2*Q(1,3), a8 = 2*Q(2,3);
    double a9 = Q(3,3);
    
    quadric->SetCoefficients(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
    
    // Update sampling bounds around the ellipsoid center
    double maxEigenval = eigenvals.maxCoeff();
    double bounds = 3.0 * std::sqrt(maxEigenval); // 3-sigma bounds
    
    sample->SetModelBounds(center.x()-bounds, center.x()+bounds,
                          center.y()-bounds, center.y()+bounds, 
                          center.z()-bounds, center.z()+bounds);
    sample->Modified();
}

void ConfidenceEllipsoidPlot::initialize() {
    // Create VTK pipeline for quadric surface rendering (based on Lab 8)
    quadric = vtkSmartPointer<vtkQuadric>::New();
    sample = vtkSmartPointer<vtkSampleFunction>::New();
    contour = vtkSmartPointer<vtkContourFilter>::New();
    mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    actor = vtkSmartPointer<vtkActor>::New();
    
    // Set up sampling
    sample->SetImplicitFunction(quadric);
    sample->SetSampleDimensions(50, 50, 50);
    sample->ComputeNormalsOff();
    
    // Extract isosurface at level 0
    contour->SetInputConnection(sample->GetOutputPort());
    contour->SetValue(0, 0.0);
    
    // Set up mapper and actor
    mapper->SetInputConnection(contour->GetOutputPort());
    actor->SetMapper(mapper);
    
    // Default appearance
    setColor(0.0, 0.5, 1.0, 0.7); // Blue with transparency
    
    isInitialized = true;
}

void ConfidenceEllipsoidPlot::setColor(double r, double g, double b, double opacity) {
    if (actor) {
        actor->GetProperty()->SetColor(r, g, b);
        actor->GetProperty()->SetOpacity(opacity);
        actor->Modified();
    }
}

vtkActor* ConfidenceEllipsoidPlot::getActor() {
    return actor;
}

// Plot class implementation
Plot::Plot(const Camera& camera) : camera(camera) {
    // Initialize VTK rendering pipeline
    renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    
    // Left renderer for camera image
    leftRenderer = vtkSmartPointer<vtkRenderer>::New();
    leftRenderer->SetViewport(0.0, 0.0, 0.5, 1.0);
    leftRenderer->SetBackground(0.1, 0.1, 0.1);
    
    // Right renderer for 3D scene
    rightRenderer = vtkSmartPointer<vtkRenderer>::New(); 
    rightRenderer->SetViewport(0.5, 0.0, 1.0, 1.0);
    rightRenderer->SetBackground(0.2, 0.2, 0.2);
    
    renderWindow->AddRenderer(leftRenderer);
    renderWindow->AddRenderer(rightRenderer);
    renderWindow->SetSize(1280, 640); // Dual-pane layout
    
    // Initialize interactor
    interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(renderWindow);
    
    // Set up camera view for 3D scene
    rightRenderer->GetActiveCamera()->SetPosition(2, 2, 2);
    rightRenderer->GetActiveCamera()->SetFocalPoint(0, 0, 0);
    rightRenderer->GetActiveCamera()->SetViewUp(0, 0, 1);
    
    std::cout << "Plot visualization initialized with dual-pane layout" << std::endl;
}

void Plot::updateImage(const cv::Mat& image, const std::vector<cv::Point2f>& features, 
                      const std::vector<int>& featureStatus) {
    // Clone image for annotation
    cv::Mat annotatedImage = image.clone();
    
    // Draw detected features with color coding (based on assignment specs)
    for (size_t i = 0; i < features.size() && i < featureStatus.size(); i++) {
        cv::Scalar color;
        switch (featureStatus[i]) {
            case 0: color = cv::Scalar(255, 0, 0); break;     // Blue = tracked
            case 1: color = cv::Scalar(0, 0, 255); break;     // Red = visible undetected  
            default: color = cv::Scalar(0, 255, 255); break;  // Yellow = not visible
        }
        
        cv::circle(annotatedImage, features[i], 5, color, 2);
        
        // TODO: Add confidence ellipses around features (3σ)
        // This would use the confidence ellipse code from Lab 8
    }
    
    // Convert to VTK format and update left renderer
    updateImageDisplay(annotatedImage);
}

void Plot::updateScene(const Eigen::Vector3d& cameraPos, const Eigen::Matrix3d& cameraRot,
                      const Eigen::Matrix3d& cameraCov,
                      const std::vector<Eigen::Vector3d>& landmarkPositions,
                      const std::vector<Eigen::Matrix3d>& landmarkCovariances,
                      const std::vector<int>& landmarkStatus) {
    
    // Clear previous 3D scene
    rightRenderer->RemoveAllViewProps();
    
    // Add camera confidence ellipsoid
    if (cameraEllipsoid.isValid()) {
        cameraEllipsoid.updateEllipsoid(cameraPos, cameraCov, 3.0);
        cameraEllipsoid.setColor(0.0, 1.0, 0.0, 0.5); // Green for camera
        rightRenderer->AddActor(cameraEllipsoid.getActor());
    }
    
    // Add camera frustum
    if (cameraFrustum) {
        cameraFrustum->updatePose(cameraPos, cameraRot);
        rightRenderer->AddActor(cameraFrustum->getActor());
    }
    
    // Add landmark ellipsoids with color coding
    landmarkEllipsoids.resize(landmarkPositions.size());
    for (size_t i = 0; i < landmarkPositions.size() && i < landmarkCovariances.size(); i++) {
        if (i < landmarkStatus.size()) {
            landmarkEllipsoids[i].updateEllipsoid(landmarkPositions[i], landmarkCovariances[i], 3.0);
            
            // Color coding based on assignment specs
            switch (landmarkStatus[i]) {
                case 0: landmarkEllipsoids[i].setColor(0.0, 0.0, 1.0, 0.7); break; // Blue = tracked
                case 1: landmarkEllipsoids[i].setColor(1.0, 0.0, 0.0, 0.7); break; // Red = visible undetected
                case 2: landmarkEllipsoids[i].setColor(1.0, 1.0, 0.0, 0.7); break; // Yellow = not visible
                default: landmarkEllipsoids[i].setColor(0.5, 0.5, 0.5, 0.7); break;
            }
            
            rightRenderer->AddActor(landmarkEllipsoids[i].getActor());
        }
    }
    
    rightRenderer->Modified();
}

void Plot::render() {
    renderWindow->Render();
}

cv::Mat Plot::getFrame() {
    // Capture current visualization for video export
    renderWindow->Render();
    
    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = 
        vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->Update();
    
    // Convert VTK image to OpenCV Mat
    vtkImageData* vtkImage = windowToImageFilter->GetOutput();
    int dims[3];
    vtkImage->GetDimensions(dims);
    
    cv::Mat frame(dims[1], dims[0], CV_8UC3);
    unsigned char* vtkPtr = static_cast<unsigned char*>(vtkImage->GetScalarPointer());
    
    // Copy and flip (VTK has different coordinate system)
    for (int y = 0; y < dims[1]; y++) {
        for (int x = 0; x < dims[0]; x++) {
            int vtkIdx = (y * dims[0] + x) * 3;
            int cvIdx = ((dims[1] - 1 - y) * dims[0] + x) * 3;
            frame.data[cvIdx] = vtkPtr[vtkIdx + 2];     // B
            frame.data[cvIdx + 1] = vtkPtr[vtkIdx + 1]; // G  
            frame.data[cvIdx + 2] = vtkPtr[vtkIdx];     // R
        }
    }
    
    return frame;
}

void Plot::handleInteractivity(int interactive, bool isLastFrame) {
    // Handle interactive modes per assignment specification
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

void Plot::updateImageDisplay(const cv::Mat& image) {
    // Convert OpenCV Mat to VTK format for left pane display
    cv::Mat flippedImage;
    cv::flip(image, flippedImage, 0); // VTK coordinate system
    
    vtkSmartPointer<vtkImageImport> importer = vtkSmartPointer<vtkImageImport>::New();
    importer->SetDataSpacing(1, 1, 1);
    importer->SetDataOrigin(0, 0, 0);
    importer->SetWholeExtent(0, image.cols-1, 0, image.rows-1, 0, 0);
    importer->SetDataExtentToWholeExtent();
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents(image.channels());
    importer->SetImportVoidPointer(flippedImage.data);
    importer->Update();
    
    // Update image actor
    if (!imageActor) {
        imageActor = vtkSmartPointer<vtkImageActor>::New();
        leftRenderer->AddActor(imageActor);
    }
    
    imageActor->SetInputData(importer->GetOutput());
    leftRenderer->ResetCamera();
    leftRenderer->Modified();
}