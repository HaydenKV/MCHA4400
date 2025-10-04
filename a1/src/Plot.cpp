#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <memory>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>


#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>

#define vtkRenderingContext2D_AUTOINIT 1(vtkRenderingContextOpenGL2)
#define vtkRenderingCore_AUTOINIT 3(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingOpenGL2)
#define vtkRenderingOpenGL2_AUTOINIT 1(vtkRenderingGL2PSOpenGL2)

#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkAxisFollower.h>
#include <vtkBMPWriter.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCaptionActor2D.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkColor.h>
#include <vtkContextInteractorStyle.h>
#include <vtkContourFilter.h>
#include <vtkCubeAxesActor.h>
#include <vtkDataSetMapper.h>
#include <vtkGeometryFilter.h>
#include <vtkImageActor.h>
#include <vtkImageCast.h>
#include <vtkImageConstantPad.h>
#include <vtkImageData.h>
#include <vtkImageGradient.h>
#include <vtkImageImport.h>
#include <vtkImageLuminance.h>
#include <vtkImageMapper.h>
#include <vtkInteractorStyleImage.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkJPEGWriter.h>
#include <vtkLine.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkOutlineFilter.h>
#include <vtkPlaneSource.h>
#include <vtkPNGWriter.h>
#include <vtkPNMWriter.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPostScriptWriter.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkPyramid.h>
#include <vtkQuadric.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSampleFunction.h>
#include <vtkSmartPointer.h>
#include <vtkStripper.h>
#include <vtkTextProperty.h>
#include <vtkThreshold.h>
#include <vtkTIFFWriter.h>
#include <vtkTransform.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkWindowToImageFilter.h>

// For compatibility with new VTK generic data arrays
#ifdef vtkGenericDataArray_h
#define InsertNextTupleValue InsertNextTypedTuple
#endif

#include "Camera.h"
#include "GaussianInfo.hpp"
#include "rotation.hpp"
#include "SystemSLAM.h"
#include "MeasurementSLAM.h"
#include "MeasurementSLAMPointBundle.h"
#include "MeasurementSLAMUniqueTagBundle.h"
#include "Plot.h"

// Forward declarations
static void hsv2rgb(const double & h, const double & s, const double & v, double & r, double & g, double & b);
static void plotGaussianConfidenceEllipse(cv::Mat & img, const GaussianInfo<double> & prQOi, const Eigen::Vector3d & color);
static void openCV2VTK(const cv::Mat & viewCVRGB, vtkImageData * viewVTK);

// NEW: project 3D landmark position covariance to 2D pixel covariance (3σ ellipse source)
static GaussianInfo<double>
projectLandmarkPosToPixel(const GaussianInfo<double>& posDen,
                          const Camera& camera,
                          const Eigen::Vector3d& rCNn,
                          const Eigen::Matrix3d& Rnc);

// -------------------------------------------------------
// Bounds
// -------------------------------------------------------

Bounds::Bounds()
    : xmin(0), xmax(0)
    , ymin(0), ymax(0)
    , zmin(0), zmax(0)
{}

void Bounds::getVTKBounds(double * bounds) const
{
    bounds[0] = xmin;
    bounds[1] = xmax;
    bounds[2] = ymin;
    bounds[3] = ymax;
    bounds[4] = zmin;
    bounds[5] = zmax;
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

void Bounds::calculateMaxMinSigmaPoints(const GaussianInfo<double> & positionDensity, const double sigma)
{
    assert(positionDensity.dim() == 3);

    // Marginals
    GaussianInfo px = positionDensity.marginal(Eigen::seqN(0, 1));
    GaussianInfo py = positionDensity.marginal(Eigen::seqN(1, 1));
    GaussianInfo pz = positionDensity.marginal(Eigen::seqN(2, 1));

    double mux = px.mean()(0);
    double muy = py.mean()(0);
    double muz = pz.mean()(0);

    double Sxx = px.sqrtCov()(0, 0);
    double Syy = py.sqrtCov()(0, 0);
    double Szz = pz.sqrtCov()(0, 0);

    xmin = mux - sigma*std::abs(Sxx);
    xmax = mux + sigma*std::abs(Sxx);

    ymin = muy - sigma*std::abs(Syy);
    ymax = muy + sigma*std::abs(Syy);

    zmin = muz - sigma*std::abs(Szz);
    zmax = muz + sigma*std::abs(Szz);
}

// -------------------------------------------------------
// QuadricPlot
// -------------------------------------------------------

QuadricPlot::QuadricPlot()
    : contourActor(vtkSmartPointer<vtkActor>::New())
    , contours(vtkSmartPointer<vtkContourFilter>::New())
    , contourMapper(vtkSmartPointer<vtkPolyDataMapper>::New())
    , quadric(vtkSmartPointer<vtkQuadric>::New())
    , sample(vtkSmartPointer<vtkSampleFunction>::New())
    , value(0.0)
    , isInit(false)
{
    int ns          = 25;
    sample->SetSampleDimensions(ns, ns, ns);
    sample->SetImplicitFunction(quadric);

    // create the 0 isosurface
    contours->SetInputConnection(sample->GetOutputPort());
    contours->GenerateValues(1, value, value);

    contourMapper->SetInputConnection(contours->GetOutputPort());
    contourMapper->ScalarVisibilityOff();

    contourActor->SetMapper(contourMapper);

    isInit = true;
}

void QuadricPlot::update(const GaussianInfo<double> & positionDensity)
{ 
    assert(isInit);

    assert(positionDensity.dim() == 3);

    bounds.calculateMaxMinSigmaPoints(positionDensity, 6);

    // Get quadric surface coefficients from Gaussian position density
    Eigen::Matrix4d Q = positionDensity.quadricSurface(3);
    double a0, a1, a2, a3, a4, a5, a6, a7, a8, a9;

    // pull out in easy to read form
    const Eigen::Matrix3d& L = Q.topLeftCorner<3,3>();    // Λ
    const Eigen::Vector3d  eta = -Q.topRightCorner<3,1>(); // since Q stores -η
    const double d = Q(3,3);
    
    a0 = L(0,0);     // TODO: Lab 8
    a1 = L(1,1);     // TODO: Lab 8
    a2 = L(2,2);     // TODO: Lab 8
    a3 = 2.0 * L(0,1);     // TODO: Lab 8
    a4 = 2.0 * L(1,2);     // TODO: Lab 8
    a5 = 2.0 * L(0,2);     // TODO: Lab 8
    a6 = -2.0 * eta(0);     // TODO: Lab 8
    a7 = -2.0 * eta(1);     // TODO: Lab 8
    a8 = -2.0 * eta(2);     // TODO: Lab 8
    a9 = d;     // TODO: Lab 8

    quadric->SetCoefficients(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);

    double boundsVTK[6];
    bounds.getVTKBounds(boundsVTK);
    sample->SetModelBounds(boundsVTK);
}

vtkActor * QuadricPlot::getActor() const
{ 
    assert(isInit);
    return contourActor;
}

// -------------------------------------------------------
// FrustumPlot
// -------------------------------------------------------

FrustumPlot::FrustumPlot(const Camera & camera)
    : pyramidActor(vtkSmartPointer<vtkActor>::New())
    , cells(vtkSmartPointer<vtkCellArray>::New())
    , mapper(vtkSmartPointer<vtkDataSetMapper>::New())
    , pyramidPts(vtkSmartPointer<vtkPoints>::New())
    , pyramid(vtkSmartPointer<vtkPyramid>::New())
    , ug(vtkSmartPointer<vtkUnstructuredGrid>::New())
    , rPCc(Eigen::MatrixXd::Zero(3,5))
    , rPNn(Eigen::MatrixXd::Zero(3,5))
    , isInit(false)
{
    int nu, nv;
    nu = camera.imageSize.width;
    nv = camera.imageSize.height;

    std::vector<cv::Point2f> p_cv;
    p_cv.push_back(cv::Point2f(   0,    0));
    p_cv.push_back(cv::Point2f(nu-1,    0));
    p_cv.push_back(cv::Point2f(nu-1, nv-1));
    p_cv.push_back(cv::Point2f(   0, nv-1));

    std::vector<cv::Point2f> rZCc2_cv;
    cv::undistortPoints(p_cv, rZCc2_cv, camera.cameraMatrix, camera.distCoeffs);

    Eigen::MatrixXd rZCc2(2,4);
    for (int i = 0; i < rZCc2.cols(); ++i)
    {
        rZCc2.col(i) << rZCc2_cv[i].x,  rZCc2_cv[i].y;
    } 

    Eigen::MatrixXd rZCc(3,4), nrZCc;
    rZCc.fill(1);
    rZCc.topRows(2)     = rZCc2;
    nrZCc               = rZCc.colwise().squaredNorm().cwiseSqrt();

    for (int i = 0; i < rZCc.cols(); ++i)
    {
        rZCc.col(i)            = rZCc.col(i) / nrZCc(0,i);
    }
    
    double d            = 0.5;
    rPCc.block(0,0,3,4) = d*rZCc;
    rPCc.block(0,4,3,1) << 0,0,0;

    pyramidPts->SetNumberOfPoints(5);

    pyramid->GetPointIds()->SetId(0, 0);
    pyramid->GetPointIds()->SetId(1, 1);
    pyramid->GetPointIds()->SetId(2, 2);
    pyramid->GetPointIds()->SetId(3, 3);
    pyramid->GetPointIds()->SetId(4, 4);

    cells->InsertNextCell(pyramid);

    ug->SetPoints(pyramidPts);
    ug->InsertNextCell(pyramid->GetCellType(), pyramid->GetPointIds());

    mapper->SetInputData(ug);

    vtkNew<vtkNamedColors> colors;
    pyramidActor->SetMapper(mapper);
    pyramidActor->GetProperty()->SetColor(colors->GetColor3d("Tomato").GetData());
    pyramidActor->GetProperty()->SetOpacity(0.1);   

    isInit = true;
}

void FrustumPlot::update(const Eigen::Vector3d & rCNn, const Eigen::Vector3d & Thetanc)
{   
    assert(isInit);
    Eigen::Matrix3d Rnc = rpy2rot(Thetanc);

    rPNn = (Rnc*rPCc).colwise() + rCNn;

    pyramidPts->SetPoint(0, rPNn.col(0).data());
    pyramidPts->SetPoint(1, rPNn.col(1).data());
    pyramidPts->SetPoint(2, rPNn.col(2).data());
    pyramidPts->SetPoint(3, rPNn.col(3).data());
    pyramidPts->SetPoint(4, rPNn.col(4).data());
    
    pyramidPts->Modified();
    ug->Modified();
    mapper->Modified();
}

vtkActor * FrustumPlot::getActor() const
{
    assert(isInit);
    return pyramidActor;
}


// -------------------------------------------------------
// AxisPlot
// -------------------------------------------------------

AxisPlot::AxisPlot()
    : cubeAxesActor(vtkSmartPointer<vtkCubeAxesActor>::New())
    , isInit(false)
{
    vtkNew<vtkNamedColors> colors;
    axis1Color = colors->GetColor3d("Salmon");
    axis2Color = colors->GetColor3d("PaleGreen");
    axis3Color = colors->GetColor3d("LightSkyBlue");
}

void AxisPlot::init(vtkCamera *cam)
{
    int fontsize    = 48;

    cubeAxesActor->SetCamera(cam);
    cubeAxesActor->GetTitleTextProperty(0)->SetColor(axis1Color.GetData());
    cubeAxesActor->GetTitleTextProperty(0)->SetFontSize(fontsize);
    cubeAxesActor->GetLabelTextProperty(0)->SetColor(axis1Color.GetData());
    cubeAxesActor->GetLabelTextProperty(0)->SetFontSize(fontsize);

    cubeAxesActor->GetTitleTextProperty(1)->SetColor(axis2Color.GetData());
    cubeAxesActor->GetTitleTextProperty(1)->SetFontSize(fontsize);
    cubeAxesActor->GetLabelTextProperty(1)->SetColor(axis2Color.GetData());
    cubeAxesActor->GetLabelTextProperty(1)->SetFontSize(fontsize);

    cubeAxesActor->GetTitleTextProperty(2)->SetColor(axis3Color.GetData());
    cubeAxesActor->GetTitleTextProperty(2)->SetFontSize(fontsize);
    cubeAxesActor->GetLabelTextProperty(2)->SetColor(axis3Color.GetData());
    cubeAxesActor->GetLabelTextProperty(2)->SetFontSize(fontsize);
    
    cubeAxesActor->SetXTitle("N - [m]");
    cubeAxesActor->SetYTitle("E - [m]");
    cubeAxesActor->SetZTitle("D - [m]");

    cubeAxesActor->XAxisMinorTickVisibilityOn();
    cubeAxesActor->YAxisMinorTickVisibilityOn();
    cubeAxesActor->ZAxisMinorTickVisibilityOn();

    // cubeAxesActor->SetFlyModeToStaticEdges();
    cubeAxesActor->SetFlyModeToFurthestTriad();
    cubeAxesActor->SetUseTextActor3D(1); 

    isInit  = true;
}

void AxisPlot::update(const Bounds & bounds)
{
    assert(isInit);

    double boundsVTK[6];
    bounds.getVTKBounds(boundsVTK);

    cubeAxesActor->SetBounds(boundsVTK);
}

vtkActor * AxisPlot::getActor() const
{
    assert(isInit);
    return cubeAxesActor;
}

// -------------------------------------------------------
// BasisPlot
// -------------------------------------------------------

BasisPlot::BasisPlot()
    : axesActor(vtkSmartPointer<vtkAxesActor>::New())
    , transform(vtkSmartPointer<vtkTransform>::New())
    , isInit(false)
{
    // Set the length of the axes
    axesActor->SetTotalLength(0.2, 0.2, 0.2); 

    // Disable axis label text
    axesActor->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->SetOpacity(0);
    axesActor->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->SetOpacity(0);
    axesActor->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->SetOpacity(0);

    // Set line thickness
    const double radius = 0.02;
    axesActor->SetShaftTypeToCylinder();
    axesActor->SetCylinderRadius(radius);

    // Set default transformation matrix and apply it to the axes actor
    transform->Identity();
    axesActor->SetUserTransform(transform);

    isInit = true;
}

void BasisPlot::update(const Eigen::Vector3d & rCNn, const Eigen::Vector3d & Thetanc)
{
    assert(isInit);
    
    // Convert Thetanc to a rotation matrix
    Eigen::Matrix3d Rnc = rpy2rot(Thetanc);

    // Update homogeneous transformation matrix via a map
    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> Tnc(transform->GetMatrix()->GetData());
    Tnc << Rnc, rCNn, 
           0, 0, 0, 1;

    // Notify VTK that the axes actor has been modified
    axesActor->Modified();
}

vtkProp3D * BasisPlot::getActor() const
{
    assert(isInit);
    return axesActor;
}

// -------------------------------------------------------
// ImagePlot
// -------------------------------------------------------

ImagePlot::ImagePlot()
    : viewVTK(vtkSmartPointer<vtkImageData>::New())
    , imageActor2d(vtkSmartPointer<vtkActor2D>::New())
    , imageMapper(vtkSmartPointer<vtkImageMapper>::New())
    , width(0)
    , height(0)
    , isInit(false)
{
    imageMapper->SetInputData(viewVTK);
    imageMapper->SetColorWindow(255.0);
    imageMapper->SetColorLevel(127.5);
    
    imageActor2d->SetMapper(imageMapper);
}

void ImagePlot::init(double rendererWidth, double rendererHeight)
{
    width   = rendererWidth;
    height  = rendererHeight;

    isInit  = true;
}

void ImagePlot::update(const cv::Mat & view)
{
    assert(isInit);

    cv::Mat viewCVrgb, tmp;
    cv::resize(view, tmp, cv::Size(static_cast<int>(width), static_cast<int>(height)), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(tmp, viewCVrgb, cv::COLOR_BGR2RGB);
    cv::flip(viewCVrgb, cvVTKBuffer, 0);
    openCV2VTK(cvVTKBuffer, viewVTK);
}

vtkActor2D * ImagePlot::getActor() const
{
    assert(isInit);
    return imageActor2d;
}


// -------------------------------------------------------
// Plot
// -------------------------------------------------------

void Plot::setData(const SystemSLAM & system, const MeasurementSLAM & measurement)
{
    pSystem.reset(system.clone());
    pMeasurement.reset(measurement.clone());
}


cv::Mat Plot::getFrame() const
{
    cv::Mat frame;
    int *size = renderWindow->GetSize();
    int & w = size[0];
    int & h = size[1];
    std::shared_ptr<unsigned char[]> pixels(renderWindow->GetPixelData(0, 0, w - 1, h - 1, 0));
    cv::Mat frameBufferRGB(h, w, CV_8UC3, pixels.get());
    cv::Mat frameBufferBGR;
    cv::cvtColor(frameBufferRGB, frameBufferBGR, cv::COLOR_RGB2BGR);
    cv::flip(frameBufferBGR, frame, 0); // Flip vertically
    return frame;
}

cv::Size Plot::renderSize() const
{
    int* size = renderWindow->GetSize();
    return cv::Size(size[0], size[1]);
}

Plot::Plot(const Camera & camera)
    : camera(camera)
    , renderWindow(vtkSmartPointer<vtkRenderWindow>::New())
    , threeDimRenderer(vtkSmartPointer<vtkRenderer>::New())
    , imageRenderer(vtkSmartPointer<vtkRenderer>::New())
    , interactor(vtkSmartPointer<vtkRenderWindowInteractor>::New())
    , fp(camera)
    , ap()
    , bp()
    , ip()
{
    double aspectRatio  = (1.0*camera.imageSize.width)/camera.imageSize.height;

    // double windowHeight       = 2*540;
    double windowHeight       = 1*540;
    double windowWidth        = 2*aspectRatio*windowHeight;

    vtkNew<vtkNamedColors> colors;
    double quadricViewport[4]       = {0.5, 0.0, 1.0, 1.0};
    threeDimRenderer->SetViewport(quadricViewport);
    threeDimRenderer->SetBackground(colors->GetColor3d("slategray").GetData());

    double imageViewport[4]         = {0.0, 0.0, 0.5, 1.0};
    imageRenderer->SetViewport(imageViewport);
    imageRenderer->SetBackground(colors->GetColor3d("white").GetData());

    renderWindow->SetSize(windowWidth, windowHeight);

    renderWindow->SetMultiSamples(0);
    renderWindow->AddRenderer(threeDimRenderer);
    renderWindow->AddRenderer(imageRenderer);

    ap.init(threeDimRenderer->GetActiveCamera());
    ip.init(windowWidth/2, windowHeight);

    // Quadric surfaces
    qpLandmarks.clear();

    threeDimRenderer->AddActor(ap.getActor());
    threeDimRenderer->AddActor(bp.getActor());
    threeDimRenderer->AddActor(fp.getActor());
    threeDimRenderer->AddActor(qpCamera.getActor());
    imageRenderer->AddActor2D(ip.getActor());

    // ----------------------------------------------------------------- Alter camera view setting for VTK (x, y, z)
    threeDimRenderer->GetActiveCamera()->Azimuth(0);
    threeDimRenderer->GetActiveCamera()->Elevation(200);
    // rFNn
    threeDimRenderer->GetActiveCamera()->SetFocalPoint(0,0,2);
    // rCNn
    double sc = 10; // Scale factor - increase to intially zoom out (was originally on 2)
    threeDimRenderer->GetActiveCamera()->SetPosition(0.15*sc,-0.15*sc,-0.6*sc);
    threeDimRenderer->GetActiveCamera()->SetViewUp(0,-1,0); // (0,0,-1)

    vtkNew<vtkInteractorStyleTrackballCamera> interactorStyle;
    interactor->SetInteractorStyle(interactorStyle);
    interactor->SetRenderWindow(renderWindow);
    interactor->Initialize();
}

void Plot::render()
{
    double r,g,b;   
    hsv2rgb(330, 1., 1., r, g, b);
    qpCamera.update(pSystem->cameraPositionDensity(camera));
    qpCamera.getActor()->GetProperty()->SetOpacity(0.1);
    qpCamera.getActor()->GetProperty()->SetColor(r,g,b);

    Bounds globalBounds;
    qpCamera.bounds.setExtremity(globalBounds); 

    // Grow/shrink landmark quadric plots to match number of landmarks
    while (qpLandmarks.size() < pSystem->numberLandmarks())
    {
        QuadricPlot qp;
        qpLandmarks.push_back(qp);
        threeDimRenderer->AddActor(qpLandmarks.back().getActor());
    }

    while (qpLandmarks.size() > pSystem->numberLandmarks())
    {
        threeDimRenderer->RemoveActor(qpLandmarks.back().getActor());
        qpLandmarks.pop_back();
    }    

    // Current camera pose (means)
    const Eigen::Vector3d rCNn    = pSystem->cameraPositionDensity(camera).mean();
    const Eigen::Vector3d Thetanc = pSystem->cameraOrientationEulerDensity(camera).mean();
    const Eigen::Matrix3d Rnc     = rpy2rot(Thetanc);
    const Eigen::Matrix3d Rcn     = Rnc.transpose();

    // Get association data
    const auto* pBundle = dynamic_cast<const MeasurementPointBundle*>(pMeasurement.get());
    const std::vector<int>* assocPtr = nullptr;
    if (pBundle) assocPtr = &pBundle->idxFeatures();

    // Visibility helper (geometric check only)
    auto isVisibleStrict = [&](const Eigen::Vector3d& rPNn)->bool
    {
        const Eigen::Vector3d rPCc = Rcn * (rPNn - rCNn);
        if (!std::isfinite(rPCc.x()) || !std::isfinite(rPCc.y()) || !std::isfinite(rPCc.z()))
            return false;
        if (rPCc.z() <= 1e-9) return false;

        const Eigen::Vector2d pix = camera.vectorToPixel(rPCc);
        if (!std::isfinite(pix.x()) || !std::isfinite(pix.y())) return false;

        return (pix.x() >= 0.0 && pix.x() < camera.imageSize.width &&
                pix.y() >= 0.0 && pix.y() < camera.imageSize.height);
    };

    // Process each landmark
    for (std::size_t i = 0; i < pSystem->numberLandmarks(); ++i)
    {
        // 3D position density
        const GaussianInfo<double> posDen = pSystem->landmarkPositionDensity(i);
        const Eigen::Vector3d rPNn_mean   = posDen.mean();

        // VISIBILITY: Geometric FOV check
        const bool visible = isVisibleStrict(rPNn_mean);

        // ASSOCIATION: Has persistent identity? (fixed!)
        bool associated = false;
        if (const auto* tagMeas = dynamic_cast<const MeasurementSLAMUniqueTagBundle*>(pMeasurement.get())) {
            associated = tagMeas->isEffectivelyAssociated(i);  // Uses tag ID now!
        } else if (assocPtr && i < assocPtr->size()) {
            associated = ((*assocPtr)[i] >= 0);
        }

        // COLOR SEMANTICS (corrected)
        double cr, cg, cb;
        if (!visible) {
            // Yellow: Behind camera or out of FOV
            cr = 1.0; cg = 1.0; cb = 0.0;
        } else if (associated) {
            // Blue: Has tag ID (part of map) - stays blue even during dropout!
            cr = 0.0; cg = 0.5; cb = 1.0;
        } else {
            // Red: In FOV but no tag ID (should never happen in Scenario 1)
            cr = 1.0; cg = 0.25; cb = 0.25;
        }

        Eigen::Vector3d rgb2d(cr*255.0, cg*255.0, cb*255.0);

        // ===== LEFT PANE =====
        if (visible) {
            // This correctly propagates ALL uncertainties (camera + landmark + measurement)
            GaussianInfo<double> prQOi = pMeasurement->predictFeatureDensity(*pSystem, i);
            plotGaussianConfidenceEllipse(pSystem->view(), prQOi, rgb2d);
        }

        // ===== RIGHT PANE =====
        QuadricPlot & qp = qpLandmarks[i];
        qp.update(posDen);
        qp.getActor()->GetProperty()->SetOpacity(0.5);
        qp.getActor()->GetProperty()->SetColor(cr, cg, cb);
        qp.bounds.setExtremity(globalBounds); 
    }

    ap.update(globalBounds);
    bp.update(rCNn, Thetanc);
    fp.update(rCNn, Thetanc);
    ip.update(pSystem->view());

    renderWindow->Render();
}


// void Plot::render()
// {
//     double r,g,b;   
//     hsv2rgb(330, 1., 1., r, g, b);
//     qpCamera.update(pSystem->cameraPositionDensity(camera));
//     qpCamera.getActor()->GetProperty()->SetOpacity(0.1);
//     qpCamera.getActor()->GetProperty()->SetColor(r,g,b);

//     Bounds globalBounds;
//     qpCamera.bounds.setExtremity(globalBounds); 

//     // Grow landmark quadric plots to match number of landmarks
//     while (qpLandmarks.size() < pSystem->numberLandmarks())
//     {
//         QuadricPlot qp;
//         qpLandmarks.push_back(qp);
//         threeDimRenderer->AddActor(qpLandmarks.back().getActor());
//     }

//     // Shrink landmark quadric plots to match number of landmarks
//     while (qpLandmarks.size() > pSystem->numberLandmarks())
//     {
//         threeDimRenderer->RemoveActor(qpLandmarks.back().getActor());
//         qpLandmarks.pop_back();
//     }    

//     // --- camera pose (means) for this frame ---
//     Eigen::Vector3d rCNn   = pSystem->cameraPositionDensity(camera).mean();
//     Eigen::Vector3d Thetanc= pSystem->cameraOrientationEulerDensity(camera).mean();
//     Eigen::Matrix3d Rnc    = rpy2rot(Thetanc);

//     // Build Tnc and convert to Tnb via your helper
//     Posed Tnc;
//     Tnc.rotationMatrix    = Rnc;
//     Tnc.translationVector = rCNn;
//     Posed Tnb = camera.cameraToBody(Tnc); // Tnb = Tnc * Tcb^{-1}

//     // Per-frame associations from the active measurement (UniqueTagBundle derives from PointBundle)
//     const auto* pBundle = dynamic_cast<const MeasurementPointBundle*>(pMeasurement.get());
//     const std::vector<int>* assocPtr = nullptr;
//     if (pBundle) assocPtr = &pBundle->idxFeatures();

//     // Strict visibility check helper (front-facing + in-bounds pixel)
//     auto isVisibleStrict = [&](const Eigen::Vector3d& rPNn)->bool
//     {
//         // camera pose for this frame (already computed above)
//         // rCNn, Rnc available; need Rcn
//         const Eigen::Matrix3d Rcn = Rnc.transpose();

//         // point in camera frame
//         const Eigen::Vector3d rPCc = Rcn * (rPNn - rCNn);
//         if (!std::isfinite(rPCc.x()) || !std::isfinite(rPCc.y()) || !std::isfinite(rPCc.z()))
//             return false;

//         // strictly in front of camera
//         if (rPCc.z() <= 1e-9) return false;

//         // pixel in-bounds
//         const Eigen::Vector2d pix = camera.vectorToPixel(rPCc);
//         if (!std::isfinite(pix.x()) || !std::isfinite(pix.y())) return false;

//         const bool inBounds = (pix.x() >= 0.0 && pix.x() < camera.imageSize.width &&
//                             pix.y() >= 0.0 && pix.y() < camera.imageSize.height);
//         return inBounds;
//     };

//     for (std::size_t i = 0; i < pSystem->numberLandmarks(); ++i)
//     {
//         // predicted pixel covariance (left pane ellipse)
//         Eigen::Vector3d rgb2d;
//         double cr, cg, cb;

//         // Landmark position mean (tag centre for Scenario 1)
//         const GaussianInfo<double> posDen = pSystem->landmarkPositionDensity(i);
//         const Eigen::Vector3d rPNn_mean   = posDen.mean();

//         // VISIBILITY (strict, pixel-in-bounds)
//         const bool visible = isVisibleStrict(rPNn_mean);

//         // ASSOCIATED this frame (>=0 means detected & matched)
//         bool associated = false;

//         // Old way
//         // if (assocPtr && i < assocPtr->size())
//         //     associated = ((*assocPtr)[i] >= 0);

//         // Try to cast to unique tag bundle (supports grace period)
//         auto* tagMeas = dynamic_cast<const MeasurementSLAMUniqueTagBundle*>(pMeasurement.get());

//         if (tagMeas != nullptr) {
//             // Use grace period logic - stays blue during miss counter period
//             associated = tagMeas->isEffectivelyAssociated(i);
//         } else if (assocPtr && i < assocPtr->size()) {
//             // Fallback for other measurement types
//             associated = ((*assocPtr)[i] >= 0);
//         }

//         // Colour semantics:
//         // Blue   = visible + associated
//         // Red    = visible + !associated
//         // Yellow = !visible
//         if (!visible)        { cr = 1.0; cg = 1.0; cb = 0.0; }     // Yellow
//         else if (associated) { cr = 0.0; cg = 0.5; cb = 1.0; }     // Blue-ish
//         else                 { cr = 1.0; cg = 0.25; cb = 0.25; }   // Red-ish

//         // 2D ellipse colour
//         rgb2d(0) = cr*255.0;
//         rgb2d(1) = cg*255.0;
//         rgb2d(2) = cb*255.0;

//         GaussianInfo<double> prQOi = pMeasurement->predictFeatureDensity(*pSystem, i);
//         plotGaussianConfidenceEllipse(pSystem->view(), prQOi, rgb2d);

//         // 3D quadric colour
//         QuadricPlot & qp = qpLandmarks[i];
//         qp.update(posDen);
//         qp.getActor()->GetProperty()->SetOpacity(0.5);
//         qp.getActor()->GetProperty()->SetColor(cr, cg, cb);
//         qp.bounds.setExtremity(globalBounds); 
//     }

//     // Axes / frustum / image panes (unchanged)
//     ap.update(globalBounds);
//     bp.update(rCNn, Thetanc);
//     fp.update(rCNn, Thetanc);
//     ip.update(pSystem->view());

//     renderWindow->Render();
// }


void Plot::start() const
{
    interactor->Start(); // block on interactor
}


// -------------------------------------------------------
// Miscellaneous
// -------------------------------------------------------


// Inputs
// H \in [0, 360]
// S \in [0, 1]
// V \in [0, 1]
// Outputs
// R \in [0, 1]
// G \in [0, 1]
// B \in [0, 1]
void hsv2rgb(const double & h, const double & s, const double & v, double & r, double & g, double & b)
{
    assert(0 <= h && h <=  360.0);
    assert(0 <= s && s <=  1.0);
    assert(0 <= v && v <=  1.0);

    // https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB

    double  c, x, r1 = 0, g1 = 0, b1 = 0, m;
    int hp;
    // shift the hue to the range [0, 360] before performing calculations
    hp  = (int)(h / 60.);
    c   = v*s;
    x   = c * (1 - std::abs((hp % 2) - 1));

    switch(hp) {
        case 0: r1 = c; g1 = x; b1 = 0; break;
        case 1: r1 = x; g1 = c; b1 = 0; break;
        case 2: r1 = 0; g1 = c; b1 = x; break;
        case 3: r1 = 0; g1 = x; b1 = c; break;
        case 4: r1 = x; g1 = 0; b1 = c; break;
        case 5: r1 = c; g1 = 0; b1 = x; break;
    }
    m   = v - c;
    r   = r1 + m;
    g   = g1 + m;
    b   = b1 + m;
}

void openCV2VTK(const cv::Mat & viewCVRGB, vtkImageData* viewVTK)
{
    assert( viewCVRGB.data != NULL );

    vtkNew<vtkImageImport> importer;
    if ( viewVTK )
    {
        importer->SetOutput( viewVTK );
    }
    importer->SetDataSpacing( 1, 1, 1 );
    importer->SetDataOrigin( 0, 0, 0 );
    importer->SetWholeExtent(   0, viewCVRGB.size().width-1, 0,
                            viewCVRGB.size().height-1, 0, 0 );
    importer->SetDataExtentToWholeExtent();
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents( viewCVRGB.channels() );
    importer->SetImportVoidPointer( viewCVRGB.data );
    importer->Update();
}

void plotGaussianConfidenceEllipse(cv::Mat & img, const GaussianInfo<double> & prQOi, const Eigen::Vector3d & color)
{
    assert(prQOi.dim() == 2);

    int markerSize              = 24;
    int markerThickness         = 2;

    Eigen::MatrixXd rQOi_ellipse = prQOi.confidenceEllipse(3, 100);

    cv::Scalar bgr(color(2), color(1), color(0));

    Eigen::VectorXd murQOi = prQOi.mean();
    cv::drawMarker(img, cv::Point(murQOi(0), murQOi(1)), bgr,   cv::MARKER_CROSS,   markerSize, markerThickness);
    Eigen::VectorXd rQOi_seg1, rQOi_seg2;

    for (int i = 0; i < rQOi_ellipse.cols()-1; ++i)
    {
        rQOi_seg1   = rQOi_ellipse.col(i);
        rQOi_seg2   = rQOi_ellipse.col(i+1);

        bool isInWidth1  = 0 <= rQOi_seg1(0) && rQOi_seg1(0) <= img.cols-1;
        bool isInHeight1 = 0 <= rQOi_seg1(1) && rQOi_seg1(1) <= img.rows-1;
        
        bool isInWidth2  = 0 <= rQOi_seg2(0) && rQOi_seg2(0) <= img.cols-1;
        bool isInHeight2 = 0 <= rQOi_seg2(1) && rQOi_seg2(1) <= img.rows-1;
        bool plotLine   = isInWidth1 && isInHeight1 && isInWidth2 && isInHeight2;
        if (plotLine)
        {
            cv::line(img, 
                cv::Point(rQOi_seg1(0), rQOi_seg1(1)),
                cv::Point(rQOi_seg2(0), rQOi_seg2(1)),
                bgr,
                2);
        }
    }
}

// static void plotGaussianConfidenceEllipse(cv::Mat & img,
//                                           const GaussianInfo<double> & prQOi,
//                                           const Eigen::Vector3d & color)
// {
//     assert(prQOi.dim() == 2);

//     // k-sigma ellipse
//     const double k = 3.0;
//     const int N = 100;

//     // Recover covariance: Σ = S Sᵀ
//     const Eigen::Matrix2d S = prQOi.sqrtCov();
//     Eigen::Matrix2d Sigma = S * S.transpose();

//     // Numerical guard (in case of near-singular)
//     Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(Sigma);
//     Eigen::Vector2d evals = es.eigenvalues().cwiseMax(1e-12).cwiseSqrt(); // std devs
//     Eigen::Matrix2d U = es.eigenvectors();

//     const Eigen::Vector2d mu = prQOi.mean();

//     // Build polyline points
//     std::vector<cv::Point> pts;
//     pts.reserve(N);
//     for (int i = 0; i < N; ++i) {
//         const double th = 2.0 * M_PI * (static_cast<double>(i) / (N - 1));
//         const Eigen::Vector2d unit(std::cos(th), std::sin(th));
//         const Eigen::Vector2d xy = mu + k * (U * evals.asDiagonal() * unit);
//         pts.emplace_back(cvRound(xy.x()), cvRound(xy.y()));
//     }

//     // Color (BGR)
//     const cv::Scalar bgr(color(2), color(1), color(0));

//     // Draw center
//     cv::drawMarker(img, cv::Point(cvRound(mu.x()), cvRound(mu.y())),
//                    bgr, cv::MARKER_CROSS, 24, 2);

//     // Draw ellipse as a polyline (clip by bounds to avoid wrap artifacts)
//     for (int i = 0; i < N - 1; ++i) {
//         const cv::Point p1 = pts[i];
//         const cv::Point p2 = pts[i + 1];

//         const bool in1 = (0 <= p1.x && p1.x < img.cols && 0 <= p1.y && p1.y < img.rows);
//         const bool in2 = (0 <= p2.x && p2.x < img.cols && 0 <= p2.y && p2.y < img.rows);
//         if (in1 && in2) {
//             cv::line(img, p1, p2, bgr, 2, cv::LINE_AA);
//         }
//     }
// }
