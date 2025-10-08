#ifndef DUCKDETECTORBASE_H
#define DUCKDETECTORBASE_H

#include <cstdint>
#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>

class DuckDetectorBase
{
public:
    virtual ~DuckDetectorBase();
    virtual cv::Mat detect(const cv::Mat & image) = 0;

    const std::vector<cv::Point2f>& last_centroids() const noexcept { return centroids_; }
    const std::vector<double>&      last_areas()     const noexcept { return areas_; }

protected:
    void preprocess(const cv::Mat & img, std::vector<float> & input_tensor_values);
    void postprocess(const std::vector<float> & class_scores_data, const std::vector<float> & mask_probs_data,
                     const std::vector<std::int64_t> & class_scores_shape, const std::vector<std::int64_t> & mask_probs_shape,
                     cv::Mat & imgout);

    std::vector<cv::Point2f> centroids_;
    std::vector<double>      areas_;
};

#endif
