#include "MeasurementSLAMUniqueTagBundle.h"
#include <unordered_map>

const std::vector<int>&
MeasurementSLAMUniqueTagBundle::associate(const SystemSLAM& system,
                                          const std::vector<std::size_t>& idxLandmarks)
{
    // Build a lookup: tag ID -> column index in Y_
    std::unordered_map<int,int> id2feat;
    id2feat.reserve(ids_.size());
    for (int i = 0; i < static_cast<int>(ids_.size()); ++i)
        id2feat.emplace(ids_[i], i);

    // Ensure mapping vector is at least number of landmarks long
    if (id_by_landmark_.size() < system.numberLandmarks())
        id_by_landmark_.resize(system.numberLandmarks(), -1);

    // Fill idxFeatures_ for ALL landmarks (size == numberLandmarks)
    idxFeatures_.assign(system.numberLandmarks(), -1);

    // For each landmark j, find its tag ID and map to the current feature column if present
    for (std::size_t j = 0; j < system.numberLandmarks(); ++j)
    {
        const int tagId = (j < id_by_landmark_.size()) ? id_by_landmark_[j] : -1;
        if (tagId < 0) continue;

        auto it = id2feat.find(tagId);
        if (it != id2feat.end())
            idxFeatures_[j] = it->second; // column index in Y_
    }

    return idxFeatures_;
}
