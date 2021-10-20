#include <iostream>
#include "LightGBM/c_api.h"

int main(int argc, char* argv[]) {
    int model_num_trees;
    int num_features;
    BoosterHandle model_handle;
    DatasetHandle data_handle;

    const std::string model_filename = "C:\\Users\\jeomhove\\source\\lightgbm-benchmark\\data\\tests\\fusionmodel.txt";
    const std::string data_filename = "C:\\Users\\jeomhove\\source\\lightgbm-benchmark\\data\\tests\\data.txt";

    std::cout << "Hello, world!\n";

    if (LGBM_BoosterCreateFromModelfile(model_filename.c_str(), &model_num_trees, &model_handle) != 0) {
        throw std::runtime_error("Could not load LightGBM model from file " + model_filename);
    } else {
        std::cout << "Successfully loaded LightGBM model from file" + model_filename + "\n";
    }

    if (LGBM_BoosterGetNumFeature(model_handle, &num_features) != 0) {
        throw std::runtime_error("Could not get number of features from model");
    } else {
        std::cout << "num_features=" << num_features << "\n";
    }

    if (LGBM_DatasetCreateFromFile(data_filename.c_str(), "", nullptr, &data_handle) != 0) {
        throw std::runtime_error("Could not load LightGBM data from file " + data_filename);
    } else {
        std::cout << "Successfully loaded LightGBM data from file" + data_filename + "\n";
    }

    //LGBM_BoosterPredictForCSRSingleRow(model_handle, )
    // LGBM_DatasetGetNumData(DatasetHandle handle, int *out)
    // LGBM_DatasetGetNumFeature(DatasetHandle handle, int *out)
    //FastConfigHandle fast_config;
    //if (LGBM_BoosterPredictForCSRSingleRowFastInit(model_handle, C_API_PREDICT_NORMAL, 0, 0, C_API_DTYPE_FLOAT32, num_features, nullptr, &fast_config) != 0) {
    //    throw std::runtime_error("Could not create fast config for model");
    //} else {
    //    std::cout << "Initialized for FastPredict\n";
    //}

    // LGBM_BoosterPredictForCSRSingleRowFast
    if (model_handle != nullptr) {
        std::cout << "Free booster\n";
        LGBM_BoosterFree(model_handle);
    }

    std::cout << "Done.\n";
}
