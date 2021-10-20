#include <iostream>
#include "LightGBM/c_api.h"

int main(int argc, char* argv[]) {
    int model_num_trees;
    int num_features;
    BoosterHandle model_handle;
    DatasetHandle data_handle;
    //std::vector<float*> sample_data{sample_data_row};
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

    // auto dataset = reinterpret_cast<Dataset*>(handle);
    // void CopySubrow(const Dataset* fullset, const data_size_t* used_indices, data_size_t num_used_indices, bool need_meta_data);
    // LGBM_BoosterPredictForMats(iterator->second->getHandle(), (const void**)features.second.data(), C_API_DTYPE_FLOAT32, nrows, ncols,
    //                               C_API_PREDICT_NORMAL, 0, "", &outputLength, outResult.data());

    float * sample_data_mat_row = new float[301];
    size_t sample_data_mat_num_features = 301;
    int64_t out_len = 0;
    double * out_result = new double[1];

    try {
        if (LGBM_BoosterPredictForMatSingleRow(model_handle, (void*)sample_data_mat_row, C_API_DTYPE_FLOAT32, sample_data_mat_num_features, 1, C_API_PREDICT_NORMAL, 0, 0, "", &out_len, out_result) != 0) {
            std::cout << "failed prediction for some reason";
        } else {
            std::cout << "prediction=" << out_result[0] << "\n";
        }
    } catch (std::exception e) {
        std::cout << "exception";
    }

    int32_t * sample_data_csr_row_headers = new int32_t[301];
    int32_t * sample_data_csr_indices = new int32_t[301];
    float * sample_data_csr_row = new float[301];
    size_t sample_data_csr_num_features = 301;
    int64_t sample_data_nindptr = 1; // number of rows in sample data
    int64_t sample_data_null_elem = 0;

    try {
        if (LGBM_BoosterPredictForCSRSingleRow(model_handle, (void*)sample_data_csr_row_headers, C_API_DTYPE_INT32, sample_data_csr_indices, sample_data_csr_row, C_API_DTYPE_FLOAT32, sample_data_nindptr, sample_data_null_elem, sample_data_csr_num_features, C_API_PREDICT_NORMAL, 0, 0, "", &out_len, out_result) != 0) {
            std::cout << "failed prediction for some reason";
        } else {
            std::cout << "prediction=" << out_result[0] << "\n";
        }
    } catch (std::exception e) {
        std::cout << "exception";
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
