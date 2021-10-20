#include <iostream>
#include <chrono>
#include "LightGBM/c_api.h"


struct CSRDataRow_t {
    // arguments for call to LGBM_BoosterPredictForCSRSingleRow()
    // see https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForCSRSingleRow
    int32_t * row_headers;
    int32_t * indices;
    float * row;
    size_t num_features;
    int64_t nindptr;
    int64_t null_elem;
};

class LibSVMReader {
    private:
        int row_counter = 0;
        int max_rows = 10;

    public:
        LibSVMReader(const std::string file_path) {
            this->row_counter = 0;
            this->max_rows = 10;
        };
        ~LibSVMReader() {};

        /* iterates on the svm file and returns a row ready to predict */
        CSRDataRow_t * iter(CSRDataRow_t * replace_row = nullptr) {
            CSRDataRow_t * csr_row;

            // let's fake it for now
            if (this->row_counter >= this->max_rows) {
                return nullptr;
            } else {
                this->row_counter++;
            }

            if (replace_row == nullptr) {
                csr_row = new CSRDataRow_t;
            } else {
                csr_row = replace_row;
            }

            // let's fake it for now
            csr_row->row_headers = new int32_t[301];

            for (int32_t i=0; i<301; i++) {
                csr_row->row_headers[i] = i;
            }

            csr_row->indices = new int32_t[301];
            for (int32_t i=0; i<301; i++) {
                csr_row->indices[i] = i;
            }

            csr_row->row = new float[301];
            csr_row->num_features = 301;
            csr_row->nindptr = 1; // number of rows in sample data
            csr_row->null_elem = 0;

            return csr_row;
        };
};

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

    LibSVMReader * data_reader = new LibSVMReader(data_filename);

    while (CSRDataRow_t * csr_row = data_reader->iter()) {
        using std::chrono::high_resolution_clock;
        using std::chrono::duration_cast;
        using std::chrono::duration;
        using std::chrono::milliseconds;

        try {
            auto t1 = high_resolution_clock::now();
            if (LGBM_BoosterPredictForCSRSingleRow(model_handle, (void*)csr_row->row_headers, C_API_DTYPE_INT32, csr_row->indices, csr_row->row, C_API_DTYPE_FLOAT32, csr_row->nindptr, csr_row->null_elem, csr_row->num_features, C_API_PREDICT_NORMAL, 0, 0, "", &out_len, out_result) != 0) {
                std::cout << "failed prediction for some reason";
            } else {
                std::cout << "prediction=" << out_result[0] << "\n";
            }
            auto t2 = high_resolution_clock::now();
            duration<double, std::milli> ms_double = t2 - t1;
            std::cout << ms_double.count() << "ms\n";

        } catch (std::exception e) {
            std::cout << "exception";
        }
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
