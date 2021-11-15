/*
This program uses the LightGBM C API to run predictions on a file dataset
in order to measure inferencing time for production scenarios.

Usage:
    lightgbm_predict_telemetry.exe MODELFILEPATH DATAFILEPATH CUSTOMPARAMS

Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
*/
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "LightGBM/c_api.h" // in LightGBM includes
#include "LightGBM/dataset.h" // in LightGBM includes

#include "custom_loader.hpp" // in common folder

using std::cout; 
using std::cerr;
using std::endl;
using std::string;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


// main function obviously
int main(int argc, char* argv[]) {
    // ****************
    // ARGUMENT PARSING
    // ****************

    if (argc < 3) {
        cerr << "You need to provide at least 2 arguments:\n\n     lightgbm_predict_telemetry.exe MODELFILEPATH DATAFILEPATH [CUSTOMPARAMS]";
        return EXIT_FAILURE;
    }
    string model_filename = string(argv[1]);
    string data_filename = string(argv[2]);
    string custom_params = "";

    // everything after model+data is considered custom params for LightGBM
    if (argc > 3) {
        for (int i=3; i<argc; i++) {
            if (i>1) {
                custom_params += " ";
            }
            custom_params += argv[i];
        }
    }
    cout << "LOG running with model=" << model_filename << " data=" << data_filename << " custom_params=" << custom_params << endl;

    // *************
    // LOADING MODEL
    // *************

    BoosterHandle model_handle;
    int model_num_trees;
    int num_features;
    int num_classes;

    if (LGBM_BoosterCreateFromModelfile(model_filename.c_str(), &model_num_trees, &model_handle) != 0) {
        throw std::runtime_error("ERROR Could not load LightGBM model from file " + model_filename);
    } else {
        std::cout << "LOG Successfully loaded LightGBM model from file " + model_filename << endl;
    }

    // we need number of features for the libsvm parser
    if (LGBM_BoosterGetNumFeature(model_handle, &num_features) != 0) {
        throw std::runtime_error("Could not get number of features from model");
    } else {
        std::cout << "PROPERTY inference_data_width=" << num_features << endl;
    }

    // we need number of outputs to allocate memory later
    if (LGBM_BoosterGetNumClasses(model_handle, &num_classes) != 0) {
        throw std::runtime_error("Could not get number of classes from model");
    } else {
        std::cout << "PROPERTY num_classes=" << num_classes << endl;
    }

    // **********************
    // LOOP ON DATA + PREDICT
    // **********************

    // variables for predictions output
    int64_t out_len = 0;
    double * out_result = new double[num_classes];

    // custom class for reading libsvm
    LightGBMDataReader * data_reader = new LightGBMDataReader();
    data_reader->open(data_filename, num_features);

    // variables for metrics
    double prediction_per_request = 0.0;
    int count_request = 0;

    // initialize fast handle for fast predictions (-20%)
    FastConfigHandle fastConfig;
    LGBM_BoosterPredictForCSRSingleRowFastInit(model_handle, C_API_PREDICT_NORMAL, 0, 0, C_API_DTYPE_FLOAT32, num_features, custom_params.c_str(), &fastConfig);

    // loop on each data row
    while (CSRDataRow_t * csr_row = data_reader->iter()) {
        // NOTE: we add an exception here just in case
        try {
            // start a clock
            auto t1 = high_resolution_clock::now();

            // call for LightGBM C API
            int ret_val = LGBM_BoosterPredictForCSRSingleRowFast(fastConfig, (void*)csr_row->row_headers, C_API_DTYPE_INT32, csr_row->indices, csr_row->row, csr_row->nindptr, csr_row->null_elem, &out_len, out_result);

            // stop the clock
            auto t2 = high_resolution_clock::now();

            if (ret_val != 0) {
                std::cout << endl << "ERROR failed prediction for some reason" << endl;
            } else {
                std::cout << " prediction=" << out_result[0];
            }

            // compute metric
            duration<double> ms_double = t2 - t1;
            std::cout << " time_usecs=" << ms_double.count()*1000000 << endl;

            // record the rest and iterate
            prediction_per_request += ms_double.count();
            count_request++;
        } catch (std::exception e) {
            std::cout << endl << "ERROR exception" << endl;
        }
    }

    // print out summary metrics
    cout << "METRIC time_inferencing=" << prediction_per_request << endl;
    cout << "METRIC prediction_per_request_usecs=" << prediction_per_request/count_request*1000000 << endl;
    cout << "PROPERTY inference_data_length=" << count_request << endl;

    // free resources
    data_reader->close();

    if (model_handle != nullptr) {
        std::cout << "LOG Free booster\n";
        LGBM_BoosterFree(model_handle);
    }

    std::cout << "LOG Done.\n";
}
