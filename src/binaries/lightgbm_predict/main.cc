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
#include "LightGBM/c_api.h"
#include "LightGBM/dataset.h"

using std::cout; 
using std::cerr;
using std::endl;
using std::string;
using std::ifstream;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using LightGBM::Parser;


// struct to organize arguments for call to LGBM_BoosterPredictForCSRSingleRow()
// see https://lightgbm.readthedocs.io/en/latest/C-API.html#c.LGBM_BoosterPredictForCSRSingleRow
struct CSRDataRow_t {
    // array of size nindptr (number of rows)
    // row_headers[n] = mem index start of row n
    // row_headers[n+1] = mem index end of row n
    int32_t * row_headers = nullptr;

    // array of size maximum num_features
    // sparse indices in the row
    int32_t * indices = nullptr;

    // array of size maximum num_features
    // sparse values in the row
    float * row = nullptr;

    // maximum number of features (total)
    size_t num_features;

    // number of rows
    int64_t nindptr;

    // number of null elements in the row (?)
    int64_t null_elem;
};

// class to read libsvm file and iterate on each line
class LightGBMDataReader {
    private:
        // counts how many rows have been processed so far
        int row_counter;
        int num_features;
        ifstream * file_handler;
        Parser * lightgbm_parser;

    public:
        // Constructor
        LightGBMDataReader() {
            this->row_counter = 0;
            this->num_features = 0;
            this->file_handler = nullptr;
            this->lightgbm_parser = nullptr;
        };
        // Destructor
        ~LightGBMDataReader() {};

        // open the file for parsing
        int open(const string file_path, int32_t num_features) {
            // use Parser class from LightGBM source code
#ifdef USE_LIGHTGBM_V321_PARSER
            // LightGBM::Parser <3.2.1 uses 4 arguments, not 5
            this->lightgbm_parser = Parser::CreateParser(file_path.c_str(), false, num_features, 0);
#else
            this->lightgbm_parser = Parser::CreateParser(file_path.c_str(), false, num_features, 0, false);
#endif
            if (this->lightgbm_parser == nullptr) {
                throw std::runtime_error("Could not recognize data format");
            }

            // but handle file reading separately
            this->num_features = num_features;
            this->file_handler = new ifstream(file_path);
            string line;

            if (!this->file_handler->is_open()) {
                cerr << "Could not open the file - '" << file_path << "'" << endl;
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;            
        };

        // close the file handler (duh)
        void close() {
            this->file_handler->close();
        };

        // allocate ONE new row of data in the given struct
        CSRDataRow_t * init_row(CSRDataRow_t * row, int32_t num_features) {
            // if null provide, allocate completely new struct
            if (row == nullptr) {
                row = new CSRDataRow_t;
            }

            // if previous struct provided, free each member + reallocate
            if (row->row_headers != nullptr) {
                free(row->row_headers);
            }
            row->row_headers = new int32_t[2]; // one for start of row (0), one for end of row (tbd)
            row->row_headers[0] = 0;
            row->row_headers[1] = 0;

            if (row->indices != nullptr) {
                free(row->indices);
            }
            row->indices = new int32_t[num_features];

            if (row->row != nullptr) {
                free(row->row);
            }
            row->row = new float[num_features];

            // just making sure we're not keeping old data
            for (int i=0; i<num_features; i++) {
                row->indices[i] = -1;
                row->row[i] = 0.0;
            }

            row->num_features = 0; // tbd during parsing

            // number of rows, that's 1
            row->nindptr = 1;

            // number of null cells, tbd
            row->null_elem = 0;

            return row;
        };

        // Iterates on the svm file and returns ONE row ready to predict
        // returns nullptr when finished
        CSRDataRow_t * iter(CSRDataRow_t * replace_row = nullptr) {
            CSRDataRow_t * csr_row;
            string input_line;
            std::vector<std::pair<int, double>> oneline_features;
            double row_label;

            if (this->file_handler == nullptr) {
                throw std::runtime_error("You need to open() the file before iterating on it.");
            }

            // get a line from the file handler
            if(getline(*this->file_handler, input_line)) {
                this->row_counter++;
                cout << "ROW line=" << this->row_counter;
            } else {
                // if we're done
                return nullptr;
            }

            // allocate or re-allocate a new row struct
            csr_row = this->init_row(replace_row, num_features);
            csr_row->row_headers[0] = 0; // memory index begin of row (0, duh)

            oneline_features.clear();
            this->lightgbm_parser->ParseOneLine(input_line.c_str(), &oneline_features, &row_label);

            cout << " label=" << row_label;

            // convert output from Parser into expected format for C API call
            for (std::pair<int, double>& inner_data : oneline_features) {
                csr_row->indices[csr_row->num_features] = inner_data.first;
                csr_row->row[csr_row->num_features] = inner_data.second;
                csr_row->num_features++;
                
                csr_row->row_headers[1] = csr_row->num_features; // memory index end of row (actual length)
            }

            // number of null elements
            csr_row->null_elem = num_features - csr_row->num_features;

            // finalize number of features
            // check consistency
            int max_feature_index = 0;
            for (int i=0; i<csr_row->num_features; i++) {
                if (csr_row->indices[i] > max_feature_index) {
                    max_feature_index = csr_row->indices[i];
                }
            }
            if (max_feature_index > num_features) {
                cerr << "Number of features found in row line=" << this->row_counter << " is " << max_feature_index << " >= num_features" << endl;
                csr_row->num_features = max_feature_index;
            } else {
                csr_row->num_features = num_features;
            }

            cout << " null_elem=" << csr_row->null_elem;
            return csr_row;
        };
};

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
