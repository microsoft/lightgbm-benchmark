/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
*/

#ifndef LIGHTGBM_BENCHMARK_COMMON_CUSTOM_LOADER_H_
#define LIGHTGBM_BENCHMARK_COMMON_CUSTOM_LOADER_H_

#include <stdint.h>
#include "LightGBM/dataset.h"


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

    // some metadata for debugging
    int32_t file_line_index;
    float row_label;
};

// class to read libsvm file and iterate on each line
class LightGBMDataReader {
    private:
        // counts how many rows have been processed so far
        int row_counter;
        int num_features;
        std::ifstream * file_handler;
        LightGBM::Parser * lightgbm_parser;

    public:
        // Constructor
        LightGBMDataReader();

        // Destructor
        ~LightGBMDataReader();

        // open the file for parsing
        int open(const std::string file_path, int32_t init_num_features);

        // close the file handler (duh)
        void close();

        // allocate ONE new row of data in the given struct
        CSRDataRow_t * init_row(CSRDataRow_t * row, int32_t num_features);

        // Iterates on the svm file and returns ONE row ready to predict
        // returns nullptr when finished
        CSRDataRow_t * iter(CSRDataRow_t * replace_row = nullptr);
};

#endif  /* LIGHTGBM_BENCHMARK_COMMON_CUSTOM_LOADER_H_ */
