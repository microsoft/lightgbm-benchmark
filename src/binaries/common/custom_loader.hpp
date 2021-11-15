#ifndef LIGHTGBM_BENCHMARK_COMMON_CUSTOM_LOADER_H_
#define LIGHTGBM_BENCHMARK_COMMON_CUSTOM_LOADER_H_

#include <stdint.h>
#include "LightGBM/dataset.h"

using std::cout; 
using std::cerr;
using std::endl;
using std::string;
using std::ifstream;
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
            try {
#ifdef USE_LIGHTGBM_V321_PARSER
            // LightGBM::Parser <3.2.1 uses 4 arguments, not 5
                this->lightgbm_parser = Parser::CreateParser(file_path.c_str(), false, 0, 0);
#else
                this->lightgbm_parser = Parser::CreateParser(file_path.c_str(), false, 0, 0, false);
#endif
            } catch (...) {
                cerr << "Failed during Parser::CreateParser() call";
                throw;
            }

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
            
            bool fetched_parsable_line = false;
            do {
                if(getline(*this->file_handler, input_line)) {
                    if (!input_line.empty() && input_line.back() == '\n')
                    {
                        input_line.pop_back();
                    }
                    if (!input_line.empty() && input_line.back() == '\r')
                    {
                        input_line.pop_back();
                    }
                    if (input_line.empty())
                    {
                        cout << "Empty line" << endl;
                        return nullptr;
                    }
                    this->row_counter++;
                    cout << "ROW line=" << this->row_counter;
                } else {
                    // if we're done, let's just return
                    return nullptr;
                }

                oneline_features.clear();

                // let's make sure the line is parsable
                try {
                    this->lightgbm_parser->ParseOneLine(input_line.c_str(), &oneline_features, &row_label);

                    // if we go that far, it means the line has been parsed
                    fetched_parsable_line = true;
                } catch (...) {
                    cout << " FAILED" << endl;
                    cout << "Line: " << input_line << endl;
                }
            } while (!fetched_parsable_line);

            cout << " label=" << row_label;

            // allocate or re-allocate a new row struct
            csr_row = this->init_row(replace_row, num_features);
            csr_row->row_headers[0] = 0; // memory index begin of row (0, duh)

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
#endif  /* LIGHTGBM_BENCHMARK_COMMON_CUSTOM_LOADER_H_ */
