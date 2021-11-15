#include <iostream>
#include <fstream>
#include <string>
#include <stdint.h>
#include "LightGBM/dataset.h"
#include "custom_loader.hpp"

using std::cout; 
using std::cerr;
using std::endl;
using std::string;
using std::ifstream;
using LightGBM::Parser;


// Constructor
LightGBMDataReader::LightGBMDataReader() {
    row_counter = 0;
    num_features = 0;
    file_handler = nullptr;
    lightgbm_parser = nullptr;
};

// Destructor
LightGBMDataReader::~LightGBMDataReader() {};


// open the file for parsing
int LightGBMDataReader::open(const string file_path, int32_t init_num_features) {
    // use Parser class from LightGBM source code
    try {
#ifdef USE_LIGHTGBM_V321_PARSER
    // LightGBM::Parser <3.2.1 uses 4 arguments, not 5
        lightgbm_parser = Parser::CreateParser(file_path.c_str(), false, init_num_features, 0);
#else
        lightgbm_parser = Parser::CreateParser(file_path.c_str(), false, init_num_features, 0, false);
#endif
    } catch (...) {
        cerr << "Failed during Parser::CreateParser() call";
        throw;
    }

    if (lightgbm_parser == nullptr) {
        throw std::runtime_error("Could not recognize data format");
    }

    // but handle file reading separately
    num_features = init_num_features;
    file_handler = new ifstream(file_path);
    string line;

    if (!file_handler->is_open()) {
        cerr << "Could not open the file - '" << file_path << "'" << endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;            
};

// close the file handler (duh)
void LightGBMDataReader::close() {
    file_handler->close();
};

// allocate ONE new row of data in the given struct
CSRDataRow_t * LightGBMDataReader::init_row(CSRDataRow_t * row, int32_t num_features) {
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
CSRDataRow_t * LightGBMDataReader::iter(CSRDataRow_t * replace_row) {
    CSRDataRow_t * csr_row;
    string input_line;
    std::vector<std::pair<int, double>> oneline_features;
    double row_label;

    if (file_handler == nullptr) {
        throw std::runtime_error("You need to open() the file before iterating on it.");
    }

    // get a line from the file handler
    
    bool fetched_parsable_line = false;
    do {
        if(getline(*file_handler, input_line)) {
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
                cout << "LOG empty line at " << row_counter+1 << endl;
                return nullptr;
            }
            row_counter++;
        } else {
            // if we're done, let's just return
            return nullptr;
        }

        oneline_features.clear();

        // let's make sure the line is parsable
        try {
            lightgbm_parser->ParseOneLine(input_line.c_str(), &oneline_features, &row_label);

            // if we go that far, it means the line has been parsed
            fetched_parsable_line = true;
        } catch (...) {
            cout << "FAILED at line " << row_counter << " : " << input_line << endl;
        }
    } while (!fetched_parsable_line);

    // allocate or re-allocate a new row struct
    csr_row = init_row(replace_row, num_features);
    csr_row->row_headers[0] = 0; // memory index begin of row (0, duh)
    csr_row->row_label = row_label;
    csr_row->file_line_index = row_counter;

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
        cerr << "Number of features found in row line=" << row_counter << " is " << max_feature_index << " >= num_features" << endl;
        csr_row->num_features = max_feature_index;
    } else {
        csr_row->num_features = num_features;
    }

    return csr_row;
};
