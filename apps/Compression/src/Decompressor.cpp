#include "Decompressor.hpp"
#include "DiscreteWaveletTransform2D.hpp"
#include "bitreverse.hpp"
#include "utils.hpp"
#include "hyperparameters.hpp"

#include <memory>
#include <fstream>

using namespace std;
using namespace cv;
using namespace Typedefs;
using namespace compression;

auto Decompressor::dequantize () -> void {
  const double tau = pow(2, R - c + (double)levels) * (1. + f / pow(2, 11));
  const int rows = coeff[0].size();
  const int cols = coeff[0][0].size();
  for (size_t c = 0; c < coeff.size(); ++c)
    for (size_t i = 0; i < coeff[0].size(); ++i)
      for (size_t j = 0; j < coeff[0][0].size(); ++j){
        int k = levels - countSubdivisions(i, j, rows, cols, levels+1);
        if (k == levels) coeff[c][i][j] *= tau / pow(2, k);
        else coeff[c][i][j] *= tau / pow(2, k-compression_coeff);
      }
}

auto read_bits(ifstream& file) -> vector<bool> {
    vector<bool> bits;
    char byte;
    while (file.read(&byte, 1)) {
        for (int i = 7; i >= 0; --i) {
            bits.push_back((byte >> i) & 1);
        }
    }
    return bits;
}

auto Decompressor::HuffmanDecoding(const std::string& filename) -> void {
    // 1. Read the encoded data
    ifstream encoded_file(filename, ios::binary);
    if (!encoded_file.is_open()) {
        cout << "Could not open file encoded_data.txt" << '\n';
        return;
    }

    int rows, cols;

    // 2. Reconstruct the Huffman tree
    encoded_file.read((char*)&img_size, sizeof(img_size));
    encoded_file.read((char*)&rows, sizeof(rows));
    encoded_file.read((char*)&cols, sizeof(cols));
    encoded_file.read((char*)&levels, sizeof(levels));
    size_t size;
    encoded_file.read((char*)&size, sizeof(size));

    vector<int> keys(size);
    vector<char> code_sizes(size);
    vector<vector<bool>> values;

    encoded_file.read((char*)keys.data(), keys.size() * sizeof(int));
    encoded_file.read((char*)code_sizes.data(), code_sizes.size() * sizeof(char));

    vector<char> tmp;
    auto total_code_size = 0;
    for (auto& code_size : code_sizes) total_code_size += code_size;

    tmp.resize(total_code_size / 8 + 1);
    encoded_file.read((char*)tmp.data(), total_code_size / 8 + 1);

    int cnt = 0;
    for (size_t i = 0; i < size; ++i) {
      vector<bool> code;
      for (int j = 0; j < code_sizes[i]; ++j) {
        code.push_back((tmp[cnt >> 3] >> (7 - (cnt & 7))) & 1);
        cnt++;
      }
      values.push_back(code);
    }

    unordered_map<vector<bool>, int> reverse_huffman_code_table;
    for (size_t i = 0; i < size; ++i) {
      reverse_huffman_code_table[values[i]] = keys[i];
    }

    vector<bool> encoded_data = read_bits(encoded_file);
    encoded_file.close();

    // 3. Decode the data
    vector<double> decoded_coefficients;
    vector<bool> current_code;
    for (auto bit : encoded_data) {
      current_code.push_back(bit);
      if (reverse_huffman_code_table.find(current_code) != reverse_huffman_code_table.end()) {
        decoded_coefficients.push_back(reverse_huffman_code_table[current_code]);
        current_code.clear();
      }
    }

    // 4. Reshape the flat coefficient vector back into the 3D matrix format
    coeff.resize(3, vector<vector<double>>(rows, vector<double>(cols, 0)));
    size_t idx = 0;
    for (size_t c = 0; c < coeff.size(); ++c){
        for (size_t i = 0; i < coeff[0].size(); ++i){
            for (size_t j = 0; j < coeff[0][0].size(); ++j){
                coeff[c][i][j] = decoded_coefficients[idx++];
            }
        }
    }
}

auto Decompressor::apply_idwt() -> void {
    reverse_bit_reverse_image(coeff, levels);
    DiscreteWaveletTransform2D dwt(tr, levels);
    dwt(coeff, true);
}

auto Decompressor::decompress(const std::string& filename, cv::Mat& img) -> void {
    HuffmanDecoding(filename);
    dequantize();
    apply_idwt();
    
    img.create(img_size, CV_64FC3);
    for (auto c = 0ull; c < coeff.size(); ++c)
        for (auto i = 0; i < img_size.height; ++i)
            for (auto j = 0; j < img_size.width; ++j)
                img.at<cv::Vec3d>(i, j)[c] = coeff[c][i][j];
    img.convertTo(img, CV_8UC3);

}