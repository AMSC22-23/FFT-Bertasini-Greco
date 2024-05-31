#include "Compressor.hpp"
#include "Transform.hpp"
#include "DiscreteWaveletTransform2D.hpp"
#include "bitreverse.hpp"
#include "utils.hpp"

#include <memory>
#include <fstream>

using namespace std;
using namespace cv;
using namespace Typedefs;

//q(t)=sgn(t)td
auto quantize_value(double& value, const double& step) -> void {
    value = copysign(1.0, value) * floor(abs(value) / step);
}

auto Compressor::apply_dwt() -> void {
  DiscreteWaveletTransform2D dwt(TRANSFORM_MATRICES::DAUBECHIES_D20, levels);
  auto in_space = DiscreteWaveletTransform2D<4>::InputSpace(img);
  auto &in_data = in_space.data;

  dwt.computeDWT2D(in_data, false);
  bit_reverse_image(in_data, levels);

  coeff = in_data;
}

const double compression_coeff = 1; // the higher the more compression
const double R = 8.;
const double c = 8.5; // exponent
const double f = 8;   // mantissa

auto Compressor::quantize () -> void {
  const double tau = pow(2, R - c + (double)levels) * (1. + f / pow(2, 11));
  const int rows = coeff[0].size();
  for (size_t c = 0; c < coeff.size(); ++c)
    for (size_t i = 0; i < coeff[0].size(); ++i)
      for (size_t j = 0; j < coeff[0][0].size(); ++j){
        int k = levels - countSubdivisions(i, j, rows, levels+1);
        if (k == levels) quantize_value(coeff[c][i][j], tau / pow(2, k));
        else quantize_value(coeff[c][i][j], tau / pow(2, k-compression_coeff));
      }
}

auto Compressor::dequantize () -> void {
  const double tau = pow(2, R - c + (double)levels) * (1. + f / pow(2, 11));
  const int rows = coeff[0].size();
  for (size_t c = 0; c < coeff.size(); ++c)
    for (size_t i = 0; i < coeff[0].size(); ++i)
      for (size_t j = 0; j < coeff[0][0].size(); ++j){
        int k = levels - countSubdivisions(i, j, rows, levels+1);
        if (k == levels) coeff[c][i][j] *= tau / pow(2, k);
        else coeff[c][i][j] *= tau / pow(2, k-compression_coeff);
      }
}

auto write_bits(ofstream& file, vector<bool>& bits) -> void {
  int n = 0;
  char byte = 0;
  for (auto bit : bits) {
    byte = byte << 1 | bit;
    n++;
    if (n == 8) {
      file.write(&byte, 1);
      n = 0;
      byte = 0;
    }
  }

  if (n != 0) {
    byte = byte << (8 - n);
    file.write(&byte, 1);
  }
}

auto Compressor::HuffmanEncoding() -> void {
  // 1. Flatten the 3D matrix into a 1D vector
  vector<int> flat_coeff;
  for (size_t c = 0; c < coeff.size(); ++c)
    for (size_t i = 0; i < coeff[0].size(); ++i)
      for (size_t j = 0; j < coeff[0][0].size(); ++j)
        flat_coeff.push_back(coeff[c][i][j]);

  // 2. Create a frequency table
  unordered_map<int, int> freq_table;
  for (auto& val : flat_coeff)
    freq_table[val]++;
  
  // 3. Create a Huffman tree
  auto huffman_tree = make_shared<HuffmanTree<int>>(freq_table);

  // 4. Create a Huffman code table
  auto huffman_code_table = huffman_tree->get_code_table();

  // 5. Encode the data
  vector<bool> encoded_data;
  for (auto& val : flat_coeff) {
    auto code = huffman_code_table[val];
    encoded_data.insert(encoded_data.end(), code.begin(), code.end());
  }

  // 6. Save the encoded data
  ofstream encoded_file("encoded_data.txt", ios::binary);
  if (!encoded_file.is_open()) {
    cout << "Could not open file encoded_data.txt" << '\n';
    return;
  }

  // 7. Write the Huffman map to the file
  size_t size = huffman_code_table.size();
  encoded_file.write((char*)&size, sizeof(size));
  vector<int> keys;
  vector<bool> values;
  vector<char> code_sizes;
  for (auto& [key, value] : huffman_code_table) {
    keys.push_back(key);
    values.insert(values.end(), value.begin(), value.end());
    code_sizes.push_back(value.size());
  }

  encoded_file.write((char*)keys.data(), keys.size() * sizeof(int));
  encoded_file.write((char*)code_sizes.data(), code_sizes.size() * sizeof(char));
  write_bits(encoded_file, values);


  // 8. Write the encoded data to the file
  write_bits(encoded_file, encoded_data);
  encoded_file.close();
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

auto Compressor::HuffmanDecoding() -> void {
    // 1. Read the encoded data
    ifstream encoded_file("encoded_data.txt", ios::binary);
    if (!encoded_file.is_open()) {
        cout << "Could not open file encoded_data.txt" << '\n';
        return;
    }

    // 2. Reconstruct the Huffman tree
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
    size_t idx = 0;
    for (size_t c = 0; c < coeff.size(); ++c){
        for (size_t i = 0; i < coeff[0].size(); ++i){
            for (size_t j = 0; j < coeff[0][0].size(); ++j){
                coeff[c][i][j] = decoded_coefficients[idx++];
            }
        }
    }
}

auto Compressor::apply_idwt() -> void {
    reverse_bit_reverse_image(coeff, levels);
    DiscreteWaveletTransform2D dwt(TRANSFORM_MATRICES::DAUBECHIES_D20, levels);
    dwt.computeDWT2D(coeff, true);
}