#include "Compressor.hpp"
#include "HuffmanTree.hpp"
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
using namespace tr;

auto Compressor::apply_dwt() -> void {
  DiscreteWaveletTransform2D dwt(tr_mat, levels);
  auto in_space = dwt.get_input_space(img);
  auto &in_data = dynamic_cast<DiscreteWaveletTransform2D::InputSpace&>(*in_space).data;

  img_size = img.size();

  dwt(in_data, false);
  bitreverse::bit_reverse_image(in_data, levels);

  coeff = in_data;
}

//q(t)=sgn(t)td
auto Compressor::quantize_value(double& value, const double& step) -> void {
    value = copysign(1.0, value) * floor(abs(value) / step);
}

auto Compressor::quantize () -> void {
  const double tau = pow(2, R - c + (double)levels) * (1. + f / pow(2, 11));
  const int rows = coeff[0].size();
  const int cols = coeff[0][0].size();
  for (size_t c = 0; c < coeff.size(); ++c)
    for (size_t i = 0; i < coeff[0].size(); ++i)
      for (size_t j = 0; j < coeff[0][0].size(); ++j){
        int k = levels - tr::utils::countSubdivisions(i, j, rows, cols, levels+1);
        if (k == levels) quantize_value(coeff[c][i][j], tau / pow(2, k));
        else quantize_value(coeff[c][i][j], tau / pow(2, k-compression_coeff));
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

auto Compressor::HuffmanEncoding(const string& filename) -> void {
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
  ofstream encoded_file(filename, ios::binary);
  if (!encoded_file.is_open()) {
    cout << "Could not open file encoded_data.txt" << '\n';
    return;
  }

  // 7. Write the Huffman map to the file
  int rows = coeff[0].size();
  int cols = coeff[0][0].size();
  encoded_file.write((char*)&img_size, sizeof(img_size));
  encoded_file.write((char*)&rows, sizeof(int));
  encoded_file.write((char*)&cols, sizeof(int));
  encoded_file.write((char*)&levels, sizeof(levels));
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

auto Compressor::compress(const string& filename, const cv::Mat& _img, const int _levels) -> void {
    img = _img;
    levels = _levels;
    apply_dwt();
    quantize();
    HuffmanEncoding(filename);
}