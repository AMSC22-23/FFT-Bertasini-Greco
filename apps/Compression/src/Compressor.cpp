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
const double f = 8;  // mantissa

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

auto Compressor::HuffmanEncoding() -> shared_ptr<HuffmanTree<int>> {
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
    // (huffman_tree->tmp).insert((huffman_tree->tmp).end(), code.begin(), code.end());
    encoded_data.insert(encoded_data.end(), code.begin(), code.end());
  }
  cout << "Size of orginal code: " << flat_coeff.size() << endl;

  // 6. Save the encoded data
  ofstream encoded_file("encoded_data.txt", ios::binary);
  if (!encoded_file.is_open()) {
    cout << "Could not open file encoded_data.txt" << '\n';
    return nullptr;
  }

  // 7. Write the Huffman map to the file
  // size_t size = huffman_code_table.size();
  // encoded_file.write((char*)&size, sizeof(size));
  // for (auto& [key, value] : huffman_code_table) {
  //   encoded_file << key << " ";
  //   for (auto bit : value) {
  //     encoded_file << bit;
  //   }
  //   encoded_file << '\n';
  // }

  write_bits(encoded_file, encoded_data);
  encoded_file.close();

  return huffman_tree;
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

auto Compressor::HuffmanDecoding(shared_ptr<HuffmanTree<int>>& huffman_tree) -> void {
    // 1. Read the encoded data
    ifstream encoded_file("encoded_data.txt", ios::binary);
    if (!encoded_file.is_open()) {
        cout << "Could not open file encoded_data.txt" << '\n';
        return;
    }

    vector<bool> encoded_data = read_bits(encoded_file);
    encoded_file.close();

    // 2. Reconstruct the Huffman tree
    // Note: In practice, you would store and read the frequency table or the code table from the file.
    auto reverse_huffman_code_table = huffman_tree->get_reversed_table();

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

auto plot_imeage (vec3D &coeff, string title) -> void {
  cv::Mat dwt_image_colored;
  auto tmp = coeff;
  dwt_image_colored.create(tmp[0].size(), tmp[0][0].size(), CV_64FC3);
  for (size_t c = 0; c < tmp.size(); ++c)
      for (size_t i = 0; i < tmp[0].size(); ++i)
          for (size_t j = 0; j < tmp[0][0].size(); ++j)
              dwt_image_colored.at<cv::Vec3d>(i, j)[c] = tmp[c][i][j];

  dwt_image_colored.convertTo(dwt_image_colored, CV_8UC3);
  imshow(title, dwt_image_colored);
}

auto Compressor::apply_idwt() -> void {
    // for (int i = 0; i < levels * levels; i++) {
    //   string title = "IDWT Image " + to_string(i);
    //   bit_reverse_image(coeff, levels-1);
    //   plot_imeage(coeff, title);
    //   waitKey(0);
    //   }
    reverse_bit_reverse_image(coeff, levels);
    DiscreteWaveletTransform2D dwt(TRANSFORM_MATRICES::DAUBECHIES_D20, levels);
    dwt.computeDWT2D(coeff, true);
}