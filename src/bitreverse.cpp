#include <omp.h>
#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdint>

#include "bitreverse.hpp"
#include "typedefs.hpp"

using namespace Typedefs;

auto bit_reversed_index(int index, uint8_t levels) -> int
{
    int rev = 0;
    for (int i = 0; i < levels; ++i) {
        rev <<= 1;
        rev |= (index & 1);
        index >>= 1;
    }
    return rev;
}

auto partial_bit_reverse(vec& signal, size_t n, uint8_t levels) -> void
{
   std::vector<std::pair<int, double>> temp(n);


    // Compute bit-reversed indices and store in a temporary vector
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        int rev_index = bit_reversed_index(i, levels);
        temp[i] = {rev_index, signal[i]};
    }

    // Sort the temporary vector by bit-reversed indices, preserving original order on ties
    std::stable_sort(temp.begin(), temp.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
        return a.first < b.first;
    });

    // Copy sorted values back to the original signal vector
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        signal[i] = temp[i].second;
    }
}

auto bit_reverse(size_t i, size_t N) -> size_t
{
    auto m = static_cast<size_t>(log2(N));
    std::bitset<sizeof(size_t) * 8> b(i);
    std::bitset<sizeof(size_t) * 8> reversed_b;

    for (size_t j = 0; j < m; j++) {
        reversed_b[j] = b[m - j - 1];
    }
    return reversed_b.to_ullong();
}

template <typename T>
auto bit_reverse_copy(T& v) -> void
{
    size_t N = v.size();
    T v_copy = v;
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++){
        v[i] = v_copy[bit_reverse(i, N)];
    }
}

auto next_multiple_of_levels(size_t n, size_t m) -> size_t
{
    size_t mask = (1 << m) - 1;
    size_t lowerMultiple = n & ~mask;
    return lowerMultiple + (1 << m);   
}

auto bit_reverse_image(Typedefs::vec3D &image, uint8_t user_levels) -> void
{
  int channels = image.size();
  int rows = image[0].size();
  int cols = image[0][0].size();

  for (uint8_t l = 0; l < user_levels; l++)
  {
    for (int c = 0; c < channels; ++c)
    {
      for (int i = 0; i < rows; ++i)
        partial_bit_reverse(image[c][i], cols, 1);
      for (int j = 0; j < cols; j++)
      {
        vec temp;
        for (int i = 0; i < rows; ++i)
          temp.push_back(image[c][i][j]);
        partial_bit_reverse(temp, rows, 1);
        for (int i = 0; i < rows; ++i)
          image[c][i][j] = temp[i];
      }
    }
    rows /= 2;
    cols /= 2;
  }
}

auto reverse_bit_reverse_image(vec3D &bit_reversed_image, uint8_t levels) -> void
{
  int channels = bit_reversed_image.size();

  int og_rows = bit_reversed_image[0].size();
  int og_cols = bit_reversed_image[0][0].size();

  int rows = og_rows / pow(2, levels - 1);
  int cols = og_cols / pow(2, levels - 1);

  // auto tmp_mat = bit_reversed_image;
  auto original_image_inverse_rows = bit_reversed_image;
  auto original_image_inverse_cols = bit_reversed_image;

  for (uint8_t l = 0; l < levels; ++l)
  {
    for (int c = 0; c < channels; ++c)
    {
      for (int i = 0; i < rows / 2; ++i){
        for (int j = 0; j < cols; j++){
          original_image_inverse_rows[c][i+i][j] = bit_reversed_image[c][i][j];
          original_image_inverse_rows[c][rows - i - i - 1][j] = bit_reversed_image[c][rows-i-1][j];
        }
      }
      for (int j = 0; j < cols / 2; ++j)
      {
        for (int i = 0; i < rows; ++i)
        {
          original_image_inverse_cols[c][i][j+j] = original_image_inverse_rows[c][i][j];
          original_image_inverse_cols[c][i][cols - j - j - 1] = original_image_inverse_rows[c][i][cols - j - 1];
        }
      }
    }
    bit_reversed_image = original_image_inverse_cols;
    rows *= 2;
    cols *= 2;
  }
  // return tmp_mat;
}

template void bit_reverse_copy<vec>(vec &v);
template void bit_reverse_copy<vcpx>(vcpx &v);