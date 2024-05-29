#ifndef TRANSFORM_MAT
#define TRANSFORM_MAT

#include <array>

namespace TRANSFORM_MATRICES
{
  namespace TRANSFORM
  {
    template <typename T, std::size_t N>
    constexpr auto generate_forward(const std::array<T, N> &arr)
    {
      std::array<T, N * 2> forward;
      for (std::size_t i = 0; i < N; ++i)
        forward[i] = arr[i] * M_SQRT1_2;
      for (std::size_t i = 0; i < N; ++i)
        forward[N + i] = (i % 2 == 1 ? -1 : 1) * arr[N - i - 1] * M_SQRT1_2;
      return forward;
    }

    template <typename T, std::size_t N>
    constexpr auto generate_inverse(const std::array<T, N> &arr)
    {
      std::array<T, N> inverse;
      for (std::size_t i = N / 2 - 2; (int)i >= 0; i -= 2)
      {
        inverse[N / 2 - i - 2] = arr[i];
        inverse[N / 2 - i - 1] = arr[i + N / 2];
      }
      for (std::size_t i = N / 2 - 1; (int)i >= 1; i -= 2)
      {
        inverse[N - i - 1] = arr[i];
        inverse[N - i] = arr[i + N / 2];
      }
      return inverse;
    }
  };
    constexpr std::array <double, 2>  HAAR            = {1, 1};
    constexpr std::array <double, 4>  DAUBECHIES_D4   = {0.6830127, 1.1830127, 0.3169873, -0.1830127};
    constexpr std::array <double, 6>  DAUBECHIES_D6   = {0.47046721,1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175};
    constexpr std::array <double, 8>  DAUBECHIES_D8   = {0.32580343, 1.01094572, 0.89220014, -0.03957503, -0.26450717, 0.0436163, 0.0465036, -0.01498699};
    constexpr std::array <double, 10> DAUBECHIES_D10  = {0.22641898, 0.85394354, 1.02432694, 0.19576696, -0.34265671, -0.04560113, 0.10970265, -0.00882680, -0.01779187, 0.00471743};
    constexpr std::array <double, 16> DAUBECHIES_D16  = {0.07695562, 0.44246725, 0.95548615, 0.82781653, -0.02238574, -0.40165863, 6.68194092e-4, 0.18207636, -0.02456390, -0.06235021, 0.01977216, 0.01236884, -6.88771926e-3, -5.54004549e-4, 9.55229711e-4, -1.66137261e-4};
    constexpr std::array <double, 20> DAUBECHIES_D20  = {0.03771716, 0.26612218, 0.74557507, 0.97362811, 0.39763774, -0.35333620, -0.27710988, 0.18012745, 0.13160299, -0.10096657, -0.04165925, 0.04696981, 0.005100436, -0.01517900, 0.00197333, 0.00281769, -0.000969947, -0.000164709, 1.32354367e-4, -1.875841e-5};

    constexpr std::array <double, 6>  COIFFLET_6      = {-0.10285945, 0.47785945, 1.20571891, 0.54428108, -0.10285945, -0.02214054};
}

#endif