/**
* @file TransformMatrices.hpp
* @brief Defines transform matrices for various wavelet families.
*/

#ifndef TRANSFORM_MAT
#define TRANSFORM_MAT

#include <array>

#include "typedefs.hpp"

namespace TRANSFORM_MATRICES {
   namespace TRANSFORM {
       /**
        * @brief Generate the forward transform matrix from a scaling matrix.
        * @tparam T The data type of the matrix elements.
        * @tparam N The size of the scaling matrix.
        * @param arr The scaling matrix.
        * @return The forward transform matrix.
        */
       template <typename T, std::size_t N>
       constexpr auto generate_forward(const std::array<T, N>& arr) {
           std::array<T, N * 2> forward;
           for (std::size_t i = 0; i < N; ++i)
               forward[i] = arr[i] / 2;
           for (std::size_t i = 0; i < N; ++i)
               forward[N + i] = (i % 2 == 1 ? -1 : 1) * arr[N - i - 1] / 2;
           return forward;
       }

       /**
        * @brief Generate the inverse transform matrix from a scaling matrix.
        * @tparam T The data type of the matrix elements.
        * @tparam N The size of the scaling matrix.
        * @param arr The scaling matrix.
        * @return The inverse transform matrix.
        */
       template <typename T, std::size_t N>
       constexpr auto generate_inverse(const std::array<T, N>& arr) {
           std::array<T, N> inverse;
           for (std::size_t i = N / 2 - 2; (int)i >= 0; i -= 2) {
               inverse[N / 2 - i - 2] = arr[i] * 2;
               inverse[N / 2 - i - 1] = arr[i + N / 2] * 2;
           }
           for (std::size_t i = N / 2 - 1; (int)i >= 1; i -= 2) {
               inverse[N - i - 1] = arr[i] * 2;
               inverse[N - i] = arr[i + N / 2] * 2;
           }
           return inverse;
       }
   }

   /**
    * @brief Struct representing a transform matrix with forward and inverse matrices.
    * @tparam T The data type of the matrix elements.
    * @tparam N The size of the scaling matrix.
    */
   template <typename T, std::size_t N>
   struct TransformMatrix {
   public:
       std::array<T, N * 2> forward; /**< The forward transform matrix. */
       std::array<T, N * 2> inverse; /**< The inverse transform matrix. */

       /**
        * @brief Constructor for the TransformMatrix struct.
        * @param scaling_matrix The scaling matrix.
        */
       constexpr TransformMatrix(const std::array<T, N>& scaling_matrix)
           : forward(TRANSFORM::generate_forward(scaling_matrix)),
             inverse(TRANSFORM::generate_inverse(forward)) {}
   };

    constexpr TransformMatrix <Typedefs::DType, 2>  HAAR            = {{1, 1}};
    constexpr TransformMatrix <Typedefs::DType, 4>  DAUBECHIES_D4   = {{0.6830127, 1.1830127, 0.3169873, -0.1830127}};
    constexpr TransformMatrix <Typedefs::DType, 6>  DAUBECHIES_D6   = {{0.47046721,1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175}};
    constexpr TransformMatrix <Typedefs::DType, 8>  DAUBECHIES_D8   = {{0.32580343, 1.01094572, 0.89220014, -0.03957503, -0.26450717, 0.0436163, 0.0465036, -0.01498699}};
    constexpr TransformMatrix <Typedefs::DType, 10> DAUBECHIES_D10  = {{0.22641898, 0.85394354, 1.02432694, 0.19576696, -0.34265671, -0.04560113, 0.10970265, -0.00882680, -0.01779187, 0.00471743}};
    constexpr TransformMatrix <Typedefs::DType, 16> DAUBECHIES_D16  = {{0.07695562, 0.44246725, 0.95548615, 0.82781653, -0.02238574, -0.40165863, 6.68194092e-4, 0.18207636, -0.02456390, -0.06235021, 0.01977216, 0.01236884, -6.88771926e-3, -5.54004549e-4, 9.55229711e-4, -1.66137261e-4}};
    constexpr TransformMatrix <Typedefs::DType, 20> DAUBECHIES_D20  = {{0.03771716, 0.26612218, 0.74557507, 0.97362811, 0.39763774, -0.35333620, -0.27710988, 0.18012745, 0.13160299, -0.10096657, -0.04165925, 0.04696981, 0.005100436, -0.01517900, 0.00197333, 0.00281769, -0.000969947, -0.000164709, 1.32354367e-4, -1.875841e-5}};
    constexpr TransformMatrix <Typedefs::DType, 40> DAUBECHIES_D40  = {{0.0011030209784695594, 0.014919096953430685, 0.08969477050220678, 0.311045119921389, 0.668493356148703, 0.863367818244038, 0.5112414537062122, -0.19687562291211747, -0.4621463251783223, -0.023655675145709147, 0.32285230023793116, 0.05635675900449043, -0.21985187363979633, -0.034954872441101424, 0.14466233657495667, 0.007965199892237419, -0.08728936175821633, 0.008308054692890832, 0.04567103638375342, -0.012429982511485459, -0.01953103336666836, 0.009505816492072603, 0.006251590996805293, -0.005064997755501315, -0.0011760065027620996, 0.001969376700059894, -7.565702926820514e-05, -0.0005446203585447162, 0.0001435891896925773, 9.580279822815831e-05, -5.247561304911024e-05, -6.188802000796386e-06, 1.0240671536938985e-05, -1.4311756540090427e-06, -9.683232828726295e-07, 3.7249313630540966e-07, 2.8486815246919524e-10, -2.5665759353266726e-08, 5.736229892669851e-09, -4.240995234958965e-10}};

    constexpr TransformMatrix <Typedefs::DType, 6>  COIFFLET_6      = {{-0.10285945, 0.47785945, 1.20571891, 0.54428108, -0.10285945, -0.02214054}};
    constexpr TransformMatrix <Typedefs::DType, 30> COIFFLET_30     = {{-0.0002999290456692, 0.0005071055047161, 0.0030805734519904, -0.0058821563280714, -0.0143282246988201, 0.0331043666129858, 0.0398380343959686, -0.1299967565094460, -0.0736051069489375, 0.5961918029174380, 1.0950165427080700, 0.6194005181568410, -0.0877346296564723, -0.1492888402656790, 0.0583893855505615, 0.0462091445541337, -0.0279425853727641, -0.0129534995030117, 0.0095622335982613, 0.0034387669687710, -0.0023498958688271, -0.0009016444801393, 0.0004268915950172, 0.0001984938227975, -0.0000582936877724, -0.0000300806359640, 0.0000052336193200, 0.0000029150058427, -0.0000002296399300, -0.0000001358212135}};
}

#endif