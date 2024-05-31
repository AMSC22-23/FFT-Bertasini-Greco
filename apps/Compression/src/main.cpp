#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "typedefs.hpp"
#include "Compressor.hpp"

using namespace std;
using namespace cv;
using namespace Typedefs;

auto plot_image (vec3D &coeff, string title) -> void {
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

int main()
{
  string img_path = "input/lena_chonky.png";
  Mat og_image = imread(img_path);
  // Mat og_image = imread("input/milano.jpg");
  if (og_image.empty())
  {
    cout << "Failed to load the image" << endl;
    return -1;
  }
  Compressor compressor(og_image, 5);

  compressor.apply_dwt();
  plot_image(compressor.coeff, "DWT Image");
  compressor.quantize();
  plot_image(compressor.coeff, "Quantized Image");
  auto tree = compressor.HuffmanEncoding();
  compressor.HuffmanDecoding(tree);
  plot_image(compressor.coeff, "Decoded Image");
  auto coeff_post = compressor.coeff;
  compressor.dequantize();
  plot_image(compressor.coeff, "Dequantized Image");
  compressor.apply_idwt();
  imshow("Original Image", og_image);
  plot_image(compressor.coeff, "IDWT Image");
  waitKey(0);
}