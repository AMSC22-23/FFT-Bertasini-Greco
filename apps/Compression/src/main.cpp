#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "typedefs.hpp"
#include "Compressor.hpp"
#include "Decompressor.hpp"

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
  string img_path = "input/lena.png";
  string tmp;
  cout << "Insert the path to the image:[newline for default] ";
  getline(cin, tmp);
  if (tmp.empty()) cout << "Using default image: " << img_path << endl;
  else img_path = tmp;

  int n_levels = 2;
  cout << "Insert the number of levels for the transform:[newline for default] ";
  getline(cin, tmp);
  if (tmp.empty()) cout << "Using default levels: " << n_levels << endl;
  else n_levels = stoi(tmp);

  Mat og_image = imread(img_path), decompressed_image;
  // Mat og_image = imread("input/milano.jpg");
  if (og_image.empty())
  {
    cout << "Failed to load the image" << endl;
    return -1;
  }

  string base_name = img_path.substr(img_path.find_last_of("/\\") + 1);
  base_name = base_name.substr(0, base_name.find_last_of("."));

  Compressor compressor; Decompressor decompressor;

  string compressed_file = "output/" + base_name + ".gb";

  compressor.compress(compressed_file, og_image, n_levels);
  decompressor.decompress(compressed_file, decompressed_image);

  imshow("Original Image", og_image);
  imshow("Decompressed Image", decompressed_image);
  imwrite("output/" + base_name + ".jpg", og_image);
  imwrite("output/" + base_name + "_compressed.jpg", decompressed_image);

  waitKey(0);
}