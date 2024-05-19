#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>

#include "Image.hpp"
#include "FourierTransform2D.hpp"
#include "typedefs.hpp"

using namespace std;
using namespace cv;
using namespace Typedefs;

int main () {
    Mat og_image = imread("input/lena.png");
    if (og_image.empty())
    {
        cout << "Failed to load the image" << endl;
        return -1;
    }
    
    shared_ptr<Transform<Mat>> fft_obj = make_shared<FourierTransform2D<IterativeFastFourierTransform>>();

    Image img(og_image, fft_obj);

    img.transform_signal();
    img.filter_magnitude(0.95);

    Mat output_image = img.get_image();
    Mat fft_image = img.get_fft_freqs();

    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", og_image);

    namedWindow("FFT Image", WINDOW_NORMAL);
    imshow("FFT Image", fft_image);

    namedWindow("Inverse FFT Image", WINDOW_NORMAL);
    imshow("Inverse FFT Image", output_image);
    waitKey(0);
}