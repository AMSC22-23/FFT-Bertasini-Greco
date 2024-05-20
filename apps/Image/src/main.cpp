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

    Mat fft_unfiltered, fft_filtered, output_image;
    
    img.transform_signal();
    fft_unfiltered = img.get_fft_freqs();
    img.filter_magnitude(0.95);
    fft_filtered = img.get_fft_freqs();

    output_image = img.get_image();

    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", og_image);

    namedWindow("FFT Image", WINDOW_NORMAL);
    imshow("FFT Image", fft_unfiltered);

    namedWindow("FFT Image filtered", WINDOW_NORMAL);
    imshow("FFT Image filtered", fft_filtered);

    namedWindow("Inverse FFT Image", WINDOW_NORMAL);
    imshow("Inverse FFT Image", output_image);

    waitKey(0);
}