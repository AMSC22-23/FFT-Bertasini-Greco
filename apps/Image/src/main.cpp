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

/*int main()
{
    // Load the input image
    Mat og_image = imread("input/lena.png");
    if (og_image.empty())
    {
        cout << "Failed to load the image" << endl;
        return -1;
    }

    I

    // Convert the input image to double precision
    Mat input_image;
    og_image.convertTo(input_image, CV_64FC3);

    unique_ptr<Transform<Mat>> fft_obj = make_unique<FourierTransform2D<IterativeFastFourierTransform>>();

    auto in_space = fft_obj->get_input_space(input_image);
    auto out_space = fft_obj->get_output_space();

    // Compute the 2D FFT
    vcpx3D fft_coeff;
    fft_obj->operator()(*in_space, *out_space, false);

    // Display the original image and the FFT image
    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", og_image);

    // pass_filter(fft_coeff, .95, true);
    out_space->compress("filter_magnitude", 0.95);

    // Create grayscale image from FFT image
    Mat fft_image_colored = out_space->get_plottable_representation();
    imshow("FFT Image", fft_image_colored);

    // Compute the inverse 2D FFT
    fft_obj->operator()(*in_space, *out_space, true);

    Mat output_image = in_space->get_data();

    // Display the inverse FFT image
    namedWindow("Inverse FFT Image", WINDOW_NORMAL);
    imshow("Inverse FFT Image", output_image);
    waitKey(0);

    return 0;
}*/