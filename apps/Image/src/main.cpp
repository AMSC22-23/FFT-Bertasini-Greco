#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>

#include "Image.hpp"
#include "FourierTransform2D.hpp"
#include "typedefs.hpp"
#include "DiscreteWaveletTransform2D.hpp"

using namespace std;
using namespace cv;
using namespace Typedefs;

int main () {
    // Mat og_image = imread("input/lena.png");
    Mat og_image = imread("input/milano.jpg");
    if (og_image.empty())
    {
        cout << "Failed to load the image" << endl;
        return -1;
    }
    
    shared_ptr<Transform<Mat>> tr_obj = make_shared<FourierTransform2D<IterativeFastFourierTransform>>();
    // shared_ptr<Transform<Mat>> tr_obj = make_shared<DiscreteWaveletTransform2D<4>>(TRANSFORM_MATRICES::HAAR, 2);

    Image img(og_image, tr_obj);

    Mat tr_unfiltered, tr_filtered, output_image;
    
    img.transform();
    tr_unfiltered = img.get_tr_coeff();
    img.compress(0.95,"filter_magnitude");
    tr_filtered = img.get_tr_coeff();

    output_image = img.get_image();

    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", og_image);

    namedWindow("TR Image", WINDOW_NORMAL);
    imshow("TR Image",tr_unfiltered);

    namedWindow("TR Image filtered", WINDOW_NORMAL);
    imshow("TR Image filtered", tr_filtered);

    namedWindow("Inverse TR Image", WINDOW_NORMAL);
    imshow("Inverse TR Image", output_image);

    waitKey(0);
}