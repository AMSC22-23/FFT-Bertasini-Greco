#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>

#include "Image.hpp"
#include "FourierTransform2D.hpp"
#include "typedefs.hpp"
#include "DiscreteWaveletTransform2D.hpp"
#include "TransformMatrices.hpp"

using namespace std;
using namespace cv;
using namespace Typedefs;
using namespace tr;

int main () {
    unique_ptr<Transform<Mat>> tr_obj = make_unique<DiscreteWaveletTransform2D>(TRANSFORM_MATRICES::HAAR, 3);
    double percentile_cutoff = 0.75;
    string compression_method = "levels_cutoff";
    string path = "input/lena.png";
    string tmp;

    cout << "Insert the path to the image:[newline for default] ";
    getline(cin, tmp);
    if (tmp.empty()){
        cout << "Using default image: " << path << endl;
        cout << "Using default transform: Discrete Wavelet Transform HAAR with 3 levels" << endl;
        cout << "Compressing removing " << (int)(percentile_cutoff*100) << "%" << endl;
    }
    else path = tmp;
    Mat og_image = imread(path);
    // Mat og_image = imread("input/milano.jpg");
    if (og_image.empty())
    {
        cout << "Failed to load the image" << endl;
        return -1;
    }
    if (!tmp.empty()){
    int transform_choice;
    cout << "Choose the transform to apply:" << endl;
    cout << "1. Discrete Wavelet Transform" << endl;
    cout << "2. Discrete Fourier Transform" << endl;
    cout << "3. Fast Recursive Fourier Transform" << endl;
    cout << "4. Fast Iterative Fourier Transform" << endl;

    cin >> transform_choice;
    
    switch (transform_choice)
    {
    case 1:
        int levels;
        cout << "Insert the number of levels for the DWT: ";
        cin >> levels;
        tr_obj = make_unique<DiscreteWaveletTransform2D>(TRANSFORM_MATRICES::HAAR, levels);
        break;
    case 2: 
        tr_obj = make_unique<FourierTransform2D<DiscreteFourierTransform>>();
        break;
    case 3:
        tr_obj = make_unique<FourierTransform2D<RecursiveFastFourierTransform>>();
        break;
    case 4:
        tr_obj = make_unique<FourierTransform2D<IterativeFastFourierTransform>>();
        break;
    default:
        cerr << "Invalid choice" << endl;
        return -1;
    }
    
    cout << "Insert the percentile cutoff for the compression: ";
    cin >> percentile_cutoff;

    if (transform_choice == 1)
        compression_method = "levels_cutoff";
    
    else
    {
        cout << "Choose the compression method: " << endl;
        int compression_choice;
        cout << "1. Magnitude filtering" << endl;
        cout << "2. Frequency cutoff" << endl;
        cin >> compression_choice;
        switch (compression_choice)
        {   
            case 1:
                compression_method = "filter_magnitude";
                break;
            case 2:
                compression_method = "filter_freqs";
                break;
            default:
                cerr << "Invalid choice" << endl;
                return -1;
        }
    }
    }
    Image img(og_image, std::move(tr_obj));

    Mat tr_unfiltered, tr_filtered, output_image;
    
    img.transform();
    tr_unfiltered = img.get_tr_coeff();
    img.compress(percentile_cutoff, compression_method);
    tr_filtered = img.get_tr_coeff();

    output_image = img.get_image();

    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", og_image);

    namedWindow("TR Image", WINDOW_NORMAL);
    imshow("TR Image",tr_unfiltered);
    imwrite("output/tr_unfiltered.png", tr_unfiltered);

    namedWindow("TR Image filtered", WINDOW_NORMAL);
    imshow("TR Image filtered", tr_filtered);
    imwrite("output/tr_filtered.png", tr_filtered);

    namedWindow("Inverse TR Image", WINDOW_NORMAL);
    imshow("Inverse TR Image", output_image);
    imwrite("output/output_image.png", output_image);

    waitKey(0);
}