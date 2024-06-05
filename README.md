# DWT&FFT by Bertasini&Greco

This README is also available on the [doxygen documentation](https://amsc22-23.github.io/FFT-Bertasini-Greco/) of the project.

## Prerequisites

In order to compile and run this program, you need the following:

- C++ compiler
- make
- python3
- matplotlib
- numpy
- opencv (c++)
- CUDA (optional)

Install the required packages:

```
sudo apt-get install g++
sudo apt-get install make
sudo apt-get install python3
sudo apt-get install python3-matplotlib
sudo apt-get install python3-numpy
sudo apt-get install libopencv-dev
```

## Usage

Clone the repository:

```
git clone https://github.com/AMSC22-23/FFT-Bertasini-Greco.git
```

The library provides both a cpu and gpu backend, the gpu backend is implemented using the CUDA library. In order to choose which backend to use, an enviroment variable can be set:

```
export USE_CUDA=1 # to use the gpu backend
export USE_CUDA=0 # to use the cpu backend
```

By default if the nvcc compiler is not found the cpu backend will be used, to use the gpu backend the CUDA toolkit must be installed.

Before using any of the provided applications it is needed to compile the library:

```
cd FFT-Bertasini-Greco
make
```

The library will be compiled in the `lib` folder.

## Applications

Below a general overview of the provided applications, specific information can be found in the next subsections.

### Overview

Three applications are provided in the `apps` folder:

1. `apps/Signal/`: computes a transformation of a regular with noise and denoises it, plots the result.
2. `apps/Image/`: computes a transformation of a given image and plots the result.
3. `apps/Compression/`: compresses an image using the DWT and generates a compressed file.

Each Application can be compiled using the following command:

```
make
```

Once compiled the application can be run using the following command:

```
make run
```

For all of them an interactive session will start asking the user with some kind of parameters, for a faster demo run is also possible to just press enter to use the default parameters.

### Signal

The signal application is loaded with a very simple signal coupled with some noise. The application computes asks the user to choose the transformation to apply to the signal and then plots the results

1. Go to the `apps/Signal` folder

```
cd apps/Signal
```

2. Compile the application

```
make
```

3. Run the application

```
make run
```

### Image

The image application when run asks the user various information about the image to load and the transformation to apply to it, as well as the parameters of the transformation. 

1. Go to the `apps/Image` folder

```
cd apps/Image
```

2. Compile the application

```
make
```

3. Run the application

```
make run
```

By default the application will load the lena image and apply the DWT transformation to it with 3 levels of decomposition and 75% of the coefficients discarded. 

The result will be shown in a window, to close the window press any key. Moreover the application will save the transformed image in the `output` folder.

N.B: for the dwt filtering the application asks for a percentage of the coefficients to discard, since the DWT does not provide any way to filter a specific percentage of the coefficients, the application will discard as many levels as needed to not exceed the percentage of discarded coefficients, eg (extremes included):
0 - 0.74              -> no compression
0.75 - 0.9374         -> 1 level of compression
0.9375 - 0.984374     -> 2 levels of compression
0.984375 - 0.99609375 -> 3 levels of compression


### Compression

The compression application when run asks the user which image to compress and the level (integer) of compression to apply, the bigger the image the higher the level of compression that can be applied. For the stock lena image a level of 2-3 is suggested.

1. Go to the `apps/Compression` folder

```
cd apps/Compression
```

2. Compile the application

```
make
```

3. Run the application

```
make run
```

## Authors

* **Luca Venerando Greco**
* **Maria Aurora Bertasini**