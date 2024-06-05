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

Compile the library:

```
cd FFT-Bertasini-Greco
make
```

The library will be compiled in the `lib` folder.

## Applications

Three applications are provided in the `app` folder:

1. `app/Signal`: computes a transformation of a regular with noise and denoises it, plots the result.
2. `app/Image`: computes a ètransformation of a given image and plots the result.
3. `app/Compression`: compresses an image using the DWT and generates a compressed file.

Each Application can be compiled using the following command:

```
make
```

Once compiled the application can be run using the following command:

```
make run
```

For all of them an interactive session will start asking the user with some kind of parameters, for a faster demo run is also possible to just press enter to use the default parameters.


## Authors

* **Luca Venerando Greco**
* **Maria Aurora Bertasini**