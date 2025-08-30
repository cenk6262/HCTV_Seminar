## 1. Environmental Requirements  
### To run the reconstruction demo, the following dependencies are required:  
* Python 3.10.X  ***(Important)***
* PyTorch 2.0.0
* torchkbnufft 1.4.0
* [tiny-cuda-nn 1.7](https://github.com/NVlabs/tiny-cuda-nn)
  * tiny cuda instalation guide:
    Begin by cloning this repository and all its submodules using the following command:
```sh
$ git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
$ cd tiny-cuda-nn
```

Then, use CMake to build the project: (on Windows, this must be in a [developer command prompt](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-160#developer_command_prompt))
```sh
tiny-cuda-nn$ cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
tiny-cuda-nn$ cmake --build build --config RelWithDebInfo -j
```

If compilation fails inexplicably or takes longer than an hour, you might be running out of memory. Try running the above command without `-j` in that case.

* imageio 2.18.0
* torchvision, tensorboard, h5py, scikit-image, tqdm, numpy, scipy

## PyTorch extension

__tiny-cuda-nn__ comes with a [PyTorch](https://github.com/pytorch/pytorch) extension that allows using the fast MLPs and input encodings from within a [Python](https://www.python.org/) context.
These bindings can be significantly faster than full Python implementations; in particular for the [multiresolution hash encoding](https://raw.githubusercontent.com/NVlabs/tiny-cuda-nn/master/data/readme/multiresolution-hash-encoding-diagram.png).

> The overheads of Python/PyTorch can nonetheless be extensive if the batch size is small.
> For example, with a batch size of 64k, the bundled `mlp_learning_an_image` example is __~2x slower__ through PyTorch than native CUDA.
> With a batch size of 256k and higher (default), the performance is much closer.

Begin by setting up a Python 3.X environment with a recent, CUDA-enabled version of PyTorch. Then, invoke
```sh
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Alternatively, if you would like to install from a local clone of __tiny-cuda-nn__, invoke
```sh
tiny-cuda-nn$ cd bindings/torch
tiny-cuda-nn/bindings/torch$ python setup.py install
```

## 2. Sample Data
Download the sample data from [https://drive.google.com/file/d/1DIdtHcHUDEqx-qL4930-pz9mxCI8OYMR/view?usp=sharing](https://drive.google.com/file/d/1DIdtHcHUDEqx-qL4930-pz9mxCI8OYMR/view?usp=sharing) and put it into the root directory

