# Lightgbm Predict Executable

A simple executable using LightGBM C API calls to run predictions and simulate production inferencing scenarios.

## To build (on windows)

```bash
git clone --recursive https://github.com/microsoft/LightGBM.git
mkdir build
cd build
cmake -A x64 ..
cmake --build . --target ALL_BUILD --config Release
```
