# Lightgbm Predict Executable

A simple executable using LightGBM C API calls to run predictions and simulate production inferencing scenarios.

Those building instructions matter only when running the executable locally. You should not need to follow those instructions if you plan to use the executable from AzureML.

## To build locally on Windows

### Build LightGBM first

If you don't already have an existing build of LightGBM:

```bash
git clone --recursive https://github.com/microsoft/LightGBM.git
mkdir build
cd build
cmake -A x64 ..
cmake --build . --target _lightgbm --config Release
```

Do not build with `ALL_BUILD`, as you don't need all the artefacts, just the lightgbm lib. Once build, the directory `LightGBM/Release/` should contain `lib_lightgbm.lib`, `lib_lightgbm.dll` and `lib_lightgbm.exp` (only).

### Build lightgbm-benchmark binaries

(For now, only lightgbm_predict)

Run the following in `src/binaries/`. You will beed to provode paths:
- `-DLIGHTGBM_LIB`: the path to built LightGBM library,
- `-DLIGHTGBM_INC`: the path to the LightGBM includes,
- `-DLIGHTGBM_SRC`: the path to the LightGBM sources.

```bash
mkdir build
cd build
cmake -A x64 -DLIGHTGBM_LIB=___ -DLIGHTGBM_INC=___ -DLIGHTGBM_SRC=___..
cmake --build . --target lightgbm_predict --config Release
```
