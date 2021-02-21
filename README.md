A simple differentiable renderer based on the adjoint method. Written for educational purposes. Building the command-line tool requires CMake (>=3.5) and can be done by running the following commands from the project's root directory:

```
mkdir build
cd build
cmake ..
cmake --build .
```

This will compile the tool using the default release build configuration. On the other hand, compilation using the debug configuration can be performed by running the following instead:

```
cmake .. -D CMAKE_BUILD_TYPE Debug
cmake --build . --config Debug
```
