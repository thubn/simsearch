# Gettings started

```
conan install . --output-folder=. --build=missing
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```
OR
```
conan install . -s build_type=Debug --output-folder=. --build=missing
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```