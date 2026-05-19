# Build Diagnosis

## Original configure attempt

Command attempted before source-code changes:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Result: configure failed before generating build files.

Important log tail:

```text
CMake Error at CMakeLists.txt:35 (message):
  PYARROW_DIR must be defined
```

Additional preset issue:

```bash
cmake --list-presets
```

failed because `CMakeUserPresets.json` includes missing Conan-generated files:

```text
build/Release/generators/CMakePresets.json
build/Debug/generators/CMakePresets.json
```

## Diagnosis

The checked-in CMake file requires Arrow and Parquet libraries from a PyArrow
installation via `-DPYARROW_DIR=<pyarrow package directory>`. System Python in
this environment did not have `pyarrow` installed during the first configure
attempt.

## Minimal fix path

Use a local ignored virtual environment and configure CMake with both the venv
Python executable and its PyArrow package directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install pyarrow numpy pandas
PYARROW_DIR="$(python - <<'PY'
import os, pyarrow
print(os.path.dirname(pyarrow.__file__))
PY
)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPYARROW_DIR="$PYARROW_DIR" \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build -j"$(nproc)"
```
