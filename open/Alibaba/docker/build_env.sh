set -e
set -x
SRC_ROOT=$1
BUILD_ROOT=$2
mkdir -p $BUILD_ROOT && cd $BUILD_ROOT

cp -r $SRC_ROOT/* ./

cd $BUILD_ROOT/ComputeLibrary
scons -j96 arch=armv8.6-a-sve2 build=native opencl=yes neon=true embed_kernels=true cppthreads=False extra_cxx_flags="-fPIC" internal_only=0 build_dir=$BUILD_ROOT/ComputeLibrary/build install_dir=$BUILD_ROOT/ComputeLibrary/install benchmark_examples=true Werror=0

cd $BUILD_ROOT/NNPACK
mkdir build && cd build
# Generate ninja build rule and add shared library in configuration
cmake -G Ninja -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCPUINFO_SOURCE_DIR=$BUILD_ROOT/NNPACK/deps/cpuinfo \
  -DFP16_SOURCE_DIR=$BUILD_ROOT/NNPACK/deps/fp16 \
  -DFXDIV_SOURCE_DIR=$BUILD_ROOT/NNPACK/deps/fxdiv \
  -DPSIMD_SOURCE_DIR=$BUILD_ROOT/NNPACK/deps/psimd \
  -DPTHREADPOOL_SOURCE_DIR=$BUILD_ROOT/NNPACK/deps/pthreadpool \
  -DGOOGLETEST_SOURCE_DIR=$BUILD_ROOT/NNPACK/deps/googletest \
  ..

ninja
ninja install

# Add NNPACK lib folder in your ldconfig
echo "/usr/local/lib" >/etc/ld.so.conf.d/nnpack.conf
ldconfig

cd $BUILD_ROOT
cd flatbuffers-1.12.0
rm -f CMakeCache.txt
mkdir build
cd build
CXXFLAGS="-fPIC" cmake .. -DFLATBUFFERS_BUILD_FLATC=1 \
  -DCMAKE_INSTALL_PREFIX:PATH=$BUILD_ROOT/flatbuffers \
  -DFLATBUFFERS_BUILD_TESTS=0
make all install -j96

cd $BUILD_ROOT
cd tensorflow-2.5.1

EXTRA_CXXFLAGS="-include /usr/include/c++/11/limits -DFLATBUFFERS_LOCALE_INDEPENDENT=1" \
  bash ./tensorflow/lite/tools/make/build_aarch64_lib.sh

cd $BUILD_ROOT/tvm
mkdir build && cd build
cmake -DUSE_PROFILER=ON -DUSE_LLVM=/usr/bin/llvm-config-13 \
  -DUSE_BLAS=openblas -DUSE_ARM_COMPUTE_LIB=ON -DUSE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR=$BUILD_ROOT/ComputeLibrary/install \
  -DUSE_RELAY_DEBUG=ON -DUSE_PT_TVMDSOOP=ON -DUSE_TARGET_ONNX=ON -DUSE_NNPACK=$BUILD_ROOT/NNPACK \
  -DUSE_TFLITE=$BUILD_ROOT/tensorflow-2.5.1/tensorflow/lite/tools/make/gen/linux_aarch64/lib \
  -DUSE_TENSORFLOW_PATH=$BUILD_ROOT/tensorflow-2.5.1 -DUSE_FLATBUFFERS_PATH=$BUILD_ROOT/flatbuffers -DUSE_EDGETPU=OFF ..

make -j64

cd $BUILD_ROOT/tvm
cd python
python setup.py install
cd ..

echo "export PYTHONPATH=$BUILD_ROOT/tvm/python" >>~/.bashrc
source ~/.bashrc

cd $BUILD_ROOT/tophub
cp -r tophub ~/.tvm/
