set -e
set -x
SRC_ROOT=$1
mkdir -p $SRC_ROOT && cd $SRC_ROOT
# download acl
git clone https://github.com/ARM-software/ComputeLibrary.git
cd ComputeLibrary
git checkout v21.11

cd $SRC_ROOT
git clone --recursive https://github.com/Maratyszcza/NNPACK.git
cd NNPACK
git checkout c07e3a0
# Add PIC option in CFLAG and CXXFLAG to build NNPACK shared library
sed -i "s|gnu99|gnu99 -fPIC|g" CMakeLists.txt
sed -i "s|gnu++11|gnu++11 -fPIC|g" CMakeLists.txt

mkdir deps && cd deps
git clone https://github.com/Maratyszcza/cpuinfo.git
git clone https://github.com/Maratyszcza/FP16.git fp16
git clone https://github.com/Maratyszcza/FXdiv.git fxdiv
wget https://github.com/google/googletest/archive/release-1.8.0.zip &&
  unzip release-1.8.0.zip && rm -f release-1.8.0.zip && mv googletest* googletest
git clone https://github.com/Maratyszcza/psimd.git
git clone https://github.com/Maratyszcza/pthreadpool.git
wget https://pypi.python.org/packages/bf/3e/31d502c25302814a7c2f1d3959d2a3b3f78e509002ba91aea64993936876/enum34-1.1.6.tar.gz &&
  tar xvfz enum* && rm -f *.tar.gz && mv enum* enum
wget https://pypi.python.org/packages/e8/59/8c2e293c9c8d60f206fd5d0f6c8236a2e0a97832379ac319077441552c6a/opcodes-0.3.13.tar.gz &&
  tar xvfz opcodes* && rm -f *.tar.gz && mv opcodes* opcodes
git clone https://github.com/Maratyszcza/PeachPy.git peachpy
wget https://pypi.python.org/packages/16/d8/bc6316cf98419719bd59c91742194c111b6f2e85abac88e496adefaf7afe/six-1.11.0.tar.gz &&
  tar xvfz six* && rm -f *.tar.gz && mv six* six
#cd ../
#cd ../

cd $SRC_ROOT
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
git checkout c3ace209
cd ../

cd $SRC_ROOT
git clone https://github.com/tlc-pack/tophub

cd $SRC_ROOT
wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.5.1.tar.gz
tar xfz v2.5.1.tar.gz
cd tensorflow-2.5.1
bash ./tensorflow/lite/tools/make/download_dependencies.sh

cd $SRC_ROOT
wget -O flatbuffers-1.12.0.tar.gz https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz
tar xfz flatbuffers-1.12.0.tar.gz
