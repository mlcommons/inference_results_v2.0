#!/bin/bash
# Edit path as appropriate
BUILD_DIRECTORY=$(dirname $(dirname $(dirname $(realpath "$BASH_SOURCE"))))

# Other relative paths used during build
MLPERF_DIR=${BUILD_DIRECTORY}/MLPerf-Intel-openvino
DEPS_DIR=${MLPERF_DIR}/dependencies
OPENVINO_DIR=${DEPS_DIR}/openvino-repo
TEMPCV_DIR=${OPENVINO_DIR}/inference-engine/temp/opencv_4*
OPENCV_DIRS=$(ls -d -1 ${TEMPCV_DIR} )

# Libraries
OPENVINO_LIBRARIES=${OPENVINO_DIR}/bin/intel64/Release
OMP_LIBRARY=${OPENVINO_DIR}/inference-engine/temp/omp/lib
OPENCV_LIBRARIES=${OPENCV_DIRS[0]}/opencv/lib
BOOST_LIBRARIES=${DEPS_DIR}/boost/boost_1_72_0/stage/lib
GFLAGS_LIBRARIES=${DEPS_DIR}/gflags

#Back up LD_LIBRARY_PATH
export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=${OPENVINO_LIBRARIES}:${OMP_LIBRARY}:${OPENCV_LIBRARIES}:${BOOST_LIBRARIES}:${GFLAGS_LIBRARIES}
export BUILD_DIRECTORY=${BUILD_DIRECTORY}
