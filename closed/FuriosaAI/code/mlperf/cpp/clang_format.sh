#!/usr/bin/env bash

if [[ $# -eq 1 ]] && [[ $1 == "fix" ]];
  then
    echo "Correct the format"
    find . -iname "*.hpp" -o -iname "*.h" -o -iname "*.cpp" -o -iname ".cc" | grep -v spdlog | grep -v build | grep -v cmake-build-debug | xargs clang-format --verbose -i
elif [[ $# -eq 1 ]] && [[ $1 == "check" ]];
  then
    echo "Check the format"
    find . -iname "*.hpp" -o -iname "*.h" -o -iname "*.cpp" -o -iname ".cc" | grep -v spdlog | grep -v build | grep -v cmake-build-debug | xargs clang-format --verbose --dry-run --Werror
else
    echo "Error: Not supported Arguments"
    exit -1
fi
