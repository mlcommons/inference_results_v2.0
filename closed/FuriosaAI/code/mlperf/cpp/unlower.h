#pragma once
#include <tuple>
#include <utility>
#include <variant>

struct LoweredShapeFromRust {
  size_t ho_stride;
  size_t co_stride;
  size_t hi_stride;
  size_t ci_stride;
  size_t w_stride;
  size_t slice_height;
  size_t slice_channel;

  size_t index(size_t c, size_t h, size_t w) const noexcept {
    auto ho = h / slice_height;
    auto hi = h % slice_height;
    auto co = c / slice_channel;
    auto ci = c % slice_channel;

    return ho * ho_stride + co * co_stride + hi * hi_stride + ci * ci_stride + w * w_stride;
  }
};
