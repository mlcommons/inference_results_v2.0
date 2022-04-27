#pragma once
#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <optional>
#include <queue>
#include <string.h>
#include <vector>

#include "bindings.h"
#include "unlower.h"

namespace ssd_small {
constexpr const int num_classes = 91;
constexpr const int channel_count = 1917;
constexpr const int feature_map_shapes[] = {19, 10, 5, 3, 2, 1};
constexpr const int num_anchors[] = {3, 6, 6, 6, 6, 6};
constexpr const float coder_weight[6] = {10, 10, 5, 5};
int prior_base_index[13];
CenteredBox box_priors[channel_count];

LoweredShapeFromRust output_lowering[12];

float output_dequantization_tables[12][256];
float output_exp_dequantization_tables[12][256];

int16_t score_thresholds[6];
float SCORE_THRESHOLD = 0.3f;
float NMS_THRESHOLD = 0.6f;

void init(float *output_deq_tables_ptr, float *output_exp_scale_deq_tables_ptr,
          LoweredShapeFromRust *score_lowered_shapes_ptr, LoweredShapeFromRust *box_lowered_shapes_ptr,
          CenteredBox *box_priors_ptr) {
  memcpy(output_dequantization_tables, output_deq_tables_ptr, sizeof(output_dequantization_tables));
  memcpy(output_exp_dequantization_tables, output_exp_scale_deq_tables_ptr, sizeof(output_exp_dequantization_tables));
  memcpy(box_priors, box_priors_ptr, channel_count * sizeof(CenteredBox));

  prior_base_index[0] = 0;
  for (int i = 0; i < 5; ++i) {
    prior_base_index[i + 1] = prior_base_index[i] + num_anchors[i] * feature_map_shapes[i] * feature_map_shapes[i];
  }

  for (int i = 0; i < 6; ++i) {
    prior_base_index[i + 6] = prior_base_index[i];
  }

  for (int i = 0; i < 6; ++i) {
    output_lowering[i] = score_lowered_shapes_ptr[i];
    output_lowering[i + 6] = box_lowered_shapes_ptr[i];
  }

  for (int i = 0; i < 6; ++i) {
    score_thresholds[i] = 128;
    for (int j = -128; j <= 127; ++j) {
      uint8_t index = static_cast<uint8_t>(j);
      if (output_dequantization_tables[i][index] > SCORE_THRESHOLD) {
        score_thresholds[i] = j;
        break;
      }
    }
  }
}

inline BoundingBox fill_boxes(const U8Slice *buffers, int output_index, int anchor_index, int f_y, int f_x) {
  int tile_index = output_index - 6;
  auto info = output_lowering[output_index];

  const int feature_index = f_y * feature_map_shapes[tile_index] + f_x;

  auto q0 = (unsigned char)buffers[output_index].ptr[info.index(anchor_index * 4 + 0, f_y, f_x)];
  auto q1 = (unsigned char)buffers[output_index].ptr[info.index(anchor_index * 4 + 1, f_y, f_x)];
  auto q2 = (unsigned char)buffers[output_index].ptr[info.index(anchor_index * 4 + 2, f_y, f_x)];
  auto q3 = (unsigned char)buffers[output_index].ptr[info.index(anchor_index * 4 + 3, f_y, f_x)];

  CenteredBox box{output_dequantization_tables[output_index][q0], output_dequantization_tables[output_index][q1],
                  output_exp_dequantization_tables[output_index][q2],
                  output_exp_dequantization_tables[output_index][q3]};
  CenteredBox box_prior =
      box_priors[prior_base_index[tile_index] + feature_index * num_anchors[tile_index] + anchor_index];
  CenteredBox adjusted = adjust(&box_prior, box);

  return BoundingBox{py1(&adjusted), px1(&adjusted), py2(&adjusted), px2(&adjusted)};
}

inline float iou(const BoundingBox &b1, const BoundingBox &b2) {
  const float eps = 1e-5f;
  const auto clamp_lower = [](float x) -> float {
    if (x < 0)
      return 0;
    return x;
  };
  float area1 = (b1.px2 - b1.px1) * (b1.py2 - b1.py1);
  float area2 = (b2.px2 - b2.px1) * (b2.py2 - b2.py1);
  float cw = clamp_lower(std::min(b1.px2, b2.px2) - std::max(b1.px1, b2.px1));
  float ch = clamp_lower(std::min(b1.py2, b2.py2) - std::max(b1.py1, b2.py1));
  float overlap = cw * ch;
  return overlap / (area1 + area2 - overlap + eps);
}

inline void filter_results_per_class(const U8Slice *buffers,
                                     std::array<std::vector<std::tuple<float, BoundingBox>>, num_classes> &picked,
                                     int class_index) {
  using value_t = std::tuple<float, BoundingBox>;
  using container_t = std::vector<value_t>;
  auto comp = [](const value_t &left, const value_t &right) { return std::get<0>(left) < std::get<0>(right); };

  container_t inner;
  inner.reserve(200);

  std::priority_queue<value_t, container_t, decltype(comp)> heap{comp, std::move(inner)};

  for (int output_index = 0; output_index < 6; ++output_index) {
    int tile_index = output_index;
    auto info = output_lowering[output_index];
    int16_t score_threshold = score_thresholds[tile_index];
    // if score_threshold is greater then or equal to 128, it means there is
    // no element that exceeds SCORE_THRESHOLD.
    if (score_threshold < 128) {
      for (int anchor_index = 0; anchor_index < num_anchors[tile_index]; ++anchor_index) {
        for (int f_y = 0; f_y < feature_map_shapes[tile_index]; ++f_y) {
          for (int f_x = 0; f_x < feature_map_shapes[tile_index]; ++f_x) {

            int score_index = info.index(anchor_index * num_classes + class_index, f_y, f_x);
            int8_t q = static_cast<int8_t>(buffers[tile_index].ptr[score_index]);
            if (q >= static_cast<int8_t>(score_threshold)) {
              float score = output_dequantization_tables[tile_index][static_cast<uint8_t>(q)];
              auto box = fill_boxes(buffers, output_index + 6, anchor_index, f_y, f_x);
              heap.emplace(score, box);
            }
          }
        }
      }
    }
  }

  std::vector<std::tuple<float, BoundingBox>> &l_picked = picked[class_index];
  l_picked.reserve(200);
  l_picked.clear();
  for (; !heap.empty(); heap.pop()) {
    auto &[l_score, l_box] = heap.top();

    bool can_pick = std::all_of(l_picked.begin(), l_picked.end(), [&](auto &r) {
      auto &[r_score, r_box] = r;
      return iou(l_box, r_box) <= NMS_THRESHOLD;
    });

    if (can_pick) {
      l_picked.emplace_back(l_score, l_box);
    }
  }
}

inline void filter_results(const U8Slice *buffers,
                           std::array<std::vector<std::tuple<float, BoundingBox>>, num_classes> &picked) {
  for (int class_index = 1; class_index < num_classes; ++class_index) {
    filter_results_per_class(buffers, picked, class_index);
  }
}

inline void filter_results_parallel(const U8Slice *buffers,
                                    std::array<std::vector<std::tuple<float, BoundingBox>>, num_classes> &picked) {

#pragma omp parallel for num_threads(16)
  for (int class_index = 1; class_index < num_classes; ++class_index) {
    filter_results_per_class(buffers, picked, class_index);
  }
}

template <bool parallel>
size_t post_inference(float result_index, const U8Slice *buffers, DetectionResult *result_ptr) {

  thread_local std::array<std::vector<std::tuple<float, BoundingBox>>, num_classes> picked;
  if (parallel) {
    filter_results_parallel(buffers, picked);
  } else {
    filter_results(buffers, picked);
  }

  int result_count = 0;
  for (int idx = 0; idx < num_classes; ++idx) {
    for (auto &[score, box] : picked[idx]) {
      if (result_count >= 200)
        break;

      result_ptr[result_count] = DetectionResult{result_index, box, score, static_cast<float>(idx)};
      result_count++;
    }
  }

  return result_count;
}
} // namespace ssd_small
