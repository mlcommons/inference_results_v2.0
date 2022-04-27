#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>
#include <string.h>
#include <vector>

#include "bindings.h"
#include "unlower.h"

namespace ssd_large {
constexpr const int num_classes = 81;
constexpr const int channel_count = 15130;
constexpr const int feature_map_shapes[] = {50, 25, 13, 7, 3, 3};
constexpr const int num_anchors[] = {4, 6, 6, 6, 4, 4};
constexpr const float coder_weight[6] = {10, 10, 5, 5};
int prior_base_index[13];
CenteredBox box_priors[channel_count];
const int CHANNEL_COUNT_POW2 = 16384; // 2^n large than channe lcount

LoweredShapeFromRust output_lowering[12];

float output_dequantization_tables[12][256];
float output_exp_dequantization_tables[12][256];
float output_exp_scale_dequantization_tables[12][256];

float SCORE_THRESHOLD = 0.05f;
const int max_detections = 200;

void init(float *output_deq_tables_ptr, float *output_exp_deq_tables_ptr, float *output_exp_scale_deq_tables_ptr,
          LoweredShapeFromRust *score_lowered_shapes_ptr, LoweredShapeFromRust *box_lowered_shapes_ptr,
          CenteredBox *box_priors_ptr) {
  memcpy(output_dequantization_tables, output_deq_tables_ptr, sizeof(output_dequantization_tables));
  memcpy(output_exp_dequantization_tables, output_exp_deq_tables_ptr, sizeof(output_exp_dequantization_tables));
  memcpy(output_exp_scale_dequantization_tables, output_exp_scale_deq_tables_ptr,
         sizeof(output_exp_scale_dequantization_tables));
  memcpy(box_priors, box_priors_ptr, channel_count * sizeof(CenteredBox));

  prior_base_index[0] = 0;
  for (int i = 0; i < 5; i++) {
    prior_base_index[i + 1] = prior_base_index[i] + num_anchors[i] * feature_map_shapes[i] * feature_map_shapes[i];
  }

  for (int i = 0; i < 6; i++) {
    prior_base_index[i + 6] = prior_base_index[i];
  }

  for (int i = 0; i < 6; i++) {
    output_lowering[i] = score_lowered_shapes_ptr[i];
    output_lowering[i + 6] = box_lowered_shapes_ptr[i];
  }
}

inline void fill_boxes(std::array<BoundingBox, channel_count> &boxes, const U8Slice *buffers, int output_index,
                       int anchor_index, int f_y, int f_x) {
  int tile_index = output_index - 6;
  auto info = output_lowering[output_index];
  auto num_anchor = num_anchors[tile_index];
  const int feature_index = f_y * feature_map_shapes[tile_index] + f_x;

  auto q0 = (unsigned char)buffers[output_index].ptr[info.index(anchor_index + num_anchor * 0, f_y, f_x)];
  auto q1 = (unsigned char)buffers[output_index].ptr[info.index(anchor_index + num_anchor * 1, f_y, f_x)];
  auto q2 = (unsigned char)buffers[output_index].ptr[info.index(anchor_index + num_anchor * 2, f_y, f_x)];
  auto q3 = (unsigned char)buffers[output_index].ptr[info.index(anchor_index + num_anchor * 3, f_y, f_x)];

  CenteredBox box{output_dequantization_tables[output_index][q1], output_dequantization_tables[output_index][q0],
                  output_exp_scale_dequantization_tables[output_index][q3],
                  output_exp_scale_dequantization_tables[output_index][q2]};
  const int prior_index = prior_base_index[tile_index] + feature_index +
                          anchor_index * feature_map_shapes[tile_index] * feature_map_shapes[tile_index];
  CenteredBox box_prior = box_priors[prior_index];
  CenteredBox adjusted = adjust(&box_prior, box);

  auto idx = anchor_index * feature_map_shapes[tile_index] * feature_map_shapes[tile_index] +
             f_y * feature_map_shapes[tile_index] + f_x + prior_base_index[tile_index];
  boxes[idx] = BoundingBox{py1(&adjusted), px1(&adjusted), py2(&adjusted), px2(&adjusted)};
}

inline void decode_score(const U8Slice *buffers, std::array<float, channel_count> *scores,
                         std::array<float, 16> *scores_sum) {
#pragma omp parallel for num_threads(82)
  for (int class_index = 0; class_index < num_classes; class_index++)
    for (int output_index = 0; output_index < 6; output_index++) {
      auto info = output_lowering[output_index];
      auto tile_index = output_index;
      auto num_anchor = num_anchors[tile_index];

      for (int anchor_index = 0; anchor_index < num_anchor; anchor_index++)
        for (int h = 0; h < feature_map_shapes[tile_index]; h++)
          for (int w = 0; w < feature_map_shapes[tile_index]; w++) {
            const auto q = buffers[output_index].ptr[info.index(class_index * num_anchor + anchor_index, h, w)];
            const auto score = output_exp_dequantization_tables[output_index][q];
            const int scores_sum_index =
                anchor_index * feature_map_shapes[tile_index] * feature_map_shapes[tile_index] +
                h * feature_map_shapes[tile_index] + w + prior_base_index[output_index];
            scores[class_index][scores_sum_index] = score;
          }
    }

#pragma omp parallel for num_threads(82)
  for (int i = 0; i < channel_count; i++) {
    scores_sum[i][0] = 0;
    for (int class_index = 0; class_index < num_classes; class_index++)
      scores_sum[i][0] += scores[class_index][i];
  }
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

inline void filter_results_per_class(std::array<float, channel_count> *scores, std::array<float, 16> *scores_sum,
                                     const U8Slice *buffers, std::array<BoundingBox, channel_count> &boxes,
                                     std::array<std::vector<std::tuple<float, int>>, num_classes> &picked,
                                     float nms_threshold, int class_index) {
  std::vector<std::tuple<float, int>> filtered;
  filtered.reserve(50 * 50 * 4);

  for (int output_index = 0; output_index < 6; output_index++) {
    int tile_index = output_index;

    for (int anchor_index = 0; anchor_index < num_anchors[tile_index]; ++anchor_index) {
      for (int f_y = 0; f_y < feature_map_shapes[tile_index]; f_y++)
        for (int f_x = 0; f_x < feature_map_shapes[tile_index]; f_x++) {

          int score_index = anchor_index * feature_map_shapes[tile_index] * feature_map_shapes[tile_index] +
                            f_y * feature_map_shapes[tile_index] + f_x;
          int channel_index = score_index + prior_base_index[tile_index];
          if (scores[class_index][channel_index] / scores_sum[channel_index][0] > SCORE_THRESHOLD) {
            fill_boxes(boxes, buffers, output_index + 6, anchor_index, f_y, f_x);
            float score = scores[class_index][channel_index] / scores_sum[channel_index][0];
            assert(score > SCORE_THRESHOLD);
            filtered.emplace_back(score,
                                  anchor_index * feature_map_shapes[tile_index] * feature_map_shapes[tile_index] +
                                      f_y * feature_map_shapes[tile_index] + f_x + prior_base_index[tile_index]);
          }
        }
    }
  }

  // look through boxes in the descending order
  auto part_end2 = filtered.size() > max_detections ? filtered.begin() + max_detections : filtered.end();
  std::partial_sort(filtered.begin(), part_end2, filtered.end(),
                    [](auto l, auto r) { return std::get<0>(l) > std::get<0>(r); });
  filtered.erase(part_end2, filtered.end());

  std::vector<std::tuple<float, int>> &l_picked = picked[class_index];

  l_picked.reserve(max_detections);
  l_picked.clear();
  for (auto &[l_score, l_boxes_index] : filtered) {
    bool can_pick = std::all_of(l_picked.begin(), l_picked.end(), [&](auto &r) {
      auto [r_score, r_boxes_index] = r;
      return iou(boxes[l_boxes_index], boxes[r_boxes_index % CHANNEL_COUNT_POW2]) <= nms_threshold;
    });
    if (can_pick) {
      l_picked.emplace_back(l_score, l_boxes_index + class_index * CHANNEL_COUNT_POW2);
    }
  }
}

void filter_results(std::array<float, channel_count> *scores, std::array<float, 16> *scores_sum, const U8Slice *buffers,
                    std::array<BoundingBox, channel_count> &boxes,
                    std::array<std::vector<std::tuple<float, int>>, num_classes> &picked, float nms_threshold = 0.5f) {
  for (int class_index = 1; class_index < num_classes; ++class_index) {
    filter_results_per_class(scores, scores_sum, buffers, boxes, picked, nms_threshold, class_index);
  }
}

void filter_results_parallel(std::array<float, channel_count> *scores, std::array<float, 16> *scores_sum,
                             const U8Slice *buffers, std::array<BoundingBox, channel_count> &boxes,
                             std::array<std::vector<std::tuple<float, int>>, num_classes> &picked,
                             float nms_threshold = 0.5f) {
#pragma omp parallel for num_threads(82)
  for (int class_index = 1; class_index < num_classes; ++class_index) {
    filter_results_per_class(scores, scores_sum, buffers, boxes, picked, nms_threshold, class_index);
  }
}

template <bool parallel>
size_t post_inference(float result_index, const U8Slice *buffers, DetectionResult *result_ptr) {

  std::array<BoundingBox, channel_count> boxes;

  thread_local std::vector<std::array<float, 16>> scores_sum_;
  scores_sum_.resize(channel_count);
  std::array<float, 16> *scores_sum = &scores_sum_[0];
  thread_local std::vector<std::array<float, channel_count>> scores;
  scores.resize(num_classes);
  decode_score(buffers, &scores[0], scores_sum);

  thread_local std::array<std::vector<std::tuple<float, int>>, num_classes> picked;
  if (parallel) {
    filter_results_parallel(&scores[0], scores_sum, buffers, boxes, picked);
  } else {
    filter_results(&scores[0], scores_sum, buffers, boxes, picked);
  }

  thread_local std::vector<std::tuple<float, int>> results;
  results.reserve(num_classes * max_detections);
  results.clear();
  for (int class_index = 1; class_index < num_classes; ++class_index) {
    results.insert(results.end(), picked[class_index].begin(), picked[class_index].end());
  }

  if (results.size() > max_detections) {
    std::partial_sort(results.begin(), results.begin() + max_detections, results.end(),
                      [&](const auto &l, const auto &r) { return std::get<0>(l) > std::get<0>(r); });
    results.erase(results.begin() + max_detections, results.end());
  }

  int result_count = 0;
  for (auto &[score, class_index_and_box] : results) {
    if (result_count >= max_detections)
      break;
    auto boxes_idx = class_index_and_box % CHANNEL_COUNT_POW2;
    auto class_index = class_index_and_box / CHANNEL_COUNT_POW2;

    result_ptr[result_count] = DetectionResult{result_index, boxes[boxes_idx], score, static_cast<float>(class_index)};
    result_count++;
  }

  return result_count;
}

} // namespace ssd_large
