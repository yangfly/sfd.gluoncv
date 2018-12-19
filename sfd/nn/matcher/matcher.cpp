/*!
 * \file AdditionOP.cpp
 * \brief Mobula op for compensate_matching
 * \paper S3FD: Scale compensation anchor matching strategy
 * \auther: yangfly
 */
#include <iostream>
#include <utility>
#include <algorithm>
#include "mobula_op.h"
using namespace mobula;
using namespace std;

/**
 * \brief the forward kernel implementation of compensate
 * \param gtids          the matched most gtid of each anchor
 * \param ious           iou between each anchor and it's matched most gt
 * \param matches        return match id of each anchor
 * \param batch_size     number of samples
 * \param num_anchor     number of anchors
 * \param num_gt         number of ground truth bboxes
 * \param thre1          typically 0.35 and above necessarily match
 * \param thre2          typically 0.1 and above optionally match
 * \param topk           typically 6
 */
template <typename T1, typename T2>
MOBULA_KERNEL compensate_kernel(const int batch_size,
                         const T1* gtids, const T2* ious, T2* matches,
                         const size_t num_anchor, const size_t num_gt,
                         const float thre1, const float thre2, const size_t topk) {
    parfor(batch_size, [&](int b) {
        size_t offset = b * num_anchor;
        const T1* ids = gtids + offset;
        const T2* iou = ious + offset;
        T2* match = matches + offset;
        vector<size_t> cnts(num_gt, 0);
        vector<vector<pair<T1, T2>>> records(num_gt);
        for (size_t i = 0; i < num_anchor; i++) {
            if (iou[i] >= thre2) {
                records[ids[i]].push_back( {i,iou[i]} );
                if (iou[i] >= thre1) {
                    match[i] = ids[i];
                    cnts[ids[i]]++;
                }
                else
                    match[i] = -1;
            }
            else
                match[i] = -1;
        }
        for (size_t i = 0; i < num_gt; i++) {
            if (cnts[i] < topk) {
                if (records[i].size() > topk) {
                    stable_sort(records[i].begin(), records[i].end(),
                                [](const pair<T1,T2>& x, const pair<T1,T2>& y) { return x.second > y.second; });
                    for (size_t j = 0; j < topk; j++)
                        match[records[i][j].first] = i;
                }
                else {
                    for (auto p : records[i])
                        match[p.first] = i;
                }
            }
        }
    });
}
