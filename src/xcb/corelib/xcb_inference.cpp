#include "xcb_inference.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/LU>
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>
#include <math.h>
#include <random>

typedef Eigen::SparseMatrix<float> eigen_colmajor;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> eigen_rowmajor;
typedef std::size_t size_t;
typedef std::pair<size_t, float> score;
typedef std::pair<int, size_t> node;
typedef std::pair<node, size_t> chunk;

float combine(float x, float y, std::string combiner) {
  if (combiner.compare("noop") == 0) {
    return y;
  }
  if (combiner.compare("add") == 0) {
    return x + y;
  }
  if (combiner.compare("multiply") == 0) {
    return x * y;
  }
  throw "combiner has not been implemented!";
  return 0;
}

float transform(float x, std::string post_processor) {
  if (post_processor.compare("noop") == 0) {
    return x;
  }
  if (post_processor.compare("sigmoid") == 0) {
    return 1.0 / (1.0 + std::exp(-x));
  }
  if (post_processor.compare("l3-hinge") == 0) {
    float z = std::max(0.0, 1.0 - x);
    return std::exp(-std::pow(z, 3));
  }
  throw "post_processor has not been implemented!";
}

/*!  Adds elements to the next level of the model_chain.
    It takes in weight matrix and cluster matrix, and pushes them at the end of
    the weights and clusters.

*/
void ModelChain::add_elements(eigen_colmajor weight, eigen_colmajor cluster,
                              eigen_colmajor regression_weight) {
  this->weights.push_back(weight);
  this->clusters.push_back(cluster);
  this->regression_weights.push_back(regression_weight);
  this->size = weights.size();
  this->num_labels = weight.cols();
}

/*!  Perform xcb beam search

    in_vector: input sparse vector
    beam_size: beam_size to be used
    topk: number of labels to fetch
    num_explore: number of slots allowed for exploration
    multiplier: falcon parameter

    returns:
    prediction and score of singleton arms
    mapping of singleton arms to chunks if required

*/
std::pair<std::vector<score>, std::vector<chunk>> ModelChain::beam_search(
    eigen_rowmajor &in_vector, int beam_size, int topk, int num_explore,
    float multiplier, std::string post_processor, std::string combiner,
    bool explore_in_routing, std::string explore_strategy, float alpha) {
  std::vector<score> codes;
  score cp(0, 1.0);
  codes.push_back(cp);
  std::vector<score> top_scores;
  std::vector<size_t> labels;
  std::vector<chunk> selected_chunks;
  std::vector<node> rejected_nodes;
  for (int d = 0; d < this->size; ++d) {
    std::vector<score> labels = this->get_labels(codes, this->clusters[d]);
    int get_size;
    if (d == this->size - 1) {
      get_size = topk;
    } else {
      get_size = beam_size;
    }
    top_scores = this->get_top_labels(
        in_vector, d, labels, get_size, num_explore, multiplier, post_processor,
        combiner, explore_in_routing, explore_strategy, rejected_nodes,
        selected_chunks, alpha);
    // get top scores for level d in the model chain
    codes.clear();
    codes = top_scores;
    // only do the beam search starting from these top-score labels in the next
    // round.
  }
  std::pair<std::vector<score>, std::vector<chunk>> result(top_scores,
                                                           selected_chunks);
  return result;
}

/*!  Get all labels that connect with the given codes on a given level.
    It takes in the codes as a vector and cluster matrix.

    codes: the input codes for the level
    cluster_mat: the clustering matrix from this->clusters[level] is used to
   infer the connections.

*/
std::vector<score> ModelChain::get_labels(std::vector<score> &codes,
                                          eigen_colmajor &cluster_mat) {
  std::vector<score> labels;
  for (int i = 0; i < codes.size(); i++) {
    size_t k = codes[i].first;
    float val = codes[i].second;
    if ((k >= cluster_mat.cols()) || (k < 0)) {
      throw "column does not exist in cluster matrix!";
      continue;
    }
    for (eigen_colmajor::InnerIterator it(cluster_mat, k); it; ++it) {
      score cp(it.index(), val);
      labels.push_back(cp);  // inner index, here it is equal to it.row()
    }
  }
  return labels;
}

/*!  Get a random arm from the subtree.

*/

size_t ModelChain::get_random_arm(node &current_node) {
  auto d = current_node.first;
  std::vector<score> codes;
  score cp(current_node.second, 1.0);
  codes.push_back(cp);
  for (int i = d + 1; i < this->size; i++) {
    std::vector<score> labels = this->get_labels(codes, this->clusters[i]);
    int index = std::rand() % labels.size();
    codes.clear();
    codes.push_back(labels[index]);
  }
  return codes[0].first;
}

/*  Get all top labels from a level given input std::vector and a list of
   allowed labels.

    in_vector: input vector
    weights: weight matrix for a given level
    labels: allowed set of labels
    topk: number of top labels to withdraw

*/
std::vector<score> ModelChain::get_top_labels(
    eigen_rowmajor &in_vector, int level, std::vector<score> &labels, int topk,
    int num_explore, float multiplier, std::string post_processor,
    std::string combiner, bool explore_in_routing, std::string explore_strategy,
    std::vector<node> &rejected_nodes, std::vector<chunk> &selected_chunks,
    float alpha) {
  eigen_colmajor &weights = this->weights[level];
  if (level == this->size - 1) {
    weights = this->regression_weights[level];
  }  // singleton arms need regression weights
  std::vector<std::pair<float, size_t>> score_list;
  for (int i = 0; i < labels.size(); i++) {
    size_t k = labels[i].first;
    float prev_score = labels[i].second;
    if ((k >= weights.cols()) || (k < 0)) {
      throw "column does not exist in weight matrix!";
      continue;
    }
    float sum = 0;
    for (eigen_rowmajor::InnerIterator it(in_vector, 0); it; ++it) {
      sum += it.value() * weights.coeffGet(it.index(), k);
    }
    if (level < this->size - 1) {
      sum = transform(sum, post_processor);  // using post processing
      sum = combine(prev_score, sum, combiner);
    }
    std::pair<float, size_t> cp(sum, k);
    score_list.push_back(cp);  // pushing f(x; l) to the score list
  }
  // add rejected nodes to scores.
  if (level == this->size - 1) {
    for (int i = 0; i < rejected_nodes.size(); i++) {
      node n = rejected_nodes[i];
      weights = this->regression_weights[n.first];
      size_t k = n.second;
      if ((k >= weights.cols()) || (k < 0)) {
        throw "column does not exist in weight matrix!";
        continue;
      }
      float sum = 0;
      for (eigen_rowmajor::InnerIterator it(in_vector, 0); it; ++it) {
        sum += it.value() * weights.coeffGet(it.index(), k);
      }
      std::pair<float, size_t> cp(sum, 2 * this->num_labels + i);
      score_list.push_back(cp);
    }
  }
  auto top_scores =
      this->get_top_scores(score_list, topk, level, num_explore, multiplier,
                           explore_in_routing, explore_strategy, alpha);
  // adding to rejected nodes
  if (level < this->size - 1) {
    std::map<size_t, int> accepted;
    for (int i = 0; i < top_scores.size(); i++) {
      accepted[top_scores[i].first] = 1;
    }
    for (int i = 0; i < labels.size(); i++) {
      auto elem = labels[i].first;
      if (accepted.find(elem) == accepted.end()) {
        node cp(level, elem);
        rejected_nodes.push_back(cp);
      }
    }
    return top_scores;
  } else {
    std::vector<score> actual_scores;  // mapping from chunks
    for (int i = 0; i < top_scores.size(); i++) {
      auto elem = top_scores[i];
      if (elem.first < this->num_labels) {
        actual_scores.push_back(elem);
      } else if (elem.first - 2 * this->num_labels >= 0) {
        node current_node = rejected_nodes[elem.first - 2 * this->num_labels];
        auto arm = this->get_random_arm(current_node);
        score cp(arm, elem.second);
        actual_scores.push_back(cp);
        chunk selected(current_node, arm);
        selected_chunks.push_back(selected);
      }
    }
    return actual_scores;
  }
}

/*! get top scores given a score list along with exploration.



*/

std::vector<score> ModelChain::get_top_scores(
    std::vector<std::pair<float, size_t>> &score_list, int topk, int level,
    int num_explore, float multiplier, bool explore_in_routing,
    std::string explore_strategy, float alpha) {
  std::vector<score> top_scores;
  int num_exp;
  float mult = multiplier;
  if (level < this->size - 1) {
    num_exp = 1;
    if (!explore_in_routing) {
      mult = -1.0;
    }
  } else {
    num_exp = num_explore;
  }
  if ((mult < 0) || (score_list.size() < topk)) {
    // exploit only mode
    std::priority_queue<std::pair<float, size_t>> score_q;
    for (int i = 0; i < score_list.size(); i++) {
      std::pair<float, size_t> cp(score_list[i].first, score_list[i].second);
      score_q.push(cp);
    }

    for (int k = 0; k < topk; ++k) {
      if (score_q.empty()) {
        break;
      }
      auto top_elem = score_q.top();
      score cp(top_elem.second, top_elem.first);
      top_scores.push_back(cp);
      score_q.pop();
    }

  } else {
    // exploration is allowed
    int count = 0;
    int num_exploit = topk - num_exp;
    while (count < topk) {
      auto cp = this->get_sample(score_list, multiplier, level, num_exploit,
                                 explore_strategy, alpha);
      top_scores.push_back(cp);
      count++;
    }
  }
  return top_scores;
}

/* Get sample from Falcon distribution
    score_list: list for scores and corresponding indices
    multiplier: falcon parameter
    level: level in the tree
    num_exploit: top num_exploit slots are always fetched and then exploration
   begins.
*/
score ModelChain::get_sample(std::vector<std::pair<float, size_t>> &score_list,
                             float multiplier, int level, int &num_exploit,
                             std::string explore_strategy, float alpha) {
  int k = score_list.size();
  float maxima = -std::numeric_limits<float>::infinity();
  int max_index = -1;
  int ind;
  std::random_device rd;
  std::mt19937 gen(rd());
  for (int i = 0; i < k; i++) {
    if (score_list[i].first > maxima) {
      maxima = score_list[i].first;
      max_index = i;
    }
  }
  if (num_exploit > 0) {
    ind = max_index;
    num_exploit--;
  } else {
    std::vector<float> weights;
    float psum = 0;
    float eta = std::pow(float(k) * multiplier, alpha);
    // creating distributions
    if (explore_strategy.compare("falcon") == 0) {
      for (int i = 0; i < k; i++) {
        if (i != max_index) {
          float p = 1.0 / (float(k) + eta * (maxima - score_list[i].first));
          weights.push_back(p);
          psum = psum + p;
        } else {
          weights.push_back(0.0);
        }
      }
      weights[max_index] = 1.0 - psum;
    } else if (explore_strategy.compare("e-greedy") == 0) {
      for (int i = 0; i < k; i++) {
        if (i != max_index) {
          float p = multiplier / float(k);
          weights.push_back(p);
          psum = psum + p;
        } else {
          weights.push_back(0.0);
        }
      }
      weights[max_index] = 1.0 - psum;
    } else if (explore_strategy.compare("boltzmann") == 0) {
      for (int i = 0; i < k; i++) {
        float p = std::exp(multiplier * score_list[i].first);
        weights.push_back(p);
        psum = psum + p;
      }
      for (int i = 0; i < k; i++) {
        weights[i] /= psum;
      }
    } else {
      throw "explore strategy not implemented!";
    }
    std::discrete_distribution<> d(weights.begin(), weights.end());
    ind = d(gen);
  }
  auto elem = score_list[ind];
  score cp(elem.second, elem.first);
  score_list[ind] = score_list[score_list.size() - 1];
  score_list.pop_back();
  // cannot select the same index twice, so it is pushed out.
  return cp;
}
