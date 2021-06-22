#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/LU>
#include <queue>
#include <vector>

typedef Eigen::SparseMatrix<float> eigen_colmajor;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> eigen_rowmajor;
typedef std::size_t size_t;
typedef std::pair<size_t, float> score;
typedef std::pair<int, size_t> node;
typedef std::pair<node, size_t> chunk;

float combine(float x, float y, std::string combiner);
float transform(float x, std::string post_processor);

/*! Class to store and perform realtime inference in xcb mode
    for pecos xlinear model.
*/
class ModelChain {
 private:
  int size;  // meant to store the number of levels in the model chain
  std::vector<eigen_colmajor> weights;  // vector of weight matrices for routing
  std::vector<eigen_colmajor> clusters;  // vector of cluster matrices
  size_t num_labels;                     // total number of labels
  std::vector<eigen_colmajor>
      regression_weights;  // vector of regression weights

 public:
  // returns number of levels in the model chain
  int get_size() { return this->size; }
  // add elements one level at the time.
  void add_elements(eigen_colmajor weight, eigen_colmajor cluster,
                    eigen_colmajor regression_weight);
  // get the labels that are connected to the given codes for a particular level
  // in the model chain
  std::vector<score> get_labels(std::vector<score>& codes,
                                eigen_colmajor& cluster_mat);
  // get a random label in the subtree of the current node
  size_t get_random_arm(node& current_node);
  // get top scores with exploration enabled from score_list
  std::vector<score> get_top_scores(
      std::vector<std::pair<float, size_t>>& score_list, int topk, int level,
      int num_explore, float multiplier, bool explore_in_routing,
      std::string explore_strategy, float alpha);
  // get top labels given input vector and the level of the replace model
  std::vector<score> get_top_labels(
      eigen_rowmajor& in_vector, int level, std::vector<score>& labels,
      int topk, int num_explore, float multiplier, std::string post_processor,
      std::string combiner, bool explore_in_routing,
      std::string explore_strategy, std::vector<node>& rejected_nodes,
      std::vector<chunk>& selected_chunks, float alpha);
  // perform beam search given an input vector
  std::pair<std::vector<score>, std::vector<chunk>> beam_search(
      eigen_rowmajor& in_vector, int beam_size, int topk, int num_explore,
      float multiplier, std::string post_processor, std::string combiner,
      bool explore_in_routing, std::string explore_strategy, float alpha);
  // sample according to falcon distribution given a score_list
  score get_sample(std::vector<std::pair<float, size_t>>& score_list,
                   float multiplier, int level, int& num_exploit,
                   std::string explore_strategy, float alpha);
};