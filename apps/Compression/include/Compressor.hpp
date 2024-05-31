#ifndef COMPRESSOR_HPP
#define COMPRESSOR_HPP

#include <typedefs.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

template <typename T>
class HuffmanTree {
public:
  std::vector<bool> tmp;
  struct Node {
    T value;
    int freq;
    shared_ptr<Node> left;
    shared_ptr<Node> right;
    Node(T value, int freq) : value(value), freq(freq) {}
  };

  struct Compare {
    bool operator()(const shared_ptr<Node>& a, const shared_ptr<Node>& b) {
      return a->freq > b->freq;
    }
  };

  HuffmanTree(unordered_map<T, int>& freq_table) {
    priority_queue<shared_ptr<Node>, vector<shared_ptr<Node>>, Compare> pq;
    for (auto& [val, freq] : freq_table) {
      auto node = make_shared<Node>(val, freq);
      pq.push(node);
    }

    while (pq.size() > 1) {
      auto left = pq.top();
      pq.pop();
      auto right = pq.top();
      pq.pop();

      auto parent = make_shared<Node>(std::numeric_limits<T>::max(), left->freq + right->freq);
      parent->left = left;
      parent->right = right;
      pq.push(parent);
    }

    root = pq.top();
    pq.pop();
    create_code_table(root, vector<bool>());
  }

  auto get_code_table() -> unordered_map<T, vector<bool>> {
    return code_table;
  }

  // auto get_reversed_table() -> unordered_map<vector<bool>, T> {
  //   return revs_table;
  // }

private:
  shared_ptr<Node> root;
  unordered_map<T, vector<bool>> code_table;

  auto create_code_table(shared_ptr<Node> node, vector<bool> code) -> void {
    if (node->value != std::numeric_limits<T>::max()) {
      // check if there is the same code in the table
      if (code_table.find(node->value) != code_table.end()) {
        cout << "Error: Duplicate code found for value " << node->value << endl;
        return;
      }
      // if (revs_table.find(code) != revs_table.end()) {
      //   cout << "Error: Duplicate code found for value " << node->value << endl;
      //   return;
      // }
      code_table[node->value] = code;
      // revs_table[code] = node->value;
      return;
    }

    if (node->left != nullptr) {
      auto left_code = code;
      left_code.push_back(0);
      create_code_table(node->left, left_code);
    }

    if (node->right != nullptr) {
      auto right_code = code;
      right_code.push_back(1);
      create_code_table(node->right, right_code);
    }
  }
};

class Compressor {
    public:
    cv::Mat img;
    Typedefs::vec3D coeff;
    const int levels;
    Compressor(const cv::Mat& img, const int levels = 3) : img(img), levels(levels) {}
    auto apply_dwt() -> void;
    auto apply_idwt() -> void;
    auto quantize () -> void;
    auto dequantize () -> void;
    auto HuffmanEncoding() -> void;
    auto HuffmanDecoding() -> void;
};

#endif