#ifndef HUFFMAN_TREE_HPP
#define HUFFMAN_TREE_HPP

#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

template <typename T>
class HuffmanTree {
public:
  std::vector<bool> tmp;
  struct Node {
    T value;
    int freq;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
    Node(T value, int freq) : value(value), freq(freq) {}
  };

  struct Compare {
    bool operator()(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
      return a->freq > b->freq;
    }
  };

  HuffmanTree(std::unordered_map<T, int>& freq_table) {
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, Compare> pq;
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
    create_code_table(root, std::vector<bool>());
  }

  auto get_code_table() -> std::unordered_map<T, std::vector<bool>> {
    return code_table;
  }

  // auto get_reversed_table() -> std::unordered_map<std::vector<bool>, T> {
  //   return revs_table;
  // }

private:
  std::shared_ptr<Node> root;
  std::unordered_map<T, std::vector<bool>> code_table;

  auto create_code_table(std::shared_ptr<Node> node, std::vector<bool> code) -> void {
    if (node->value != std::numeric_limits<T>::max()) {
      // check if there is the same code in the table
      if (code_table.find(node->value) != code_table.end()) {
        std::cout << "Error: Duplicate code found for value " << node->value << std::endl;
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

#endif