/**
* @file HuffmanTree.hpp
* @brief Defines the HuffmanTree class for constructing a Huffman tree and generating Huffman codes.
*/

#ifndef HUFFMAN_TREE_HPP
#define HUFFMAN_TREE_HPP

#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

/**
* @tparam T The data type of the values in the tree.
*/
template <typename T>
class HuffmanTree {
public:
   /**
    * @brief Node structure for the Huffman tree.
    */
   struct Node {
       T value;
       int freq;
       std::shared_ptr<Node> left;
       std::shared_ptr<Node> right;
       Node(T value, int freq) : value(value), freq(freq) {}
   };

   /**
    * @brief Comparator for the priority queue used in the Huffman tree construction.
    */
   struct Compare {
       bool operator()(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
           return a->freq > b->freq;
       }
   };

   /**
    * @brief Constructor for the HuffmanTree.
    * @param freq_table An unordered map containing the frequency table for the values.
    */
   HuffmanTree(std::unordered_map<T, int>& freq_table) {
       std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, Compare> pq;
       for (auto& [val, freq] : freq_table) {
           auto node = std::make_shared<Node>(val, freq);
           pq.push(node);
       }

       while (pq.size() > 1) {
           auto left = pq.top();
           pq.pop();
           auto right = pq.top();
           pq.pop();

           auto parent = std::make_shared<Node>(std::numeric_limits<T>::max(), left->freq + right->freq);
           parent->left = left;
           parent->right = right;
           pq.push(parent);
       }

       root = pq.top();
       pq.pop();
       create_code_table(root, std::vector<bool>());
   }

   /**
    * @brief Get the Huffman code table.
    * @return An unordered map containing the Huffman codes for the values.
    */
   auto get_code_table() -> std::unordered_map<T, std::vector<bool>> {
       return code_table;
   }

private:
   std::shared_ptr<Node> root; /**< The root of the Huffman tree. */
   std::unordered_map<T, std::vector<bool>> code_table; /**< The Huffman code table. */

   /**
    * @brief Create the Huffman code table by traversing the Huffman tree.
    * @param node The current node in the traversal.
    * @param code The current code being constructed.
    */
   auto create_code_table(std::shared_ptr<Node> node, std::vector<bool> code) -> void {
       if (node->value != std::numeric_limits<T>::max()) {
           code_table[node->value] = code;
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