/*
 * Author: Guillaume HELBECQUE (Université du Luxembourg)
 * Date: 10/10/2024
 *
 * Description:
 * This program solves the N-Queens problem using a sequential Depth-First tree-Search
 * (DFS) algorithm. It serves as a basis for task-parallel implementations.
 */

#include <iostream>
#include <cstring>
#include <utility>
#include <vector>
#include <chrono>
#include <stack>
#include <cuda_runtime.h>
#include "parser.hpp"

// N-Queens node
struct Node {
  int depth; // depth in the tree
  std::vector<int> board; // board configuration (permutation), vector of size N with the position of the queens
  std::vector<std::vector<bool>> domains;

  Node(size_t N, const std::vector<std::vector<bool>>& initial_domains): depth(0), board(N), domains(initial_domains){
    for (int i = 0; i < N; i++) {
      board[i] = i;
    }
  }
  Node(const Node&) = default;
  Node(Node&&) = default;
  Node() = default;
};

// check if placing a queen is safe (i.e., check if all the queens already placed share
// a same diagonal)
bool isSafe(const std::vector<int>& board, const int row, const int col, const Data& data)
{
  for (int i = 0; i < row; ++i) {
    if (board[i] == col - row + i || board[i] == col + row - i) {
      return false;
    }
  }

  // Check additional constraints from data
  for (size_t i = 0; i < row; ++i) {
    if (data.get_C_at(i, row) && !(board[i] < col)) {
      return false;
    }
  }
  return true;
}

// CUDA kernel to update domains
__global__ void updateDomainsKernel(bool* domains, int N, int row, int col) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    domains[(row + 1) * N + idx] = false;
  }
}

// update domains to remove impossible values
void updateDomains(std::vector<std::vector<bool>>& domains, int row, int col, int N) {
  bool* d_domains;
  cudaMalloc(d_domains, N * sizeof(bool*));
  // Calculate the total size of the domains
  size_t totalSize = 0;
  for (const auto& domain : domains) {
    totalSize += domain.size();
  }

  // Flatten the domains vector for copying to device
  std::vector<bool> flatDomains;
  for (const auto& domain : domains) {
    flatDomains.insert(flatDomains.end(), domain.begin(), domain.end());
  }

  cudaMemcpy(d_domains, static_cast<const void*>(flatDomains.data()), totalSize * sizeof(bool), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  updateDomainsKernel<<<numBlocks, blockSize>>>(d_domains, N, row, col);

  // Copy back the flattened domains
  cudaMemcpy(flatDomains.data(), d_domains, totalSize * sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(d_domains);

  // Unflatten the domains vector
  size_t offset = 0;
  for (auto& domain : domains) {
    std::copy(flatDomains.begin() + offset, flatDomains.begin() + offset + domain.size(), domain.begin());
    offset += domain.size();
  }
}

// evaluate a given node (i.e., check its board configuration) and branch it if it is valid
// (i.e., generate its child nodes.)
void evaluate_and_branch(const Node& parent, std::stack<Node>& pool, size_t& tree_loc, size_t& num_sol, const Data& data)
{
  int depth = parent.depth;
  int N = parent.board.size();

  // if the given node is a leaf, then update counter and do nothing
  if (depth == N) {
    num_sol++;
  }
  // if the given node is not a leaf, then update counter and evaluate/branch it
  else {
    for (int j = depth; j < N; j++) {
      if (parent.domains[depth][j] && isSafe(parent.board, depth, parent.board[j], data)) {
        Node child(parent);
        child.board[depth] = j;
        child.depth++;
        updateDomains(child.domains, depth, j, N);
        pool.push(std::move(child));
        tree_loc++;
      }
    }
  }
}

int main(int argc, char** argv) {
  // helper
  if (argc != 2) {
    std::cout << "usage: " << argv[0] << " <number of queens> " << std::endl;
    exit(1);
  }

  // Read input data
  Data data;
  if (!data.read_input(argv[1])) {
    return 1;
  }

  // problem size (number of queens)
  size_t N = data.get_n();
  std::cout << "Solving " << N << "-Queens problem\n" << std::endl;

  // initialization of the root node (the board configuration where no queen is placed)
  Node root(N, data.get_domains());

  // initialization of the pool of nodes (stack -> DFS exploration order)
  std::stack<Node> pool;
  pool.push(std::move(root));

  // statistics to check correctness (number of nodes explored and number of solutions found)
  size_t exploredTree = 0;
  size_t exploredSol = 0;

  // beginning of the Depth-First tree-Search
  auto start = std::chrono::steady_clock::now();

  while (pool.size() != 0) {
    // get a node from the pool
    Node currentNode(std::move(pool.top()));
    pool.pop();

    // check the board configuration of the node and branch it if it is valid.
    evaluate_and_branch(currentNode, pool, exploredTree, exploredSol, data);
  }

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // outputs
  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
  std::cout << "Total solutions: " << exploredSol << std::endl;
  std::cout << "Size of the explored tree: " << exploredTree << std::endl;

  // write results to file
  std::ofstream outfile("time_analysis.csv", std::ios_base::app);
  outfile << N << ", " << duration.count() << "\n";
  outfile.close();

  return 0;
}