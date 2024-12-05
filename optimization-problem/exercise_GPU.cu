#include <iostream>
#include <cstring>
#include <utility>
#include <vector>
#include <chrono>
#include <numeric>
#include <stack>
#include <cassert>
#include "cuda_runtime.h"
// #include "parser.hpp"
#include "kernels.cu"

constexpr bool print = false;

// Node data structure
class Node {
  private:
    int N; // number of variables
    int variable_index; // depth in the tree
    int assignment; // assignment value for variable at variable_index
    std::vector<int>  assignments; // vector of size N with assigned values
    std::vector<bool> domains; // domains for each variable
    std::vector<int>  domain_upperbounds; // size of the domain for each variable
    std::vector<int>  offset;
    std::vector<bool> singleton;
    std::vector<int>  singleton_values;

  public:
    // Constructor for the root node with empty domains and domain sizes
    Node(size_t N): N(N), variable_index(-1), assignment(-1), assignments(N, -1){}

    // Constructor for creating a child node with specified domains and domain sizes
    Node(size_t N, int variable_index, int assignment, std::vector<int> assignments, std::vector<bool> domains, std::vector<int> domain_sizes)
      : N(N),
        variable_index(variable_index),
        assignment(assignment),
        assignments(assignments),
        domains(domains),
        domain_upperbounds(domain_sizes) {}

    // Init domains and domain sizes
    void init_domains(const int* d) {
      // Init domains
      domain_upperbounds.resize(N);
      offset.resize(N);
      singleton.resize(N, false);
      singleton_values.resize(N, -1);

      int dom_size = 0;
      // Compute dimensions of domains
      for (int i = 0; i < N; i++) {
        domain_upperbounds[i] = d[i];
        dom_size += d[i] + 1;
      }
      offset[0] = 0;
      for (int i = 1; i < N; ++i) offset[i] += offset[i - 1] + domain_upperbounds[i - 1] + 1;
      domains.resize(dom_size, true);
    }
    
    // Get domains
    bool isInDomain(int var, int val) {
      return domains[offset[var] + val];
    }
    bool isSingleton(int var) {
      return singleton[var];
    }
    bool checkSingleton(int var) {
      int count = 0;
      int val = -1;
      for (int i = 0; i <= domain_upperbounds[var]; i++) {
        if (domains[offset[var] + i]) {
          count++;
          val = i;
        }
      }
      if (count == 1) {
        singleton[var] = true;
        singleton_values[var] = val;
        return true;
      }
      return false;
    }
  
    // Set domain value to false
    void setDomainValue(int var, int val) {
      domains[offset[var] + val] = false;
    }
    // Setters
    inline void set_variable_index(int index) { variable_index = index; }
    inline void set_assignment(int val) { assignment = val; assignments[variable_index] = val; }
    inline void set_assignments_for_singleton(int var, int val) { assignments[var] = val; }
    inline void set_domains(bool* new_domains) { 
      for (int i = 0; i < domains.size(); i++) {
        domains[i] = new_domains[i];
      }  
    }
    inline void set_singleton(bool* new_singleton) { 
      for (int i = 0; i < N; i++) {
        singleton[i] = new_singleton[i];
      }  
    }
    // Getters
    inline int get_N() const { return N; }
    inline int get_variable_index() const { return variable_index; }
    inline int get_assignment() const { return assignment; }

    inline std::vector<int> get_assignments() { return assignments; }
    inline std::vector<bool> get_domains() { return domains; }
    inline std::vector<int> get_domain_upperbounds() { return domain_upperbounds; }
    inline std::vector<int> get_offset() { return offset; }
    inline std::vector<bool> get_singleton() { return singleton; }
    inline std::vector<int> get_singleton_values() { return singleton_values; }

    inline std::vector<int>& get_assignments_pointer() { return assignments; }
    inline std::vector<bool>& get_domains_pointer() { return domains; }
    inline std::vector<int>& get_domain_upperbounds_pointer() { return domain_upperbounds; }
    inline std::vector<int>& get_offset_pointer() { return offset; }
    inline std::vector<bool>& get_singleton_pointer() { return singleton; }
    inline std::vector<int>& get_singleton_values_pointer() { return singleton_values; }
    
    // Printers
    void print_assignments() const{
      for (int assignment : assignments) {
        std::cout << assignment << " ";
      }
      std::cout << std::endl;
    }
    void print_domains() const{
      for (bool domain : domains) {
        std::cout << domain << " ";
      }
      std::cout << std::endl;
    }
    void print_domain_sizes() const{
      for (int size : domain_upperbounds) {
        std::cout << size << " ";
      }
      std::cout << std::endl;
    }
    void print_domain_upperbounds() const{
      for (int size : domain_upperbounds) {
        std::cout << size << " ";
      }
      std::cout << std::endl;
    }
    void print_singleton() const{
      for (bool s : singleton) {
        std::cout << s << " ";
      }
      std::cout << std::endl;
    }

    Node(const Node&) = default;
    Node(Node&&) = default;
    Node() = default;
};

// Check if the constraint are respected
bool isNotSafe(const Data &data, int var_1, int var_2){
  return data.get_C_at(var_1, var_2) == 1;
}

void updateDomain(Node &node, const int var, const int assignments, const Data &data){
  for (int i = var + 1; i < node.get_N(); i++){
    if (assignments <= node.get_domain_upperbounds()[i] && isNotSafe(data, var, i)){
      node.setDomainValue(i, assignments); // Set the domain value to false
    }
  }
}
// The same update domain but you change also variable that can be update before the variable index
void updateDomainSingleton(Node &node, const int var, const int assignments, const Data &data){
  for (int i = node.get_variable_index() + 1; i < node.get_N(); i++){
    if (assignments <= node.get_domain_upperbounds()[i] && isNotSafe(data, var, i)){
      node.setDomainValue(i, assignments); // Set the domain value to false
    }
  }
}

// Progagate the domains
bool fixpointGPU(Node& node, const int var, const int assignments, const Data& data, const int* d_C) {
  int   N = node.get_N();

  bool  solution_found = true;
  bool  old_solution_found = false;

  bool* d_found_solution;

  bool*   d_domains;
  int*    d_offset;
  int*    d_domain_upperbounds;
  bool*   d_singleton;
  int*    d_singleton_values;

  size_t domains_size             = node.get_domains().size() * sizeof(bool);
  size_t domain_upperbounds_size  = N * sizeof(int);
  size_t singleton_size           = N * sizeof(bool);

  cudaMalloc(&d_domains, domains_size);
  cudaMalloc(&d_offset, domain_upperbounds_size);
  cudaMalloc(&d_domain_upperbounds, domain_upperbounds_size);
  cudaMalloc(&d_singleton, singleton_size);
  cudaMalloc(&d_singleton_values, domain_upperbounds_size);

  cudaMalloc(&d_found_solution, sizeof(bool));

  // Convert std::vector<bool> to bool[] for contiguous memory
  bool* domains_bool = new bool[domains_size];
  bool* singleton_bool = new bool[singleton_size];
  for (size_t i = 0; i < domains_size; i++) {
    domains_bool[i] = node.get_domains_pointer()[i];
  }
  for (size_t i = 0; i < singleton_size; i++) {
    singleton_bool[i] = node.get_singleton_pointer()[i];
  }

  // Host to Device
  cudaMemcpy(d_domains, domains_bool, domains_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_offset, node.get_offset_pointer().data(), domain_upperbounds_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_domain_upperbounds, node.get_domain_upperbounds_pointer().data(), domain_upperbounds_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_singleton, singleton_bool, singleton_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_singleton_values, node.get_singleton_values_pointer().data(), domain_upperbounds_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_found_solution, &solution_found, sizeof(bool), cudaMemcpyHostToDevice);
  bool* old_domain = new bool[domains_size];

  int blockSize = N;
  int numBlocks = 1;

  while(!std::equal(domains_bool, domains_bool + domains_size, old_domain)){
    std::copy(domains_bool, domains_bool + domains_size, old_domain);
    checkSingletonKernel<<<numBlocks, blockSize>>>(d_domains, d_offset, d_domain_upperbounds, d_singleton, d_singleton_values, N);
  
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      std::cout << "Error in checkSingletonKernel" << std::endl;
      std::exit(1);
    }
    // cudaMemcpy(domains_bool, d_domains, domains_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(singleton_bool, d_singleton, singleton_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(node.get_singleton_values_pointer().data(), d_singleton_values, singleton_size, cudaMemcpyDeviceToHost);
    
    // node.set_domains(domains_bool);
    // node.set_singleton(singleton_bool);

    // node.print_singleton();
    
    updateDomainKernel<<<numBlocks, blockSize>>>(d_domains, d_offset, d_domain_upperbounds, d_singleton, d_singleton_values, N, var, d_C);
    
    while (old_solution_found != solution_found) {
      old_solution_found = solution_found;
      checkForSolution<<<numBlocks, blockSize>>>(d_found_solution, d_singleton, var, N);
      cudaMemcpy(&solution_found, d_found_solution, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
      std::cout << "CUDA error: " << cudaGetErrorString(err2) << std::endl;
      std::cout << "Error in updateDomainKernel" << std::endl;
      std::exit(1);
    }
       
    for (size_t i = 0; i < domains_size; i++) {
      domains_bool[i] = node.get_domains_pointer()[i];
    }

    // cudaMemcpy(d_domains, domains_bool, domains_size, cudaMemcpyHostToDevice);
  }

  cudaMemcpy(d_domains, domains_bool, domains_size, cudaMemcpyHostToDevice);
  cudaMemcpy(singleton_bool, d_singleton, singleton_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(node.get_singleton_values_pointer().data(), d_singleton_values, singleton_size, cudaMemcpyDeviceToHost);

  node.set_domains(domains_bool);
  node.set_singleton(singleton_bool);

  // Device to Host
  cudaMemcpy(domains_bool, d_domains, domains_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(singleton_bool, d_singleton, singleton_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(node.get_offset_pointer().data(), d_offset, domain_upperbounds_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(node.get_domain_upperbounds_pointer().data(), d_domain_upperbounds, domain_upperbounds_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(node.get_singleton_values_pointer().data(), d_singleton_values, singleton_size, cudaMemcpyDeviceToHost);

  node.set_domains(domains_bool);
  node.set_singleton(singleton_bool);

  cudaFree(d_domains);
  cudaFree(d_domain_upperbounds);
  cudaFree(d_singleton);
  cudaFree(d_singleton_values);


  delete[] old_domain;
  delete[] domains_bool;
  delete[] singleton_bool;

  return solution_found;
}


// Evaluate and branch
void evaluate_and_branch(Node& parent, std::stack<Node>& pool, size_t& tree_loc, size_t& num_sol, const Data &data, const int* d_C) {
     
    if (print){
      std::cout << "Evaluating variable: " << parent.get_variable_index() << std::endl;
      std::cout << "With value:  " << parent.get_assignment() << std::endl;
      std::cout << "Current assignments : ";
      parent.print_assignments();
      std::cout << std::endl;
    }
    
    int depth = parent.get_variable_index(); // Current depth in the tree
    int N = parent.get_N();     // Total number of variables

    if (depth == N - 1) {
      num_sol++;
      if (print){
        std::cout << "Solution found" << std::endl;
        std::cout << "---------------------------------------------------------------" << std::endl;
      }
      return;
    }

    
    // Propagate the domain restrictions
    if(fixpointGPU(parent, depth, parent.get_assignment(), data, d_C)){
        num_sol ++;
        if (print){
          std::cout << "Solution found with propagation" << std::endl;
          std::cout << "---------------------------------------------------------------" << std::endl;
        }
      return;
    }
    if (print)
      std::cout << "---------------------------------------------------------------" << std::endl;
    // Check before create the node
    int depth_child = depth + 1;
    for(int i = depth + 1; i < N; i++){
      if(parent.isSingleton(i)){
        depth_child = i;
        parent.set_assignments_for_singleton(i, parent.get_singleton_values()[i]);
      }
      else break;
    }    
    // Find a new possible child
    for (int val = 0; val <= parent.get_domain_upperbounds()[depth_child]; ++val) {
        if (parent.isInDomain(depth_child, val)) {
          Node child = parent;
          child.set_variable_index(depth_child);
          child.set_assignment(val);
          updateDomain(child, depth_child, val, data);
          // Put child in the stack
          pool.push(std::move(child));
          tree_loc++;
        }
    }
}

int main(int argc, char** argv) {
  // helper
  if (argc != 2) {
    std::cout << "usage: " << argv[0] << " <number of varaibles> " << std::endl;
    exit(1);
  }

  // Read input data
  Data data;
  if (!data.read_input(argv[1])) {
    return 1;
  }
  
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::cout << "No CUDA device found" << std::endl;
    return 1;
  }
  std::cout << "---------------------------------------------------------------" << std::endl;
  std::cout << "\tSET-UP PROBLEM" << std::endl;
  std::cout << "---------------------------------------------------------------" << std::endl;
  std::cout << "Data read successfully" << std::endl;
  std::cout << "Number of variables: " << data.get_n() << std::endl;

  std::cout << "Devices available: " << deviceCount << std::endl;
  std::cout << "Init GPU varaibles" << std::endl;

  int** C = data.get_C();
  int* C_flat = new int[data.get_n() * data.get_n()];
  int* d_C;

  cudaMalloc(&d_C, data.get_n() * data.get_n() * sizeof(int));
  cudaMemcpy(d_C, C_flat, data.get_n() * data.get_n() * sizeof(int), cudaMemcpyHostToDevice);
  
  Node root(data.get_n());
  root.init_domains(data.get_u());
  std::cout << "Done" << std::endl;

  // Prepere the search
  std::stack<Node> stack;
  stack.push(std::move(root));
  
  // Statistics
  size_t exploredTree = 0, exploredSol = 0;
  std::cout << "---------------------------------------------------------------" << std::endl;
  std::cout << "\tINIT SEARCH" << std::endl;
  std::cout << "---------------------------------------------------------------" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  while (!stack.empty())
  {
    Node current = std::move(stack.top());
    stack.pop();
    evaluate_and_branch(current, stack, exploredTree, exploredSol, data, d_C);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "---------------------------------------------------------------" << std::endl;
  std::cout << "\tEND SEARCH" << std::endl;
  std::cout
    << "Number of solution:       \t" << exploredSol << "\n"
    << "Number of nodes explored: \t" << exploredTree << "\n"
    << "Exection time:            \t" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << 
  std::endl;

  // Insert the time into a csv file called time_results.csv
  int N = data.get_n();
  std::ofstream file;
  file.open("time_results.csv", std::ios_base::app);
  file << "gpu, "<< N << ", "<< exploredTree << ", " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "\n";
  file.close();

  return 0;
}