#include <iostream>
#include <cstring>
#include <utility>
#include <vector>
#include <chrono>
#include <numeric>
#include <stack>
#include <cassert>
#include "parser.hpp"

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
    // Getters
    inline int get_N() const { return N; }
    inline int get_variable_index() const { return variable_index; }
    inline int get_assignment() const { return assignment; }
    inline std::vector<int>& get_assignments() { return assignments; }
    inline std::vector<bool>& get_domains() { return domains; }
    inline std::vector<int>& get_domain_upperbounds() { return domain_upperbounds; }
    
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

// Progagate the domains
void fixpoint(Node &node, const int var, const int assignments, const Data &data){
  // Check if the domain are singleton
  for (int i = var + 1; i < node.get_N(); i++){
    if (node.checkSingleton(i)){
      updateDomain(node, i, assignments, data);
    }
  }
}

// Evaluate and branch
void evaluate_and_branch(Node& parent, std::stack<Node>& pool, size_t& tree_loc, size_t& num_sol, const Data &data) {
    std::cout << "Evaluating variable: " << parent.get_variable_index() << std::endl;
    std::cout << "With value:  " << parent.get_assignment() << std::endl;
    std::cout << "Current assignments : ";
    parent.print_assignments();
    std::cout << std::endl;
    
    int depth = parent.get_variable_index(); // Current depth in the tree
    int N = parent.get_N();     // Total number of variables

    if (depth == N - 1) {
        num_sol++;
        std::cout << "Solution found" << std::endl;
        std::cout << "---------------------------------------------------------------" << std::endl;
        return;
    }
    std::cout << "---------------------------------------------------------------" << std::endl;
    
    int depth_child = depth + 1;
    
    for (int val = 0; val <= parent.get_domain_upperbounds()[depth_child]; ++val) {
        if (parent.isInDomain(depth_child, val)) {
            Node child = parent;
            child.set_variable_index(depth_child);
            child.set_assignment(val);
            updateDomain(child, depth_child, val, data);
            // @todo: Propagate the domain restrictions using the function `fixpoint`
            //fixpoint(child, N, depth_child, val, data)
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
  std::cout << "\tSET-UP PROBLEM\n" << std::endl;
  std::cout << "Data read successfully" << std::endl;
  std::cout << "Number of variables: " << data.get_n() << std::endl;

  Node root(data.get_n());
  root.init_domains(data.get_u());

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
    evaluate_and_branch(current, stack, exploredTree, exploredSol, data);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "---------------------------------------------------------------" << std::endl;
  std::cout << "\tEND SEARCH" << std::endl;
  std::cout
    << "Number of solution:       \t" << exploredSol << "\n"
    << "Number of nodes explored: \t" << exploredTree << "\n"
    << "Exection time:            \t" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << 
  std::endl;
  return 0;
}