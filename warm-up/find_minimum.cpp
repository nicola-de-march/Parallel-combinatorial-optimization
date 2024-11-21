#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <fstream>

int find_min(const std::vector<int> &vet)
{
  int minimum = vet[0];
  for(auto element : vet)
  {
    if (element < minimum)
      minimum = element;
  }
  return minimum;
}

int main(int argc, char* argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <array_size>" << std::endl;
    return -1;
  }
  // Read vector from file
  std::ifstream input_file(argv[1]);
  if (!input_file) {
    std::cerr << "Error opening file" << std::endl;
    return -1;
  }

  size_t N = std::stoul(argv[2]);
  std::vector<int> array(N);

  for (size_t i = 0; i < N; i++) {
    input_file >> array[i];
  }

  auto start = std::chrono::steady_clock::now();

  int minimum = find_min(array);

  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  std::cout << "SERIAL ALGORITHM" << std::endl;
  std::cout << " Array size: " << N << std::endl;
  std::cout << " Min = " << minimum << std::endl;
  std::cout << " Time: " << duration.count()  << "us" << std::endl;

  // Print time to file
  std::ofstream output_file("time_analysis.csv", std::ios::app);
  if (!output_file) {
    std::cerr << "Error opening time file" << std::endl;
    return -1;
  }
  output_file << "optimistic," << N << "," << duration.count() << std::endl;

  return 0;
}