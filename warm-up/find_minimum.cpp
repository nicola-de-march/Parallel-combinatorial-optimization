#include <iostream>
#include <vector>
#include <array>

constexpr int N = 5;

int find_min(const std::array<int, N> &vet)
{
  int minimum = vet[0];
  for(auto element : vet)
  {
    if (element < minimum)
      minimum = element;
  }
  return minimum;
}
int main()
{
  std::array<int, N> array = {10, 20, 2, 3, 4};
  int minimum = find_min(array);
  std::cout << "Min = " << minimum << std::endl;
  return 0;
}