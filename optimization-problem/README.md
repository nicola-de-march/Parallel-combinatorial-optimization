## Example - Input data format
```
N               // indicate the number of variables in below
3
U               // indicate upper bound for each variable, e.g. [variable id];[upper bound]
0;49
1;97
2;53
C               // indicate if the pair(i,j) in C, e.g. [i,j];[boolean] if true then (i,j) in C, otherwise not.
0,0;0
0,1;1
0,2;1
1,0;1
1,1;1
1,2;1
2,0;1
2,1;0
2,2;0
```

## Example - Read input data
To read input data, you just need to `#include "parser.hpp"` in your code. Then, you can just call `read_input(filename)` to read input data.
```cpp
#include "parser.hpp"

int main(){
    Data data;
    if (data.read_input("pco_3.txt")){
        data.print_n();
        data.print_u();
        data.print_C();
    }   

    return 0;
}
```

## Example - Generate more testing scenarios
To implement more general algorithm, you can use `generate_instances.py` to generate large-scale problem instances. There is no package required in this python code.
You just need to specify how many variables in your problem instance, then the code will generate random number to be upper bound for each variable and randomly select pair of `i` and `j` to be in `C`.


All the code you can do modification to fit your scenarios.
