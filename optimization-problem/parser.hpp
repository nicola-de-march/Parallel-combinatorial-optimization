#include<utility>
#include<map>
#include<string>
#include<sstream>
#include<fstream>
#include<iostream>
#include <cassert>

class Data{
    private:
        size_t n = NULL;
        int* u;
        int** C;
    public:
        inline size_t get_n(){return n;}
        inline int* get_u(){return u;}
        inline int** get_C(){return C;}
        inline int get_u_at(size_t i){return u[i];}
        inline int get_C_at(size_t i, size_t j){return C[i][j];}

        bool read_input(std::string filename){
            std::ifstream f(filename);
            if (!f.is_open()){
                std::cerr << "[ERROR] Couldn't open " << filename << "\n";
                return false;
            }

            std::string s;
            std::string current;
            std::string delimiter1 = ";";
            std::string delimiter2 = ",";
            while(std::getline(f,s)){
                if (s == "N"){
                    current = "N";
                    continue;
                }
                if (s == "U"){
                    current = "U";
                    continue;
                }
                if (s == "C"){
                    current = "C";
                    continue;
                }

                if (current == "N"){
                    std::stringstream ss(s);
                    ss >> n;
                    u = new int[n];
                    C = new int*[n];
                    for (int i = 0; i < n; ++i) C[i] = new int[n];
                }
                if (current == "U"){
                    size_t delimiter_id = s.find(delimiter1);
                    std::string s_id = s.substr(0,delimiter_id);
                    std::string s_value = s.substr(delimiter_id+1,s.size());

                    std::stringstream ss(s_id);
                    size_t i;
                    ss >> i;
                    u[i] = std::stoi(s_value);
                }
                if (current == "C"){
                    size_t delimiter_id1 = s.find(delimiter1);
                    std::string s_id = s.substr(0,delimiter_id1);
                    std::string s_value = s.substr(delimiter_id1+1,s.size());

                    size_t delimiter_id2 = s_id.find(delimiter2);
                    std::string si = s_id.substr(0,delimiter_id2);
                    std::string sj = s_id.substr(delimiter_id2+1,s_id.size());

                    size_t sti, stj;
                    std::stringstream ss1(si);
                    ss1 >> sti;
                    std::stringstream ss2(sj);
                    ss2 >> stj;
                    C[sti][stj] = std::stoi(s_value);
                }

            }

            return true;
        }

        void print_n(){
            assert(n != NULL && "[ERROR] Haven't read input file yet");
            
            printf("n = %ld\n", n);
            return;
        }

        void print_u(){
            assert(n != NULL && "[ERROR] Haven't read input file yet");
            
            for (size_t i = 0; i < n; ++i)printf("u[%ld] = %d\n", i, u[i]);
            return;
        }

        void print_C(){
            assert(n != NULL && "[ERROR] Haven't read input file yet");

            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    printf("C[%ld,%ld] = %d, ", i, j, C[i][j]);
                printf("\n");
            return;
        }

};