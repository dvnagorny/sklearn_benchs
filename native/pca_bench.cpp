#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include <chrono>  

#include "daal.h"
#include "service.h"
namespace dm=daal::data_management;
namespace ds=daal::services;
namespace da=daal::algorithms;


#define REPS 10


template<typename T>
ds::SharedPtr<dm::HomogenNumericTable<T> > makeTable(T* data, size_t rows, size_t cols)
{
    return ds::SharedPtr<dm::HomogenNumericTable<T> >(new dm::HomogenNumericTable<T>(data, cols, rows));
}


std::vector<std::pair<int, int> > init_problem_sizes() 
{
    std::vector<std::pair<int, int> > ret_val;
    ret_val.push_back(std::make_pair<int, int>(100000, 10));
    ret_val.push_back(std::make_pair<int, int>(100000, 25));
    ret_val.push_back(std::make_pair<int, int>(100000, 50));
    ret_val.push_back(std::make_pair<int, int>(500000, 10));
    ret_val.push_back(std::make_pair<int, int>(500000, 25));
    ret_val.push_back(std::make_pair<int, int>(500000, 50));
    ret_val.push_back(std::make_pair<int, int>(1000000, 10));
    ret_val.push_back(std::make_pair<int, int>(1000000, 25));
    ret_val.push_back(std::make_pair<int, int>(1000000, 50));
    return ret_val; 
}


void pca_test(double* X, size_t rows, size_t cols)
{
    da::pca::Batch<double, da::pca::svdDense> algorithm;
    algorithm.input.set(da::pca::data, makeTable(X, rows, cols));
    algorithm.compute();
    algorithm.getResult()->get(da::pca::eigenvalues);
    algorithm.getResult()->get(da::pca::eigenvectors);
}


void bench()
{
    std::vector<std::pair<int, int> > problem_sizes = init_problem_sizes();
    std::vector<std::pair<int, int> >::iterator it; 
    for (it = problem_sizes.begin(); it != problem_sizes.end(); it++) {
        size_t size = it->first * it->second;
        double* X = new double[size];
        for(size_t i = 0; i < size; i++)
            X[i] = (double)rand() / RAND_MAX;
        std::vector<std::chrono::duration<double> > times;
        for(int i = 0; i < REPS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            pca_test(X, it->first, it->second);
            auto finish = std::chrono::high_resolution_clock::now();
            times.push_back(finish - start);
        }
        std::cout << it->first << " " << it->second 
                  << " " << std::min_element(times.begin(), times.end())->count() << std::endl;
        delete[] X;
    }
}


int main()
{
    bench();
    return 0;
}
