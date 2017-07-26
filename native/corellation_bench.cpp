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
    ret_val.push_back(std::make_pair<int, int>(500, 10000));
    ret_val.push_back(std::make_pair<int, int>(500, 50000));
    ret_val.push_back(std::make_pair<int, int>(500, 100000));
    ret_val.push_back(std::make_pair<int, int>(500, 150000));
    ret_val.push_back(std::make_pair<int, int>(500, 200000));
    ret_val.push_back(std::make_pair<int, int>(1000, 10000));
    ret_val.push_back(std::make_pair<int, int>(1000, 50000));
    ret_val.push_back(std::make_pair<int, int>(1000, 100000));
    ret_val.push_back(std::make_pair<int, int>(1000, 150000));
    ret_val.push_back(std::make_pair<int, int>(1000, 200000));
    return ret_val; 
}


void correlation_test(double* X, size_t rows, size_t cols)
{
    da::correlation_distance::Batch<> algorithm;
    algorithm.input.set(da::correlation_distance::data, makeTable(X, rows, cols));
    algorithm.compute();
    algorithm.getResult()->get(da::correlation_distance::correlationDistance);
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
            correlation_test(X, it->first, it->second);
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
