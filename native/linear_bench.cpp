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
namespace dal=da::linear_regression;

#define REPS 10


template<typename T>
ds::SharedPtr<dm::HomogenNumericTable<T> > makeTable(T* data, size_t rows, size_t cols)
{
    return ds::SharedPtr<dm::HomogenNumericTable<T> >(new dm::HomogenNumericTable<T>(data, cols, rows));
}


std::vector<std::pair<int, int> > init_problem_sizes() 
{
    std::vector<std::pair<int, int> > ret_val;
    ret_val.push_back(std::make_pair<int, int>(1000000, 5));
    ret_val.push_back(std::make_pair<int, int>(1000000, 25));
    ret_val.push_back(std::make_pair<int, int>(1000000, 50));
    ret_val.push_back(std::make_pair<int, int>(5000000, 5));
    ret_val.push_back(std::make_pair<int, int>(5000000, 25));
    ret_val.push_back(std::make_pair<int, int>(5000000, 50));
    ret_val.push_back(std::make_pair<int, int>(10000000, 5));
    ret_val.push_back(std::make_pair<int, int>(10000000, 25));
    ret_val.push_back(std::make_pair<int, int>(10000000, 50));
    return ret_val; 
}

ds::SharedPtr<dal::training::Result> trainingResult;
void linear_fit_test(double* X, double* Y, size_t rows, size_t cols)
{
    dal::training::Batch<> algorithm;
    algorithm.input.set(dal::training::data, makeTable(X, rows, cols));
    algorithm.input.set(dal::training::dependentVariables, makeTable(Y, rows, cols));
    algorithm.compute();
    trainingResult = algorithm.getResult();
}

void linear_predict_test(double* X, size_t rows, size_t cols)
{
    dal::prediction::Batch<> algorithm;
    algorithm.input.set(dal::prediction::data, makeTable(X, rows, cols));
    algorithm.input.set(dal::prediction::model, trainingResult->get(dal::training::model));
    algorithm.compute();
    algorithm.getResult()->get(dal::prediction::prediction);
}

void bench()
{
    std::vector<std::pair<int, int> > problem_sizes = init_problem_sizes();
    std::vector<std::pair<int, int> >::iterator it; 
    for (it = problem_sizes.begin(); it != problem_sizes.end(); it++) {
        size_t size = it->first * it->second;
        double* X = new double[size];
        double* Xp = new double[size];
        double* Y = new double[size];
        for(size_t i = 0; i < size; i++) {
            X[i] = (double)rand() / RAND_MAX;
            Xp[i] = (double)rand() / RAND_MAX;
            Y[i] = (double)rand() / RAND_MAX;
        }
        std::vector<std::chrono::duration<double> > times_fit;
        std::vector<std::chrono::duration<double> > times_predict;
        for(int i = 0; i < REPS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            linear_fit_test(X, Y, it->first, it->second);
            auto finish = std::chrono::high_resolution_clock::now();
            times_fit.push_back(finish - start);

            start = std::chrono::high_resolution_clock::now();
            linear_predict_test(Xp, it->first, it->second);
            finish = std::chrono::high_resolution_clock::now();
            times_predict.push_back(finish - start);
        }
        std::cout << it->first << " " << it->second 
                  << " " << std::min_element(times_fit.begin(), times_fit.end())->count() << " "
                  << " " << std::min_element(times_predict.begin(), times_predict.end())->count() << std::endl;
        delete[] X;
        delete[] Xp;
        delete[] Y;
    }
}


int main()
{
    bench();
    return 0;
}
