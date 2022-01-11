#ifndef XTESTNET_UTILS_H
#define XTESTNET_UTILS_H

#include "NvInfer.h"
#include "iostream"
#include "map"
#include "assert.h"
#include "fstream"

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

using namespace nvinfer1;

std::map<std::string, Weights> loadWeights(const std::string file);

void printWeightKeys(const std::map<std::string, Weights> *mp);

#endif //XTESTNET_UTILS_H
