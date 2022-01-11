#ifndef XTESTNET_LAYERS_H
#define XTESTNET_LAYERS_H

#include "NvInfer.h"
#include "map"
#include "math.h"
#include "assert.h"

using namespace nvinfer1;

IScaleLayer *addBatchNorm2d(INetworkDefinition *network,
                            std::map<std::string, Weights> &weightMap,
                            ITensor &input,
                            std::string lname,
                            float eps);

IActivationLayer *addConvBlock(INetworkDefinition *network,
                               std::map<std::string, Weights> &weightMap,
                               ITensor &input,
                               int outch, std::string lname,
                               int ks, int padding);

IElementWiseLayer *addResidualBlock(INetworkDefinition *network,
                                    std::map<std::string, Weights> &weightMap,
                                    ITensor &input,
                                    int inch, int outch, std::string lname);


#endif //XTESTNET_LAYERS_H
