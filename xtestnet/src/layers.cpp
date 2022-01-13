#include "layers.h"
#include "iostream"

IScaleLayer *addBatchNorm2d(INetworkDefinition *network,
                            std::map<std::string, Weights> &weightMap,
                            ITensor &input,
                            std::string lname,
                            float eps) {
    float *gamma = (float *) weightMap[lname + ".weight"].values;
    float *beta = (float *) weightMap[lname + ".bias"].values;
    float *mean = (float *) weightMap[lname + ".running_mean"].values;
    float *var = (float *) weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

//    std::cout << len << std::endl;

    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;

    IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);

    return scale_1;
}

IActivationLayer *addConvBlock(INetworkDefinition *network,
                               std::map<std::string, Weights> &weightMap,
                               ITensor &input,
                               int outch, std::string lname,
                               int ks, int padding) {
    IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, DimsHW{ks, ks},
                                                         weightMap[lname + "0.weight"],
                                                         weightMap[lname + "0.bias"]);
    assert(conv1);
    conv1->setPaddingNd(DimsHW{padding, padding});

    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);

    IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    return relu1;

}

IElementWiseLayer *addResidualBlock(INetworkDefinition *network,
                                    std::map<std::string, Weights> &weightMap,
                                    ITensor &input,
                                    int inch, int outch, std::string lname) {
    IActivationLayer *res1 = addConvBlock(network, weightMap, input, outch, "res1.", 3, 1);
    IActivationLayer *res2 = addConvBlock(network, weightMap, *res1->getOutput(0), outch, "res2.", 3, 1);
    IActivationLayer *updated_ip = addConvBlock(network, weightMap, input, outch, "res_same.", 1, 0);
    IElementWiseLayer *addLayer = network->addElementWise(*updated_ip->getOutput(0), *res2->getOutput(0),
                                                          ElementWiseOperation::kSUM);
    return addLayer;
}