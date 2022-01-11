#ifndef XTESTNET_XTESTNET_H
#define XTESTNET_XTESTNET_H

#include "memory"
#include "logging.h"
#include "utils.h"
#include "layers.h"

class XTestNet {
public:
//    XTestNet();

    ~XTestNet();

    ICudaEngine *createEngine(IBuilder *builder, IBuilderConfig *config);

    void serializeEngine();

    void deserializeEngine();

    void init();

    void inferenceOnce(IExecutionContext &context, float *input, float *output, int input_h, int input_w);

private:
    Logger gLogger;
    std::shared_ptr<nvinfer1::IRuntime> mRuntime;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext;
    DataType dt = DataType::kFLOAT;

    const char *input_name_ = "input";
    const char *output_name_ = "output";

};


#endif //XTESTNET_XTESTNET_H
