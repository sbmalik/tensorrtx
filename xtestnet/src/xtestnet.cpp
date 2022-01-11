#include "xtestnet.h"

XTestNet::~XTestNet() {}

ICudaEngine *XTestNet::createEngine(IBuilder *builder, IBuilderConfig *config) {
    std::map<std::string, Weights> weightMap = loadWeights("../xtestnet.wts");

    INetworkDefinition *network = builder->createNetworkV2(0U);

    ITensor *data = network->addInput(input_name_, dt, Dims3{3, 28, 28});
    assert(data);

    IActivationLayer *cb1 = addConvBlock(network, weightMap, *data, 16, "cb1.", 5, 0);
    assert(cb1);

    IActivationLayer *cb2 = addConvBlock(network, weightMap, *cb1->getOutput(0), 32, "cb2.", 5, 0);
    assert(cb1);


    IElementWiseLayer *resBlock = addResidualBlock(network, weightMap, *cb2->getOutput(0), 32, 64, "res1.");
    assert(resBlock);


    IActivationLayer *cb3 = addConvBlock(network, weightMap, *resBlock->getOutput(0), 128, "cb3.", 5, 0);
    assert(cb3);


    IActivationLayer *cb4 = addConvBlock(network, weightMap, *cb3->getOutput(0), 8, "cb4.", 13, 0);
    assert(cb4);

    IFullyConnectedLayer *fc = network->addFullyConnected(*cb4->getOutput(0), 10,
                                                          weightMap["fc.1.weight"],
                                                          weightMap["fc.1.bias"]);
    assert(fc);

    ISoftMaxLayer *softmax = network->addSoftMax(*fc->getOutput(0));
    assert(softmax);

    softmax->getOutput(0)->setName(output_name_);
    network->markOutput(*softmax->getOutput(0));

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1 << 20);

    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    network->destroy();

    for (auto &mem: weightMap) {
        free((void *) (mem.second.values));
    }

    return engine;

}

void XTestNet::serializeEngine() {
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create Engine
    ICudaEngine *engine = createEngine(builder, config);
    assert(engine != nullptr);

    // Serialize Engine
    IHostMemory *modelStream{nullptr};
    modelStream = engine->serialize();
    assert(modelStream != nullptr);

    // MUST: Destroy to free up extra memory
    engine->destroy();
    builder->destroy();

    // Write file
    std::ofstream p("../xtestnet.plan", std::ios::binary | std::ios::out);
    if (!p) {
        std::cerr << "Could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return;
}

void XTestNet::deserializeEngine() {}

void XTestNet::init() {}

void XTestNet::inferenceOnce(IExecutionContext &context, float *input, float *output, int input_h, int input_w) {}