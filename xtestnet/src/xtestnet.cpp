#include "xtestnet.h"

#define batchSize 1

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

    // /////////////////////////
    // ADDED CUSTOM PLUGIN /////
    // /////////////////////////
    //auto creator = getPluginRegistry()->registerCreator();
    auto creator = getPluginRegistry()->getPluginCreator("MPlugin_TRT", "1");
    assert(creator && "Plugin failed");
    //PluginFieldCollection pfc;
    //IPluginV2 *pluginObj = creator->createPlugin("MPlugin", &pfc);

    //ITensor *inputTensors[] = {softmax->getOutput(0)};
    //auto mPluginLayer = network->addPluginV2(inputTensors, 1, *pluginObj);
    //assert(mPluginLayer);

//    mPluginLayer->getOutput(0)->setName(output_name_);
//    network->markOutput(*mPluginLayer->getOutput(0));

    softmax->getOutput(0)->setName(output_name_);
    network->markOutput(*softmax->getOutput(0));
    //pluginObj->destroy();

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

void XTestNet::deserializeEngine() {
    std::ifstream file("../xtestnet.plan", std::ios::binary | std::ios::in);
    if (file.good()) {
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        char *trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(trtModelStream, size),
                                                         InferDeleter());
        assert(mEngine != nullptr);
    }
}

void XTestNet::init() {
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(gLogger), InferDeleter());
    assert(mRuntime != nullptr);

    std::cout << "Deserialize Engine" << std::endl;
    deserializeEngine();

    mContext = std::shared_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext(), InferDeleter());
    assert(mContext != nullptr);

    float input[3 * 28 * 28];
    for (int i = 0; i < 3 * 28 * 28; i++)
        input[i] = 1.0;

    float output[10];
    for (int i = 0; i < 10; i++)
        output[i] = 1.0;

    // mContext->setOptimizationProfile(0);
    std::cout << "Finished init" << std::endl;
    inferenceOnce(*mContext, input, output, 28, 28);


    std::cout << "\nOutputs:\n";
    for (auto &ops: output)
        std::cout << ops << "--";
    std::cout << std::endl;
}

void XTestNet::inferenceOnce(IExecutionContext &context, float *input, float *output, int input_h, int input_w) {
    // Get engine from the context
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(input_name_);
    const int outputIndex = engine.getBindingIndex(output_name_);

    context.setBindingDimensions(inputIndex, Dims3(3, input_h, input_w));

    // Create GPU buffers on device -- allocate memory for input and output
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * input_h * input_w * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 10 * sizeof(float)));

    // create CUDA stream for simultaneous CUDA operations
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // copy input from host (CPU) to device (GPU)  in stream
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * input_h * input_w * sizeof(float),
                          cudaMemcpyHostToDevice,
                          stream));

    // execute inference using context provided by engine
    context.enqueue(1, buffers, stream, nullptr);

    // copy output back from device (GPU) to host (CPU)
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 10 * sizeof(float), cudaMemcpyDeviceToHost,
                          stream));

    // synchronize the stream to prevent issues
    //      (block CUDA and wait for CUDA operations to be completed)
    cudaStreamSynchronize(stream);

    // Release stream and buffers (memory)
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}