#include "mPlugin.h"

// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char *&buffer, const T &val) {
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template<typename T>
T readFromBuffer(const char *&buffer) {
    T val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
    return val;
}

using namespace MPlugin;

static const char *M_PLUGIN_VERSION{"1"};
static const char *M_PLUGIN_NAME{"MPlugin_TRT"};

namespace nvinfer1 {
    // ///////////////////////////////////////////////////////////////////////////
    // Custom Plugin /////////////////////////////////////////////////////////////
    // ///////////////////////////////////////////////////////////////////////////
    MPlugin::MPlugin(const char *name)
            : mLayerName(name) {
    }

    MPlugin::MPlugin(const std::string name, size_t copy_size)
            : mLayerName(name), mCopySize(copy_size) {
    }

    MPlugin::MPlugin(const std::string name, const void *data, size_t length)
            : mLayerName(name) {
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        mCopySize = readFromBuffer<size_t>(d);
        assert(d == a + length);
    }

    int MPlugin::getNbOutputs() const noexcept {
        return 1;
    }

    Dims MPlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept {
        assert(nbInputDims == 2);
        assert(index == 0);
        assert(inputs[1].nbDims == 4);
        return Dims3(inputs[1].d[1], inputs[1].d[2], inputs[1].d[3]);
    }

    int MPlugin::initialize() noexcept {
        // return STATUS_SUCCESS;
        return 1;
    }

    size_t MPlugin::getWorkspaceSize(int) const noexcept {
        return 0;
    }

//    DataType MPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept {
//        assert(index == 0);
//        return DataType::kFLOAT;
//    }

    void MPlugin::serialize(void *buffer) const noexcept {
        char *d = reinterpret_cast<char *>(buffer), *a = d;
        writeToBuffer<size_t>(d, mCopySize);
        assert(d == a + getSerializationSize());
    }

    void MPlugin::terminate() noexcept {}

    void MPlugin::destroy() noexcept {
        delete this;
    }

    size_t MPlugin::getSerializationSize() const noexcept {
        return sizeof(size_t);
    }

    const char *MPlugin::getPluginType() const noexcept {
        return M_PLUGIN_NAME;
    }

    const char *MPlugin::getPluginVersion() const noexcept {
        return M_PLUGIN_VERSION;
    }

    void MPlugin::setPluginNamespace(const char *libNamespace) noexcept {
        mNamespace = libNamespace;
    }

    const char *MPlugin::getPluginNamespace() const noexcept {
        return mNamespace.c_str();
    }

    bool MPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept {
        return (type == DataType::kFLOAT && format == PluginFormat::kCHW32);
    }

    void
    MPlugin::configureWithFormat(Dims const *inputDims, int32_t nbInputs, Dims const *outputDims, int32_t nbOutputs,
                                 DataType type, PluginFormat format, int32_t maxBatchSize) noexcept {

    }

    int32_t MPlugin::enqueue(int32_t batchSize, void const *const *inputs, void *const *outputs, void *workspace,
                             cudaStream_t stream) noexcept {
        float *output = reinterpret_cast<float *>(outputs[0]);
        // expand to batch size
        for (int i = 0; i < batchSize; i++) {
            auto ret = cudaMemcpyAsync(output, inputs[0], mCopySize, cudaMemcpyDeviceToDevice, stream);
//            output = output + num_to_add;
//        auto ret = cudaMemcpyAsync(output + i * mCopySize, inputs[1], mCopySize, cudaMemcpyDeviceToDevice, stream);
            if (ret != 0) {
                std::cout << "Cuda failure: " << ret;
                abort();
            }

        }
        return 0;
    }

// ///////////////////////////////////////////////////////////////////////////
// Plugin Creator ////////////////////////////////////////////////////////////
// ///////////////////////////////////////////////////////////////////////////
    PluginFieldCollection MPluginCreator::mFC{};

    MPluginCreator::MPluginCreator() {
        mFC.nbFields = 0;
        mFC.fields = nullptr;
    }

    const char *MPluginCreator::getPluginName() const noexcept {
        return M_PLUGIN_NAME;
    }

    const char *MPluginCreator::getPluginVersion() const noexcept {
        return M_PLUGIN_VERSION;
    }

    const PluginFieldCollection *MPluginCreator::getFieldNames() noexcept {
        return &mFC;
    }


    IPluginV2 *MPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept {
        auto *plugin = new MPlugin(name);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    IPluginV2 *
    MPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept {
        return new MPlugin(name, serialData, serialLength);
    }

}

