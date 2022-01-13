#ifndef XPRAC_MPLUGIN_H
#define XPRAC_MPLUGIN_H

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "iostream"
#include "cassert"
#include "vector"
#include "cuda_runtime.h"

//namespace MPlugin {
//    static const float num_to_add = 2.0;
//}

namespace nvinfer1 {
    class MPlugin : public IPluginV2Ext {
    public:
        MPlugin(const char *name);

        MPlugin(const std::string name, size_t copy_size);

        MPlugin(const char *name, const void *serialData, size_t serialLength);

        ~MPlugin();

        // ///////////////////
        // For Plugin Setup
        // //////////////////
        int getNbOutputs() const noexcept override;

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override;

        DataType
        getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override;

        bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

        size_t getWorkspaceSize(int) const noexcept override;

        void serialize(void *buffer) const noexcept override;

        size_t getSerializationSize() const noexcept override;

        // ////////////////////////////
        // For Plugin Build & Inference
        // ///////////////////////////
        int initialize() noexcept override;

        void terminate() noexcept override;

        void destroy() noexcept override;

        IPluginV2Ext *clone() const noexcept override;

        const char *getPluginType() const

        noexcept override;

        const char *getPluginVersion() const

        noexcept override;

        void setPluginNamespace(const char *libNamespace) noexcept override;

        const char *getPluginNamespace() const noexcept override;

        void configurePlugin(Dims const *inputDims, int32_t nbInputs, Dims const *outputDims, int32_t nbOutputs,
                             DataType const *inputTypes, DataType const *outputTypes, bool const *inputIsBroadcast,
                             bool const *outputIsBroadcast, PluginFormat floatFormat,
                             int32_t maxBatchSize) noexcept override;
//        virtual void
//        configureWithFormat(Dims const *inputDims, int32_t nbInputs, Dims const *outputDims, int32_t nbOutputs,
//                            DataType type, PluginFormat format, int32_t maxBatchSize) noexcept;

        //int32_t enqueue(int32_t batchSize, void const *const *inputs, void ** outputs, void *workspace, cudaStream_t stream) override;
        int32_t enqueue(int32_t batchSize, void const *const *inputs, void *const *outputs, void *workspace,
                        cudaStream_t stream) noexcept override;

        /* For Broadcast support
         * -- canBroadcastInputAcrossBatch()
         * -- isOutputBroadcastAcrossBatch
         * */
        bool isOutputBroadcastAcrossBatch(
                int32_t outputIndex, bool const *inputIsBroadcasted, int32_t nbInputs) const noexcept override;

        bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

    private:
        const std::string mLayerName;
        size_t mCopySize;
        std::string mNamespace;

    };

    class MPluginCreator : public IPluginCreator {
    public:
        MPluginCreator();

        ~MPluginCreator() override = default;

        const char *getPluginName() const

        noexcept override;

        const char *getPluginVersion() const

        noexcept override;

        void setPluginNamespace(const char *libNamespace)

        noexcept override {
            mNamespace = libNamespace;
        }

        const char *getPluginNamespace() const

        noexcept override {
            return mNamespace.c_str();
        }

        const PluginFieldCollection *getFieldNames()

        noexcept override;

        IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc)

        noexcept override;

        IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength)

        noexcept override;


    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
//REGISTER_TENSORRT_PLUGIN(MPluginCreator);

};


#endif //XPRAC_MPLUGIN_H
