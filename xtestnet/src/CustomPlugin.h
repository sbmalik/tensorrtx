/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUSTOM_PLUGIN_H
#define CUSTOM_PLUGIN_H

#include "NvInferPlugin.h"

#include <vector>
#include <string.h>
#include <iostream>
#include "NvInfer.h"
#include "cassert"
#include "cuda_runtime.h"

using namespace nvinfer1;
using namespace std;

class CustomPlugin : public IPluginV2 {
public:
    CustomPlugin(const string name);

    CustomPlugin() = delete;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int batchSize) const noexcept override;

    int enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void *buffer) const noexcept override;

    void configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type,
                             PluginFormat format, int maxBatchSize) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char *getPluginType() const noexcept override;

    const char *getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2 *clone() const noexcept override;

    void setPluginNamespace(const char *pluginNamespace) noexcept override;

    const char *getPluginNamespace() const noexcept override;

private:
    const string mLayerName;
    string mNamespace;
};

class CustomPluginCreator : public IPluginCreator {
public:
    CustomPluginCreator();

    const char *getPluginName() const noexcept override;

    const char *getPluginVersion() const noexcept override;

    const PluginFieldCollection *getFieldNames() noexcept override;

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char *pluginNamespace) noexcept override;

    const char *getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static vector<PluginField> mPluginAttributes;
    string mNamespace;
};

#endif /* CustomPlugin.h */
