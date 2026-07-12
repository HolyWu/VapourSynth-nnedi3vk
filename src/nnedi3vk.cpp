/*
    Copyright (C) 2026  Holy Wu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>

#include <volk.h>

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

using namespace std::string_literals;

// Must match common.glsl.
constexpr int MARGIN_H = 24;
constexpr int MARGIN_V = 3;

const uint32_t prescreenU8Spv[] = {
#include "prescreen_u8.h"
};
const uint32_t prescreenU16Spv[] = {
#include "prescreen_u16.h"
};
const uint32_t prescreenF16Spv[] = {
#include "prescreen_f16.h"
};
const uint32_t prescreenF32Spv[] = {
#include "prescreen_f32.h"
};
const uint32_t predictU8Spv[] = {
#include "predict_u8.h"
};
const uint32_t predictU16Spv[] = {
#include "predict_u16.h"
};
const uint32_t predictF16Spv[] = {
#include "predict_f16.h"
};
const uint32_t predictF32Spv[] = {
#include "predict_f32.h"
};
const uint32_t predictCvU8Spv[] = {
#include "predict_cv_u8.h"
};
const uint32_t predictCvU16Spv[] = {
#include "predict_cv_u16.h"
};
const uint32_t predictCvF16Spv[] = {
#include "predict_cv_f16.h"
};
const uint32_t predictCvF32Spv[] = {
#include "predict_cv_f32.h"
};

struct SpvBlob {
    const uint32_t* code;
    size_t size;
};

// Indexed [kernel][pixelType]; kernels: 0 = prescreen, 1 = predict,
// 2 = predict (cooperative vector).
constexpr SpvBlob kSpv[3][4] = {
    { { prescreenU8Spv, sizeof(prescreenU8Spv) },
      { prescreenU16Spv, sizeof(prescreenU16Spv) },
      { prescreenF16Spv, sizeof(prescreenF16Spv) },
      { prescreenF32Spv, sizeof(prescreenF32Spv) } },
    { { predictU8Spv, sizeof(predictU8Spv) },
      { predictU16Spv, sizeof(predictU16Spv) },
      { predictF16Spv, sizeof(predictF16Spv) },
      { predictF32Spv, sizeof(predictF32Spv) } },
    { { predictCvU8Spv, sizeof(predictCvU8Spv) },
      { predictCvU16Spv, sizeof(predictCvU16Spv) },
      { predictCvF16Spv, sizeof(predictCvF16Spv) },
      { predictCvF32Spv, sizeof(predictCvF32Spv) } },
};

#define VK_CHECK(expr)                                                                             \
    do {                                                                                           \
        const VkResult vkCheckResult_ = (expr);                                                    \
        if (vkCheckResult_ != VK_SUCCESS)                                                          \
            throw std::runtime_error(#expr " failed with VkResult "s +                             \
                                     std::to_string(static_cast<int>(vkCheckResult_)));            \
    } while (0)

VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) {
    return (v + a - 1) & ~(a - 1);
}

// Sub-allocation alignment within the shared upload/device/readback buffers
// (the spec caps minStorageBufferOffsetAlignment at 256).
constexpr VkDeviceSize BUF_ALIGN = 256;

// Reserves bytes at the next aligned offset and returns that offset.
VkDeviceSize suballoc(VkDeviceSize& off, VkDeviceSize bytes) {
    off = alignUp(off, BUF_ALIGN);
    const VkDeviceSize o = off;
    off += bytes;
    return o;
}

bool envFlag(const char* name) {
    const char* env = std::getenv(name);
    return env && env[0] && env[0] != '0';
}

bool hasExtension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    return std::any_of(exts.begin(), exts.end(), [&](const VkExtensionProperties& e) {
        return std::strcmp(e.extensionName, name) == 0;
    });
}

VkResult waitTimeline(VkDevice device, VkSemaphore semaphore, uint64_t value) {
    const VkSemaphoreWaitInfo wi{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                  .semaphoreCount = 1,
                                  .pSemaphores = &semaphore,
                                  .pValues = &value };
    return vkWaitSemaphores(device, &wi, UINT64_MAX);
}

// Round-to-nearest-even float32 -> float16 conversion.
uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t absx = x & 0x7fffffffu;

    if (absx >= 0x7f800000u) // inf/nan
        return static_cast<uint16_t>(sign | 0x7c00u | ((absx > 0x7f800000u) ? 0x200u : 0u));
    if (absx >= 0x477ff000u) // overflows to inf
        return static_cast<uint16_t>(sign | 0x7c00u);
    if (absx < 0x38800000u) { // subnormal or zero
        if (absx < 0x33000001u)
            return static_cast<uint16_t>(sign);
        const int shift = 125 - static_cast<int>(absx >> 23);
        const uint32_t mant = (absx & 0x7fffffu) | 0x800000u;
        uint32_t half = mant >> (shift + 1);
        const uint32_t rem = mant & ((2u << shift) - 1);
        const uint32_t halfway = 1u << shift;
        if (rem > halfway || (rem == halfway && (half & 1u)))
            half++;
        return static_cast<uint16_t>(sign | half);
    }

    uint32_t half = ((absx >> 13) & 0x3ffu) | ((((absx >> 23) - 112u) & 0x1fu) << 10);
    const uint32_t rem = absx & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
        half++;
    return static_cast<uint16_t>(sign | half);
}

//////////////////////////////////////////
// Shared Vulkan instance/device management

struct VulkanGlobals {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> physicalDevices;

    ~VulkanGlobals() {
        if (messenger)
            vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
        if (instance)
            vkDestroyInstance(instance, nullptr);
    }
};

struct DeviceCtx {
    std::shared_ptr<VulkanGlobals> globals;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkQueue queue = VK_NULL_HANDLE;
    std::mutex queueMutex;
    // Dedicated transfer queue for readback copies (equal to the compute
    // queue/family when the device has no separate transfer family).
    uint32_t transferFamily = 0;
    VkQueue transferQueue = VK_NULL_HANDLE;
    std::mutex transferMutex;
    VkPhysicalDeviceProperties props{};
    bool hasFloat16 = false;
    uint32_t subgroupSize = 32; // required subgroup size for the predict kernel
    bool canRequireSubgroupSize = false;
    bool fullSubgroups = false;
    bool hasExecutableProperties = false;
    bool hasCoopVecF16 = false; // VK_NV_cooperative_vector with an f16 x f16 -> f16 combination
    uint32_t maxCoopVecComponents = 0;

    ~DeviceCtx() {
        if (allocator)
            vmaDestroyAllocator(allocator);
        if (device)
            vkDestroyDevice(device, nullptr);
    }
};

std::mutex g_vkMutex;
std::weak_ptr<VulkanGlobals> g_globals;
std::map<uint32_t, std::weak_ptr<DeviceCtx>> g_devices;

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT types,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             [[maybe_unused]] void* userData) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::fprintf(stderr, "nnedi3vk validation: %s\n", data->pMessage);
    return VK_FALSE;
}

// Must be called with g_vkMutex held.
std::shared_ptr<VulkanGlobals> acquireGlobals() {
    if (auto g = g_globals.lock())
        return g;

    static std::once_flag volkOnce;
    static VkResult volkResult;
    std::call_once(volkOnce, [] { volkResult = volkInitialize(); });
    if (volkResult != VK_SUCCESS)
        throw std::runtime_error("failed to load the Vulkan loader");

    if (volkGetInstanceVersion() < VK_API_VERSION_1_4)
        throw std::runtime_error("Vulkan instance version 1.4 is required");

    auto g = std::make_shared<VulkanGlobals>();

    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, exts.data());

    auto hasExt = [&](const char* name) { return hasExtension(exts, name); };

    std::vector<const char*> enabledExts;
    std::vector<const char*> enabledLayers;
    VkInstanceCreateFlags flags = 0;

    if (hasExt(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) { // MoltenVK
        enabledExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    const bool validation = envFlag("NNEDI3VK_VALIDATION");
    if (validation) {
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> layers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
        const bool hasLayer = std::any_of(layers.begin(), layers.end(), [](const VkLayerProperties& l) {
            return std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0;
        });
        if (hasLayer) {
            enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
            if (hasExt(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
                enabledExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }

    VkApplicationInfo appInfo{ .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                               .pApplicationName = "nnedi3vk",
                               .apiVersion = VK_API_VERSION_1_4 };

    VkInstanceCreateInfo ici{ .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                              .flags = flags,
                              .pApplicationInfo = &appInfo,
                              .enabledLayerCount = static_cast<uint32_t>(enabledLayers.size()),
                              .ppEnabledLayerNames = enabledLayers.data(),
                              .enabledExtensionCount = static_cast<uint32_t>(enabledExts.size()),
                              .ppEnabledExtensionNames = enabledExts.data() };

    VK_CHECK(vkCreateInstance(&ici, nullptr, &g->instance));
    volkLoadInstance(g->instance);

    if (!enabledLayers.empty() && vkCreateDebugUtilsMessengerEXT) {
        VkDebugUtilsMessengerCreateInfoEXT dci{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
        };
        vkCreateDebugUtilsMessengerEXT(g->instance, &dci, nullptr, &g->messenger);
    }

    uint32_t devCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(g->instance, &devCount, nullptr));
    g->physicalDevices.resize(devCount);
    VK_CHECK(vkEnumeratePhysicalDevices(g->instance, &devCount, g->physicalDevices.data()));

    g_globals = g;
    return g;
}

// Must be called with g_vkMutex held.
std::shared_ptr<DeviceCtx> acquireDevice(uint32_t deviceIndex) {
    if (auto it = g_devices.find(deviceIndex); it != g_devices.end())
        if (auto d = it->second.lock())
            return d;

    auto globals = acquireGlobals();

    if (deviceIndex >= globals->physicalDevices.size())
        throw std::runtime_error("device_index out of range: " + std::to_string(deviceIndex) + " (" +
                                 std::to_string(globals->physicalDevices.size()) + " device(s) available)");

    auto d = std::make_shared<DeviceCtx>();
    d->globals = globals;
    d->physicalDevice = globals->physicalDevices[deviceIndex];

    vkGetPhysicalDeviceProperties(d->physicalDevice, &d->props);
    if (d->props.apiVersion < VK_API_VERSION_1_4)
        throw std::runtime_error("device '"s + d->props.deviceName + "' does not support Vulkan 1.4");

    {
        VkPhysicalDeviceVulkan13Properties props13{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES };
        VkPhysicalDeviceVulkan11Properties props11{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
                                                    .pNext = &props13 };
        VkPhysicalDeviceProperties2 props2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                            .pNext = &props11 };
        vkGetPhysicalDeviceProperties2(d->physicalDevice, &props2);

        constexpr VkSubgroupFeatureFlags needed = VK_SUBGROUP_FEATURE_BASIC_BIT |
            VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT;
        if ((props11.subgroupSupportedOperations & needed) != needed)
            throw std::runtime_error("device does not support subgroup arithmetic/ballot operations");

        d->canRequireSubgroupSize = (props13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
        if (d->canRequireSubgroupSize)
            d->subgroupSize = std::clamp(32u, props13.minSubgroupSize, props13.maxSubgroupSize);
        else
            d->subgroupSize = props11.subgroupSize;
    }
    // fullSubgroups is filled in below once features are queried.

    // Queue family: prefer a dedicated compute queue.
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d->physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(d->physicalDevice, &qfCount, qfs.data());

    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; i++) {
        const auto f = qfs[i].queueFlags;
        if ((f & VK_QUEUE_COMPUTE_BIT) && !(f & VK_QUEUE_GRAPHICS_BIT)) {
            family = i;
            break;
        }
    }
    if (family == UINT32_MAX)
        for (uint32_t i = 0; i < qfCount; i++)
            if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                family = i;
                break;
            }
    if (family == UINT32_MAX)
        throw std::runtime_error("no compute queue found");
    d->queueFamily = family;

    // Dedicated transfer family (DMA engine) so readback copies overlap with
    // compute instead of serializing on the compute queue.
    uint32_t transferFamily = family;
    for (uint32_t i = 0; i < qfCount; i++) {
        const auto f = qfs[i].queueFlags;
        if ((f & VK_QUEUE_TRANSFER_BIT) && !(f & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT))) {
            transferFamily = i;
            break;
        }
    }
    d->transferFamily = transferFamily;

    // Device extensions.
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(d->physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(d->physicalDevice, nullptr, &extCount, exts.data());

    auto hasExt = [&](const char* name) { return hasExtension(exts, name); };

    std::vector<const char*> enabledExts;
    const bool hasBudget = hasExt(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    const bool hasPriority = hasExt(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
    const bool hasMaxReconv = hasExt(VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME);
    const bool hasWgExplicit = hasExt(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
    const bool hasExecProps = hasExt(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    const bool hasCoopVec = hasExt(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
    if (hasBudget)
        enabledExts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (hasPriority)
        enabledExts.push_back(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
    if (hasMaxReconv)
        enabledExts.push_back(VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME);
    if (hasWgExplicit)
        enabledExts.push_back(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
    if (hasExecProps)
        enabledExts.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    if (hasCoopVec)
        enabledExts.push_back(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
    if (hasExt("VK_KHR_portability_subset")) // MoltenVK
        enabledExts.push_back("VK_KHR_portability_subset");

    // Features: query supported, then enable what we use.
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR supExecProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR };
    VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR supMaxReconv{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR };
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR supWgExplicit{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR };
    VkPhysicalDeviceMemoryPriorityFeaturesEXT supMemPriority{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT };
    VkPhysicalDeviceCooperativeVectorFeaturesNV supCoopVec{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV };

    const auto chainIf = [](void*& head, bool cond, auto& s) {
        if (cond) {
            s.pNext = head;
            head = &s;
        }
    };

    void* extChain = nullptr;
    chainIf(extChain, hasExecProps, supExecProps);
    chainIf(extChain, hasMaxReconv, supMaxReconv);
    chainIf(extChain, hasWgExplicit, supWgExplicit);
    chainIf(extChain, hasPriority, supMemPriority);
    chainIf(extChain, hasCoopVec, supCoopVec);

    VkPhysicalDeviceVulkan14Features sup14{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
                                            .pNext = extChain };
    VkPhysicalDeviceVulkan13Features sup13{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &sup14 };
    VkPhysicalDeviceVulkan12Features sup12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &sup13 };
    VkPhysicalDeviceVulkan11Features sup11{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &sup12 };
    VkPhysicalDeviceFeatures2 sup2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &sup11 };
    vkGetPhysicalDeviceFeatures2(d->physicalDevice, &sup2);

    auto require = [](VkBool32 supported, const char* name) {
        if (!supported)
            throw std::runtime_error("required Vulkan feature not supported: "s + name);
    };
    require(sup11.storageBuffer16BitAccess, "storageBuffer16BitAccess");
    require(sup12.storageBuffer8BitAccess, "storageBuffer8BitAccess");
    require(sup2.features.shaderInt16, "shaderInt16");
    require(sup12.shaderInt8, "shaderInt8");
    require(sup12.scalarBlockLayout, "scalarBlockLayout");
    require(sup12.timelineSemaphore, "timelineSemaphore");
    require(sup12.hostQueryReset, "hostQueryReset");
    require(sup12.bufferDeviceAddress, "bufferDeviceAddress");
    require(sup12.vulkanMemoryModel, "vulkanMemoryModel");
    require(sup13.synchronization2, "synchronization2");
    require(sup13.maintenance4, "maintenance4");
    require(sup14.maintenance5, "maintenance5");
    require(sup14.maintenance6, "maintenance6");
    require(sup14.pushDescriptor, "pushDescriptor");

    d->hasFloat16 = sup12.shaderFloat16;
    d->fullSubgroups = sup13.computeFullSubgroups && sup13.subgroupSizeControl;
    d->hasExecutableProperties = hasExecProps && supExecProps.pipelineExecutableInfo;

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR enExecProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
        .pipelineExecutableInfo = supExecProps.pipelineExecutableInfo };
    VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR enMaxReconv{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR,
        .shaderMaximalReconvergence = supMaxReconv.shaderMaximalReconvergence };
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR enWgExplicit{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
        .workgroupMemoryExplicitLayout = supWgExplicit.workgroupMemoryExplicitLayout,
        .workgroupMemoryExplicitLayoutScalarBlockLayout = supWgExplicit.workgroupMemoryExplicitLayoutScalarBlockLayout,
        .workgroupMemoryExplicitLayout8BitAccess = supWgExplicit.workgroupMemoryExplicitLayout8BitAccess,
        .workgroupMemoryExplicitLayout16BitAccess = supWgExplicit.workgroupMemoryExplicitLayout16BitAccess };
    VkPhysicalDeviceMemoryPriorityFeaturesEXT enMemPriority{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
        .memoryPriority = supMemPriority.memoryPriority };
    VkPhysicalDeviceCooperativeVectorFeaturesNV enCoopVec{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV,
        .cooperativeVector = supCoopVec.cooperativeVector };

    void* enExtChain = nullptr;
    chainIf(enExtChain, hasExecProps, enExecProps);
    chainIf(enExtChain, hasMaxReconv, enMaxReconv);
    chainIf(enExtChain, hasWgExplicit, enWgExplicit);
    chainIf(enExtChain, hasPriority, enMemPriority);
    chainIf(enExtChain, hasCoopVec, enCoopVec);

    // Cooperative vector: usable when the feature is present together with an
    // f16 x f16 -> f16 matrix-vector combination.
    if (hasCoopVec && supCoopVec.cooperativeVector && vkGetPhysicalDeviceCooperativeVectorPropertiesNV) {
        VkPhysicalDeviceCooperativeVectorPropertiesNV cvProps{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_PROPERTIES_NV };
        VkPhysicalDeviceProperties2 cvProps2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                              .pNext = &cvProps };
        vkGetPhysicalDeviceProperties2(d->physicalDevice, &cvProps2);
        d->maxCoopVecComponents = cvProps.maxCooperativeVectorComponents;

        uint32_t comboCount = 0;
        vkGetPhysicalDeviceCooperativeVectorPropertiesNV(d->physicalDevice, &comboCount, nullptr);
        std::vector<VkCooperativeVectorPropertiesNV> combos(
            comboCount, { .sType = VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV });
        vkGetPhysicalDeviceCooperativeVectorPropertiesNV(d->physicalDevice, &comboCount, combos.data());

        d->hasCoopVecF16 = std::any_of(combos.begin(), combos.end(), [](const VkCooperativeVectorPropertiesNV& c) {
            return c.inputType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                c.inputInterpretation == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                c.matrixInterpretation == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                c.resultType == VK_COMPONENT_TYPE_FLOAT16_KHR;
        });
    }

    VkPhysicalDeviceVulkan14Features en14{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
        .pNext = enExtChain,
        .shaderSubgroupRotate = sup14.shaderSubgroupRotate,
        .shaderSubgroupRotateClustered = sup14.shaderSubgroupRotateClustered,
        .shaderFloatControls2 = sup14.shaderFloatControls2,
        .shaderExpectAssume = sup14.shaderExpectAssume,
        .maintenance5 = VK_TRUE,
        .maintenance6 = VK_TRUE,
        .pipelineRobustness = sup14.pipelineRobustness,
        .pushDescriptor = VK_TRUE };
    VkPhysicalDeviceVulkan13Features en13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &en14,
        .pipelineCreationCacheControl = sup13.pipelineCreationCacheControl,
        .shaderTerminateInvocation = sup13.shaderTerminateInvocation,
        .subgroupSizeControl = sup13.subgroupSizeControl,
        .computeFullSubgroups = sup13.computeFullSubgroups,
        .synchronization2 = VK_TRUE,
        .shaderZeroInitializeWorkgroupMemory = sup13.shaderZeroInitializeWorkgroupMemory,
        .maintenance4 = VK_TRUE };
    VkPhysicalDeviceVulkan12Features en12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &en13,
        .storageBuffer8BitAccess = VK_TRUE,
        .uniformAndStorageBuffer8BitAccess = sup12.uniformAndStorageBuffer8BitAccess,
        .shaderFloat16 = sup12.shaderFloat16,
        .shaderInt8 = VK_TRUE,
        .scalarBlockLayout = VK_TRUE,
        .shaderSubgroupExtendedTypes = sup12.shaderSubgroupExtendedTypes,
        .hostQueryReset = VK_TRUE,
        .timelineSemaphore = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
        .vulkanMemoryModel = VK_TRUE,
        .vulkanMemoryModelDeviceScope = sup12.vulkanMemoryModelDeviceScope };
    VkPhysicalDeviceVulkan11Features en11{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &en12,
        .storageBuffer16BitAccess = VK_TRUE,
        .uniformAndStorageBuffer16BitAccess = sup11.uniformAndStorageBuffer16BitAccess };
    VkPhysicalDeviceFeatures2 en2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &en11 };
    en2.features.shaderInt16 = sup2.features.shaderInt16;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qcis[2];
    qcis[0] = VkDeviceQueueCreateInfo{ .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                       .queueFamilyIndex = family,
                                       .queueCount = 1,
                                       .pQueuePriorities = &priority };
    qcis[1] = VkDeviceQueueCreateInfo{ .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                       .queueFamilyIndex = d->transferFamily,
                                       .queueCount = 1,
                                       .pQueuePriorities = &priority };

    VkDeviceCreateInfo dci{ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                            .pNext = &en2,
                            .queueCreateInfoCount = d->transferFamily != family ? 2u : 1u,
                            .pQueueCreateInfos = qcis,
                            .enabledExtensionCount = static_cast<uint32_t>(enabledExts.size()),
                            .ppEnabledExtensionNames = enabledExts.data() };

    VK_CHECK(vkCreateDevice(d->physicalDevice, &dci, nullptr, &d->device));
    vkGetDeviceQueue(d->device, family, 0, &d->queue);
    if (d->transferFamily != family)
        vkGetDeviceQueue(d->device, d->transferFamily, 0, &d->transferQueue);
    else
        d->transferQueue = d->queue;

    VmaVulkanFunctions vmaFuncs{ .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
                                 .vkGetDeviceProcAddr = vkGetDeviceProcAddr };

    VmaAllocatorCreateFlags vmaFlags = VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT |
        VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT |
        VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
        VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT |
        VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
    if (hasBudget)
        vmaFlags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    if (hasPriority && supMemPriority.memoryPriority)
        vmaFlags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;

    VmaAllocatorCreateInfo aci{ .flags = vmaFlags,
                                .physicalDevice = d->physicalDevice,
                                .device = d->device,
                                .pVulkanFunctions = &vmaFuncs,
                                .instance = globals->instance,
                                .vulkanApiVersion = VK_API_VERSION_1_4 };
    VK_CHECK(vmaCreateAllocator(&aci, &d->allocator));

    g_devices[deviceIndex] = d;
    return d;
}

//////////////////////////////////////////
// NNEDI3 weights (ported from znedi3)

constexpr size_t NNEDI3_WEIGHTS_SIZE = 13574928;
constexpr unsigned NNEDI3_XDIM[] = { 8, 16, 32, 48, 8, 16, 32 };
constexpr unsigned NNEDI3_YDIM[] = { 6, 6, 6, 6, 4, 4, 4 };
constexpr unsigned NNEDI3_NNS[] = { 16, 32, 64, 128, 256 };

struct PrescreenerOldCoefficients {
    float kernel_l0[4][12 * 4];
    float bias_l0[4];

    float kernel_l1[4][4];
    float bias_l1[4];

    float kernel_l2[4][8];
    float bias_l2[4];
};

struct PrescreenerNewCoefficients {
    float kernel_l0[4][16 * 4];
    float bias_l0[4];

    float kernel_l1[4][4];
    float bias_l1[4];
};

struct PredictorModel {
    unsigned xdim = 0, ydim = 0, nns = 0;
    std::vector<float> softmax_q1, elliott_q1, softmax_bias_q1, elliott_bias_q1;
    std::vector<float> softmax_q2, elliott_q2, softmax_bias_q2, elliott_bias_q2;
};

double vecMean(const float* buf, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; i++)
        acc += buf[i];
    return acc / n;
}

template<typename Coeffs>
void subtractMean(Coeffs& coeffs, double pixelHalf) {
    for (unsigned n = 0; n < 4; n++) {
        const double m = vecMean(coeffs.kernel_l0[n], std::size(coeffs.kernel_l0[n]));
        for (float& x : coeffs.kernel_l0[n])
            x = static_cast<float>((x - m) / pixelHalf);
    }
}

void subtractMean(PredictorModel& model) {
    const size_t filterSize = model.xdim * model.ydim;
    const unsigned nns = model.nns;

    std::vector<double> softmaxMeans(nns);       // average of individual softmax filters
    std::vector<double> elliottMeans(nns);       // average of individual elliott filters
    std::vector<double> meanFilter(filterSize);  // pointwise average of all softmax filters
    double meanBias;

    const auto onePass = [&](float* softmax, float* elliott, float* softmaxBias) {
        std::fill(meanFilter.begin(), meanFilter.end(), 0.0);

        for (unsigned nn = 0; nn < nns; nn++) {
            softmaxMeans[nn] = vecMean(softmax + nn * filterSize, filterSize);
            elliottMeans[nn] = vecMean(elliott + nn * filterSize, filterSize);

            for (size_t k = 0; k < filterSize; k++)
                meanFilter[k] += softmax[nn * filterSize + k] - softmaxMeans[nn];
        }
        for (size_t k = 0; k < filterSize; k++)
            meanFilter[k] /= nns;
        meanBias = vecMean(softmaxBias, nns);

        for (unsigned nn = 0; nn < nns; nn++) {
            for (size_t k = 0; k < filterSize; k++) {
                softmax[nn * filterSize + k] -= static_cast<float>(softmaxMeans[nn] + meanFilter[k]);
                elliott[nn * filterSize + k] -= static_cast<float>(elliottMeans[nn]);
            }
            softmaxBias[nn] -= static_cast<float>(meanBias);
        }
    };

    onePass(model.softmax_q1.data(), model.elliott_q1.data(), model.softmax_bias_q1.data());
    onePass(model.softmax_q2.data(), model.elliott_q2.data(), model.softmax_bias_q2.data());
}

// Walks the nnedi3 weights file exactly like znedi3's read_nnedi3_weights and
// extracts the prescreeners plus the single predictor model that was asked
// for (etypeSel: 0 = abs, 1 = mse).
void readNNEDI3Weights(const float* data, unsigned nsizeSel, unsigned nnsSel, unsigned etypeSel,
                       PrescreenerOldCoefficients& psOld, PrescreenerNewCoefficients psNew[3],
                       PredictorModel& model) {
    const float* ptr = data;
    auto read = [&](float* dst, size_t n) {
        std::copy_n(ptr, n, dst);
        ptr += n;
    };

    // Old prescreener data.
    read(&psOld.kernel_l0[0][0], 4 * 48);
    read(psOld.bias_l0, 4);
    read(&psOld.kernel_l1[0][0], 4 * 4);
    read(psOld.bias_l1, 4);
    read(&psOld.kernel_l2[0][0], 4 * 8);
    read(psOld.bias_l2, 4);

    // New prescreener data.
    for (unsigned i = 0; i < 3; i++) {
        float kernelL0Shuffled[4 * 64];
        float kernelL1Shuffled[4 * 4];

        read(kernelL0Shuffled, 4 * 64);
        read(psNew[i].bias_l0, 4);
        read(kernelL1Shuffled, 4 * 4);
        read(psNew[i].bias_l1, 4);

        // Convert kernels back to row-major order.
        for (unsigned n = 0; n < 4; n++) {
            for (unsigned k = 0; k < 64; k++)
                psNew[i].kernel_l0[n][k] = kernelL0Shuffled[(k / 8) * 32 + n * 8 + k % 8];
            for (unsigned k = 0; k < 4; k++)
                psNew[i].kernel_l1[n][k] = kernelL1Shuffled[k * 4 + n];
        }
    }

    // ABS models, then MSE models; grouped by neuron count, then window size.
    for (unsigned m = 0; m < 2; m++) {
        for (unsigned i = 0; i < 5; i++) {
            for (unsigned j = 0; j < 7; j++) {
                const unsigned nns = NNEDI3_NNS[i];
                const size_t filterSize = NNEDI3_XDIM[j] * NNEDI3_YDIM[j];

                if (m == etypeSel && i == nnsSel && j == nsizeSel) {
                    model.xdim = NNEDI3_XDIM[j];
                    model.ydim = NNEDI3_YDIM[j];
                    model.nns = nns;

                    model.softmax_q1.resize(nns * filterSize);
                    model.elliott_q1.resize(nns * filterSize);
                    model.softmax_bias_q1.resize(nns);
                    model.elliott_bias_q1.resize(nns);
                    model.softmax_q2.resize(nns * filterSize);
                    model.elliott_q2.resize(nns * filterSize);
                    model.softmax_bias_q2.resize(nns);
                    model.elliott_bias_q2.resize(nns);

                    read(model.softmax_q1.data(), nns * filterSize);
                    read(model.elliott_q1.data(), nns * filterSize);
                    read(model.softmax_bias_q1.data(), nns);
                    read(model.elliott_bias_q1.data(), nns);
                    read(model.softmax_q2.data(), nns * filterSize);
                    read(model.elliott_q2.data(), nns * filterSize);
                    read(model.softmax_bias_q2.data(), nns);
                    read(model.elliott_bias_q2.data(), nns);
                } else {
                    ptr += 4 * nns * filterSize + 4 * nns;
                }
            }
        }
    }

    assert(static_cast<size_t>(ptr - data) == NNEDI3_WEIGHTS_SIZE / sizeof(float));
}

//////////////////////////////////////////
// Filter data

// Field order and types must match the PC push-constant block in
// shaders/common.glsl (scalar layout makes the correspondence purely
// positional).
struct PushConstants {
    int32_t width;
    int32_t rows;
    int32_t padStride;
    int32_t peak;
};

static_assert(sizeof(PushConstants) == 4 * 4, "must stay in sync with the PC block in common.glsl");

// Field order must match the constant_id assignments in the shaders (see
// specEntries).
struct SpecData {
    uint32_t wgSize;
    int32_t pscrn;
    int32_t xdim;
    int32_t ydim;
    int32_t nns;
    int32_t qual;
    int32_t sgSize;
    int32_t subgroups; // pixels per predict workgroup
    VkBool32 useList;
    uint32_t cvChunkBytes; // cooperative vector only
    int32_t cvMcu;
    int32_t cvNmc;
};

// Storage-buffer binding indices; must match the layout(binding = N)
// declarations across the kernels.
enum Binding : uint32_t {
    BindPad = 0,
    BindDst,
    BindPsW,
    BindPdW,
    BindPdB,
    BindList,
    BindCnt,
    BindCount,
};

struct PlaneSetup {
    bool process = false;
    int width = 0, height = 0; // output plane dimensions
    int rows = 0;              // interpolated rows (= field height)
    int padStride = 0, padHeight = 0;
    VkDeviceSize padOffset = 0, padBytes = 0;   // upload buffer
    VkDeviceSize dstOffset = 0, dstBytes = 0;   // device buffer
    VkDeviceSize listOffset = 0, listBytes = 0; // device buffer
    VkDeviceSize cntOffset = 0;                 // device buffer, 16 bytes
    VkDeviceSize rbOffset = 0;                  // readback buffer
};

struct StreamCtx {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandPool tpool = VK_NULL_HANDLE;    // transfer queue (readback copies)
    VkCommandBuffer tcmd = VK_NULL_HANDLE;
    VkBuffer upload = VK_NULL_HANDLE, devbuf = VK_NULL_HANDLE;
    VmaAllocation uploadAlloc = VK_NULL_HANDLE, devAlloc = VK_NULL_HANDLE;
    uint8_t* uploadMap = nullptr;
    VkSemaphore timeline = VK_NULL_HANDLE;
    uint64_t timelineValue = 0;
    VkQueryPool queryPool = VK_NULL_HANDLE; // profiling only
};

// Readback buffers are pooled separately from streams: a stream can be
// released (and start the next frame) while the previous frame's rows are
// still being copied out of its readback slot.
struct RbSlot {
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    uint8_t* map = nullptr;
};

struct NNEDI3Data {
    VSNode* node = nullptr;
    VSVideoInfo vi{};
    int field = 0;
    bool dh = false;
    int qual = 1, pscrn = 2;
    int peak = 0, pixelType = 0;

    std::shared_ptr<DeviceCtx> dev;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule prescreenModule = VK_NULL_HANDLE, predictModule = VK_NULL_HANDLE;
    VkPipeline prescreenPipe = VK_NULL_HANDLE, predictPipe = VK_NULL_HANDLE;
    VkBuffer weightsBuf = VK_NULL_HANDLE;
    VmaAllocation weightsAlloc = VK_NULL_HANDLE;
    VkDeviceSize psWOffset = 0, psWBytes = 0;
    VkDeviceSize pdWOffset = 0, pdWBytes = 0;
    VkDeviceSize pdBOffset = 0, pdBBytes = 0;
    PlaneSetup planes[3];
    VkDeviceSize uploadSize = 0, devbufSize = 0, rbSize = 0;

    uint32_t prescreenWG = 128;      // prescreen workgroup size (threads)
    uint32_t pixelsPerPredictWG = 4; // predict: pixels per workgroup

    bool useCoopVec = false; // predict via VK_NV_cooperative_vector
    uint32_t cvChunkBytes = 0;
    int32_t cvMCU = 0, cvNMC = 0;

    std::vector<std::unique_ptr<StreamCtx>> streams;
    std::mutex streamMutex;
    std::condition_variable streamCv;
    std::vector<int> freeStreams;

    std::vector<RbSlot> rbSlots;
    std::mutex rbMutex;
    std::condition_variable rbCv;
    std::vector<int> freeRbSlots;

    bool profile = false;
    std::mutex profileMutex;
    double prescreenMs = 0.0, predictMs = 0.0, copyMs = 0.0;
    int64_t profiledFrames = 0;

    void destroy() {
        if (!dev)
            return;
        const VkDevice device = dev->device;
        for (auto& s : streams) {
            if (s->timeline && s->timelineValue)
                waitTimeline(device, s->timeline, s->timelineValue);
            if (s->queryPool)
                vkDestroyQueryPool(device, s->queryPool, nullptr);
            if (s->timeline)
                vkDestroySemaphore(device, s->timeline, nullptr);
            if (s->pool)
                vkDestroyCommandPool(device, s->pool, nullptr);
            if (s->tpool)
                vkDestroyCommandPool(device, s->tpool, nullptr);
            if (s->upload)
                vmaDestroyBuffer(dev->allocator, s->upload, s->uploadAlloc);
            if (s->devbuf)
                vmaDestroyBuffer(dev->allocator, s->devbuf, s->devAlloc);
        }
        streams.clear();
        for (auto& slot : rbSlots)
            if (slot.buf)
                vmaDestroyBuffer(dev->allocator, slot.buf, slot.alloc);
        rbSlots.clear();
        if (prescreenPipe)
            vkDestroyPipeline(device, prescreenPipe, nullptr);
        if (predictPipe)
            vkDestroyPipeline(device, predictPipe, nullptr);
        if (prescreenModule)
            vkDestroyShaderModule(device, prescreenModule, nullptr);
        if (predictModule)
            vkDestroyShaderModule(device, predictModule, nullptr);
        if (pipelineLayout)
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (dsl)
            vkDestroyDescriptorSetLayout(device, dsl, nullptr);
        if (weightsBuf)
            vmaDestroyBuffer(dev->allocator, weightsBuf, weightsAlloc);

        if (profile && profiledFrames > 0)
            std::fprintf(stderr,
                         "nnedi3vk profile: frames=%lld prescreen=%.3fms predict=%.3fms copy=%.3fms (per frame GPU time)\n",
                         static_cast<long long>(profiledFrames), prescreenMs / profiledFrames,
                         predictMs / profiledFrames, copyMs / profiledFrames);
    }
};

//////////////////////////////////////////
// CPU-side padding

// Builds the padded field plane in the mapped upload buffer. Padded row i
// corresponds to source field row clamp(i - (MARGIN_V - fp), 0, fieldH - 1)
// (znedi3's PadFilter geometry, fp = !src_parity); horizontal margins
// replicate the edge pixels. Nothing is ever read back from the
// write-combined mapping.
template<typename T>
void copyPadPlane(const uint8_t* field8, ptrdiff_t fieldStrideBytes, T* pad, int padStride, int width,
                  int fieldH, int fp) {
    const int padH = fieldH + MARGIN_V * 2;

    for (int i = 0; i < padH; i++) {
        const int f = std::clamp(i - (MARGIN_V - fp), 0, fieldH - 1);
        const T* line = reinterpret_cast<const T*>(field8 + fieldStrideBytes * f);
        T* dst = pad + static_cast<ptrdiff_t>(padStride) * i;

        std::memcpy(dst + MARGIN_H, line, static_cast<size_t>(width) * sizeof(T));

        const T l = line[0];
        const T r = line[width - 1];
        for (int x = 0; x < MARGIN_H; x++) {
            dst[x] = l;
            dst[MARGIN_H + width + x] = r;
        }
    }
}

//////////////////////////////////////////
// GPU recording

void cmdMemoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                      VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    const VkMemoryBarrier2 mb{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                               .srcStageMask = srcStage,
                               .srcAccessMask = srcAccess,
                               .dstStageMask = dstStage,
                               .dstAccessMask = dstAccess };
    const VkDependencyInfo dep{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                .memoryBarrierCount = 1,
                                .pMemoryBarriers = &mb };
    vkCmdPipelineBarrier2(cmd, &dep);
}

// Submits one command buffer on `queue` (guarded by `mutex`), optionally waiting
// on `sem` reaching `waitValue` (waitValue == 0 means no wait; timeline values
// always start at 1) and signalling `sem` with `signalValue`.
void submitTimeline(std::mutex& mutex, VkQueue queue, VkCommandBuffer cmd, VkSemaphore sem,
                    uint64_t waitValue, VkPipelineStageFlags2 waitStage, uint64_t signalValue) {
    const VkCommandBufferSubmitInfo cbsi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                          .commandBuffer = cmd };
    const VkSemaphoreSubmitInfo wait{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                      .semaphore = sem,
                                      .value = waitValue,
                                      .stageMask = waitStage };
    const VkSemaphoreSubmitInfo signal{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                        .semaphore = sem,
                                        .value = signalValue,
                                        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT };
    const VkSubmitInfo2 si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                            .waitSemaphoreInfoCount = waitValue ? 1u : 0u,
                            .pWaitSemaphoreInfos = &wait,
                            .commandBufferInfoCount = 1,
                            .pCommandBufferInfos = &cbsi,
                            .signalSemaphoreInfoCount = 1,
                            .pSignalSemaphoreInfos = &signal };
    std::lock_guard<std::mutex> lock(mutex);
    VK_CHECK(vkQueueSubmit2(queue, 1, &si, VK_NULL_HANDLE));
}

void recordAndSubmit(NNEDI3Data* d, StreamCtx& s, VkBuffer readback) {
    const VkDevice device = d->dev->device;
    // With a dedicated transfer family the readback copies go on the DMA
    // queue, overlapping the next frame's compute.
    const bool split = d->dev->transferQueue != d->dev->queue;

    VK_CHECK(vkResetCommandPool(device, s.pool, 0));
    if (split)
        VK_CHECK(vkResetCommandPool(device, s.tpool, 0));
    if (d->profile)
        vkResetQueryPool(device, s.queryPool, 0, 4);

    VkCommandBufferBeginInfo bi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                 .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VK_CHECK(vkBeginCommandBuffer(s.cmd, &bi));

    if (d->profile)
        vkCmdWriteTimestamp2(s.cmd, VK_PIPELINE_STAGE_2_NONE, s.queryPool, 0);

    auto pushPlaneDescriptors = [&](const PlaneSetup& p) {
        const bool hasList = d->pscrn > 0;

        VkDescriptorBufferInfo bufs[BindCount];
        bufs[BindPad] = { s.upload, p.padOffset, p.padBytes };
        bufs[BindDst] = { s.devbuf, p.dstOffset, p.dstBytes };
        bufs[BindPsW] = { d->weightsBuf, d->psWOffset, d->psWBytes };
        bufs[BindPdW] = { d->weightsBuf, d->pdWOffset, d->pdWBytes };
        bufs[BindPdB] = { d->weightsBuf, d->pdBOffset, d->pdBBytes };
        bufs[BindList] = { s.devbuf, hasList ? p.listOffset : 0, hasList ? p.listBytes : VK_WHOLE_SIZE };
        bufs[BindCnt] = { s.devbuf, hasList ? p.cntOffset : 0, hasList ? 16 : VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[BindCount];
        for (uint32_t i = 0; i < BindCount; i++)
            writes[i] = VkWriteDescriptorSet{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                              .dstBinding = i,
                                              .descriptorCount = 1,
                                              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              .pBufferInfo = &bufs[i] };
        vkCmdPushDescriptorSet(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipelineLayout, 0, BindCount, writes);
    };

    auto pushPlaneConstants = [&](const PlaneSetup& p) {
        const PushConstants pcv{ .width = p.width, .rows = p.rows, .padStride = p.padStride, .peak = d->peak };
        vkCmdPushConstants(s.cmd, d->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcv), &pcv);
    };

    const int numPlanes = d->vi.format.numPlanes;

    if (d->pscrn > 0) {
        // Reset the per-plane counters and indirect dispatch arguments.
        for (int plane = 0; plane < numPlanes; plane++) {
            const PlaneSetup& p = d->planes[plane];
            if (!p.process)
                continue;
            vkCmdFillBuffer(s.cmd, s.devbuf, p.cntOffset, 8, 0);      // predCount, groupsX
            vkCmdFillBuffer(s.cmd, s.devbuf, p.cntOffset + 8, 8, 1);  // groupsY, groupsZ
        }

        cmdMemoryBarrier(s.cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->prescreenPipe);
        for (int plane = 0; plane < numPlanes; plane++) {
            const PlaneSetup& p = d->planes[plane];
            if (!p.process)
                continue;
            pushPlaneDescriptors(p);
            pushPlaneConstants(p);
            const int pixPerThread = (d->pscrn == 1) ? 1 : 4;
            const uint32_t threads = static_cast<uint32_t>(p.rows) *
                ((static_cast<uint32_t>(p.width) + pixPerThread - 1) / pixPerThread);
            vkCmdDispatch(s.cmd, (threads + d->prescreenWG - 1) / d->prescreenWG, 1, 1);
        }
    }

    if (d->profile)
        vkCmdWriteTimestamp2(s.cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, s.queryPool, 1);

    if (d->pscrn > 0)
        cmdMemoryBarrier(s.cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

    vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->predictPipe);
    for (int plane = 0; plane < numPlanes; plane++) {
        const PlaneSetup& p = d->planes[plane];
        if (!p.process)
            continue;
        pushPlaneDescriptors(p);
        pushPlaneConstants(p);
        if (d->pscrn > 0) {
            vkCmdDispatchIndirect(s.cmd, s.devbuf, p.cntOffset + 4);
        } else {
            const uint32_t pixels = static_cast<uint32_t>(p.width) * static_cast<uint32_t>(p.rows);
            vkCmdDispatch(s.cmd, (pixels + d->pixelsPerPredictWG - 1) / d->pixelsPerPredictWG, 1, 1);
        }
    }

    if (d->profile)
        vkCmdWriteTimestamp2(s.cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, s.queryPool, 2);

    VkCommandBuffer copyCmd = split ? s.tcmd : s.cmd;
    if (split) {
        VK_CHECK(vkEndCommandBuffer(s.cmd));
        VkCommandBufferBeginInfo tbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        VK_CHECK(vkBeginCommandBuffer(s.tcmd, &tbi));
        // The timeline wait below makes the compute writes visible; no
        // barrier is needed before the copies.
    } else {
        cmdMemoryBarrier(s.cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    }

    for (int plane = 0; plane < numPlanes; plane++) {
        const PlaneSetup& p = d->planes[plane];
        if (!p.process)
            continue;
        VkBufferCopy region{ .srcOffset = p.dstOffset, .dstOffset = p.rbOffset, .size = p.dstBytes };
        vkCmdCopyBuffer(copyCmd, s.devbuf, readback, 1, &region);
    }

    cmdMemoryBarrier(copyCmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);

    if (d->profile)
        vkCmdWriteTimestamp2(copyCmd, VK_PIPELINE_STAGE_2_COPY_BIT, s.queryPool, 3);

    VK_CHECK(vkEndCommandBuffer(copyCmd));

    if (split) {
        const uint64_t computeDone = ++s.timelineValue;
        const uint64_t copyDone = ++s.timelineValue;
        submitTimeline(d->dev->queueMutex, d->dev->queue, s.cmd, s.timeline, 0, 0, computeDone);
        submitTimeline(d->dev->transferMutex, d->dev->transferQueue, s.tcmd, s.timeline, computeDone,
                       VK_PIPELINE_STAGE_2_COPY_BIT, copyDone);
    } else {
        s.timelineValue++;
        submitTimeline(d->dev->queueMutex, d->dev->queue, s.cmd, s.timeline, 0, 0, s.timelineValue);
    }

    VK_CHECK(waitTimeline(device, s.timeline, s.timelineValue));

    if (d->profile) {
        uint64_t ts[4] = {};
        if (vkGetQueryPoolResults(device, s.queryPool, 0, 4, sizeof(ts), ts, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            const double period = d->dev->props.limits.timestampPeriod * 1e-6; // ns -> ms
            std::lock_guard<std::mutex> lock(d->profileMutex);
            d->prescreenMs += (ts[1] - ts[0]) * period;
            d->predictMs += (ts[2] - ts[1]) * period;
            d->copyMs += (ts[3] - ts[2]) * period;
            d->profiledFrames++;
        }
    }
}

//////////////////////////////////////////
// getFrame

const VSFrame* VS_CC nnedi3GetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData, VSFrameContext* frameCtx, VSCore* core,
                                    const VSAPI* vsapi) {
    auto d = static_cast<NNEDI3Data*>(instanceData);

    // Source frame number (field > 1 doubles the output frame rate).
    const int sn = d->field > 1 ? n / 2 : n;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(sn, d->node, frameCtx);
        return nullptr;
    }

    if (activationReason != arAllFramesReady)
        return nullptr;

    const VSFrame* src = vsapi->getFrameFilter(sn, d->node, frameCtx);

    VSFrame* dst;
    if (!d->dh) {
        const VSFrame* planeSrc[3] = {};
        int planeNo[3] = {};
        for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
            planeSrc[plane] = d->planes[plane].process ? nullptr : src;
            planeNo[plane] = plane;
        }
        dst = vsapi->newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, planeSrc, planeNo, src, core);
    } else {
        dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
    }

    // Source field parity, mirroring vsznedi3's get_src_parity exactly.
    // parity == 1 means the source is (treated as) the bottom field.
    const VSMap* srcProps = vsapi->getFramePropertiesRO(src);
    const int defaultParity = (d->field == 0 || d->field == 2) ? 1 : 0;
    int parity;
    int err;
    if (d->dh) {
        const int fieldProp = vsapi->mapGetIntSaturated(srcProps, "_Field", 0, &err);
        parity = err ? defaultParity : fieldProp;
    } else if (d->field > 1) {
        const int fieldBased = vsapi->mapGetIntSaturated(srcProps, "_FieldBased", 0, &err);
        parity = fieldBased == VSC_FIELD_BOTTOM ? 1 : fieldBased == VSC_FIELD_TOP ? 0 : defaultParity;
        if (n % 2)
            parity = !parity;
    } else {
        parity = d->field == 0 ? 1 : 0;
    }
    parity = !!parity;

    const int fp = !parity; // znedi3 PadFilter parity
    const int bps = d->vi.format.bytesPerSample;

    // Pass-through rows straight from the source field; done before a stream
    // is held so it overlaps with other frames' GPU work.
    for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
        const PlaneSetup& p = d->planes[plane];
        if (!p.process)
            continue;

        const uint8_t* srcp = vsapi->getReadPtr(src, plane);
        const ptrdiff_t srcStride = vsapi->getStride(src, plane);
        const uint8_t* fieldp = srcp + (d->dh ? 0 : parity * srcStride);
        const ptrdiff_t fieldStride = srcStride * (d->dh ? 1 : 2);
        uint8_t* dstp = vsapi->getWritePtr(dst, plane);
        const ptrdiff_t dstStride = vsapi->getStride(dst, plane);
        vsh::bitblt(dstp + dstStride * parity, dstStride * 2, fieldp, fieldStride,
                    static_cast<size_t>(p.width) * bps, p.rows);
    }

    // Acquire a stream.
    int streamIdx;
    {
        std::unique_lock<std::mutex> lock(d->streamMutex);
        d->streamCv.wait(lock, [&] { return !d->freeStreams.empty(); });
        streamIdx = d->freeStreams.back();
        d->freeStreams.pop_back();
    }
    StreamCtx& s = *d->streams[streamIdx];

    auto releaseStream = [&] {
        if (streamIdx < 0)
            return;
        {
            std::lock_guard<std::mutex> lock(d->streamMutex);
            d->freeStreams.push_back(streamIdx);
        }
        d->streamCv.notify_one();
        streamIdx = -1;
    };

    int rbIdx = -1;
    bool failed = false;
    std::string errorMsg;

    try {
        for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
            const PlaneSetup& p = d->planes[plane];
            if (!p.process)
                continue;

            const uint8_t* srcp = vsapi->getReadPtr(src, plane);
            const ptrdiff_t srcStride = vsapi->getStride(src, plane);
            const uint8_t* fieldp = srcp + (d->dh ? 0 : parity * srcStride);
            const ptrdiff_t fieldStride = srcStride * (d->dh ? 1 : 2);
            uint8_t* pad = s.uploadMap + p.padOffset;

            switch (bps) {
            case 1:
                copyPadPlane<uint8_t>(fieldp, fieldStride, pad, p.padStride, p.width, p.rows, fp);
                break;
            case 2:
                copyPadPlane<uint16_t>(fieldp, fieldStride, reinterpret_cast<uint16_t*>(pad), p.padStride,
                                       p.width, p.rows, fp);
                break;
            default:
                copyPadPlane<uint32_t>(fieldp, fieldStride, reinterpret_cast<uint32_t*>(pad), p.padStride,
                                       p.width, p.rows, fp);
                break;
            }
        }

        vmaFlushAllocation(d->dev->allocator, s.uploadAlloc, 0, VK_WHOLE_SIZE);

        // Acquire a readback slot (there are more slots than streams, so this
        // cannot deadlock while holding the stream).
        {
            std::unique_lock<std::mutex> lock(d->rbMutex);
            d->rbCv.wait(lock, [&] { return !d->freeRbSlots.empty(); });
            rbIdx = d->freeRbSlots.back();
            d->freeRbSlots.pop_back();
        }
        const RbSlot& slot = d->rbSlots[rbIdx];

        recordAndSubmit(d, s, slot.buf);

        // GPU work is complete; free the stream before the CPU copies the
        // rows out of the readback slot.
        releaseStream();

        vmaInvalidateAllocation(d->dev->allocator, slot.alloc, 0, VK_WHOLE_SIZE);

        for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
            const PlaneSetup& p = d->planes[plane];
            if (!p.process)
                continue;

            const size_t rowBytes = static_cast<size_t>(p.width) * bps;
            uint8_t* dstp = vsapi->getWritePtr(dst, plane);
            const ptrdiff_t dstStride = vsapi->getStride(dst, plane);

            // Interpolated rows from the GPU.
            const uint8_t* rb = slot.map + p.rbOffset;
            for (int r = 0; r < p.rows; r++)
                std::memcpy(dstp + dstStride * (!parity + 2 * r), rb + static_cast<size_t>(r) * rowBytes,
                            rowBytes);
        }
    } catch (const std::exception& e) {
        failed = true;
        errorMsg = e.what();
    }

    releaseStream();
    if (rbIdx >= 0) {
        {
            std::lock_guard<std::mutex> lock(d->rbMutex);
            d->freeRbSlots.push_back(rbIdx);
        }
        d->rbCv.notify_one();
    }

    vsapi->freeFrame(src);

    if (failed) {
        vsapi->setFilterError(("NNEDI3VK: "s + errorMsg).c_str(), frameCtx);
        vsapi->freeFrame(dst);
        return nullptr;
    }

    VSMap* props = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetInt(props, "_FieldBased", VSC_FIELD_PROGRESSIVE, maReplace);
    vsapi->mapDeleteKey(props, "_Field");

    if (d->field > 1) {
        int errNum, errDen;
        int64_t durationNum = vsapi->mapGetInt(props, "_DurationNum", 0, &errNum);
        int64_t durationDen = vsapi->mapGetInt(props, "_DurationDen", 0, &errDen);
        if (!errNum && !errDen) {
            vsh::muldivRational(&durationNum, &durationDen, 1, 2);
            vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
            vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
        }
    }

    return dst;
}

void VS_CC nnedi3Free(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d = static_cast<NNEDI3Data*>(instanceData);
    d->destroy();
    {
        std::lock_guard<std::mutex> lock(g_vkMutex);
        d->dev.reset();
    }
    vsapi->freeNode(d->node);
    delete d;
}

//////////////////////////////////////////
// Creation / Vulkan object setup

// Dump compiled pipeline statistics (registers, spills, occupancy hints) via
// VK_KHR_pipeline_executable_properties.
void dumpPipelineStats(VkDevice device, VkPipeline pipeline, const std::string& label) {
    VkPipelineInfoKHR pi{ .sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipeline };
    uint32_t execCount = 0;
    if (vkGetPipelineExecutablePropertiesKHR(device, &pi, &execCount, nullptr) != VK_SUCCESS || execCount == 0)
        return;
    std::vector<VkPipelineExecutablePropertiesKHR> props(
        execCount, { .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR });
    vkGetPipelineExecutablePropertiesKHR(device, &pi, &execCount, props.data());

    for (uint32_t e = 0; e < execCount; e++) {
        std::string line = "nnedi3vk pipeline stats [" + label + "] '" + props[e].name +
            "' subgroupSize=" + std::to_string(props[e].subgroupSize);

        VkPipelineExecutableInfoKHR ei{ .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
                                        .pipeline = pipeline,
                                        .executableIndex = e };
        uint32_t statCount = 0;
        vkGetPipelineExecutableStatisticsKHR(device, &ei, &statCount, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> stats(
            statCount, { .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR });
        vkGetPipelineExecutableStatisticsKHR(device, &ei, &statCount, stats.data());

        for (const auto& st : stats) {
            line += " | "s + st.name + "=";
            switch (st.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                line += st.value.b32 ? "true" : "false";
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                line += std::to_string(st.value.i64);
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                line += std::to_string(st.value.u64);
                break;
            default:
                line += std::to_string(st.value.f64);
                break;
            }
        }
        std::fprintf(stderr, "%s\n", line.c_str());
    }
}

// Uploads the prepared weight blobs into a device-local buffer via a one-shot
// staging copy.
void uploadWeights(NNEDI3Data* d, const void* data, VkDeviceSize bytes) {
    const VkDevice device = d->dev->device;

    {
        VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = bytes,
                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
        VmaAllocationCreateInfo vaci{ .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, .priority = 1.0f };
        VK_CHECK(vmaCreateBuffer(d->dev->allocator, &bci, &vaci, &d->weightsBuf, &d->weightsAlloc, nullptr));
    }

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;

    try {
        VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = bytes,
                                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
        VmaAllocationCreateInfo vaci{ .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                               VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST };
        VmaAllocationInfo info{};
        VK_CHECK(vmaCreateBuffer(d->dev->allocator, &bci, &vaci, &staging, &stagingAlloc, &info));

        std::memcpy(info.pMappedData, data, bytes);
        vmaFlushAllocation(d->dev->allocator, stagingAlloc, 0, VK_WHOLE_SIZE);

        VkCommandPoolCreateInfo cpi{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                     .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                     .queueFamilyIndex = d->dev->queueFamily };
        VK_CHECK(vkCreateCommandPool(device, &cpi, nullptr, &pool));

        VkCommandBufferAllocateInfo cbai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                          .commandPool = pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                          .commandBufferCount = 1 };
        VkCommandBuffer cmd;
        VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd));

        VkCommandBufferBeginInfo cbbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
        VkBufferCopy region{ .size = bytes };
        vkCmdCopyBuffer(cmd, staging, d->weightsBuf, 1, &region);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkCommandBufferSubmitInfo cbsi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                        .commandBuffer = cmd };
        VkSubmitInfo2 si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                          .commandBufferInfoCount = 1,
                          .pCommandBufferInfos = &cbsi };
        {
            std::lock_guard<std::mutex> lock(d->dev->queueMutex);
            VK_CHECK(vkQueueSubmit2(d->dev->queue, 1, &si, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(d->dev->queue));
        }
    } catch (...) {
        if (pool)
            vkDestroyCommandPool(device, pool, nullptr);
        if (staging)
            vmaDestroyBuffer(d->dev->allocator, staging, stagingAlloc);
        throw;
    }

    vkDestroyCommandPool(device, pool, nullptr);
    vmaDestroyBuffer(d->dev->allocator, staging, stagingAlloc);
}

void setupVulkanObjects(NNEDI3Data* d, int numStreams, int32_t xdim, int32_t ydim, int32_t nns) {
    const VkDevice device = d->dev->device;
    const int pixelType = d->pixelType;

    if (d->pscrn > 0) {
        VkShaderModuleCreateInfo smci{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                       .codeSize = kSpv[0][pixelType].size,
                                       .pCode = kSpv[0][pixelType].code };
        VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &d->prescreenModule));
    }
    {
        const SpvBlob& blob = kSpv[d->useCoopVec ? 2 : 1][pixelType];
        VkShaderModuleCreateInfo smci{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                       .codeSize = blob.size,
                                       .pCode = blob.code };
        VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &d->predictModule));
    }

    VkDescriptorSetLayoutBinding bindings[BindCount];
    for (uint32_t i = 0; i < BindCount; i++)
        bindings[i] = VkDescriptorSetLayoutBinding{ .binding = i,
                                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                    .descriptorCount = 1,
                                                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dslci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
                                           .bindingCount = BindCount,
                                           .pBindings = bindings };
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &d->dsl));

    VkPushConstantRange pcr{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(PushConstants) };
    VkPipelineLayoutCreateInfo plci{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .setLayoutCount = 1,
                                     .pSetLayouts = &d->dsl,
                                     .pushConstantRangeCount = 1,
                                     .pPushConstantRanges = &pcr };
    VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &d->pipelineLayout));

    static constexpr VkSpecializationMapEntry specEntries[] = {
        { 0, offsetof(SpecData, wgSize), sizeof(uint32_t) },
        { 1, offsetof(SpecData, pscrn), sizeof(int32_t) },
        { 2, offsetof(SpecData, xdim), sizeof(int32_t) },
        { 3, offsetof(SpecData, ydim), sizeof(int32_t) },
        { 4, offsetof(SpecData, nns), sizeof(int32_t) },
        { 5, offsetof(SpecData, qual), sizeof(int32_t) },
        { 6, offsetof(SpecData, sgSize), sizeof(int32_t) },
        { 7, offsetof(SpecData, subgroups), sizeof(int32_t) },
        { 8, offsetof(SpecData, useList), sizeof(VkBool32) },
        { 9, offsetof(SpecData, cvChunkBytes), sizeof(uint32_t) },
        { 10, offsetof(SpecData, cvMcu), sizeof(int32_t) },
        { 11, offsetof(SpecData, cvNmc), sizeof(int32_t) },
    };
    constexpr uint32_t specEntryCount = static_cast<uint32_t>(std::size(specEntries));

    const uint32_t sgSize = d->dev->subgroupSize;
    const uint32_t maxWG = std::min(d->dev->props.limits.maxComputeWorkGroupInvocations,
                                    d->dev->props.limits.maxComputeWorkGroupSize[0]);
    d->prescreenWG = std::min(128u, maxWG);

    // Fallback predict: PX pixels per subgroup (blocked GEMM). Cooperative
    // vector predict: one invocation per pixel.
    const uint32_t gemvPPL = (static_cast<uint32_t>(nns) + sgSize - 1) / sgSize;
    const uint32_t gemvFS = static_cast<uint32_t>(xdim * ydim);
    // Must match PX in predict.comp.
    const uint32_t gemvPX = (d->pixelType != 2 && gemvPPL <= 2 && gemvFS <= 128) ? 8 : 4;
    const uint32_t subgroupsPerWG = std::max(1u, std::min(4u, maxWG / sgSize));
    const uint32_t predictWG = d->useCoopVec ? std::min(64u, maxWG) : sgSize * subgroupsPerWG;
    d->pixelsPerPredictWG = d->useCoopVec ? predictWG : subgroupsPerWG * gemvPX;

    // Note: constant_id 7 differs per kernel — the prescreen kernel uses it
    // as "pixels per predict workgroup" (indirect dispatch sizing), the GEMV
    // predict kernel as its actual subgroup count.
    const SpecData spec{ .wgSize = 0, // filled in per pipeline
                         .pscrn = d->pscrn,
                         .xdim = xdim,
                         .ydim = ydim,
                         .nns = nns,
                         .qual = d->qual,
                         .sgSize = static_cast<int32_t>(sgSize),
                         .subgroups = 0, // filled in per pipeline
                         .useList = d->pscrn > 0 ? VK_TRUE : VK_FALSE,
                         .cvChunkBytes = d->cvChunkBytes,
                         .cvMcu = d->cvMCU,
                         .cvNmc = d->cvNMC };

    const bool captureStats = d->profile && d->dev->hasExecutableProperties;
    const VkPipelineCreateFlags createFlags =
        captureStats ? VkPipelineCreateFlags(VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR) : 0;

    if (d->pscrn > 0) {
        SpecData pspec = spec;
        pspec.wgSize = d->prescreenWG;
        pspec.subgroups = static_cast<int32_t>(d->pixelsPerPredictWG);
        VkSpecializationInfo specInfo{ .mapEntryCount = specEntryCount,
                                       .pMapEntries = specEntries,
                                       .dataSize = sizeof(pspec),
                                       .pData = &pspec };
        VkComputePipelineCreateInfo cpci{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .flags = createFlags,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                       .module = d->prescreenModule,
                       .pName = "main",
                       .pSpecializationInfo = &specInfo },
            .layout = d->pipelineLayout,
        };
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &d->prescreenPipe));
        if (captureStats)
            dumpPipelineStats(device, d->prescreenPipe, "prescreen");
    }

    {
        SpecData pspec = spec;
        pspec.wgSize = predictWG;
        pspec.subgroups = static_cast<int32_t>(subgroupsPerWG);
        VkSpecializationInfo specInfo{ .mapEntryCount = specEntryCount,
                                       .pMapEntries = specEntries,
                                       .dataSize = sizeof(pspec),
                                       .pData = &pspec };

        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSg{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
            .requiredSubgroupSize = sgSize };

        // The fallback kernel relies on subgroup-per-pixel mapping, so it pins
        // the subgroup size and requires full subgroups; the cooperative
        // vector kernel is per-invocation and needs neither.
        const bool pinSubgroups = !d->useCoopVec;

        VkComputePipelineCreateInfo cpci{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .flags = createFlags,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       .pNext = (pinSubgroups && d->dev->canRequireSubgroupSize) ? &reqSg : nullptr,
                       .flags = (pinSubgroups && d->dev->fullSubgroups)
                                    ? VkPipelineShaderStageCreateFlags(VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT)
                                    : 0,
                       .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                       .module = d->predictModule,
                       .pName = "main",
                       .pSpecializationInfo = &specInfo },
            .layout = d->pipelineLayout,
        };
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &d->predictPipe));
        if (captureStats)
            dumpPipelineStats(device, d->predictPipe, d->useCoopVec ? "predict_cv" : "predict");
    }

    // Per-stream resources.
    const bool crossQueue = d->dev->transferQueue != d->dev->queue;
    const uint32_t bothFamilies[2] = { d->dev->queueFamily, d->dev->transferFamily };

    for (int i = 0; i < numStreams; i++) {
        auto s = std::make_unique<StreamCtx>();

        VkCommandPoolCreateInfo cpi{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                     .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                     .queueFamilyIndex = d->dev->queueFamily };
        VK_CHECK(vkCreateCommandPool(device, &cpi, nullptr, &s->pool));

        VkCommandBufferAllocateInfo cbai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                          .commandPool = s->pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                          .commandBufferCount = 1 };
        VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &s->cmd));

        if (crossQueue) {
            VkCommandPoolCreateInfo tcpi{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                          .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                          .queueFamilyIndex = d->dev->transferFamily };
            VK_CHECK(vkCreateCommandPool(device, &tcpi, nullptr, &s->tpool));

            VkCommandBufferAllocateInfo tcbai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                               .commandPool = s->tpool,
                                               .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                               .commandBufferCount = 1 };
            VK_CHECK(vkAllocateCommandBuffers(device, &tcbai, &s->tcmd));
        }

        VkSemaphoreTypeCreateInfo stci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                        .initialValue = 0 };
        VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &stci };
        VK_CHECK(vkCreateSemaphore(device, &sci, nullptr, &s->timeline));

        if (d->profile) {
            VkQueryPoolCreateInfo qpci{ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                        .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                        .queryCount = 4 };
            VK_CHECK(vkCreateQueryPool(device, &qpci, nullptr, &s->queryPool));
            vkResetQueryPool(device, s->queryPool, 0, 4);
        }

        // Upload buffer: prefer DEVICE_LOCAL | HOST_VISIBLE (ReBAR).
        {
            VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                    .size = d->uploadSize,
                                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
            VmaAllocationCreateInfo vaci{ .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                   VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                          .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                          .priority = 0.9f };
            VmaAllocationInfo info{};
            VK_CHECK(vmaCreateBuffer(d->dev->allocator, &bci, &vaci, &s->upload, &s->uploadAlloc, &info));
            s->uploadMap = static_cast<uint8_t*>(info.pMappedData);
        }

        // Device-local scratch + output buffer (read by the transfer queue,
        // hence concurrent sharing when the families differ).
        {
            VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                    .size = d->devbufSize,
                                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                    .sharingMode = crossQueue ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
                                    .queueFamilyIndexCount = crossQueue ? 2u : 0u,
                                    .pQueueFamilyIndices = crossQueue ? bothFamilies : nullptr };
            VmaAllocationCreateInfo vaci{ .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, .priority = 1.0f };
            VK_CHECK(vmaCreateBuffer(d->dev->allocator, &bci, &vaci, &s->devbuf, &s->devAlloc, nullptr));
        }

        d->freeStreams.push_back(i);
        d->streams.push_back(std::move(s));
    }

    // Readback slots: host-cached, a couple more than there are streams so a
    // stream never stalls behind a frame still copying its rows out.
    const int numRbSlots = numStreams + 2;
    for (int i = 0; i < numRbSlots; i++) {
        RbSlot slot;
        VkBufferCreateInfo bci{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = d->rbSize,
                                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                .sharingMode = crossQueue ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
                                .queueFamilyIndexCount = crossQueue ? 2u : 0u,
                                .pQueueFamilyIndices = crossQueue ? bothFamilies : nullptr };
        VmaAllocationCreateInfo vaci{ .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                               VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .priority = 0.5f };
        VmaAllocationInfo info{};
        VK_CHECK(vmaCreateBuffer(d->dev->allocator, &bci, &vaci, &slot.buf, &slot.alloc, &info));
        slot.map = static_cast<uint8_t*>(info.pMappedData);

        d->freeRbSlots.push_back(i);
        d->rbSlots.push_back(slot);
    }
}

int getIntDef(const VSAPI* vsapi, const VSMap* in, const char* name, int def) {
    int err;
    int v = vsapi->mapGetIntSaturated(in, name, 0, &err);
    return err ? def : v;
}

void VS_CC nnedi3Create(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d = std::make_unique<NNEDI3Data>();
    int err;

    try {
        if (!!vsapi->mapGetInt(in, "list_device", 0, &err)) {
            std::lock_guard<std::mutex> lock(g_vkMutex);
            auto globals = acquireGlobals();
            std::string msg = "NNEDI3VK: available devices:";
            for (size_t i = 0; i < globals->physicalDevices.size(); i++) {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(globals->physicalDevices[i], &props);
                msg += "\n" + std::to_string(i) + ": " + props.deviceName;
            }
            vsapi->mapSetError(out, msg.c_str());
            return;
        }

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = *vsapi->getVideoInfo(d->node);

        if (!vsh::isConstantVideoFormat(&d->vi) ||
            (d->vi.format.sampleType == stInteger && d->vi.format.bitsPerSample > 16) ||
            (d->vi.format.sampleType == stFloat && d->vi.format.bitsPerSample != 16 && d->vi.format.bitsPerSample != 32))
            throw std::runtime_error("only constant format 8-16 bit integer and 16/32 bit float input supported");

        d->field = vsapi->mapGetIntSaturated(in, "field", 0, nullptr);

        d->dh = !!vsapi->mapGetInt(in, "dh", 0, &err);

        const int numPlanes = vsapi->mapNumElements(in, "planes");

        bool process[3] = {};
        for (int i = 0; i < 3; i++)
            process[i] = (numPlanes <= 0);

        for (int i = 0; i < numPlanes; i++) {
            const int plane = vsapi->mapGetIntSaturated(in, "planes", i, nullptr);

            if (plane < 0 || plane >= d->vi.format.numPlanes)
                throw std::runtime_error("plane index out of range");

            if (process[plane])
                throw std::runtime_error("plane specified twice");

            process[plane] = true;
        }

        const int nsize = getIntDef(vsapi, in, "nsize", 6);
        const int nns = getIntDef(vsapi, in, "nns", 1);
        d->qual = getIntDef(vsapi, in, "qual", 1);
        const int etype = getIntDef(vsapi, in, "etype", 0);
        d->pscrn = getIntDef(vsapi, in, "pscrn", 2);
        const int deviceIndex = getIntDef(vsapi, in, "device_index", 0);
        const int numStreams = getIntDef(vsapi, in, "num_streams", 2);

        if (d->field < 0 || d->field > 3)
            throw std::runtime_error("field must be 0, 1, 2, or 3");

        if (!d->dh)
            for (int plane = 0; plane < d->vi.format.numPlanes; plane++)
                if (process[plane] && ((d->vi.height >> (plane > 0 ? d->vi.format.subSamplingH : 0)) & 1))
                    throw std::runtime_error("plane's height must be mod 2 when dh=False");

        if (d->dh && d->field > 1)
            throw std::runtime_error("field must be 0 or 1 when dh=True");

        if (nsize < 0 || nsize > 6)
            throw std::runtime_error("nsize must be between 0 and 6 (inclusive)");

        if (nns < 0 || nns > 4)
            throw std::runtime_error("nns must be between 0 and 4 (inclusive)");

        if (d->qual < 1 || d->qual > 2)
            throw std::runtime_error("qual must be 1 or 2");

        if (etype < 0 || etype > 1)
            throw std::runtime_error("etype must be 0 or 1");

        if (d->pscrn < 0 || d->pscrn > 4)
            throw std::runtime_error("pscrn must be between 0 and 4 (inclusive)");

        if (numStreams < 1)
            throw std::runtime_error("num_streams must be greater than or equal to 1");

        if (d->field > 1) {
            if (d->vi.numFrames > INT_MAX / 2)
                throw std::runtime_error("resulting clip is too long");
            d->vi.numFrames *= 2;

            vsh::muldivRational(&d->vi.fpsNum, &d->vi.fpsDen, 2, 1);
        }

        if (d->dh)
            d->vi.height *= 2;

        // Load the nnedi3 weights that ship next to the plugin binary.
        PrescreenerOldCoefficients psOld;
        PrescreenerNewCoefficients psNew[3];
        PredictorModel model;
        {
            const char* pluginPath = vsapi->getPluginPath(vsapi->getPluginByID("com.holywu.nnedi3vk", core));
            if (!pluginPath)
                throw std::runtime_error("cannot determine plugin path");
            std::filesystem::path weightsPath(std::u8string(reinterpret_cast<const char8_t*>(pluginPath)));
            weightsPath = weightsPath.parent_path() / "nnedi3_weights.bin";

            std::ifstream file(weightsPath, std::ios::binary);
            std::vector<float> fileData(NNEDI3_WEIGHTS_SIZE / sizeof(float));
            if (!file.read(reinterpret_cast<char*>(fileData.data()), NNEDI3_WEIGHTS_SIZE) ||
                file.gcount() != static_cast<std::streamsize>(NNEDI3_WEIGHTS_SIZE))
                throw std::runtime_error("error reading weights from " + weightsPath.string());

            readNNEDI3Weights(fileData.data(), nsize, nns, etype, psOld, psNew, model);
        }

        const bool isFloat = d->vi.format.sampleType == stFloat;
        const double pixelHalf = isFloat ? 0.5 : static_cast<double>((1 << d->vi.format.bitsPerSample) - 1) / 2.0;

        if (d->pscrn == 1)
            subtractMean(psOld, pixelHalf);
        else if (d->pscrn >= 2)
            subtractMean(psNew[d->pscrn - 2], pixelHalf);
        subtractMean(model);

        if (d->vi.format.bytesPerSample == 1)
            d->pixelType = 0;
        else if (d->vi.format.bytesPerSample == 2 && !isFloat)
            d->pixelType = 1;
        else if (d->vi.format.bytesPerSample == 2)
            d->pixelType = 2;
        else
            d->pixelType = 3;

        d->peak = (1 << d->vi.format.bitsPerSample) - 1;

        d->profile = envFlag("NNEDI3VK_PROFILE");

        {
            std::lock_guard<std::mutex> lock(g_vkMutex);
            d->dev = acquireDevice(static_cast<uint32_t>(deviceIndex));
        }

        if (d->pixelType == 2 && !d->dev->hasFloat16)
            throw std::runtime_error("FP16 input requires shaderFloat16 device support");

        const bool cvSupported = d->dev->hasCoopVecF16 && d->dev->hasFloat16 && model.xdim * model.ydim <= d->dev->maxCoopVecComponents;
        const bool cvRequested = !!vsapi->mapGetInt(in, "coopvec", 0, &err);
        if (err) {
            d->useCoopVec = cvSupported;
        } else if (cvRequested) {
            if (!cvSupported)
                throw std::runtime_error("coopvec=True requires VK_NV_cooperative_vector device support");
            d->useCoopVec = true;
        } else {
            d->useCoopVec = false;
        }

        // Plane layout / buffer sizes.
        const int bps = d->vi.format.bytesPerSample;
        VkDeviceSize uploadOff = 0, devOff = 0, rbOff = 0;
        for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
            PlaneSetup& p = d->planes[plane];
            p.process = process[plane];
            p.width = d->vi.width >> (plane > 0 ? d->vi.format.subSamplingW : 0);
            p.height = d->vi.height >> (plane > 0 ? d->vi.format.subSamplingH : 0);
            if (!p.process)
                continue;

            p.rows = p.height / 2;
            p.padStride = (p.width + MARGIN_H * 2 + 15) & ~15;
            p.padHeight = p.rows + MARGIN_V * 2;

            p.padBytes = static_cast<VkDeviceSize>(p.padStride) * p.padHeight * bps;
            p.padOffset = suballoc(uploadOff, p.padBytes);

            p.dstBytes = static_cast<VkDeviceSize>(p.width) * p.rows * bps;
            p.dstOffset = suballoc(devOff, p.dstBytes);

            if (d->pscrn > 0) {
                p.listBytes = static_cast<VkDeviceSize>(p.width) * p.rows * sizeof(uint32_t);
                p.listOffset = suballoc(devOff, p.listBytes);
                p.cntOffset = suballoc(devOff, 16);
            }

            p.rbOffset = suballoc(rbOff, p.dstBytes);
        }
        d->uploadSize = std::max(uploadOff, BUF_ALIGN);
        d->devbufSize = std::max(devOff, BUF_ALIGN);
        d->rbSize = std::max(rbOff, BUF_ALIGN);

        // Build the weight blobs.
        //
        // Prescreener: the flat float layout described in prescreen.comp,
        // with the layer-0 kernel transposed to [k][4] so the kernel fetches
        // all four neurons' weights for a window element in one vec4 load.
        std::vector<float> psBlob;
        if (d->pscrn == 1) {
            psBlob.resize(sizeof(psOld) / sizeof(float));
            const PrescreenerOldCoefficients& ps = psOld;
            float* p = psBlob.data();
            for (unsigned k = 0; k < 48; k++)
                for (unsigned n = 0; n < 4; n++)
                    *p++ = ps.kernel_l0[n][k];
            std::memcpy(p, ps.bias_l0, sizeof(psOld) - sizeof(ps.kernel_l0));
        } else if (d->pscrn >= 2) {
            psBlob.resize(sizeof(psNew[0]) / sizeof(float));
            const PrescreenerNewCoefficients& ps = psNew[d->pscrn - 2];
            float* p = psBlob.data();
            for (unsigned k = 0; k < 64; k++)
                for (unsigned n = 0; n < 4; n++)
                    *p++ = ps.kernel_l0[n][k];
            std::memcpy(p, ps.bias_l0, sizeof(psNew[0]) - sizeof(ps.kernel_l0));
        } else {
            psBlob.resize(1, 0.0f);
        }

        // Predictor: transposed (softmax, elliott) pairs plus interleaved
        // bias/rowsum pairs; see predict.comp for the exact layouts.
        const unsigned fs = model.xdim * model.ydim;
        const unsigned N = model.nns;
        const unsigned numQ = static_cast<unsigned>(d->qual);

        std::vector<float> pdBias(numQ * 4 * N);
        for (unsigned q = 0; q < numQ; q++) {
            const float* sm = q ? model.softmax_q2.data() : model.softmax_q1.data();
            const float* el = q ? model.elliott_q2.data() : model.elliott_q1.data();
            const float* smB = q ? model.softmax_bias_q2.data() : model.softmax_bias_q1.data();
            const float* elB = q ? model.elliott_bias_q2.data() : model.elliott_bias_q1.data();

            for (unsigned p = 0; p < N; p++) {
                pdBias[(q * 2 * N + p) * 2 + 0] = smB[p];
                pdBias[(q * 2 * N + p) * 2 + 1] = elB[p];

                double smSum = 0.0, elSum = 0.0;
                for (unsigned k = 0; k < fs; k++) {
                    smSum += sm[p * fs + k];
                    elSum += el[p * fs + k];
                }
                pdBias[(q * 2 * N + N + p) * 2 + 0] = static_cast<float>(smSum);
                pdBias[(q * 2 * N + N + p) * 2 + 1] = static_cast<float>(elSum);
            }
        }

        std::vector<float> pdW32;
        std::vector<uint16_t> pdW16;
        std::vector<uint8_t> pdWCv;
        if (d->useCoopVec) {
            // Row-major FP16 matrix with interleaved (softmax_i, elliott_i)
            // rows, then M-chunks of at most 128 rows converted to the
            // device's inferencing-optimal layout.
            const unsigned M = 2 * N;
            std::vector<uint16_t> rowMajor(static_cast<size_t>(numQ) * M * fs);
            for (unsigned q = 0; q < numQ; q++) {
                const float* sm = q ? model.softmax_q2.data() : model.softmax_q1.data();
                const float* el = q ? model.elliott_q2.data() : model.elliott_q1.data();
                for (unsigned i = 0; i < N; i++)
                    for (unsigned k = 0; k < fs; k++) {
                        rowMajor[(static_cast<size_t>(q) * M + 2 * i + 0) * fs + k] = floatToHalf(sm[i * fs + k]);
                        rowMajor[(static_cast<size_t>(q) * M + 2 * i + 1) * fs + k] = floatToHalf(el[i * fs + k]);
                    }
            }

            d->cvMCU = static_cast<int32_t>(M <= 128 ? M : 128); // M is always a multiple of 128 above it
            d->cvNMC = static_cast<int32_t>(M) / d->cvMCU;

            VkConvertCooperativeVectorMatrixInfoNV convInfo{
                .sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV,
                .srcSize = static_cast<size_t>(d->cvMCU) * fs * sizeof(uint16_t),
                .srcData = {.hostAddress = nullptr },
                .pDstSize = nullptr,
                .dstData = {.hostAddress = nullptr },
                .srcComponentType = VK_COMPONENT_TYPE_FLOAT16_KHR,
                .dstComponentType = VK_COMPONENT_TYPE_FLOAT16_KHR,
                .numRows = static_cast<uint32_t>(d->cvMCU),
                .numColumns = fs,
                .srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV,
                .srcStride = fs * sizeof(uint16_t),
                .dstLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV,
                .dstStride = 0 };

            size_t chunkSize = 0;
            convInfo.pDstSize = &chunkSize;
            VK_CHECK(vkConvertCooperativeVectorMatrixNV(d->dev->device, &convInfo));

            d->cvChunkBytes = static_cast<uint32_t>(alignUp(chunkSize, 64));
            pdWCv.assign(static_cast<size_t>(numQ) * d->cvNMC * d->cvChunkBytes, 0);

            for (unsigned q = 0; q < numQ; q++)
                for (int32_t mi = 0; mi < d->cvNMC; mi++) {
                    size_t dstSize = chunkSize;
                    convInfo.pDstSize = &dstSize;
                    convInfo.srcData.hostAddress =
                        rowMajor.data() + (static_cast<size_t>(q) * M + mi * d->cvMCU) * fs;
                    convInfo.dstData.hostAddress =
                        pdWCv.data() + (static_cast<size_t>(q) * d->cvNMC + mi) * d->cvChunkBytes;
                    VK_CHECK(vkConvertCooperativeVectorMatrixNV(d->dev->device, &convInfo));
                }
        } else if (d->pixelType == 2) {
            pdW16.resize(static_cast<size_t>(numQ) * (fs / 2) * N * 4);
            for (unsigned q = 0; q < numQ; q++) {
                const float* sm = q ? model.softmax_q2.data() : model.softmax_q1.data();
                const float* el = q ? model.elliott_q2.data() : model.elliott_q1.data();
                for (unsigned k2 = 0; k2 < fs / 2; k2++)
                    for (unsigned p = 0; p < N; p++) {
                        const size_t base = ((static_cast<size_t>(q) * (fs / 2) + k2) * N + p) * 4;
                        pdW16[base + 0] = floatToHalf(sm[p * fs + k2 * 2 + 0]);
                        pdW16[base + 1] = floatToHalf(sm[p * fs + k2 * 2 + 1]);
                        pdW16[base + 2] = floatToHalf(el[p * fs + k2 * 2 + 0]);
                        pdW16[base + 3] = floatToHalf(el[p * fs + k2 * 2 + 1]);
                    }
            }
        } else {
            pdW32.resize(static_cast<size_t>(numQ) * fs * N * 2);
            for (unsigned q = 0; q < numQ; q++) {
                const float* sm = q ? model.softmax_q2.data() : model.softmax_q1.data();
                const float* el = q ? model.elliott_q2.data() : model.elliott_q1.data();
                for (unsigned k = 0; k < fs; k++)
                    for (unsigned p = 0; p < N; p++) {
                        const size_t base = ((static_cast<size_t>(q) * fs + k) * N + p) * 2;
                        pdW32[base + 0] = sm[p * fs + k];
                        pdW32[base + 1] = el[p * fs + k];
                    }
            }
        }

        // Assemble the combined weights buffer.
        const void* pdWData;
        if (d->useCoopVec) {
            d->pdWBytes = pdWCv.size();
            pdWData = pdWCv.data();
        } else if (d->pixelType == 2) {
            d->pdWBytes = pdW16.size() * sizeof(uint16_t);
            pdWData = pdW16.data();
        } else {
            d->pdWBytes = pdW32.size() * sizeof(float);
            pdWData = pdW32.data();
        }
        d->psWBytes = psBlob.size() * sizeof(float);
        d->pdBBytes = pdBias.size() * sizeof(float);

        VkDeviceSize wOff = 0;
        d->psWOffset = suballoc(wOff, d->psWBytes);
        d->pdWOffset = suballoc(wOff, d->pdWBytes);
        d->pdBOffset = suballoc(wOff, d->pdBBytes);

        std::vector<uint8_t> weightsBlob(wOff);
        std::memcpy(weightsBlob.data() + d->psWOffset, psBlob.data(), d->psWBytes);
        std::memcpy(weightsBlob.data() + d->pdWOffset, pdWData, d->pdWBytes);
        std::memcpy(weightsBlob.data() + d->pdBOffset, pdBias.data(), d->pdBBytes);

        uploadWeights(d.get(), weightsBlob.data(), weightsBlob.size());

        setupVulkanObjects(d.get(), numStreams, static_cast<int32_t>(model.xdim),
                           static_cast<int32_t>(model.ydim), static_cast<int32_t>(model.nns));
    } catch (const std::exception& e) {
        vsapi->mapSetError(out, ("NNEDI3VK: "s + e.what()).c_str());
        nnedi3Free(d.release(), core, vsapi);
        return;
    }

    VSFilterDependency deps[] = { { d->node, d->field > 1 ? rpGeneral : rpStrictSpatial } };
    vsapi->createVideoFilter(out, "NNEDI3", &d->vi, nnedi3GetFrame, nnedi3Free, fmParallel, deps, 1, d.get(), core);
    d.release();
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.nnedi3vk",
                         "nnedi3vk",
                         "Neural Network Edge Directed Interpolation 3, using Vulkan compute",
                         VS_MAKE_VERSION(1, 0),
                         VAPOURSYNTH_API_VERSION,
                         0,
                         plugin);

    vspapi->registerFunction("NNEDI3",
                             "clip:vnode;"
                             "field:int;"
                             "dh:int:opt;"
                             "planes:int[]:opt;"
                             "nsize:int:opt;"
                             "nns:int:opt;"
                             "qual:int:opt;"
                             "etype:int:opt;"
                             "pscrn:int:opt;"
                             "coopvec:int:opt;"
                             "device_index:int:opt;"
                             "list_device:int:opt;"
                             "num_streams:int:opt;",
                             "clip:vnode;",
                             nnedi3Create,
                             nullptr,
                             plugin);
}
