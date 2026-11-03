// Layer 2: model parts on a live context.
//   set_component: load, replace or remove the VAE or the TAESD without a model reload. It uses the
//                  same upstream path as ControlNet (StableDiffusionGGML::apply_model_update).
//   parts:         weight memory of each part, from the ModelManager tensor states.
// See the Layer 2 register.

#include "l2.h"

#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>

#include "core/ggml_extend_backend.h"
#include "ggml-backend.h"
#include "model_manager.h"
#include "pipeline/diffusion_engine.h"

// Defined in src/stable-diffusion.cpp (hook marked BRANCUSI).
SD_API StableDiffusionGGML* bsd_l2_context(sd_ctx_t* sd_ctx);

// Friend of ModelManager (hook marked BRANCUSI in src/model_manager.h).
struct BsdPartProbe {
    struct Bytes {
        size_t declared = 0;  // Size of all registered weights.
        size_t resident = 0;  // Weights loaded in the params backend.
        size_t host     = 0;  // Part of the resident bytes that are in host memory.
        size_t staged   = 0;  // Extra copy in compute backend memory.
    };

    static std::map<ModelComponent, Bytes> collect(const ModelManager& manager) {
        std::map<ModelComponent, Bytes> out;
        std::unordered_set<const ggml_tensor*> seen;
        for (const auto& state : manager.tensor_states_) {
            if (state->tensor == nullptr || !seen.insert(state->tensor).second) {
                continue;
            }
            Bytes& b       = out[state->component];
            const size_t n = ggml_nbytes(state->tensor);
            b.declared += n;
            if (state->loaded_to_params_backend && state->tensor->buffer != nullptr) {
                b.resident += n;
                if (ggml_backend_buffer_is_host(state->tensor->buffer)) {
                    b.host += n;
                }
            }
            if (state->staged_to_compute_backend && !state->params_follow_compute_backend &&
                state->compute_backend != nullptr && !sd_backend_is_cpu(state->compute_backend)) {
                b.staged += n;
            }
        }
        return out;
    }
};

namespace {

using Sd = StableDiffusionGGML;

const char* part_prefix(bsd_component_t which) {
    return which == BSD_COMPONENT_TAESD ? "tae." : "vae.";
}

// Ids of the model files that were added with this path and this prefix.
std::set<ModelLoader::FileId> find_files(ModelLoader& loader, const std::string& path, const std::string& prefix) {
    std::set<ModelLoader::FileId> ids;
    for (const auto& entry : loader.get_tensor_storage_map()) {
        ids.insert(entry.second.file_id);
    }
    std::set<ModelLoader::FileId> found;
    for (ModelLoader::FileId id : ids) {
        if (loader.file_path(id) != path) {
            continue;
        }
        const auto tensors = loader.file_tensors(id);
        if (!tensors.empty() && tensors.begin()->first.rfind(prefix, 0) == 0) {
            found.insert(id);
        }
    }
    return found;
}

struct PartName {
    ModelComponent component;
    SDBackendModule module;
    const char* name;
};

// The names follow the bsd_component_t names where there is one.
const PartName kPartNames[] = {
    {ModelComponent::Conditioner, SDBackendModule::TE, "te"},
    {ModelComponent::Diffusion, SDBackendModule::DIFFUSION, "diffusion"},
    {ModelComponent::HighNoiseDiffusion, SDBackendModule::DIFFUSION, "high_noise_diffusion"},
    {ModelComponent::CLIPVision, SDBackendModule::CLIP_VISION, "clip_vision"},
    {ModelComponent::IPAdapter, SDBackendModule::DIFFUSION, "ip_adapter"},
    {ModelComponent::VAE, SDBackendModule::VAE, "vae"},
    {ModelComponent::PreviewVAE, SDBackendModule::VAE, "taesd"},
    {ModelComponent::AudioVAE, SDBackendModule::VAE, "audio_vae"},
    {ModelComponent::AudioEncoder, SDBackendModule::AUDIO_ENCODER, "audio_encoder"},
    {ModelComponent::ControlNet, SDBackendModule::CONTROL_NET, "control_net"},
    {ModelComponent::PhotoMaker, SDBackendModule::PHOTOMAKER, "photo_maker"},
    {ModelComponent::PuLID, SDBackendModule::PHOTOMAKER, "pulid"},
};

}  // namespace

namespace bsd_l2 {

bsd_status_t set_component(sd_ctx_t* ctx, bsd_component_t which, const std::string& path, bool& context_lost,
                           std::string& error) {
    context_lost = false;
    Sd* sd       = bsd_l2_context(ctx);
    if (sd == nullptr || sd->model_manager == nullptr || sd->config_ == nullptr) {
        error = "The engine has no model.";
        return BSD_ERR_NOT_LOADED;
    }
    Sd::ContextOperation operation(*sd);
    if (!operation.acquired) {
        error = "A job is running. Cancel it before you change a part.";
        return BSD_ERR_BUSY;
    }
    const bool is_tae        = which == BSD_COMPONENT_TAESD;
    const char* part         = is_tae ? "TAESD" : "VAE";
    const std::string prefix = part_prefix(which);
    const char* old_cstr     = is_tae ? sd->config_->params.taesd_path : sd->config_->params.vae_path;
    const std::string old_path = old_cstr != nullptr ? old_cstr : "";

    // Work on copies. The context changes only when upstream accepts the new loader.
    const ModelLoader old_loader = sd->model_manager->loader();
    ModelLoader candidate        = old_loader;
    if (!old_path.empty()) {
        for (ModelLoader::FileId id : find_files(candidate, old_path, prefix)) {
            candidate.del_file(id);
        }
    }
    if (!path.empty() && !candidate.add_file(path, prefix)) {
        error = std::string("Cannot read the ") + part + " file: " + path;
        return BSD_ERR_BAD_MODEL;
    }
    if (candidate.get_sd_version() != sd->version) {
        error = std::string("The ") + part + " changes the model family. The old part stays.";
        return BSD_ERR_BAD_MODEL;
    }

    auto next = std::make_unique<Sd::ModelConfig>(*sd->config_);
    next->strings.push_back(path);
    (is_tae ? next->params.taesd_path : next->params.vae_path) = next->strings.back().c_str();
    next->record_model(is_tae ? "taesd" : "vae", path);
    if (is_tae) {
        next->use_tae = !path.empty();
    }
    auto old_config                = std::make_unique<Sd::ModelConfig>(*sd->config_);
    const bool old_external_invalid = sd->external_vae_is_invalid;
    if (!is_tae) {
        sd->external_vae_is_invalid = false;
    }

    if (sd->apply_model_update(std::move(candidate), std::move(next), {Sd::RunnerGroup::VAE})) {
        return BSD_OK;
    }

    // Upstream drops all runners when an update fails. Build the old state again.
    sd->external_vae_is_invalid = old_external_invalid;
    if (sd->apply_model_update(old_loader, std::move(old_config), {})) {
        error = path.empty() ? std::string("The model has no built-in ") + part + ". The old part stays."
                             : std::string("The new ") + part + " does not fit the model. The old part stays.";
        return BSD_ERR_BAD_MODEL;
    }
    context_lost = true;
    error        = std::string("The engine could not rebuild the model after a failed ") + part + " change. Load the model again.";
    return BSD_ERR_INTERNAL;
}

bsd_status_t parts(sd_ctx_t* ctx, bsd_parts_t* out, std::string& error) {
    Sd* sd = bsd_l2_context(ctx);
    if (sd == nullptr || sd->model_manager == nullptr) {
        error = "The engine has no model.";
        return BSD_ERR_NOT_LOADED;
    }
    std::unique_lock<std::recursive_mutex> lock(sd->execution_mutex, std::try_to_lock);
    if (!lock.owns_lock() || sd->executing_) {
        error = "A job is running.";
        return BSD_ERR_BUSY;
    }
    const auto bytes = BsdPartProbe::collect(*sd->model_manager);
    for (const PartName& entry : kPartNames) {
        const auto it = bytes.find(entry.component);
        if (it == bytes.end() || it->second.declared == 0 || out->count >= BSD_MAX_PARTS) {
            continue;
        }
        const BsdPartProbe::Bytes& b = it->second;
        bsd_part_memory_t& part      = out->parts[out->count++];
        const char* name             = entry.name;
        // With a TAE as the only VAE, the VAE runner holds the TAE weights.
        if (entry.component == ModelComponent::VAE && sd->uses_tae() && !sd->tae_preview_only()) {
            name = "taesd";
        }
        std::snprintf(part.name, sizeof(part.name), "%s", name);
        if (sd->backend_manager.params_backend_is_disk(entry.module)) {
            std::snprintf(part.backend, sizeof(part.backend), "disk");
        } else if (ggml_backend_t backend = sd->backend_manager.params_backend(entry.module)) {
            std::snprintf(part.backend, sizeof(part.backend), "%s", ggml_backend_name(backend));
        }
        part.bytes     = b.declared;
        part.bytes_cpu = b.host;
        part.bytes_gpu = (b.resident - b.host) + b.staged;
        part.on_disk   = b.declared - b.resident;
    }
    return BSD_OK;
}

}  // namespace bsd_l2
