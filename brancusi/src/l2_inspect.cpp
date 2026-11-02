// Layer 2: inspect. Reads model file headers with upstream ModelLoader (src/model_loader.h).
// The public API has no header-only read. See the Layer 2 register.

#include "l2.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "model.h"
#include "model_loader.h"
#include "util.h"

namespace {

// Tensor name prefix of each part. The same prefixes as upstream init_model_loader()
// (src/pipeline/diffusion_engine.cpp). Keep them in sync after a rebase.
const char* part_prefix(bsd_component_t c) {
    switch (c) {
        case BSD_COMPONENT_DIFFUSION:
            return "model.diffusion_model.";
        case BSD_COMPONENT_HIGH_NOISE_DIFFUSION:
            return "model.high_noise_diffusion_model.";
        case BSD_COMPONENT_UNCOND_DIFFUSION:
            return "model.diffusion_model.uncond.";
        case BSD_COMPONENT_CLIP_L:
            return "clip_l.";
        case BSD_COMPONENT_CLIP_G:
            return "clip_g.";
        case BSD_COMPONENT_CLIP_VISION:
            return "clip_vision.";
        case BSD_COMPONENT_T5XXL:
            return "text_encoders.t5xxl.transformer.";
        case BSD_COMPONENT_PULID:
            return "model.diffusion_model.";
        case BSD_COMPONENT_LLM:
            return "text_encoders.llm.";
        case BSD_COMPONENT_LLM_VISION:
            return "text_encoders.llm.visual.";
        case BSD_COMPONENT_VAE:
            return "vae.";
        case BSD_COMPONENT_TAESD:
            return "tae.";
        case BSD_COMPONENT_AUDIO_ENCODER:
            return "wav2vec2.";
        case BSD_COMPONENT_PHOTO_MAKER:
            return "pmid.";
        default:
            return "";
    }
}

// Parts that upstream reads into the main model loader. The others (control net, tokenizer and
// so on) have their own loaders, so they do not count for the family.
bool in_main_loader(bsd_component_t c) {
    switch (c) {
        case BSD_COMPONENT_CONTROL_NET:
        case BSD_COMPONENT_IP_ADAPTER:
        case BSD_COMPONENT_MOTION_MODULE:
        case BSD_COMPONENT_TOKENIZER:
            return false;
        default:
            return true;
    }
}

void fill_types(const std::map<ggml_type, uint32_t>& stat, bsd_part_info_t& part) {
    std::vector<std::pair<ggml_type, uint32_t>> sorted(stat.begin(), stat.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    part.type_count = 0;
    part.tensors    = 0;
    for (const auto& [type, count] : sorted) {
        part.tensors += count;
        if (part.type_count < BSD_MAX_PART_TYPES) {
            bsd_type_count_t& t = part.types[part.type_count++];
            std::snprintf(t.name, sizeof(t.name), "%s", ggml_type_name(type));
            t.tensors = count;
        }
    }
    part.present = part.tensors > 0 ? 1 : 0;
}

void fill_sizes(const ModelLoader& loader, bsd_part_info_t& part) {
    uint64_t params = 0, bytes = 0;
    for (const auto& [name, storage] : loader.get_tensor_storage_map()) {
        params += static_cast<uint64_t>(storage.nelements());
        bytes += static_cast<uint64_t>(storage.nbytes());
    }
    part.params = params;
    part.bytes  = bytes;
}

}  // namespace

namespace bsd_l2 {

bsd_status_t inspect(const bsd_components_t* parts, int wtype, bsd_inspect_t* out, std::string& error) {
    ModelLoader all;
    for (int i = 0; i < BSD_COMPONENT_COUNT; i++) {
        const char* path = parts->paths[i];
        if (path == nullptr || path[0] == '\0') {
            continue;
        }
        const auto which = static_cast<bsd_component_t>(i);
        if (which == BSD_COMPONENT_TOKENIZER) {
            continue;  // A vocabulary or JSON file, not a model file.
        }
        const std::string prefix = part_prefix(which);
        // One loader for each part gives the counts and types of that part only.
        ModelLoader one;
        if (!one.init_from_file(path, prefix)) {
            error = std::string("Cannot read the part '") + bsd_component_name(which) + "': " + path;
            return BSD_ERR_BAD_MODEL;
        }
        bsd_part_info_t& part = out->parts[i];
        fill_types(one.get_wtype_stat(), part);
        fill_sizes(one, part);
        part.present = 1;
        out->tensors += part.tensors;
        out->params += part.params;
        if (in_main_loader(which) && !all.init_from_file(path, prefix)) {
            error = std::string("Cannot read the part '") + bsd_component_name(which) + "': " + path;
            return BSD_ERR_BAD_MODEL;
        }
    }

    // Upstream converts the tensor names (diffusers, split files) to its own names before the
    // model runs. Do the same, so that the family and the contents use the names upstream uses.
    all.convert_tensors_name();
    const SDVersion version = all.get_sd_version();
    if (version != VERSION_COUNT) {
        std::snprintf(out->family, sizeof(out->family), "%s", sd_version_name(version));
    }
    const ggml_type type = wtype >= 0 ? sd_type_to_ggml_type(static_cast<sd_type_t>(wtype)) : GGML_TYPE_COUNT;
    out->mem_bytes       = static_cast<uint64_t>(std::max<int64_t>(0, all.get_params_mem_size(nullptr, type)));
    fill_types(all.get_diffusion_model_wtype_stat(), out->contents[BSD_CONTENT_DIFFUSION]);
    fill_types(all.get_conditioner_wtype_stat(), out->contents[BSD_CONTENT_TEXT_ENCODER]);
    fill_types(all.get_vae_wtype_stat(), out->contents[BSD_CONTENT_VAE]);
    return BSD_OK;
}

}  // namespace bsd_l2
