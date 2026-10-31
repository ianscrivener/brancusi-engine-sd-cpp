// brancusi-engine: runs bsd_* calls for the Brancusi core.
//
// Transport: JSON lines. Commands on stdin, protocol messages on stdout, raw text on stderr.
// The engine exits when stdin closes. See brancusi-desktop _docs/sd-cpp-interface.md section 3.4.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define bsd_dup _dup
#define bsd_dup2 _dup2
#define bsd_fdopen _fdopen
#else
#include <unistd.h>
#define bsd_dup dup
#define bsd_dup2 dup2
#define bsd_fdopen fdopen
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#include "bsd.h"
#include "json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

// ---- Output: one writer thread, so callbacks never block on the pipe. ----

class Writer {
public:
    explicit Writer(FILE* out) : out_(out), thread_([this] { run(); }) {}

    ~Writer() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closing_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    void send(const json& message) {
        std::string line = message.dump(-1, ' ', false, json::error_handler_t::replace);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(std::move(line));
        }
        cv_.notify_one();
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            cv_.wait(lock, [this] { return closing_ || !queue_.empty(); });
            while (!queue_.empty()) {
                std::string line = std::move(queue_.front());
                queue_.pop_front();
                lock.unlock();
                std::fputs(line.c_str(), out_);
                std::fputc('\n', out_);
                std::fflush(out_);
                lock.lock();
            }
            if (closing_) {
                return;
            }
        }
    }

    FILE* out_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::string> queue_;
    bool closing_ = false;
    std::thread thread_;
};

// ---- Long work (load, unload, generate) runs on one worker thread. ----

class Worker {
public:
    Worker() : thread_([this] { run(); }) {}

    ~Worker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closing_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    // Returns false if work is already running or waiting.
    bool try_post(std::function<void()> task) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (busy_ || task_) {
            return false;
        }
        task_ = std::move(task);
        cv_.notify_one();
        return true;
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            cv_.wait(lock, [this] { return closing_ || task_; });
            if (!task_) {
                return;
            }
            auto task = std::move(task_);
            task_     = nullptr;
            busy_     = true;
            lock.unlock();
            task();
            lock.lock();
            busy_ = false;
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::function<void()> task_;
    bool busy_    = false;
    bool closing_ = false;
    std::thread thread_;
};

struct State {
    bsd_engine_t* engine = nullptr;
    Writer* writer       = nullptr;
    // The command and job that own the events from the engine callback.
    std::mutex current_mutex;
    std::string current_id;
    std::string current_job;
    int log_level = BSD_LOG_INFO;
};

State g;

std::string str_or(const json& j, const char* key, const std::string& fallback = "") {
    auto it = j.find(key);
    return (it != j.end() && it->is_string()) ? it->get<std::string>() : fallback;
}

const char* c_or_null(const std::string& s) {
    return s.empty() ? nullptr : s.c_str();
}

void set_current(const std::string& id, const std::string& job) {
    std::lock_guard<std::mutex> lock(g.current_mutex);
    g.current_id  = id;
    g.current_job = job;
}

json tagged(json message) {
    std::lock_guard<std::mutex> lock(g.current_mutex);
    if (!g.current_id.empty()) message["id"] = g.current_id;
    if (!g.current_job.empty()) message["job"] = g.current_job;
    return message;
}

json error_message(const std::string& id, bsd_status_t status, const std::string& message,
                   const std::string& detail = "", const std::string& job = "") {
    json m = {{"event", "error"}, {"code", bsd_status_name(status)}, {"message", message}};
    if (!id.empty()) m["id"] = id;
    if (!job.empty()) m["job"] = job;
    if (!detail.empty()) m["detail"] = detail;
    return m;
}

json engine_error(const std::string& id, bsd_status_t status, const std::string& job = "") {
    return error_message(id, status, bsd_last_error(g.engine), bsd_last_error_detail(g.engine), job);
}

json devices_json() {
    bsd_memory_t mem{};
    mem.size = sizeof(mem);
    bsd_memory(g.engine, &mem);
    json devices = json::array();
    for (int i = 0; i < mem.device_count; i++) {
        const auto& d = mem.devices[i];
        devices.push_back({{"name", d.name},
                           {"description", d.description},
                           {"type", d.type},
                           {"free_mib", d.free_bytes / (1024 * 1024)},
                           {"total_mib", d.total_bytes / (1024 * 1024)}});
    }
    return devices;
}

json capabilities_json(uint64_t caps) {
    json out = json::array();
    const std::pair<uint64_t, const char*> names[] = {
        {BSD_CAP_IMAGE, "image"},     {BSD_CAP_VIDEO, "video"},           {BSD_CAP_CANCEL, "cancel"},
        {BSD_CAP_PREVIEW, "preview"}, {BSD_CAP_MEMORY, "memory"},         {BSD_CAP_UPSCALE, "upscale"},
        {BSD_CAP_SET_COMPONENT, "set_component"}, {BSD_CAP_RUN_STAGE, "run_stage"},
    };
    for (const auto& [flag, name] : names) {
        if (caps & flag) out.push_back(name);
    }
    return out;
}

json info_json() {
    bsd_info_t info{};
    info.size = sizeof(info);
    bsd_info(g.engine, &info);
    json m = {{"bsd_api_version", info.api_version},
              {"sd_version", info.sd_version ? info.sd_version : ""},
              {"sd_commit", info.sd_commit ? info.sd_commit : ""},
              {"system_info", info.system_info ? info.system_info : ""},
              {"capabilities", capabilities_json(info.capabilities)},
              {"loaded", info.loaded != 0},
              {"devices", devices_json()}};
    if (info.model_version) m["model_version"] = info.model_version;
    return m;
}

void on_event(const bsd_event_t* ev, void*) {
    switch (ev->type) {
        case BSD_EVENT_LOG: {
            if (ev->level < g.log_level) return;
            static const char* levels[] = {"debug", "verbose", "info", "warn", "error"};
            g.writer->send(tagged({{"event", "log"}, {"level", levels[ev->level]}, {"text", ev->text}}));
            return;
        }
        case BSD_EVENT_PROGRESS:
            g.writer->send(tagged({{"event", "progress"},
                                   {"step", ev->step},
                                   {"steps", ev->steps},
                                   {"sec_per_step", ev->sec_per_step}}));
            return;
        case BSD_EVENT_STAGE:
            g.writer->send(tagged({{"event", "stage"}, {"stage", ev->stage}}));
            return;
        default:
            return;
    }
}

long long ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
}

// ---- Commands ----

void do_load(const std::string& id, const json& cmd) {
    set_current(id, "");
    // Keep the strings alive for the call.
    std::string paths[BSD_COMPONENT_COUNT];
    bsd_components_t components{};
    components.size = sizeof(components);
    const json comps = cmd.value("components", json::object());
    for (auto it = comps.begin(); it != comps.end(); ++it) {
        int index = -1;
        for (int i = 0; i < BSD_COMPONENT_COUNT; i++) {
            if (it.key() == bsd_component_name(static_cast<bsd_component_t>(i))) index = i;
        }
        if (index < 0 || !it->is_string()) {
            g.writer->send(error_message(id, BSD_ERR_INVALID_ARGUMENT, "Unknown model part: " + it.key()));
            set_current("", "");
            return;
        }
        paths[index]                = it->get<std::string>();
        components.paths[index]     = paths[index].c_str();
    }

    const json o = cmd.value("options", json::object());
    const std::string backend = str_or(o, "backend"), params_backend = str_or(o, "params_backend"),
                      max_vram = str_or(o, "max_vram"), flash = str_or(o, "flash_attn"),
                      model_args = str_or(o, "model_args"), weight_type = str_or(o, "weight_type");
    bsd_load_options_t opts{};
    opts.size              = sizeof(opts);
    opts.backend           = c_or_null(backend);
    opts.params_backend    = c_or_null(params_backend);
    opts.max_vram          = c_or_null(max_vram);
    opts.flash_attn_mode   = c_or_null(flash);
    opts.model_args        = c_or_null(model_args);
    opts.weight_type       = c_or_null(weight_type);
    opts.enable_mmap       = o.value("mmap", false) ? 1 : 0;
    opts.eager_load        = o.value("eager_load", false) ? 1 : 0;
    opts.tae_preview_only  = o.value("tae_preview_only", false) ? 1 : 0;
    opts.skip_memory_check = o.value("skip_memory_check", false) ? 1 : 0;

    const auto start        = std::chrono::steady_clock::now();
    const bsd_status_t st   = bsd_load(g.engine, &components, &opts);
    if (st != BSD_OK) {
        g.writer->send(engine_error(id, st));
    } else {
        json m    = info_json();
        m["id"]    = id;
        m["event"] = "loaded";
        m["ms"]    = ms_since(start);
        g.writer->send(m);
    }
    set_current("", "");
}

void do_generate(const std::string& id, const json& cmd) {
    const std::string job        = str_or(cmd, "job", id);
    const std::string output_dir = str_or(cmd, "output_dir");
    const json p                 = cmd.value("params", json::object());
    set_current(id, job);

    if (output_dir.empty()) {
        g.writer->send(error_message(id, BSD_ERR_INVALID_ARGUMENT, "output_dir is required.", "", job));
        set_current("", "");
        return;
    }
    const std::string prompt = str_or(p, "prompt"), negative = str_or(p, "negative_prompt"),
                      method = str_or(p, "sample_method"), scheduler = str_or(p, "scheduler"),
                      sigmas = str_or(p, "sigmas");
    bsd_image_params_t params{};
    params.size            = sizeof(params);
    params.prompt          = prompt.c_str();
    params.negative_prompt = negative.c_str();
    params.width           = p.value("width", 0);
    params.height          = p.value("height", 0);
    params.steps           = p.value("steps", 0);
    params.seed            = p.value("seed", -1LL);
    params.cfg_scale       = p.value("cfg_scale", 0.0f);
    params.sample_method   = c_or_null(method);
    params.scheduler       = c_or_null(scheduler);
    params.sigmas          = c_or_null(sigmas);
    params.batch_count     = p.value("batch_count", 1);
    params.flow_shift      = p.value("flow_shift", 0.0f);

    const auto start      = std::chrono::steady_clock::now();
    bsd_result_t* result  = nullptr;
    const bsd_status_t st = bsd_generate_image(g.engine, &params, &result);
    if (st != BSD_OK) {
        g.writer->send(engine_error(id, st, job));
        set_current("", "");
        return;
    }

    std::error_code ec;
    fs::create_directories(fs::u8path(output_dir), ec);
    json outputs = json::array();
    for (int i = 0; i < result->count; i++) {
        const bsd_image_t& img = result->images[i];
        const fs::path path    = fs::u8path(output_dir) / (std::to_string(i) + ".png");
        const std::string path_str = path.u8string();
        if (!stbi_write_png(path_str.c_str(), img.width, img.height, img.channels, img.data, img.width * img.channels)) {
            g.writer->send(error_message(id, BSD_ERR_INTERNAL, "Could not write the image to " + path_str, "", job));
            bsd_free_result(result);
            set_current("", "");
            return;
        }
        outputs.push_back({{"path", path_str}, {"width", img.width}, {"height", img.height}, {"seed", result->seeds[i]}});
    }
    bsd_free_result(result);
    g.writer->send({{"id", id}, {"event", "done"}, {"job", job}, {"outputs", outputs}, {"timings", {{"total_ms", ms_since(start)}}}});
    set_current("", "");
}

void handle(const std::string& line, Worker& worker) {
    json cmd;
    try {
        cmd = json::parse(line);
    } catch (const std::exception& ex) {
        g.writer->send(error_message("", BSD_ERR_INVALID_ARGUMENT, std::string("Bad JSON: ") + ex.what()));
        return;
    }
    if (!cmd.is_object()) {
        g.writer->send(error_message("", BSD_ERR_INVALID_ARGUMENT, "A command must be a JSON object."));
        return;
    }
    const std::string id   = str_or(cmd, "id");
    const std::string name = str_or(cmd, "cmd");

    if (name == "info") {
        json m    = info_json();
        m["id"]    = id;
        m["event"] = "info";
        g.writer->send(m);
    } else if (name == "memory") {
        g.writer->send({{"id", id}, {"event", "memory"}, {"devices", devices_json()}});
    } else if (name == "cancel") {
        const std::string job = str_or(cmd, "job");
        std::string running_job;
        {
            std::lock_guard<std::mutex> lock(g.current_mutex);
            running_job = g.current_job;
        }
        if (!running_job.empty() && (job.empty() || job == running_job)) {
            bsd_cancel(g.engine, str_or(cmd, "mode") == "new_latents" ? BSD_CANCEL_NEW_LATENTS : BSD_CANCEL_ALL);
        }
        g.writer->send({{"id", id}, {"event", "ok"}});
    } else if (name == "load" || name == "unload" || name == "generate") {
        std::function<void()> task;
        if (name == "load") {
            task = [id, cmd] { do_load(id, cmd); };
        } else if (name == "generate") {
            task = [id, cmd] { do_generate(id, cmd); };
        } else {
            task = [id] {
                const bsd_status_t st = bsd_unload(g.engine);
                g.writer->send(st == BSD_OK ? json{{"id", id}, {"event", "unloaded"}} : engine_error(id, st));
            };
        }
        if (!worker.try_post(std::move(task))) {
            g.writer->send(error_message(id, BSD_ERR_BUSY, "The engine is busy. Wait for the current command, or cancel it.",
                                         "", str_or(cmd, "job")));
        }
    } else if (name == "log_level") {
        static const char* levels[] = {"debug", "verbose", "info", "warn", "error"};
        const std::string level     = str_or(cmd, "level", "info");
        for (int i = 0; i < 5; i++) {
            if (level == levels[i]) g.log_level = i;
        }
        g.writer->send({{"id", id}, {"event", "ok"}});
    } else {
        g.writer->send(error_message(id, BSD_ERR_INVALID_ARGUMENT, "Unknown command: " + name));
    }
}

}  // namespace

int main() {
    // Only protocol messages go to stdout. Keep the real stdout for the protocol, and send
    // anything else that writes to fd 1 (upstream printf, ggml) to stderr.
    std::fflush(stdout);
    const int proto_fd = bsd_dup(1);
    bsd_dup2(2, 1);
#ifdef _WIN32
    _setmode(proto_fd, _O_BINARY);
#endif
    FILE* proto = bsd_fdopen(proto_fd, "w");
    if (proto == nullptr) {
        std::fprintf(stderr, "brancusi-engine: cannot open the protocol stream\n");
        return 1;
    }

    Writer writer(proto);
    g.writer = &writer;

    bsd_config_t config{};
    config.size = sizeof(config);
    g.engine    = bsd_create(&config);
    if (g.engine == nullptr) {
        writer.send(error_message("", BSD_ERR_INTERNAL, "Could not create the engine."));
        return 1;
    }
    bsd_set_event_cb(g.engine, on_event, nullptr);

    json hello     = info_json();
    hello["event"] = "hello";
    writer.send(hello);

    {
        Worker worker;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (line.find("\"shutdown\"") != std::string::npos) {
                try {
                    if (json::parse(line).value("cmd", "") == "shutdown") break;
                } catch (...) {
                }
            }
            handle(line, worker);
        }
        // stdin closed or shutdown: stop the running job, then let the worker finish.
        bsd_cancel(g.engine, BSD_CANCEL_ALL);
    }
    bsd_destroy(g.engine);
    g.engine = nullptr;
    return 0;
}
