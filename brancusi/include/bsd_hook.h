// Stage hooks for the upstream pipeline (Layer 2). Upstream code includes this header and marks
// the start and the end of a stage. The engine installs one function. With no function, a scope
// does nothing. See the Layer 2 register (brancusi-desktop _docs/sd-cpp-interface.md section 5.3).
#ifndef BSD_HOOK_H
#define BSD_HOOK_H

// Called with a stage name ("encode_prompt", "sample" or "decode") and begin = 1 or 0.
typedef void (*bsd_stage_hook_fn)(const char* stage, int begin);

// One shared slot. An inline function keeps a single copy in the whole program.
inline bsd_stage_hook_fn& bsd_stage_hook() {
    static bsd_stage_hook_fn hook = nullptr;
    return hook;
}

// Marks a stage from the constructor to end() or to the destructor, so an early return also
// closes the stage.
class bsd_stage_scope {
public:
    explicit bsd_stage_scope(const char* stage) : stage_(stage) {
        if (bsd_stage_hook() != nullptr) {
            bsd_stage_hook()(stage_, 1);
            open_ = true;
        }
    }
    ~bsd_stage_scope() { end(); }
    bsd_stage_scope(const bsd_stage_scope&)            = delete;
    bsd_stage_scope& operator=(const bsd_stage_scope&) = delete;

    void end() {
        if (open_) {
            open_ = false;
            if (bsd_stage_hook() != nullptr) {
                bsd_stage_hook()(stage_, 0);
            }
        }
    }

private:
    const char* stage_;
    bool open_ = false;
};

#endif  // BSD_HOOK_H
