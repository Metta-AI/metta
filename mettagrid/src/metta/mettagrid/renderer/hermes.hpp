#pragma once

#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;

class MettaGrid;

// Hermes C API --------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

struct  Hermes; typedef struct Hermes Hermes;
Hermes* Hermes_Init ();
void    Hermes_Quit (Hermes* ctx);
void    Hermes_Scene(Hermes* ctx, MettaGrid* env);
void    Hermes_Frame(Hermes* ctx);

#ifdef __cplusplus
} // extern "C"
#endif

// Hermes Python API ---------------------------------------------------------

class HermesPy {
    struct Deleter {
        void operator()(Hermes* ctx) const { Hermes_Quit(ctx); }
    };
    std::unique_ptr<Hermes, Deleter> _ctx;

public:
    HermesPy() : _ctx(Hermes_Init()) {}
    ~HermesPy() = default;

    void update(MettaGrid* env) { Hermes_Scene(_ctx.get(), env); }
    void render() { Hermes_Frame(_ctx.get()); }

    HermesPy(const HermesPy&) = delete;
    HermesPy(HermesPy&&) = delete;
    HermesPy& operator=(const HermesPy&) = delete;
    HermesPy& operator=(HermesPy&&) = delete;
};
