#pragma once

#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;

class MettaGrid;

// Metta2D C API --------------------------------------------------------------

struct   Metta2D; typedef struct Metta2D Metta2D;
Metta2D* Metta2D_Init ();
void     Metta2D_Quit (Metta2D* ctx);
void     Metta2D_Scene(Metta2D* ctx, MettaGrid* env);
void     Metta2D_Frame(Metta2D* ctx);

// Metta2D Python API ---------------------------------------------------------

class Metta2DPy {
    struct Deleter {
        void operator()(Metta2D* ctx) const { Metta2D_Quit(ctx); }
    };
    std::unique_ptr<Metta2D, Deleter> _ctx;

public:
    Metta2DPy() : _ctx(Metta2D_Init()) {}
    ~Metta2DPy() = default;

    void update(MettaGrid* env) { Metta2D_Scene(_ctx.get(), env); }
    void render() { Metta2D_Frame(_ctx.get()); }

    Metta2DPy(const Metta2DPy&) = delete;
    Metta2DPy(Metta2DPy&&) = delete;
    Metta2DPy& operator=(const Metta2DPy&) = delete;
    Metta2DPy& operator=(Metta2DPy&&) = delete;
};
