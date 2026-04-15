/**
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

// This is an example of how an external module can be added via SWIG.

%module Hypervec_example_external_module;


// Put C++ includes here
%{

#include <core/HypervecException.h>
#include <core/IDSelector.h>

%}

#pragma SWIG nowarn=322

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;

#ifdef SWIGWORDSIZE64
typedef unsigned long uint64_t;
typedef long int64_t;
#else
typedef unsigned long long uint64_t;
typedef long long int64_t;
#endif

typedef uint64_t size_t;

// This means: assume what's declared in these .h files is provided
// by the hypervec module.
%import(module="hypervec") "core/MetricType.h"
%import(module="hypervec") "core/IDSelector.h"

// functions to be parsed here

// This is important to release GIL and do hypervec exception handing
%exception {
    Py_BEGIN_ALLOW_THREADS
    try {
        $action
    } catch(hypervec::HypervecException & e) {
        PyEval_RestoreThread(_save);

        if (PyErr_Occurred()) {
            // some previous code already set the error type.
        } else {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        SWIG_fail;
    } catch(std::bad_alloc&) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_MemoryError, "std::bad_alloc");
        SWIG_fail;
    }
    Py_END_ALLOW_THREADS
}


// any class or function declared below will be made available
// in the module.
%inline %{

struct IDSelectorModulo : hypervec::IDSelector {
    int mod;

    IDSelectorModulo(int mod): mod(mod) {}

    bool IsMember(hypervec::idx_t id) const {
        return id % mod == 0;
    }

    ~IDSelectorModulo() override {}
};

hypervec::idx_t sum_of_idx(size_t n, const hypervec::idx_t *tab) {
    hypervec::idx_t sum = 0;
    for(size_t i = 0; i < n; i++) {
        sum += tab[i];
    }
    return sum;
}

float sum_of_float32(size_t n, const float *tab) {
    float sum = 0;
    for(size_t i = 0; i < n; i++) {
        sum += tab[i];
    }
    return sum;
}

double sum_of_float64(size_t n, const double *tab) {
    double sum = 0;
    for(size_t i = 0; i < n; i++) {
        sum += tab[i];
    }
    return sum;
}

%}

/**********************************************
 * To test if passing a swig_ptr on all array types works
 **********************************************/

%define SUM_OF_TYPE(ty)

%inline %{

ty##_t sum_of_##ty (size_t n, const ty##_t * tab) {
    ty##_t sum = 0;
    for(size_t i = 0; i < n; i++) {
        sum += tab[i];
    }
    return sum;
}

%}

%enddef

SUM_OF_TYPE(uint8);
SUM_OF_TYPE(uint16);
SUM_OF_TYPE(uint32);
SUM_OF_TYPE(uint64);

SUM_OF_TYPE(int8);
SUM_OF_TYPE(int16);
SUM_OF_TYPE(int32);
SUM_OF_TYPE(int64);
