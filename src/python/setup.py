# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import glob
import os
import platform
import shutil

from setuptools import setup

# make the hypervec python package dir
shutil.rmtree("hypervec", ignore_errors=True)
os.mkdir("hypervec")
if os.path.exists("contrib"):
    if os.path.isdir("contrib"):
        shutil.copytree("contrib", "hypervec/contrib")
    else:
        shutil.copyfile("contrib", "hypervec/contrib")
# Copy every Python source module from the current directory (a faithful
# mirror of src/python produced by CMake) into the hypervec sub-package.
# Single source of truth is the directory contents rather than a hand-written
# list, so a newly added module can no longer silently drop out of the wheel
# (Issue #41). SWIG-generated wrappers are excluded here and copied below,
# conditionally on the matching compiled library actually existing.
# Note: supersedes the per-file hypervec_bundle.py copy added by #39 — the
# glob covers it (verified: 17 modules land in the wheel on both x86/ARM).
for _py_src in sorted(glob.glob("*.py")):
    if _py_src == "setup.py":
        continue
    if _py_src.startswith("swighypervec") or _py_src == "Hypervec_example_external_module.py":
        continue  # SWIG output — handled per detected architecture below
    shutil.copyfile(_py_src, os.path.join("hypervec", _py_src))

if os.path.exists("__init__.pyi"):
    shutil.copyfile("__init__.pyi", "hypervec/__init__.pyi")
if os.path.exists("py.typed"):
    shutil.copyfile("py.typed", "hypervec/py.typed")

if platform.system() != "AIX":
    ext = ".pyd" if platform.system() == "Windows" else ".so"
else:
    ext = ".a"
prefix = "Release/" * (
    platform.system() == "Windows" and os.path.exists(os.path.join("Release", "_swighypervec.pyd"))
)

swighypervec_generic_lib = f"{prefix}_swighypervec{ext}"
swighypervec_avx2_lib = f"{prefix}_swighypervec_avx2{ext}"
swighypervec_avx512_lib = f"{prefix}_swighypervec_avx512{ext}"
swighypervec_avx512_spr_lib = f"{prefix}_swighypervec_avx512_spr{ext}"
callbacks_lib = f"{prefix}libhypervec_python_callbacks{ext}"
swighypervec_sve_lib = f"{prefix}_swighypervec_sve{ext}"
external_module_lib = f"_hypervec_example_external_module{ext}"

found_swighypervec_generic = os.path.exists(swighypervec_generic_lib)
found_swighypervec_avx2 = os.path.exists(swighypervec_avx2_lib)
found_swighypervec_avx512 = os.path.exists(swighypervec_avx512_lib)
found_swighypervec_avx512_spr = os.path.exists(swighypervec_avx512_spr_lib)
found_callbacks = os.path.exists(callbacks_lib)
found_swighypervec_sve = os.path.exists(swighypervec_sve_lib)
found_external_module = os.path.exists(external_module_lib)

if platform.system() != "AIX":
    assert (
        found_swighypervec_generic
        or found_swighypervec_avx2
        or found_swighypervec_avx512
        or found_swighypervec_avx512_spr
        or found_swighypervec_sve
        or found_external_module
    ), (
        f"Could not find {swighypervec_generic_lib} or "
        f"{swighypervec_avx2_lib} or {swighypervec_avx512_lib} or {swighypervec_avx512_spr_lib} or {swighypervec_sve_lib} or {external_module_lib}. "
        f"hypervec may not be compiled yet."
    )

if found_swighypervec_generic:
    print(f"Copying {swighypervec_generic_lib}")
    shutil.copyfile("swighypervec.py", "hypervec/swighypervec.py")
    shutil.copyfile(swighypervec_generic_lib, f"hypervec/_swighypervec{ext}")

if found_swighypervec_avx2:
    print(f"Copying {swighypervec_avx2_lib}")
    shutil.copyfile("swighypervec_avx2.py", "hypervec/swighypervec_avx2.py")
    shutil.copyfile(swighypervec_avx2_lib, f"hypervec/_swighypervec_avx2{ext}")

if found_swighypervec_avx512:
    print(f"Copying {swighypervec_avx512_lib}")
    shutil.copyfile("swighypervec_avx512.py", "hypervec/swighypervec_avx512.py")
    shutil.copyfile(swighypervec_avx512_lib, f"hypervec/_swighypervec_avx512{ext}")

if found_swighypervec_avx512_spr:
    print(f"Copying {swighypervec_avx512_spr_lib}")
    shutil.copyfile("swighypervec_avx512_spr.py", "hypervec/swighypervec_avx512_spr.py")
    shutil.copyfile(swighypervec_avx512_spr_lib, f"hypervec/_swighypervec_avx512_spr{ext}")

if found_callbacks:
    print(f"Copying {callbacks_lib}")
    shutil.copyfile(callbacks_lib, f"hypervec/{callbacks_lib}")

if found_swighypervec_sve:
    print(f"Copying {swighypervec_sve_lib}")
    shutil.copyfile("swighypervec_sve.py", "hypervec/swighypervec_sve.py")
    shutil.copyfile(swighypervec_sve_lib, f"hypervec/_swighypervec_sve{ext}")

if found_external_module:
    print(f"Copying {external_module_lib}")
    shutil.copyfile(
        "Hypervec_example_external_module.py", "hypervec/Hypervec_example_external_module.py"
    )
    shutil.copyfile(
        external_module_lib,
        f"hypervec/_hypervec_example_external_module{ext}",
    )

long_description = """
hypervec is a library for efficient similarity Search and clustering of dense
vectors. It contains algorithms that Search in sets of vectors of any size,
up to ones that possibly do not fit in RAM. It also contains supporting
code for evaluation and parameter tuning. hypervec is written in C++ with
complete wrappers for Python/numpy.
"""
setup(
    name="hypervec",
    version="1.14.1",
    description="A library for efficient similarity Search and clustering of dense vectors",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/QY-Graph/hypervec",
    author="Matthijs Douze, Jeff Johnson, Herve Jegou, Lucas Hosseini",
    author_email="hypervec@meta.com",
    license="MIT",
    keywords="Search nearest neighbors",
    install_requires=["numpy", "packaging"],
    extras_require={
        "server": ["fastapi", "uvicorn", "hypercorn", "h2"],
        "grpc-server": ["grpcio>=1.83,<2", "protobuf>=7.35.1,<8"],
        "dual-server": [
            "fastapi",
            "uvicorn",
            "hypercorn",
            "h2",
            "grpcio>=1.83,<2",
            "protobuf>=7.35.1,<8",
        ],
    },
    packages=["hypervec"],
    package_data={
        "hypervec": ["*.so", "*.pyd", "*.a", "*.pyi", "py.typed"],
    },
    zip_safe=False,
)
