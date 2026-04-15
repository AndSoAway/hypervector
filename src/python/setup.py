# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import platform
import shutil

from setuptools import setup

# make the hypervec python package dir
shutil.rmtree("hypervec", ignore_errors=True)
os.mkdir("hypervec")
shutil.copyfile("contrib", "hypervec/contrib")
shutil.copyfile("__init__.py", "hypervec/__init__.py")
shutil.copyfile("loader.py", "hypervec/loader.py")
shutil.copyfile("class_wrappers.py", "hypervec/class_wrappers.py")
shutil.copyfile("extra_wrappers.py", "hypervec/extra_wrappers.py")
shutil.copyfile("array_conversions.py", "hypervec/array_conversions.py")

if os.path.exists("__init__.pyi"):
    shutil.copyfile("__init__.pyi", "hypervec/__init__.pyi")
if os.path.exists("py.typed"):
    shutil.copyfile("py.typed", "hypervec/py.typed")

if platform.system() != "AIX":
    ext = ".pyd" if platform.system() == "Windows" else ".so"
else:
    ext = ".a"
prefix = "Release/" * (platform.system() == "Windows")

swighypervec_generic_lib = f"{prefix}_swighypervec{ext}"
swighypervec_avx2_lib = f"{prefix}_swighypervec_avx2{ext}"
swighypervec_avx512_lib = f"{prefix}_swighypervec_avx512{ext}"
swighypervec_avx512_spr_lib = f"{prefix}_swighypervec_avx512_spr{ext}"
callbacks_lib = f"{prefix}libHypervec_python_callbacks{ext}"
swighypervec_sve_lib = f"{prefix}_swighypervec_sve{ext}"
Hypervec_example_external_module_lib = f"_Hypervec_example_external_module{ext}"

found_swighypervec_generic = os.path.exists(swighypervec_generic_lib)
found_swighypervec_avx2 = os.path.exists(swighypervec_avx2_lib)
found_swighypervec_avx512 = os.path.exists(swighypervec_avx512_lib)
found_swighypervec_avx512_spr = os.path.exists(swighypervec_avx512_spr_lib)
found_callbacks = os.path.exists(callbacks_lib)
found_swighypervec_sve = os.path.exists(swighypervec_sve_lib)
found_Hypervec_example_external_module_lib = os.path.exists(
    Hypervec_example_external_module_lib
)

if platform.system() != "AIX":
    assert (
        found_swighypervec_generic
        or found_swighypervec_avx2
        or found_swighypervec_avx512
        or found_swighypervec_avx512_spr
        or found_swighypervec_sve
        or found_Hypervec_example_external_module_lib
    ), (
        f"Could not find {swighypervec_generic_lib} or "
        f"{swighypervec_avx2_lib} or {swighypervec_avx512_lib} or {swighypervec_avx512_spr_lib} or {swighypervec_sve_lib} or {Hypervec_example_external_module_lib}. "
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

if found_Hypervec_example_external_module_lib:
    print(f"Copying {Hypervec_example_external_module_lib}")
    shutil.copyfile(
        "Hypervec_example_external_module.py", "hypervec/Hypervec_example_external_module.py"
    )
    shutil.copyfile(
        Hypervec_example_external_module_lib,
        f"hypervec/_Hypervec_example_external_module{ext}",
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
    packages=["hypervec"],
    package_data={
        "hypervec": ["*.so", "*.pyd", "*.a", "*.pyi", "py.typed"],
    },
    zip_safe=False,
)
