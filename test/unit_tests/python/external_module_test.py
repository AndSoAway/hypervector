# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

import unittest

import hypervec

import hypervec.Hypervec_example_external_module as external_module

import numpy as np


class TestCustomIDSelector(unittest.TestCase):
    """test if we can construct a custom IDSelector"""

    def test_IDSelector(self):
        ids = external_module.IDSelectorModulo(3)
        self.assertFalse(ids.IsMember(1))
        self.assertTrue(ids.IsMember(3))


class TestArrayConversions(unittest.TestCase):

    def test_idx_array(self):
        tab = np.arange(10).astype("int64")
        new_sum = external_module.sum_of_idx(len(tab), hypervec.swig_ptr(tab))
        self.assertEqual(new_sum, tab.sum())

    def do_array_test(self, ty):
        tab = np.arange(10).astype(ty)
        func = getattr(external_module, "sum_of_" + ty)
        print("perceived type", hypervec.swig_ptr(tab))
        new_sum = func(len(tab), hypervec.swig_ptr(tab))
        self.assertEqual(new_sum, tab.sum())

    def test_sum_uint8(self):
        self.do_array_test("uint8")

    def test_sum_uint16(self):
        self.do_array_test("uint16")

    def test_sum_uint32(self):
        self.do_array_test("uint32")

    def test_sum_uint64(self):
        self.do_array_test("uint64")

    def test_sum_int8(self):
        self.do_array_test("int8")

    def test_sum_int16(self):
        self.do_array_test("int16")

    def test_sum_int32(self):
        self.do_array_test("int32")

    def test_sum_int64(self):
        self.do_array_test("int64")

    def test_sum_float32(self):
        self.do_array_test("float32")

    def test_sum_float64(self):
        self.do_array_test("float64")
