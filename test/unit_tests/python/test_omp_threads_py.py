# Copyright (c) 2024 HyperVec Authors. All rights reserved.
#
# This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import hypervec
import unittest


class TestOpenMP(unittest.TestCase):

    def test_openmp(self):
        assert hypervec.check_openmp()
