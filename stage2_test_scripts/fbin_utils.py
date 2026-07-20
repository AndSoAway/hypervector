#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def read_fbin(filename):
    with open(filename, "rb") as f:
        n = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        d = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        data = np.frombuffer(f.read(), dtype=np.float32)
        vectors = data.reshape(n, d)
    return vectors, n, d
