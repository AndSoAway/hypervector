/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HYPERVEC_TEST_UTIL_H
#define HYPERVEC_TEST_UTIL_H

#include <hypervec/IndexIVFPQ.h>
#include <unistd.h>

struct Tempfilename {
    pthread_mutex_t* mutex;
    std::string filename;

    Tempfilename(pthread_mutex_t* mutex, std::string filename_template) {
        this->mutex = mutex;
        this->filename = filename_template;
        pthread_mutex_lock(mutex);
        int fd = mkstemp(&this->filename[0]);
        close(fd);
        pthread_mutex_unlock(mutex);
    }

    ~Tempfilename() {
        if (access(filename.c_str(), F_OK)) {
            unlink(filename.c_str());
        }
    }

    const char* c_str() {
        return filename.c_str();
    }
};

#endif // HYPERVEC_TEST_UTIL_H
