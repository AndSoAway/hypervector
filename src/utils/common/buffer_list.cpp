/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */



#include <utils/common/buffer_list.h>

#include <cstring>

namespace hypervec {

BufferList::BufferList(size_t buffer_size) : buffer_size(buffer_size) {
  wp = buffer_size;
}

BufferList::~BufferList() {
  for (int i = 0; i < buffers.size(); i++) {
    delete[] buffers[i].ids;
    delete[] buffers[i].dis;
  }
}

void BufferList::Add(idx_t id, float dis) {
  if (wp == buffer_size) {  // need new buffer
    AppendBuffer();
  }
  Buffer& buf = buffers.back();
  buf.ids[wp] = id;
  buf.dis[wp] = dis;
  wp++;
}

void BufferList::AppendBuffer() {
  Buffer buf = {new idx_t[buffer_size], new float[buffer_size]};
  buffers.push_back(buf);
  wp = 0;
}

/// copy elements ofs:ofs+n-1 seen as linear data in the buffers to
/// tables dest_ids, dest_dis
void BufferList::CopyRange(size_t ofs, size_t n, idx_t* dest_ids,
                            float* dest_dis) {
  size_t bno = ofs / buffer_size;
  ofs -= bno * buffer_size;
  while (n > 0) {
    size_t ncopy = ofs + n < buffer_size ? n : buffer_size - ofs;
    Buffer buf = buffers[bno];
    memcpy(dest_ids, buf.ids + ofs, ncopy * sizeof(*dest_ids));
    memcpy(dest_dis, buf.dis + ofs, ncopy * sizeof(*dest_dis));
    dest_ids += ncopy;
    dest_dis += ncopy;
    ofs = 0;
    bno++;
    n -= ncopy;
  }
}

}  // namespace hypervec