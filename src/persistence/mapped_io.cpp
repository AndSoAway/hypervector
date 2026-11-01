/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <persistence/mapped_io.h>
#include <utils/log/assert.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>

#if defined(__linux__) || defined(__FreeBSD__)

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#elif defined(_WIN32)

#include <Windows.h>  // @manual
#include <io.h>       // @manual

#endif

namespace hypervec {

#if defined(__linux__) || defined(__FreeBSD__)

struct MmappedFileMappingOwner::PImpl {
  void* ptr = nullptr;
  size_t ptr_size = 0;

  explicit PImpl(const std::string& filename) {
    struct FileDeleter {
      void operator()(FILE* f) const {
        if (f != nullptr) {
          fclose(f);
        }
      }
    };

    auto f = std::unique_ptr<FILE, FileDeleter>(fopen(filename.c_str(), "r"),
                                                FileDeleter{});
    HYPERVEC_THROW_IF_NOT_FMT(f.get(), "could not open %s for reading: %s",
                              filename.c_str(), strerror(errno));

    // get the size
    struct stat s;
    int status = fstat(fileno(f.get()), &s);
    HYPERVEC_THROW_IF_NOT_FMT(status >= 0, "fstat() failed: %s",
                              strerror(errno));

    HYPERVEC_THROW_IF_NOT_MSG(s.st_size > 0, "cannot memory-map an empty file");
    HYPERVEC_THROW_IF_NOT_MSG(
        static_cast<uintmax_t>(s.st_size) <=
            static_cast<uintmax_t>((std::numeric_limits<size_t>::max)()),
        "memory-mapped file is too large for this address space");
    const size_t filesize = static_cast<size_t>(s.st_size);

    void* address =
        mmap(nullptr, filesize, PROT_READ, MAP_SHARED, fileno(f.get()), 0);
    HYPERVEC_THROW_IF_NOT_FMT(address != MAP_FAILED, "could not mmap(): %s",
                              strerror(errno));

    // btw, fd can be closed here

    madvise(address, filesize, MADV_RANDOM);

    // save it
    ptr = address;
    ptr_size = filesize;
  }

  explicit PImpl(FILE* f) {
    // get the size
    struct stat s;
    int status = fstat(fileno(f), &s);
    HYPERVEC_THROW_IF_NOT_FMT(status >= 0, "fstat() failed: %s",
                              strerror(errno));

    HYPERVEC_THROW_IF_NOT_MSG(s.st_size > 0, "cannot memory-map an empty file");
    HYPERVEC_THROW_IF_NOT_MSG(
        static_cast<uintmax_t>(s.st_size) <=
            static_cast<uintmax_t>((std::numeric_limits<size_t>::max)()),
        "memory-mapped file is too large for this address space");
    const size_t filesize = static_cast<size_t>(s.st_size);

    void* address =
        mmap(nullptr, filesize, PROT_READ, MAP_SHARED, fileno(f), 0);
    HYPERVEC_THROW_IF_NOT_FMT(address != MAP_FAILED, "could not mmap(): %s",
                              strerror(errno));

    // btw, fd can be closed here

    madvise(address, filesize, MADV_RANDOM);

    // save it
    ptr = address;
    ptr_size = filesize;
  }

  ~PImpl() {
    if (ptr != nullptr && ptr_size > 0) {
      munmap(ptr, ptr_size);
    }
  }
};

#elif defined(_WIN32)

struct MmappedFileMappingOwner::PImpl {
  void* ptr = nullptr;
  size_t ptr_size = 0;
  HANDLE mapping_handle = INVALID_HANDLE_VALUE;

  PImpl(const std::string& filename) {
    HANDLE file_handle =
        CreateFile(filename.c_str(), GENERIC_READ,
                   FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                   nullptr, OPEN_EXISTING, 0, nullptr);
    if (file_handle == INVALID_HANDLE_VALUE) {
      const auto error = GetLastError();
      HYPERVEC_THROW_FMT("could not open the file, %s (error %d)",
                         filename.c_str(), error);
    }

    // get the size of the file
    LARGE_INTEGER len_li;
    if (GetFileSizeEx(file_handle, &len_li) == 0) {
      const auto error = GetLastError();

      CloseHandle(file_handle);

      HYPERVEC_THROW_FMT("could not get the file size, %s (error %d)",
                         filename.c_str(), error);
    }
    if (len_li.QuadPart <= 0 ||
        static_cast<uint64_t>(len_li.QuadPart) >
            static_cast<uint64_t>((std::numeric_limits<size_t>::max)())) {
      CloseHandle(file_handle);
      HYPERVEC_THROW_MSG("memory-mapped file size is invalid");
    }

    // create a mapping
    mapping_handle =
        CreateFileMapping(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_handle == 0) {
      const auto error = GetLastError();

      CloseHandle(file_handle);

      HYPERVEC_THROW_FMT("could not create a file mapping, %s (error %d)",
                         filename.c_str(), error);
    }
    CloseHandle(file_handle);

    char* data = static_cast<char*>(
        MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0));
    if (data == nullptr) {
      const auto error = GetLastError();

      CloseHandle(mapping_handle);
      mapping_handle = INVALID_HANDLE_VALUE;

      HYPERVEC_THROW_FMT("could not get map the file, %s (error %d)",
                         filename.c_str(), error);
    }

    ptr = data;
    ptr_size = static_cast<size_t>(len_li.QuadPart);
  }

  PImpl(FILE* f) {
    // obtain a HANDLE from a FILE
    const int fd = _fileno(f);
    if (fd == -1) {
      // no good
      HYPERVEC_THROW_MSG("could not get a HANDLE");
    }

    HANDLE file_handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (file_handle == INVALID_HANDLE_VALUE) {
      HYPERVEC_THROW_MSG("could not get an OS HANDLE");
    }

    // get the size of the file
    LARGE_INTEGER len_li;
    if (GetFileSizeEx(file_handle, &len_li) == 0) {
      const auto error = GetLastError();
      HYPERVEC_THROW_FMT("could not get the file size (error %d)", error);
    }
    HYPERVEC_THROW_IF_NOT_MSG(
        len_li.QuadPart > 0 &&
            static_cast<uint64_t>(len_li.QuadPart) <=
                static_cast<uint64_t>((std::numeric_limits<size_t>::max)()),
        "memory-mapped file size is invalid");

    // create a mapping
    mapping_handle =
        CreateFileMapping(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_handle == 0) {
      const auto error = GetLastError();
      HYPERVEC_THROW_FMT("could not create a file mapping, (error %d)", error);
    }

    // the handle is provided externally, so this is not our business
    //   to close file_handle.

    char* data = static_cast<char*>(
        MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0));
    if (data == nullptr) {
      const auto error = GetLastError();

      CloseHandle(mapping_handle);
      mapping_handle = INVALID_HANDLE_VALUE;

      HYPERVEC_THROW_FMT("could not get map the file, (error %d)", error);
    }

    ptr = data;
    ptr_size = static_cast<size_t>(len_li.QuadPart);
  }

  ~PImpl() {
    if (mapping_handle != INVALID_HANDLE_VALUE) {
      UnmapViewOfFile(ptr);
      CloseHandle(mapping_handle);

      mapping_handle = INVALID_HANDLE_VALUE;
      ptr = nullptr;
    }
  }
};

#else

struct MmappedFileMappingOwner::PImpl {
  void* ptr = nullptr;
  size_t ptr_size = 0;

  explicit PImpl(const std::string& filename) {
    HYPERVEC_THROW_MSG("Not implemented");
  }

  explicit PImpl(FILE* f) { HYPERVEC_THROW_MSG("Not implemented"); }
};

#endif

MmappedFileMappingOwner::MmappedFileMappingOwner(const std::string& filename) {
  p_impl = std::make_unique<MmappedFileMappingOwner::PImpl>(filename);
}

MmappedFileMappingOwner::MmappedFileMappingOwner(FILE* f) {
  HYPERVEC_THROW_IF_NOT_MSG(f != nullptr,
                            "memory-mapped file handle must not be null");
  p_impl = std::make_unique<MmappedFileMappingOwner::PImpl>(f);
}

MmappedFileMappingOwner::~MmappedFileMappingOwner() = default;

//
void* MmappedFileMappingOwner::data() const { return p_impl->ptr; }

size_t MmappedFileMappingOwner::size() const { return p_impl->ptr_size; }

MappedFileIOReader::MappedFileIOReader(
    const std::shared_ptr<MmappedFileMappingOwner>& owner)
    : mmap_owner(owner) {
  HYPERVEC_THROW_IF_NOT_MSG(mmap_owner != nullptr,
                            "memory-mapped reader requires an owner");
  name = "memory-mapped file";
}

// this operation performs a copy
size_t MappedFileIOReader::operator()(void* ptr, size_t size, size_t nitems) {
  if (ptr == nullptr || size == 0 || nitems == 0) {
    return 0;
  }

  void* mapped_address = nullptr;
  const size_t actual_nitems = mmap(&mapped_address, size, nitems);
  if (actual_nitems > 0) {
    memcpy(ptr, mapped_address, size * actual_nitems);
  }

  return actual_nitems;
}

// this operation returns a mmapped address, owned by mmap_owner
size_t MappedFileIOReader::mmap(void** ptr, size_t size, size_t nitems) {
  if (ptr == nullptr || size == 0 || nitems == 0 || pos >= mmap_owner->size()) {
    return 0;
  }

  const size_t available_items = (mmap_owner->size() - pos) / size;
  const size_t actual_nitems = (std::min)(nitems, available_items);
  if (actual_nitems == 0) {
    return 0;
  }

  // get an address
  *ptr = static_cast<char*>(mmap_owner->data()) + pos;

  // alter pos
  pos += size * actual_nitems;

  return actual_nitems;
}

int MappedFileIOReader::filedescriptor() {
  HYPERVEC_THROW_MSG("memory-mapped reader does not retain a file descriptor");
}

}  // namespace hypervec
