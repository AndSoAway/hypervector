# -*- coding: utf-8 -*-
"""
读写锁实现

特性：
- 多个读操作可以并发执行（search）
- 写操作与任何操作互斥（insert/flush/drop 等）
- 写操作优先，避免写饥饿
"""

import threading


class RWLock:
    """读写锁：读读并发，读写互斥，写写互斥"""

    def __init__(self):
        self._lock = threading.Lock()
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False
        self._read_ready = threading.Condition(self._lock)
        self._write_ready = threading.Condition(self._lock)

    def acquire_read(self):
        with self._lock:
            # 有写者活跃或等待时，读者让路（写优先，避免写饥饿）
            while self._writer_active or self._writers_waiting > 0:
                self._read_ready.wait()
            self._readers += 1

    def release_read(self):
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_ready.notify()

    def acquire_write(self):
        with self._lock:
            self._writers_waiting += 1
            while self._writer_active or self._readers > 0:
                self._write_ready.wait()
            self._writers_waiting -= 1
            self._writer_active = True

    def release_write(self):
        with self._lock:
            self._writer_active = False
            # 优先唤醒等待的写者，否则唤醒所有读者
            if self._writers_waiting > 0:
                self._write_ready.notify()
            else:
                self._read_ready.notify_all()

    def read_lock(self):
        return _ReadLockCtx(self)

    def write_lock(self):
        return _WriteLockCtx(self)


class _ReadLockCtx:
    def __init__(self, rwlock):
        self._rwlock = rwlock

    def __enter__(self):
        self._rwlock.acquire_read()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._rwlock.release_read()
        return False


class _WriteLockCtx:
    def __init__(self, rwlock):
        self._rwlock = rwlock

    def __enter__(self):
        self._rwlock.acquire_write()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._rwlock.release_write()
        return False
