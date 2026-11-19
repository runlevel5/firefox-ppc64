/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Declarations of the few libc functions the proxy and the helper use. Both are
// built freestanding, the helper without an x86_64 sysroot; at run time box64
// or the system supplies libc.
#pragma once
#include <stddef.h>
#include <stdint.h>

extern "C" {
typedef struct FILE FILE;
struct timespec {
  long tv_sec;
  long tv_nsec;
};

int printf(const char* fmt, ...);
int snprintf(char* buf, size_t n, const char* fmt, ...);
int fflush(FILE* f);
FILE* fopen(const char* path, const char* mode);
size_t fread(void* p, size_t size, size_t n, FILE* f);
size_t fwrite(const void* p, size_t size, size_t n, FILE* f);
int fclose(FILE* f);
void* malloc(size_t n);
void* realloc(void* p, size_t n);
void free(void* p);
void* memcpy(void* d, const void* s, size_t n);
void* memset(void* d, int c, size_t n);
int memcmp(const void* a, const void* b, size_t n);
size_t strlen(const char* s);
int system(const char* cmd);
int clock_gettime(int clk, struct timespec* ts);
int usleep(unsigned usec);
void exit(int code) __attribute__((noreturn));
int unlink(const char* path);

long read(int fd, void* p, size_t n);
long write(int fd, const void* p, size_t n);
int close(int fd);
int socketpair(int domain, int type, int protocol, int fds[2]);
int fork(void);
int execv(const char* path, char* const argv[]);
void _exit(int code) __attribute__((noreturn));
int waitpid(int pid, int* status, int options);
char* getenv(const char* name);
int setenv(const char* name, const char* value, int overwrite);
int fcntl(int fd, int cmd, ...);
char* strrchr(const char* s, int c);

void* dlopen(const char* path, int flags);
void* dlsym(void* handle, const char* name);
char* dlerror(void);
typedef struct {
  const char* dli_fname;
  void* dli_fbase;
  const char* dli_sname;
  void* dli_saddr;
} Dl_info;
int dladdr(const void* addr, Dl_info* info);
}

#define AF_UNIX 1
#define SOCK_STREAM 1
#define RTLD_NOW 2
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1
#define F_SETFD 2
#define FD_CLOEXEC 1

inline void* operator new(size_t, void* p) noexcept { return p; }
// Objects are never deleted through these; virtual destructors reference it.
__attribute__((weak)) void operator delete(void*, size_t) noexcept {}
