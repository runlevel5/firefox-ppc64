/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Wire protocol between the native proxy CDM (wvproxy.cc) and the x86_64
// helper that runs the real CDM under box64 (wvhost.cc).
//
// The CDM interface is single-threaded, so the protocol is a synchronous RPC
// over a socketpair. Each proxy -> helper message is a request (a CDM call, a
// timer, or a FileIO completion). While handling it the helper sends any number
// of Host_10 callbacks, then one kDone message with the call's results. The
// proxy forwards the callbacks to Firefox's host as they arrive.
//
// One helper serves every CDM instance of a GMP process: kCreate returns an
// instance id and each later request names the instance it is for.
//
// Message: u32 type, u32 instance, u32 payload length, payload. Payload fields
// are written in order with Writer and read back with Reader; both ends are
// little-endian.
#pragma once
#include "minilibc.h"

namespace wvipc {

enum Type : uint32_t {
  // Requests, proxy -> helper. kCreate is the first message and names the
  // key system; the helper answers it with kDone carrying a success flag.
  kCreate = 0,
  kInitialize = 1,
  kGetStatusForPolicy,
  kSetServerCertificate,
  kCreateSessionAndGenerateRequest,
  kLoadSession,
  kUpdateSession,
  kCloseSession,
  kRemoveSession,
  kTimerExpired,
  kDecrypt,
  kInitializeAudioDecoder,
  kInitializeVideoDecoder,
  kDeinitializeDecoder,
  kResetDecoder,
  kDecryptAndDecodeFrame,
  kDecryptAndDecodeSamples,
  kOnPlatformChallengeResponse,
  kOnQueryOutputProtectionStatus,
  kOnStorageId,
  kDestroy,
  kFileOpenComplete,
  kFileReadComplete,
  kFileWriteComplete,

  // Callbacks, helper -> proxy.
  kDone = 100,
  kSetTimer,
  kOnInitialized,
  kOnResolveKeyStatusPromise,
  kOnResolveNewSessionPromise,
  kOnResolvePromise,
  kOnRejectPromise,
  kOnSessionMessage,
  kOnSessionKeysChange,
  kOnExpirationChange,
  kOnSessionClosed,
  kSendPlatformChallenge,
  kEnableOutputProtection,
  kQueryOutputProtectionStatus,
  kOnDeferredInitializationDone,
  kRequestStorageId,
  kFileCreate,
  kFileOpen,
  kFileRead,
  kFileWrite,
  kFileClose,
};

constexpr uint32_t kMaxMessage = 64 << 20;
constexpr uint32_t kMaxInstances = 16;
constexpr uint32_t kNoInstance = 0xffffffff;

class Writer {
 public:
  ~Writer() { free(mData); }
  void U32(uint32_t v) { Raw(&v, 4); }
  void U64(uint64_t v) { Raw(&v, 8); }
  void I64(int64_t v) { Raw(&v, 8); }
  void F64(double v) { Raw(&v, 8); }
  void Bytes(const void* p, uint32_t n) {
    U32(n);
    Raw(p, n);
  }
  void Raw(const void* p, uint32_t n) {
    if (mLen + n > mCap) {
      while (mLen + n > mCap) mCap = mCap ? mCap * 2 : 256;
      mData = static_cast<uint8_t*>(realloc(mData, mCap));
    }
    if (n) memcpy(mData + mLen, p, n);
    mLen += n;
  }
  const uint8_t* Data() const { return mData; }
  uint32_t Length() const { return mLen; }

 private:
  uint8_t* mData = nullptr;
  uint32_t mLen = 0, mCap = 0;
};

// Reads fields from a received payload. A short payload sets mBad and yields
// zeros, so callers check Ok() once at the end.
class Reader {
 public:
  Reader(const uint8_t* p, uint32_t n) : mP(p), mEnd(p + n) {}
  uint32_t U32() { return Get<uint32_t>(); }
  uint64_t U64() { return Get<uint64_t>(); }
  int64_t I64() { return Get<int64_t>(); }
  double F64() { return Get<double>(); }
  // Returns a pointer into the payload, valid until the next message.
  const uint8_t* Bytes(uint32_t* n) {
    *n = U32();
    if (uint32_t(mEnd - mP) < *n) {
      mBad = true;
      *n = 0;
      return nullptr;
    }
    const uint8_t* p = mP;
    mP += *n;
    return p;
  }
  bool Ok() const { return !mBad; }

 private:
  template <typename T>
  T Get() {
    T v{};
    if (uint32_t(mEnd - mP) < sizeof(T)) {
      mBad = true;
      return v;
    }
    memcpy(&v, mP, sizeof(T));
    mP += sizeof(T);
    return v;
  }
  const uint8_t* mP;
  const uint8_t* mEnd;
  bool mBad = false;
};

inline bool WriteAll(int fd, const void* p, uint32_t n) {
  const uint8_t* c = static_cast<const uint8_t*>(p);
  while (n) {
    long w = write(fd, c, n);
    if (w <= 0) return false;
    c += w;
    n -= uint32_t(w);
  }
  return true;
}

inline bool ReadAll(int fd, void* p, uint32_t n) {
  uint8_t* c = static_cast<uint8_t*>(p);
  while (n) {
    long r = read(fd, c, n);
    if (r <= 0) return false;
    c += r;
    n -= uint32_t(r);
  }
  return true;
}

inline bool Send(int fd, uint32_t type, uint32_t instance, const Writer& w) {
  uint32_t hdr[3] = {type, instance, w.Length()};
  return WriteAll(fd, hdr, 12) && WriteAll(fd, w.Data(), w.Length());
}

// Receives one message into |*buf| (grown as needed).
inline bool Receive(int fd, uint32_t* type, uint32_t* instance, uint8_t** buf,
                    uint32_t* cap, uint32_t* len) {
  uint32_t hdr[3];
  if (!ReadAll(fd, hdr, 12) || hdr[2] > kMaxMessage) return false;
  if (hdr[2] > *cap) {
    *cap = hdr[2];
    *buf = static_cast<uint8_t*>(realloc(*buf, *cap));
  }
  *type = hdr[0];
  *instance = hdr[1];
  *len = hdr[2];
  return ReadAll(fd, *buf, hdr[2]);
}

}  // namespace wvipc
