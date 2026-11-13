/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Native proxy CDM. Firefox loads it as libwidevinecdm.so and it forwards every
// ContentDecryptionModule_10/11 call to the x86_64 helper (wvhost), which runs
// the real CDM under box64. See wvipc.h for the protocol.
//
// Inside Firefox the GMP child starts the helper before its sandbox engages
// (through the gmp-helper program beside this library) and passes the socket
// in MOZ_GMP_HELPER_FD; that one helper serves every CDM instance. Without
// that variable the proxy starts a helper of its own per instance, for use
// outside Firefox. Paths for that case, overridable through the environment:
//   WV_BOX64               box64 (default: box64 from PATH)
//   WV_HOST                x86_64 helper (default: <this dir>/wvhost)
//   WV_REAL_CDM            real x86_64 CDM
//                          (default: <this dir>/libwidevinecdm-x86_64.so)
//   BOX64_LD_LIBRARY_PATH  (default: <this dir>/x64lib)

#include "wvinput.h"

using namespace wvipc;

extern "C" {
int execvp(const char* file, char* const argv[]);
}

extern "C" void __cxa_pure_virtual() { _exit(70); }

#define WV_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

// Firefox passes a Host_10 or a Host_11; they differ only in ReportMetrics,
// which the real CDM (driven as a CDM 10) never calls.
class HostRef {
 public:
  virtual ~HostRef() {}
  virtual cdm::Buffer* Allocate(uint32_t capacity) = 0;
  virtual void SetTimer(int64_t delay_ms, void* context) = 0;
  virtual void OnInitialized(bool success) = 0;
  virtual void OnResolveKeyStatusPromise(uint32_t id, cdm::KeyStatus s) = 0;
  virtual void OnResolveNewSessionPromise(uint32_t id, const char* sid,
                                          uint32_t len) = 0;
  virtual void OnResolvePromise(uint32_t id) = 0;
  virtual void OnRejectPromise(uint32_t id, cdm::Exception e, uint32_t code,
                               const char* msg, uint32_t len) = 0;
  virtual void OnSessionMessage(const char* sid, uint32_t sidlen,
                                cdm::MessageType t, const char* msg,
                                uint32_t len) = 0;
  virtual void OnSessionKeysChange(const char* sid, uint32_t sidlen, bool more,
                                   const cdm::KeyInformation* keys,
                                   uint32_t count) = 0;
  virtual void OnExpirationChange(const char* sid, uint32_t len,
                                  cdm::Time t) = 0;
  virtual void OnSessionClosed(const char* sid, uint32_t len) = 0;
  virtual void SendPlatformChallenge(const char* service, uint32_t serviceLen,
                                     const char* challenge,
                                     uint32_t challengeLen) = 0;
  virtual void EnableOutputProtection(uint32_t mask) = 0;
  virtual void QueryOutputProtectionStatus() = 0;
  virtual void OnDeferredInitializationDone(cdm::StreamType type,
                                            cdm::Status status) = 0;
  virtual cdm::FileIO* CreateFileIO(cdm::FileIOClient* client) = 0;
  virtual void RequestStorageId(uint32_t version) = 0;
};

template <typename H>
class HostImpl final : public HostRef {
 public:
  explicit HostImpl(H* h) : mH(h) {}
  cdm::Buffer* Allocate(uint32_t c) override { return mH->Allocate(c); }
  void SetTimer(int64_t d, void* ctx) override { mH->SetTimer(d, ctx); }
  void OnInitialized(bool ok) override { mH->OnInitialized(ok); }
  void OnResolveKeyStatusPromise(uint32_t id, cdm::KeyStatus s) override {
    mH->OnResolveKeyStatusPromise(id, s);
  }
  void OnResolveNewSessionPromise(uint32_t id, const char* sid,
                                  uint32_t len) override {
    mH->OnResolveNewSessionPromise(id, sid, len);
  }
  void OnResolvePromise(uint32_t id) override { mH->OnResolvePromise(id); }
  void OnRejectPromise(uint32_t id, cdm::Exception e, uint32_t code,
                       const char* msg, uint32_t len) override {
    mH->OnRejectPromise(id, e, code, msg, len);
  }
  void OnSessionMessage(const char* sid, uint32_t sidlen, cdm::MessageType t,
                        const char* msg, uint32_t len) override {
    mH->OnSessionMessage(sid, sidlen, t, msg, len);
  }
  void OnSessionKeysChange(const char* sid, uint32_t sidlen, bool more,
                           const cdm::KeyInformation* keys,
                           uint32_t count) override {
    mH->OnSessionKeysChange(sid, sidlen, more, keys, count);
  }
  void OnExpirationChange(const char* sid, uint32_t len, cdm::Time t) override {
    mH->OnExpirationChange(sid, len, t);
  }
  void OnSessionClosed(const char* sid, uint32_t len) override {
    mH->OnSessionClosed(sid, len);
  }
  void SendPlatformChallenge(const char* s, uint32_t sl, const char* c,
                             uint32_t cl) override {
    mH->SendPlatformChallenge(s, sl, c, cl);
  }
  void EnableOutputProtection(uint32_t mask) override {
    mH->EnableOutputProtection(mask);
  }
  void QueryOutputProtectionStatus() override {
    mH->QueryOutputProtectionStatus();
  }
  void OnDeferredInitializationDone(cdm::StreamType t, cdm::Status s) override {
    mH->OnDeferredInitializationDone(t, s);
  }
  cdm::FileIO* CreateFileIO(cdm::FileIOClient* c) override {
    return mH->CreateFileIO(c);
  }
  void RequestStorageId(uint32_t v) override { mH->RequestStorageId(v); }

 private:
  H* mH;
};

class ProxyCdm;
class Connection;

// Forwards Firefox's FileIO completions for one helper-side FileIO.
class ProxyFileClient final : public cdm::FileIOClient {
 public:
  ProxyFileClient(ProxyCdm* cdm, uint32_t id) : mCdm(cdm), mId(id) {}
  void OnOpenComplete(Status status) override;
  void OnReadComplete(Status status, const uint8_t* data,
                      uint32_t size) override;
  void OnWriteComplete(Status status) override;

  cdm::FileIO* mFile = nullptr;

 private:
  ProxyCdm* mCdm;
  uint32_t mId;
};

constexpr uint32_t kMaxFiles = 64;

// A request raised while a call to the helper is in progress (a synchronous
// FileIO completion, for instance); it is sent once that call has returned.
struct PendingRequest {
  ProxyCdm* cdm;
  uint32_t type;
  Writer* payload;
  PendingRequest* next;
};

template <typename T, typename... Args>
T* New(Args... args) {
  return new (malloc(sizeof(T))) T(args...);
}

template <typename T>
void Delete(T* p) {
  if (p) {
    p->~T();
    free(p);
  }
}

const char* Env(const char* name, const char* fallback) {
  const char* v = getenv(name);
  return v && *v ? v : fallback;
}

// Directory this library was loaded from, with a trailing slash.
void SelfDir(char* buf, size_t n) {
  Dl_info info;
  buf[0] = 0;
  if (dladdr(reinterpret_cast<void*>(&SelfDir), &info) && info.dli_fname) {
    const char* slash = strrchr(info.dli_fname, '/');
    size_t len = slash ? size_t(slash - info.dli_fname + 1) : 0;
    if (len < n) {
      memcpy(buf, info.dli_fname, len);
      buf[len] = 0;
    }
  }
}

void PathFrom(char* buf, size_t n, const char* env, const char* dir,
              const char* leaf) {
  const char* v = getenv(env);
  if (v && *v) {
    snprintf(buf, n, "%s", v);
  } else {
    snprintf(buf, n, "%s%s", dir, leaf);
  }
}

// The socket to one helper, shared by the CDM instances it serves.
class Connection {
 public:
  // The helper Firefox started, or null outside Firefox.
  static Connection* Shared();
  // A helper of our own under box64; null if it could not be started.
  static Connection* Spawn();

  Connection(int fd, int pid) : mFd(fd), mPid(pid) {}
  ~Connection() {
    if (mFd >= 0) close(mFd);
    if (mPid > 0) {
      int status;
      waitpid(mPid, &status, 0);
    }
    free(mBuf);
  }

  // Sends one request for |who| and serves the helper's callbacks until its
  // kDone, whose payload the returned Reader covers (until the next call).
  Reader Call(ProxyCdm* who, uint32_t type, const Writer& w);
  // Sends a request now, or queues it while a call is in progress.
  void Request(ProxyCdm* who, uint32_t type, Writer* w);
  void Flush();

 private:
  void Dispatch(uint32_t type, Reader& r);
  HostRef* Host();

  int mFd;
  int mPid;
  bool mDead = false;
  ProxyCdm* mInCall = nullptr;
  uint8_t* mBuf = nullptr;
  uint32_t mCap = 0, mLen = 0;
  PendingRequest* mPending = nullptr;
  PendingRequest** mPendingTail = &mPending;
  ProxyFileClient* mFiles[kMaxFiles] = {};
};

class ProxyCdm final : public cdm::ContentDecryptionModule_10,
                       public cdm::ContentDecryptionModule_11 {
 public:
  ProxyCdm(HostRef* host, Connection* conn, bool ownsConn)
      : mHost(host), mConn(conn), mOwnsConn(ownsConn) {}
  ~ProxyCdm() {
    if (mOwnsConn) Delete(mConn);
    Delete(mHost);
  }

  bool Create(const char* keySystem, uint32_t len) {
    Writer w;
    w.Bytes(keySystem, len);
    Reader r = mConn->Call(this, kCreate, w);
    bool ok = r.U32() != 0;
    mId = r.U32();
    return ok && r.Ok();
  }

  HostRef* Host() const { return mHost; }
  uint32_t Id() const { return mId; }

  // cdm::ContentDecryptionModule_10 and _11 (identical method sets).
  void Initialize(bool distinctive, bool persistent, bool hw) override {
    Writer w;
    w.U32(distinctive);
    w.U32(persistent);
    w.U32(hw);
    Call(kInitialize, w);
  }
  void GetStatusForPolicy(uint32_t id, const cdm::Policy& policy) override {
    Writer w;
    w.U32(id);
    w.U32(policy.min_hdcp_version);
    Call(kGetStatusForPolicy, w);
  }
  void SetServerCertificate(uint32_t id, const uint8_t* cert,
                            uint32_t len) override {
    Writer w;
    w.U32(id);
    w.Bytes(cert, len);
    Call(kSetServerCertificate, w);
  }
  void CreateSessionAndGenerateRequest(uint32_t id, cdm::SessionType st,
                                       cdm::InitDataType it,
                                       const uint8_t* data,
                                       uint32_t len) override {
    Writer w;
    w.U32(id);
    w.U32(st);
    w.U32(it);
    w.Bytes(data, len);
    Call(kCreateSessionAndGenerateRequest, w);
  }
  void LoadSession(uint32_t id, cdm::SessionType st, const char* sid,
                   uint32_t len) override {
    Writer w;
    w.U32(id);
    w.U32(st);
    w.Bytes(sid, len);
    Call(kLoadSession, w);
  }
  void UpdateSession(uint32_t id, const char* sid, uint32_t sidlen,
                     const uint8_t* resp, uint32_t len) override {
    Writer w;
    w.U32(id);
    w.Bytes(sid, sidlen);
    w.Bytes(resp, len);
    Call(kUpdateSession, w);
  }
  void CloseSession(uint32_t id, const char* sid, uint32_t len) override {
    SessionCall(kCloseSession, id, sid, len);
  }
  void RemoveSession(uint32_t id, const char* sid, uint32_t len) override {
    SessionCall(kRemoveSession, id, sid, len);
  }
  void TimerExpired(void* context) override {
    Writer* w = New<Writer>();
    w->U64(reinterpret_cast<uint64_t>(context));
    mConn->Request(this, kTimerExpired, w);
  }
  cdm::Status Decrypt(const cdm::InputBuffer_2& in,
                      cdm::DecryptedBlock* block) override {
    Writer w;
    WriteInput(w, in);
    Reader r = mConn->Call(this, kDecrypt, w);
    auto st = cdm::Status(r.U32());
    uint32_t n;
    const uint8_t* data = r.Bytes(&n);
    int64_t ts = r.I64();
    if (!r.Ok()) st = cdm::kDecryptError;
    if (st == cdm::kSuccess) {
      block->SetDecryptedBuffer(CopyOut(data, n));
      block->SetTimestamp(ts);
    }
    mConn->Flush();
    return st;
  }
  cdm::Status InitializeAudioDecoder(
      const cdm::AudioDecoderConfig_2& cfg) override {
    Writer w;
    w.U32(cfg.codec);
    w.U32(uint32_t(cfg.channel_count));
    w.U32(uint32_t(cfg.bits_per_channel));
    w.U32(uint32_t(cfg.samples_per_second));
    w.Bytes(cfg.extra_data, cfg.extra_data_size);
    w.U32(uint32_t(cfg.encryption_scheme));
    return StatusCall(kInitializeAudioDecoder, w);
  }
  cdm::Status InitializeVideoDecoder(
      const cdm::VideoDecoderConfig_2& cfg) override {
    Writer w;
    w.U32(cfg.codec);
    w.U32(cfg.profile);
    w.U32(cfg.format);
    w.U32(uint32_t(cfg.coded_size.width));
    w.U32(uint32_t(cfg.coded_size.height));
    w.Bytes(cfg.extra_data, cfg.extra_data_size);
    w.U32(uint32_t(cfg.encryption_scheme));
    return StatusCall(kInitializeVideoDecoder, w);
  }
  void DeinitializeDecoder(cdm::StreamType type) override {
    Writer w;
    w.U32(type);
    Call(kDeinitializeDecoder, w);
  }
  void ResetDecoder(cdm::StreamType type) override {
    Writer w;
    w.U32(type);
    Call(kResetDecoder, w);
  }
  cdm::Status DecryptAndDecodeFrame(const cdm::InputBuffer_2& in,
                                    cdm::VideoFrame* frame) override {
    Writer w;
    WriteInput(w, in);
    Reader r = mConn->Call(this, kDecryptAndDecodeFrame, w);
    auto st = cdm::Status(r.U32());
    auto format = cdm::VideoFormat(r.U32());
    cdm::Size size;
    size.width = int32_t(r.U32());
    size.height = int32_t(r.U32());
    uint32_t offset[3], stride[3];
    for (int i = 0; i < 3; i++) {
      offset[i] = r.U32();
      stride[i] = r.U32();
    }
    int64_t ts = r.I64();
    uint32_t n;
    const uint8_t* data = r.Bytes(&n);
    if (!r.Ok()) st = cdm::kDecodeError;
    if (st == cdm::kSuccess) {
      frame->SetFormat(format);
      frame->SetSize(size);
      frame->SetFrameBuffer(CopyOut(data, n));
      for (int i = 0; i < 3; i++) {
        frame->SetPlaneOffset(cdm::VideoPlane(i), offset[i]);
        frame->SetStride(cdm::VideoPlane(i), stride[i]);
      }
      frame->SetTimestamp(ts);
    }
    mConn->Flush();
    return st;
  }
  cdm::Status DecryptAndDecodeSamples(const cdm::InputBuffer_2& in,
                                      cdm::AudioFrames* frames) override {
    Writer w;
    WriteInput(w, in);
    Reader r = mConn->Call(this, kDecryptAndDecodeSamples, w);
    auto st = cdm::Status(r.U32());
    auto format = cdm::AudioFormat(r.U32());
    uint32_t n;
    const uint8_t* data = r.Bytes(&n);
    if (!r.Ok()) st = cdm::kDecodeError;
    if (st == cdm::kSuccess) {
      frames->SetFormat(format);
      frames->SetFrameBuffer(CopyOut(data, n));
    }
    mConn->Flush();
    return st;
  }
  void OnPlatformChallengeResponse(
      const cdm::PlatformChallengeResponse& resp) override {
    Writer* w = New<Writer>();
    w->Bytes(resp.signed_data, resp.signed_data_length);
    w->Bytes(resp.signed_data_signature, resp.signed_data_signature_length);
    w->Bytes(resp.platform_key_certificate,
             resp.platform_key_certificate_length);
    mConn->Request(this, kOnPlatformChallengeResponse, w);
  }
  void OnQueryOutputProtectionStatus(cdm::QueryResult result, uint32_t links,
                                     uint32_t mask) override {
    Writer* w = New<Writer>();
    w->U32(result);
    w->U32(links);
    w->U32(mask);
    mConn->Request(this, kOnQueryOutputProtectionStatus, w);
  }
  void OnStorageId(uint32_t version, const uint8_t* id, uint32_t len) override {
    Writer* w = New<Writer>();
    w->U32(version);
    w->Bytes(id, len);
    mConn->Request(this, kOnStorageId, w);
  }
  void Destroy() override {
    Writer w;
    Call(kDestroy, w);
    Delete(this);
  }

  // Called by ProxyFileClient.
  void FileComplete(uint32_t type, uint32_t id, uint32_t status,
                    const uint8_t* data, uint32_t len) {
    Writer* w = New<Writer>();
    w->U32(id);
    w->U32(status);
    if (type == kFileReadComplete) w->Bytes(data, len);
    mConn->Request(this, type, w);
  }

 private:
  void Call(uint32_t type, const Writer& w) {
    mConn->Call(this, type, w);
    mConn->Flush();
  }
  void SessionCall(uint32_t type, uint32_t id, const char* sid, uint32_t len) {
    Writer w;
    w.U32(id);
    w.Bytes(sid, len);
    Call(type, w);
  }
  cdm::Status StatusCall(uint32_t type, const Writer& w) {
    Reader r = mConn->Call(this, type, w);
    auto st = cdm::Status(r.U32());
    if (!r.Ok()) st = cdm::kInitializationError;
    mConn->Flush();
    return st;
  }
  cdm::Buffer* CopyOut(const uint8_t* data, uint32_t n) {
    cdm::Buffer* buf = mHost->Allocate(n);
    if (!buf) return nullptr;
    if (n) memcpy(buf->Data(), data, n);
    buf->SetSize(n);
    return buf;
  }

  HostRef* mHost;
  Connection* mConn;
  bool mOwnsConn;
  uint32_t mId = kNoInstance;
};

void ProxyFileClient::OnOpenComplete(Status status) {
  mCdm->FileComplete(kFileOpenComplete, mId, uint32_t(status), nullptr, 0);
}

void ProxyFileClient::OnReadComplete(Status status, const uint8_t* data,
                                     uint32_t size) {
  mCdm->FileComplete(kFileReadComplete, mId, uint32_t(status), data, size);
}

void ProxyFileClient::OnWriteComplete(Status status) {
  mCdm->FileComplete(kFileWriteComplete, mId, uint32_t(status), nullptr, 0);
}

Connection* Connection::Shared() {
  static Connection* shared = nullptr;
  static bool looked = false;
  if (!looked) {
    looked = true;
    const char* v = getenv("MOZ_GMP_HELPER_FD");
    if (v && *v) {
      int fd = 0;
      for (const char* p = v; *p >= '0' && *p <= '9'; p++) {
        fd = fd * 10 + (*p - '0');
      }
      shared = New<Connection>(fd, 0);
    }
  }
  return shared;
}

Connection* Connection::Spawn() {
  char dir[1024], box64[1024], host[1024], realCdm[1024], x64lib[1024];
  SelfDir(dir, sizeof(dir));
  snprintf(box64, sizeof(box64), "%s", Env("WV_BOX64", "box64"));
  PathFrom(host, sizeof(host), "WV_HOST", dir, "wvhost");
  PathFrom(realCdm, sizeof(realCdm), "WV_REAL_CDM", dir,
           "libwidevinecdm-x86_64.so");
  snprintf(x64lib, sizeof(x64lib), "%sx64lib", dir);

  int fds[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) return nullptr;
  char fdstr[16];
  snprintf(fdstr, sizeof(fdstr), "%d", fds[1]);

  int pid = fork();
  if (pid < 0) {
    close(fds[0]);
    close(fds[1]);
    return nullptr;
  }
  if (pid == 0) {
    close(fds[0]);
    if (!getenv("BOX64_LD_LIBRARY_PATH")) {
      setenv("BOX64_LD_LIBRARY_PATH", x64lib, 1);
    }
    setenv("BOX64_NOBANNER", "1", 0);
    char* argv[] = {box64, host, realCdm, fdstr, nullptr};
    execvp(box64, argv);
    _exit(127);
  }
  close(fds[1]);
  fcntl(fds[0], F_SETFD, FD_CLOEXEC);
  return New<Connection>(fds[0], pid);
}

HostRef* Connection::Host() { return mInCall ? mInCall->Host() : nullptr; }

Reader Connection::Call(ProxyCdm* who, uint32_t type, const Writer& w) {
  uint32_t instance = type == kCreate ? kNoInstance : who->Id();
  if (mDead || !Send(mFd, type, instance, w)) {
    mDead = true;
    return Reader(nullptr, 0);
  }
  ProxyCdm* outer = mInCall;
  mInCall = who;
  uint32_t t, from;
  for (;;) {
    if (!Receive(mFd, &t, &from, &mBuf, &mCap, &mLen)) {
      mDead = true;
      mLen = 0;
      break;
    }
    if (t == kDone) break;
    Reader r(mBuf, mLen);
    Dispatch(t, r);
  }
  mInCall = outer;
  return Reader(mBuf, mLen);
}

void Connection::Request(ProxyCdm* who, uint32_t type, Writer* w) {
  if (mInCall) {
    PendingRequest* p = New<PendingRequest>();
    p->cdm = who;
    p->type = type;
    p->payload = w;
    p->next = nullptr;
    *mPendingTail = p;
    mPendingTail = &p->next;
    return;
  }
  Call(who, type, *w);
  Delete(w);
  Flush();
}

void Connection::Flush() {
  while (mPending && !mInCall) {
    PendingRequest* p = mPending;
    mPending = p->next;
    if (!mPending) mPendingTail = &mPending;
    Call(p->cdm, p->type, *p->payload);
    Delete(p->payload);
    free(p);
  }
}

void Connection::Dispatch(uint32_t type, Reader& r) {
  HostRef* host = Host();
  if (!host) return;
  uint32_t n, m;
  const uint8_t *a, *b;
  switch (type) {
    case kSetTimer: {
      int64_t delay = r.I64();
      uint64_t ctx = r.U64();
      host->SetTimer(delay, reinterpret_cast<void*>(ctx));
      break;
    }
    case kOnInitialized:
      host->OnInitialized(r.U32() != 0);
      break;
    case kOnResolveKeyStatusPromise: {
      uint32_t id = r.U32();
      host->OnResolveKeyStatusPromise(id, cdm::KeyStatus(r.U32()));
      break;
    }
    case kOnResolveNewSessionPromise: {
      uint32_t id = r.U32();
      a = r.Bytes(&n);
      host->OnResolveNewSessionPromise(id, reinterpret_cast<const char*>(a), n);
      break;
    }
    case kOnResolvePromise:
      host->OnResolvePromise(r.U32());
      break;
    case kOnRejectPromise: {
      uint32_t id = r.U32();
      auto e = cdm::Exception(r.U32());
      uint32_t code = r.U32();
      a = r.Bytes(&n);
      host->OnRejectPromise(id, e, code, reinterpret_cast<const char*>(a), n);
      break;
    }
    case kOnSessionMessage: {
      a = r.Bytes(&n);
      auto t = cdm::MessageType(r.U32());
      b = r.Bytes(&m);
      host->OnSessionMessage(reinterpret_cast<const char*>(a), n, t,
                             reinterpret_cast<const char*>(b), m);
      break;
    }
    case kOnSessionKeysChange: {
      a = r.Bytes(&n);
      bool more = r.U32() != 0;
      uint32_t count = r.U32();
      if (count > 4096) count = 0;
      auto* keys = static_cast<cdm::KeyInformation*>(
          malloc((count ? count : 1) * sizeof(cdm::KeyInformation)));
      for (uint32_t i = 0; i < count; i++) {
        keys[i].key_id = r.Bytes(&keys[i].key_id_size);
        keys[i].status = cdm::KeyStatus(r.U32());
        keys[i].system_code = r.U32();
      }
      if (r.Ok()) {
        host->OnSessionKeysChange(reinterpret_cast<const char*>(a), n, more,
                                  keys, count);
      }
      free(keys);
      break;
    }
    case kOnExpirationChange: {
      a = r.Bytes(&n);
      double t = r.F64();
      host->OnExpirationChange(reinterpret_cast<const char*>(a), n, t);
      break;
    }
    case kOnSessionClosed:
      a = r.Bytes(&n);
      host->OnSessionClosed(reinterpret_cast<const char*>(a), n);
      break;
    case kSendPlatformChallenge:
      a = r.Bytes(&n);
      b = r.Bytes(&m);
      host->SendPlatformChallenge(reinterpret_cast<const char*>(a), n,
                                  reinterpret_cast<const char*>(b), m);
      break;
    case kEnableOutputProtection:
      host->EnableOutputProtection(r.U32());
      break;
    case kQueryOutputProtectionStatus:
      host->QueryOutputProtectionStatus();
      break;
    case kOnDeferredInitializationDone: {
      auto t = cdm::StreamType(r.U32());
      host->OnDeferredInitializationDone(t, cdm::Status(r.U32()));
      break;
    }
    case kRequestStorageId:
      host->RequestStorageId(r.U32());
      break;
    case kFileCreate: {
      uint32_t id = r.U32();
      if (id < kMaxFiles && !mFiles[id]) {
        mFiles[id] = New<ProxyFileClient>(mInCall, id);
        mFiles[id]->mFile = host->CreateFileIO(mFiles[id]);
      }
      break;
    }
    case kFileOpen:
    case kFileRead:
    case kFileWrite:
    case kFileClose: {
      uint32_t id = r.U32();
      ProxyFileClient* client = id < kMaxFiles ? mFiles[id] : nullptr;
      if (!client) break;
      cdm::FileIO* file = client->mFile;
      if (type == kFileClose) {
        mFiles[id] = nullptr;
        if (file) file->Close();
        Delete(client);
        break;
      }
      if (!file) {
        uint32_t done = type == kFileOpen   ? kFileOpenComplete
                        : type == kFileRead ? kFileReadComplete
                                            : kFileWriteComplete;
        mInCall->FileComplete(
            done, id, uint32_t(cdm::FileIOClient::Status::kError), nullptr, 0);
        break;
      }
      if (type == kFileOpen) {
        a = r.Bytes(&n);
        file->Open(reinterpret_cast<const char*>(a), n);
      } else if (type == kFileRead) {
        file->Read();
      } else {
        a = r.Bytes(&n);
        file->Write(a, n);
      }
      break;
    }
    default:
      break;
  }
}

}  // namespace

WV_EXPORT void InitializeCdmModule_4() {}

WV_EXPORT void DeinitializeCdmModule() {}

WV_EXPORT const char* GetCdmVersion() { return "wvproxy"; }

WV_EXPORT void* CreateCdmInstance(int version, const char* keySystem,
                                  uint32_t len, GetCdmHostFunc getHost,
                                  void* userData) {
  HostRef* host = nullptr;
  if (version == cdm::ContentDecryptionModule_11::kVersion) {
    auto* h =
        static_cast<cdm::Host_11*>(getHost(cdm::Host_11::kVersion, userData));
    if (h) host = New<HostImpl<cdm::Host_11>>(h);
  } else if (version == cdm::ContentDecryptionModule_10::kVersion) {
    auto* h =
        static_cast<cdm::Host_10*>(getHost(cdm::Host_10::kVersion, userData));
    if (h) host = New<HostImpl<cdm::Host_10>>(h);
  }
  if (!host) return nullptr;

  bool owned = false;
  Connection* conn = Connection::Shared();
  if (!conn) {
    conn = Connection::Spawn();
    owned = true;
  }
  if (!conn) {
    Delete(host);
    return nullptr;
  }
  ProxyCdm* cdm = New<ProxyCdm>(host, conn, owned);
  if (!cdm->Create(keySystem, len)) {
    Delete(cdm);
    return nullptr;
  }
  if (version == cdm::ContentDecryptionModule_11::kVersion) {
    return static_cast<cdm::ContentDecryptionModule_11*>(cdm);
  }
  return static_cast<cdm::ContentDecryptionModule_10*>(cdm);
}
