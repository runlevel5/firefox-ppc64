/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// x86_64 helper run under box64 for the native proxy CDM (wvproxy.cc). It
// loads the real CDM, executes the proxy's requests on it and forwards the
// CDM's Host_10 callbacks over the socket. It serves every CDM instance the
// proxy creates and exits when the socket closes. See wvipc.h for the
// protocol.
//
// usage: wvhost <libwidevinecdm.so> <socket fd>

#include "wvinput.h"

using namespace wvipc;

extern "C" void __cxa_pure_virtual() { _exit(70); }

namespace {

int gFd = -1;
uint32_t gInstance = kNoInstance;

void Fail(const char* what) {
  printf("wvhost: %s\n", what);
  fflush(nullptr);
  _exit(1);
}

void Callback(uint32_t type, const Writer& w) {
  if (!Send(gFd, type, gInstance, w)) Fail("lost the proxy");
}

void Callback(uint32_t type) {
  Writer w;
  Callback(type, w);
}

class LocalBuffer final : public cdm::Buffer {
 public:
  explicit LocalBuffer(uint32_t cap)
      : mData(static_cast<uint8_t*>(malloc(cap ? cap : 1))), mCap(cap) {}
  void Destroy() override {
    free(mData);
    free(this);
  }
  uint32_t Capacity() const override { return mCap; }
  uint8_t* Data() override { return mData; }
  void SetSize(uint32_t size) override { mSize = size; }
  uint32_t Size() const override { return mSize; }

 private:
  uint8_t* mData;
  uint32_t mCap;
  uint32_t mSize = 0;
};

class LocalDecryptedBlock final : public cdm::DecryptedBlock {
 public:
  void SetDecryptedBuffer(cdm::Buffer* buffer) override { mBuffer = buffer; }
  cdm::Buffer* DecryptedBuffer() override { return mBuffer; }
  void SetTimestamp(int64_t timestamp) override { mTimestamp = timestamp; }
  int64_t Timestamp() const override { return mTimestamp; }

 private:
  cdm::Buffer* mBuffer = nullptr;
  int64_t mTimestamp = 0;
};

class LocalVideoFrame final : public cdm::VideoFrame {
 public:
  void SetFormat(cdm::VideoFormat format) override { mFormat = format; }
  cdm::VideoFormat Format() const override { return mFormat; }
  void SetSize(cdm::Size size) override { mSize = size; }
  cdm::Size Size() const override { return mSize; }
  void SetFrameBuffer(cdm::Buffer* buffer) override { mBuffer = buffer; }
  cdm::Buffer* FrameBuffer() override { return mBuffer; }
  void SetPlaneOffset(cdm::VideoPlane plane, uint32_t offset) override {
    mOffset[plane] = offset;
  }
  uint32_t PlaneOffset(cdm::VideoPlane plane) override {
    return mOffset[plane];
  }
  void SetStride(cdm::VideoPlane plane, uint32_t stride) override {
    mStride[plane] = stride;
  }
  uint32_t Stride(cdm::VideoPlane plane) override { return mStride[plane]; }
  void SetTimestamp(int64_t timestamp) override { mTimestamp = timestamp; }
  int64_t Timestamp() const override { return mTimestamp; }

  cdm::VideoFormat mFormat = cdm::kUnknownVideoFormat;
  cdm::Size mSize = {0, 0};
  cdm::Buffer* mBuffer = nullptr;
  uint32_t mOffset[3] = {0, 0, 0};
  uint32_t mStride[3] = {0, 0, 0};
  int64_t mTimestamp = 0;
};

class LocalAudioFrames final : public cdm::AudioFrames {
 public:
  void SetFrameBuffer(cdm::Buffer* buffer) override { mBuffer = buffer; }
  cdm::Buffer* FrameBuffer() override { return mBuffer; }
  void SetFormat(cdm::AudioFormat format) override { mFormat = format; }
  cdm::AudioFormat Format() const override { return mFormat; }

  cdm::Buffer* mBuffer = nullptr;
  cdm::AudioFormat mFormat = cdm::kUnknownAudioFormat;
};

// The CDM's storage, backed by a FileIO the proxy creates on Firefox's host.
constexpr uint32_t kMaxFiles = 64;
cdm::FileIOClient* gFileClients[kMaxFiles];

class RemoteFileIO final : public cdm::FileIO {
 public:
  explicit RemoteFileIO(uint32_t id) : mId(id) {}
  void Open(const char* name, uint32_t size) override {
    Writer w;
    w.U32(mId);
    w.Bytes(name, size);
    Callback(kFileOpen, w);
  }
  void Read() override {
    Writer w;
    w.U32(mId);
    Callback(kFileRead, w);
  }
  void Write(const uint8_t* data, uint32_t size) override {
    Writer w;
    w.U32(mId);
    w.Bytes(data, size);
    Callback(kFileWrite, w);
  }
  void Close() override {
    Writer w;
    w.U32(mId);
    Callback(kFileClose, w);
    gFileClients[mId] = nullptr;
    free(this);
  }

 private:
  uint32_t mId;
};

class HostProxy final : public cdm::Host_10 {
 public:
  cdm::Buffer* Allocate(uint32_t capacity) override {
    return new (malloc(sizeof(LocalBuffer))) LocalBuffer(capacity);
  }
  void SetTimer(int64_t delay_ms, void* context) override {
    Writer w;
    w.I64(delay_ms);
    w.U64(reinterpret_cast<uint64_t>(context));
    Callback(kSetTimer, w);
  }
  cdm::Time GetCurrentWallTime() override {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
  }
  void OnInitialized(bool success) override {
    Writer w;
    w.U32(success);
    Callback(kOnInitialized, w);
  }
  void OnResolveKeyStatusPromise(uint32_t id, cdm::KeyStatus s) override {
    Writer w;
    w.U32(id);
    w.U32(s);
    Callback(kOnResolveKeyStatusPromise, w);
  }
  void OnResolveNewSessionPromise(uint32_t id, const char* sid,
                                  uint32_t len) override {
    Writer w;
    w.U32(id);
    w.Bytes(sid, len);
    Callback(kOnResolveNewSessionPromise, w);
  }
  void OnResolvePromise(uint32_t id) override {
    Writer w;
    w.U32(id);
    Callback(kOnResolvePromise, w);
  }
  void OnRejectPromise(uint32_t id, cdm::Exception e, uint32_t code,
                       const char* msg, uint32_t len) override {
    Writer w;
    w.U32(id);
    w.U32(e);
    w.U32(code);
    w.Bytes(msg, len);
    Callback(kOnRejectPromise, w);
  }
  void OnSessionMessage(const char* sid, uint32_t sidlen, cdm::MessageType t,
                        const char* msg, uint32_t len) override {
    Writer w;
    w.Bytes(sid, sidlen);
    w.U32(t);
    w.Bytes(msg, len);
    Callback(kOnSessionMessage, w);
  }
  void OnSessionKeysChange(const char* sid, uint32_t sidlen, bool more,
                           const cdm::KeyInformation* keys,
                           uint32_t count) override {
    Writer w;
    w.Bytes(sid, sidlen);
    w.U32(more);
    w.U32(count);
    for (uint32_t i = 0; i < count; i++) {
      w.Bytes(keys[i].key_id, keys[i].key_id_size);
      w.U32(keys[i].status);
      w.U32(keys[i].system_code);
    }
    Callback(kOnSessionKeysChange, w);
  }
  void OnExpirationChange(const char* sid, uint32_t len, cdm::Time t) override {
    Writer w;
    w.Bytes(sid, len);
    w.F64(t);
    Callback(kOnExpirationChange, w);
  }
  void OnSessionClosed(const char* sid, uint32_t len) override {
    Writer w;
    w.Bytes(sid, len);
    Callback(kOnSessionClosed, w);
  }
  void SendPlatformChallenge(const char* service, uint32_t serviceLen,
                             const char* challenge,
                             uint32_t challengeLen) override {
    Writer w;
    w.Bytes(service, serviceLen);
    w.Bytes(challenge, challengeLen);
    Callback(kSendPlatformChallenge, w);
  }
  void EnableOutputProtection(uint32_t mask) override {
    Writer w;
    w.U32(mask);
    Callback(kEnableOutputProtection, w);
  }
  void QueryOutputProtectionStatus() override {
    Callback(kQueryOutputProtectionStatus);
  }
  void OnDeferredInitializationDone(cdm::StreamType type,
                                    cdm::Status status) override {
    Writer w;
    w.U32(type);
    w.U32(status);
    Callback(kOnDeferredInitializationDone, w);
  }
  cdm::FileIO* CreateFileIO(cdm::FileIOClient* client) override {
    for (uint32_t id = 0; id < kMaxFiles; id++) {
      if (!gFileClients[id]) {
        gFileClients[id] = client;
        Writer w;
        w.U32(id);
        Callback(kFileCreate, w);
        return new (malloc(sizeof(RemoteFileIO))) RemoteFileIO(id);
      }
    }
    return nullptr;
  }
  void RequestStorageId(uint32_t version) override {
    Writer w;
    w.U32(version);
    Callback(kRequestStorageId, w);
  }
};

HostProxy* gHost;

void* GetHost(int version, void*) {
  return version == cdm::Host_10::kVersion ? gHost : nullptr;
}

cdm::FileIOClient* FileClient(uint32_t id) {
  return id < kMaxFiles ? gFileClients[id] : nullptr;
}

typedef void (*InitFn)();
typedef void* (*CreateFn)(int, const char*, uint32_t, GetCdmHostFunc, void*);

// Runs one request on the CDM and fills |out| with the kDone payload.
// Returns false once the CDM has been destroyed.
bool Handle(cdm::ContentDecryptionModule_10* cdm, uint32_t type, Reader& r,
            Writer& out) {
  static cdm::SubsampleEntry* subs;
  static uint32_t subsCap;
  uint32_t n, m, k;
  const uint8_t *a, *b, *c;
  switch (type) {
    case kInitialize: {
      uint32_t x = r.U32(), y = r.U32(), z = r.U32();
      cdm->Initialize(x, y, z);
      break;
    }
    case kGetStatusForPolicy: {
      uint32_t id = r.U32();
      cdm::Policy policy = {cdm::HdcpVersion(r.U32())};
      cdm->GetStatusForPolicy(id, policy);
      break;
    }
    case kSetServerCertificate: {
      uint32_t id = r.U32();
      a = r.Bytes(&n);
      cdm->SetServerCertificate(id, a, n);
      break;
    }
    case kCreateSessionAndGenerateRequest: {
      uint32_t id = r.U32();
      auto st = cdm::SessionType(r.U32());
      auto it = cdm::InitDataType(r.U32());
      a = r.Bytes(&n);
      cdm->CreateSessionAndGenerateRequest(id, st, it, a, n);
      break;
    }
    case kLoadSession: {
      uint32_t id = r.U32();
      auto st = cdm::SessionType(r.U32());
      a = r.Bytes(&n);
      cdm->LoadSession(id, st, reinterpret_cast<const char*>(a), n);
      break;
    }
    case kUpdateSession: {
      uint32_t id = r.U32();
      a = r.Bytes(&n);
      b = r.Bytes(&m);
      cdm->UpdateSession(id, reinterpret_cast<const char*>(a), n, b, m);
      break;
    }
    case kCloseSession:
    case kRemoveSession: {
      uint32_t id = r.U32();
      a = r.Bytes(&n);
      auto sid = reinterpret_cast<const char*>(a);
      if (type == kCloseSession) {
        cdm->CloseSession(id, sid, n);
      } else {
        cdm->RemoveSession(id, sid, n);
      }
      break;
    }
    case kTimerExpired:
      cdm->TimerExpired(reinterpret_cast<void*>(r.U64()));
      break;
    case kDecrypt: {
      cdm::InputBuffer_2 in;
      if (!ReadInput(r, in, &subs, &subsCap)) Fail("bad decrypt request");
      LocalDecryptedBlock block;
      cdm::Status st = cdm->Decrypt(in, &block);
      cdm::Buffer* buf = block.DecryptedBuffer();
      out.U32(st);
      out.Bytes(buf ? buf->Data() : nullptr, buf ? buf->Size() : 0);
      out.I64(block.Timestamp());
      if (buf) buf->Destroy();
      break;
    }
    case kInitializeAudioDecoder: {
      cdm::AudioDecoderConfig_2 cfg = {};
      cfg.codec = cdm::AudioCodec(r.U32());
      cfg.channel_count = int32_t(r.U32());
      cfg.bits_per_channel = int32_t(r.U32());
      cfg.samples_per_second = int32_t(r.U32());
      cfg.extra_data = const_cast<uint8_t*>(r.Bytes(&cfg.extra_data_size));
      cfg.encryption_scheme = cdm::EncryptionScheme(r.U32());
      out.U32(cdm->InitializeAudioDecoder(cfg));
      break;
    }
    case kInitializeVideoDecoder: {
      cdm::VideoDecoderConfig_2 cfg = {};
      cfg.codec = cdm::VideoCodec(r.U32());
      cfg.profile = cdm::VideoCodecProfile(r.U32());
      cfg.format = cdm::VideoFormat(r.U32());
      cfg.coded_size.width = int32_t(r.U32());
      cfg.coded_size.height = int32_t(r.U32());
      cfg.extra_data = const_cast<uint8_t*>(r.Bytes(&cfg.extra_data_size));
      cfg.encryption_scheme = cdm::EncryptionScheme(r.U32());
      out.U32(cdm->InitializeVideoDecoder(cfg));
      break;
    }
    case kDeinitializeDecoder:
      cdm->DeinitializeDecoder(cdm::StreamType(r.U32()));
      break;
    case kResetDecoder:
      cdm->ResetDecoder(cdm::StreamType(r.U32()));
      break;
    case kDecryptAndDecodeFrame: {
      cdm::InputBuffer_2 in;
      if (!ReadInput(r, in, &subs, &subsCap)) Fail("bad decode request");
      LocalVideoFrame frame;
      cdm::Status st = cdm->DecryptAndDecodeFrame(in, &frame);
      cdm::Buffer* buf = frame.mBuffer;
      out.U32(st);
      out.U32(frame.mFormat);
      out.U32(uint32_t(frame.mSize.width));
      out.U32(uint32_t(frame.mSize.height));
      for (int i = 0; i < 3; i++) {
        out.U32(frame.mOffset[i]);
        out.U32(frame.mStride[i]);
      }
      out.I64(frame.mTimestamp);
      out.Bytes(buf ? buf->Data() : nullptr, buf ? buf->Size() : 0);
      if (buf) buf->Destroy();
      break;
    }
    case kDecryptAndDecodeSamples: {
      cdm::InputBuffer_2 in;
      if (!ReadInput(r, in, &subs, &subsCap)) Fail("bad decode request");
      LocalAudioFrames frames;
      cdm::Status st = cdm->DecryptAndDecodeSamples(in, &frames);
      cdm::Buffer* buf = frames.mBuffer;
      out.U32(st);
      out.U32(frames.mFormat);
      out.Bytes(buf ? buf->Data() : nullptr, buf ? buf->Size() : 0);
      if (buf) buf->Destroy();
      break;
    }
    case kOnPlatformChallengeResponse: {
      cdm::PlatformChallengeResponse resp = {};
      a = r.Bytes(&n);
      b = r.Bytes(&m);
      c = r.Bytes(&k);
      resp.signed_data = a;
      resp.signed_data_length = n;
      resp.signed_data_signature = b;
      resp.signed_data_signature_length = m;
      resp.platform_key_certificate = c;
      resp.platform_key_certificate_length = k;
      cdm->OnPlatformChallengeResponse(resp);
      break;
    }
    case kOnQueryOutputProtectionStatus: {
      auto result = cdm::QueryResult(r.U32());
      uint32_t links = r.U32(), mask = r.U32();
      cdm->OnQueryOutputProtectionStatus(result, links, mask);
      break;
    }
    case kOnStorageId: {
      uint32_t version = r.U32();
      a = r.Bytes(&n);
      cdm->OnStorageId(version, a, n);
      break;
    }
    case kFileOpenComplete:
    case kFileWriteComplete: {
      cdm::FileIOClient* client = FileClient(r.U32());
      auto st = cdm::FileIOClient::Status(r.U32());
      if (client && type == kFileOpenComplete) client->OnOpenComplete(st);
      if (client && type == kFileWriteComplete) client->OnWriteComplete(st);
      break;
    }
    case kFileReadComplete: {
      cdm::FileIOClient* client = FileClient(r.U32());
      auto st = cdm::FileIOClient::Status(r.U32());
      a = r.Bytes(&n);
      if (client) client->OnReadComplete(st, a, n);
      break;
    }
    case kDestroy:
      cdm->Destroy();
      return false;
    default:
      Fail("unknown request");
  }
  if (!r.Ok()) Fail("short request");
  return true;
}

}  // namespace

extern "C" int main(int argc, char** argv) {
  if (argc != 3) {
    printf("usage: %s cdm.so fd\n", argv[0]);
    return 2;
  }
  gFd = 0;
  for (const char* p = argv[2]; *p; p++) gFd = gFd * 10 + (*p - '0');

  void* h = dlopen(argv[1], RTLD_NOW);
  if (!h) {
    printf("wvhost: dlopen %s failed: %s\n", argv[1], dlerror());
    return 1;
  }
  auto init = reinterpret_cast<InitFn>(dlsym(h, "InitializeCdmModule_4"));
  auto create = reinterpret_cast<CreateFn>(dlsym(h, "CreateCdmInstance"));
  if (!init || !create) Fail("missing CDM entry points");
  init();

  HostProxy host;
  gHost = &host;
  cdm::ContentDecryptionModule_10* cdms[kMaxInstances] = {};

  uint8_t* buf = nullptr;
  uint32_t cap = 0, len = 0, type = 0, instance = 0;
  while (Receive(gFd, &type, &instance, &buf, &cap, &len)) {
    Reader r(buf, len);
    Writer out;
    if (type == kCreate) {
      uint32_t ksLen;
      const uint8_t* ks = r.Bytes(&ksLen);
      uint32_t id = 0;
      while (id < kMaxInstances && cdms[id]) id++;
      if (id < kMaxInstances) {
        gInstance = id;
        cdms[id] = static_cast<cdm::ContentDecryptionModule_10*>(
            create(cdm::ContentDecryptionModule_10::kVersion,
                   reinterpret_cast<const char*>(ks), ksLen, GetHost, nullptr));
      }
      bool ok = id < kMaxInstances && cdms[id];
      out.U32(ok);
      out.U32(id);
      gInstance = kNoInstance;
      Send(gFd, kDone, kNoInstance, out);
      continue;
    }
    if (instance >= kMaxInstances || !cdms[instance]) Fail("bad instance");
    gInstance = instance;
    if (!Handle(cdms[instance], type, r, out)) cdms[instance] = nullptr;
    Send(gFd, kDone, instance, out);
    gInstance = kNoInstance;
  }
  return 0;
}
