/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Serialisation of cdm::InputBuffer_2, shared by the proxy and the helper.
#pragma once
#include "wvipc.h"
#include "content_decryption_module.h"

namespace wvipc {

inline void WriteInput(Writer& w, const cdm::InputBuffer_2& in) {
  w.Bytes(in.data, in.data_size);
  w.U32(uint32_t(in.encryption_scheme));
  w.Bytes(in.key_id, in.key_id_size);
  w.Bytes(in.iv, in.iv_size);
  w.U32(in.num_subsamples);
  for (uint32_t i = 0; i < in.num_subsamples; i++) {
    w.U32(in.subsamples[i].clear_bytes);
    w.U32(in.subsamples[i].cipher_bytes);
  }
  w.U32(in.pattern.crypt_byte_block);
  w.U32(in.pattern.skip_byte_block);
  w.I64(in.timestamp);
}

// |*subs| is grown as needed and owned by the caller. The pointers in |in|
// point into the message payload.
inline bool ReadInput(Reader& r, cdm::InputBuffer_2& in,
                      cdm::SubsampleEntry** subs, uint32_t* subsCap) {
  in = {};
  in.data = r.Bytes(&in.data_size);
  in.encryption_scheme = cdm::EncryptionScheme(r.U32());
  in.key_id = r.Bytes(&in.key_id_size);
  in.iv = r.Bytes(&in.iv_size);
  uint32_t n = r.U32();
  if (n > 65536) return false;
  if (n > *subsCap) {
    *subsCap = n;
    *subs = static_cast<cdm::SubsampleEntry*>(
        realloc(*subs, n * sizeof(cdm::SubsampleEntry)));
  }
  for (uint32_t i = 0; i < n; i++) {
    (*subs)[i].clear_bytes = r.U32();
    (*subs)[i].cipher_bytes = r.U32();
  }
  in.subsamples = n ? *subs : nullptr;
  in.num_subsamples = n;
  in.pattern.crypt_byte_block = r.U32();
  in.pattern.skip_byte_block = r.U32();
  in.timestamp = r.I64();
  return r.Ok();
}

}  // namespace wvipc
