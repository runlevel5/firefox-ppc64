# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
{
  'includes': [
    '../../coreconf/config.gypi'
  ],
  'target_defaults': {
    'type': 'static_library',
    'sources': [
      'gcm.c',
    ],
    'dependencies': [
      '<(DEPTH)/exports.gyp:nss_exports'
    ],
    'conditions': [
      [ 'target_arch=="ia32" or target_arch=="x64"', {
        'dependencies': [
          'ghash.gyp:ghash-aes-x86_c_lib',
        ],
        'defines': [
          'HAVE_PLATFORM_GHASH'
        ]
      }],
      [ 'disable_arm32_neon==0 and target_arch=="arm"', {
        'dependencies': [
          'ghash.gyp:ghash-aes-arm32-neon_c_lib',
        ],
        'defines': [
          'HAVE_PLATFORM_GHASH'
        ]
      }],
      [ 'target_arch=="arm64" or target_arch=="aarch64"', {
        'dependencies': [
          'ghash.gyp:ghash-aes-aarch64_c_lib',
        ],
        'defines': [
          'HAVE_PLATFORM_GHASH'
        ]
      }],
      # Only claim a platform GHASH when the implementation is actually
      # compiled: ghash-ppc.c multiplies with vpmsumd, a POWER8 instruction.
      # Defining HAVE_PLATFORM_GHASH unconditionally makes gcm.c drop its
      # stubs and reference symbols that do not exist.
      [ '(target_arch=="ppc64" or target_arch=="ppc64le") and disable_vec_crypto==0', {
        'dependencies': [
          'ghash.gyp:ghash-aes-ppc_c_lib',
        ],
        'defines': [
          'HAVE_PLATFORM_GHASH'
        ]
      }],
      # gcm.c must see the same __ALTIVEC__/__VSX__ visibility as
      # ghash-ppc.c: gcmHashContext's vec_u64 x/h fields are gated on
      # those macros, and a mismatch between translation units shifts
      # every field after them (including ghash_mul) to different
      # offsets, corrupting the hardware GHASH dispatch.
      [ 'target_arch=="ppc64" or target_arch=="ppc64le"', {
        'conditions': [
          [ 'disable_vec_crypto==0', {
            'cflags': [
              '-mcrypto',
              '-maltivec'
            ],
            'cflags_mozilla': [
              '-mcrypto',
              '-maltivec'
            ],
          }, 'disable_vec_crypto==1', {
            'cflags': [
              '-maltivec'
            ],
            'cflags_mozilla': [
              '-maltivec'
            ],
          }],
        ],
      }],
      [ 'OS=="linux"', {
        'defines': [
          'FREEBL_NO_DEPEND',
        ],
      }],
    ],
  },
  'targets': [
    {
      'target_name': 'gcm-nodepend',
      'conditions': [
        [ '(OS=="win" and cc_use_gnu_ld!=1 and (target_arch=="ia32" or target_arch=="x64")) or (target_arch=="x64" and OS!="win" and OS!="ios")', {
          'dependencies': [
            'intel-gcm-wrap.gyp:intel-gcm-wrap-nodepend_c_lib',
          ],
          'defines': [
            'HAVE_PLATFORM_GCM'
          ],
        }],
        # vcipher is POWER8, so this is only built when the target baseline
        # has it; ppc_crypto_support() still gates its use at run time.
        [ 'disable_altivec==0 and disable_vec_crypto==0 and (target_arch=="ppc64" or target_arch=="ppc64le")', {
          'dependencies': [
            'ppc-gcm-wrap.gyp:ppc-gcm-wrap-nodepend_c_lib',
          ],
          'defines': [
            'HAVE_PLATFORM_GCM'
          ],
        }],
        [ '(target_arch=="arm64" or target_arch=="aarch64") and OS!="win"', {
          'dependencies': [
            'aarch64-gcm-wrap.gyp:aarch64-gcm-wrap-nodepend_c_lib',
          ],
          'defines': [
            'HAVE_PLATFORM_GCM'
          ],
        }],
      ],
    },
    {
      'target_name': 'gcm',
      'conditions': [
        [ '(OS=="win" and cc_use_gnu_ld!=1 and (target_arch=="ia32" or target_arch=="x64")) or (target_arch=="x64" and OS!="win" and OS!="ios")', {
          'dependencies': [
            'intel-gcm-wrap.gyp:intel-gcm-wrap_c_lib',
          ],
          'defines': [
            'HAVE_PLATFORM_GCM'
          ],
        }],
        # vcipher is POWER8, so this is only built when the target baseline
        # has it; ppc_crypto_support() still gates its use at run time.
        [ 'disable_altivec==0 and disable_vec_crypto==0 and (target_arch=="ppc64" or target_arch=="ppc64le")', {
          'dependencies': [
            'ppc-gcm-wrap.gyp:ppc-gcm-wrap_c_lib',
          ],
          'defines': [
            'HAVE_PLATFORM_GCM'
          ],
        }],
        [ '(target_arch=="arm64" or target_arch=="aarch64") and OS!="win"', {
          'dependencies': [
            'aarch64-gcm-wrap.gyp:aarch64-gcm-wrap_c_lib',
          ],
          'defines': [
            'HAVE_PLATFORM_GCM'
          ],
        }],
      ],
      'defines!': [
        'FREEBL_NO_DEPEND',
      ],
    },
  ],
  'variables': {
    'module': 'nss',
  }
}
