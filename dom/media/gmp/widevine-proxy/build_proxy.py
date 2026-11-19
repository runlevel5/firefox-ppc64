# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Builds the Widevine proxy bundle: the native proxy CDM, and the x86_64
# helper and runtime library that box64 runs. Everything is freestanding, so
# the x86_64 parts need only clang and lld, not an x86_64 sysroot.

import os
import shlex
import subprocess
import tempfile

import buildconfig

X86_64 = "--target=x86_64-linux-gnu"

FLAGS = [
    "-O2",
    "-ffreestanding",
    "-nostdinc",
    "-fno-stack-protector",
    "-Wall",
]

CXX_FLAGS = [
    "-fno-exceptions",
    "-fno-rtti",
    "-fno-threadsafe-statics",
    "-Wno-unused-private-field",
]

X86_64_LINK = [
    "-nostdlib",
    "-fuse-ld=lld",
    "-Wl,-z,now",
    "-Wl,-z,noexecstack",
    "-Wl,--build-id",
]


def configured(name):
    value = buildconfig.substs[name]
    return shlex.split(value) if isinstance(value, str) else list(value)


def retarget(compiler):
    """The configured compiler, targeting x86_64 instead."""
    args = []
    skip = False
    for arg in compiler:
        if skip:
            skip = False
        elif arg == "-target":
            skip = True
        elif not arg.startswith("--target="):
            args.append(arg)
    return args + [X86_64]


def run(args):
    print(shlex.join(args), flush=True)
    subprocess.check_call(args)


def main(output, *inputs):
    """Builds the bundle into the directory of |output|; |inputs| are its
    sources, the first being wvproxy.cc."""
    srcdir = os.path.dirname(inputs[0])
    cc = configured("CC")
    cxx = configured("CXX")
    out_dir = os.path.dirname(os.path.abspath(output.name))
    resource_dir = subprocess.check_output(
        cxx + ["-print-resource-dir"], text=True
    ).strip()
    builtin = ["-isystem", os.path.join(resource_dir, "include")]
    includes = builtin + [
        "-isystem",
        os.path.join(srcdir, "shim"),
        "-I",
        srcdir,
        "-I",
        os.path.join(srcdir, "..", "widevine-adapter"),
    ]

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        # The native proxy CDM, which Firefox loads as libwidevinecdm.so.
        proxy = os.path.join(tmp, "libwidevinecdm.so")
        run(
            cxx
            + FLAGS
            + CXX_FLAGS
            + includes
            + [
                "-fPIC",
                "-fvisibility=hidden",
                "-shared",
                "-nostdlib++",
                "-Wl,--no-undefined",
                "-Wl,--version-script=" + os.path.join(srcdir, "wvproxy.map"),
                os.path.join(srcdir, "wvproxy.cc"),
                "-o",
                proxy,
            ]
        )

        # Empty libc.so.6 and libdl.so.2 for the helper to link against, so
        # that it records them as needed; box64 supplies the real ones.
        empty = os.path.join(tmp, "empty.c")
        open(empty, "w").close()
        stubs = []
        for soname in ("libc.so.6", "libdl.so.2"):
            stub = os.path.join(tmp, soname)
            run(
                retarget(cc)
                + X86_64_LINK
                + ["-shared", "-Wl,-soname," + soname, empty, "-o", stub]
            )
            stubs.append(stub)

        # The helper box64 runs to load the real CDM.
        run(
            retarget(cxx)
            + FLAGS
            + CXX_FLAGS
            + includes
            + ["-fPIE", "-fno-asynchronous-unwind-tables", "-pie"]
            + X86_64_LINK
            + [
                "-Wl,--dynamic-linker=/lib64/ld-linux-x86-64.so.2",
                "-Wl,--unresolved-symbols=ignore-all",
                os.path.join(srcdir, "start.S"),
                os.path.join(srcdir, "wvhost.cc"),
            ]
            + stubs
            + ["-o", os.path.join(out_dir, "wvhost")]
        )

        # The one x86_64 runtime library the real CDM needs besides libc.
        run(
            retarget(cc)
            + FLAGS
            + builtin
            + ["-fPIC", "-shared"]
            + X86_64_LINK
            + [
                "-Wl,-soname,libgcc_s.so.1",
                "-Wl,--version-script=" + os.path.join(srcdir, "libgcc_s.map"),
                os.path.join(srcdir, "libgcc_s.c"),
                "-o",
                os.path.join(out_dir, "libgcc_s.so.1"),
            ]
        )

        with open(proxy, "rb") as f:
            output.write(f.read())
