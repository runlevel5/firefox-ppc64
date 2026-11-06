let exp = wasmEvalText(`(module
    (memory 1)
    (export "mem" (memory 0))
    (func $f (result i32) (i32.load (i32.const 0)))
    (export "zero" (func $f))
)`).exports;

const byteLength = 65536;
let buffer = exp.mem.buffer;
// Wasm memory is little-endian, so access it with explicit byte order.
let dv = new DataView(buffer);
let zero = exp.zero;

const magic = 0xbadf00d;

assertEq(zero(), 0);
assertEq(dv.getInt32(0, true), 0);

dv.setInt32(0, magic, true);

assertEq(zero(), magic);
assertEq(dv.getInt32(0, true), magic);

assertEq(buffer.detached, false);
assertEq(buffer.byteLength, byteLength);

// Can't transfer Wasm prepared array buffers.
assertThrowsInstanceOf(() => buffer.transfer(), TypeError);

// |buffer| is still attached.
assertEq(buffer.detached, false);
assertEq(buffer.byteLength, byteLength);

// Access still returns the original value.
assertEq(zero(), magic);
assertEq(dv.getInt32(0, true), magic);
