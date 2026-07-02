var exp = wasmEvalText(`(module
    (memory 1)
    (export "mem" (memory 0))
    (func $f (result i32) (i32.load (i32.const 0)))
    (export "f" (func $f))
)`).exports;

var ab = exp.mem.buffer;
// Wasm memory is little-endian, so write with explicit byte order.
new DataView(ab).setInt32(0, 42, true);

assertEq(exp.f(), 42);

assertThrowsInstanceOf(() => detachArrayBuffer(ab), Error);
assertEq(exp.f(), 42);

assertThrowsInstanceOf(() => serialize(ab, [ab]), Error);
assertEq(exp.f(), 42);
