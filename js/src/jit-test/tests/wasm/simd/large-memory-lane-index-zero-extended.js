// |jit-test| skip-if: !wasmSimdEnabled() || !largeArrayBufferSupported(); allow-oom
//
// The i32 index of v128.loadN_lane and v128.storeN_lane is zero-extended like
// any other memory index, so an index at or above 2 GiB is not a negative
// offset. PPC64 Ion crashed on the load.

let extract = {
  8: "i8x16.extract_lane_u 0",
  16: "i16x8.extract_lane_u 0",
  32: "i32x4.extract_lane 0",
  64: "i32.wrap_i64 (i64x2.extract_lane 0",
};
let funcs = "";
for (let size of [8, 16, 32, 64]) {
  funcs += `
    (func (export "store${size}") (param $p i32)
      (v128.store${size}_lane 0 (local.get $p) (v128.const i64x2 77 0)))
    (func (export "load${size}") (param $p i32) (result i32)
      (${extract[size]}
        (v128.load${size}_lane 0 (local.get $p) (v128.const i64x2 0 0))${size == 64 ? ")" : ""}))`;
}
let ins = wasmEvalText(`(module (memory (export "mem") 32769) ${funcs})`);
assertEq(ins.exports.mem.buffer.byteLength > 0x80000000, true);
let bytes = new Uint8Array(ins.exports.mem.buffer);
for (let size of [8, 16, 32, 64]) {
  bytes[0x80000000] = 0;
  ins.exports["store" + size](0x80000000);
  assertEq(bytes[0x80000000], 77);
  assertEq(ins.exports["load" + size](0x80000000), 77);
}
