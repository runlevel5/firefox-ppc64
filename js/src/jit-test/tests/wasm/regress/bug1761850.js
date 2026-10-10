// Testing runtime execution of select + comparison operations.
// Normally they are folded into shorter/faster sequence than select alone.

const floatOps = {
    lt(a, b) { return a < b ? 0 : 1; },
    le(a, b) { return a <= b ? 0 : 1; },
    gt(a, b) { return a > b ? 0 : 1; },
    ge(a, b) { return a >= b ? 0 : 1; },
    eq(a, b) { return a === b ? 0 : 1; },
    ne(a, b) { return a !== b ? 0 : 1; },
}

for (let ty of ['f32', 'f64']) {
    for (let op of ['lt', 'le', 'gt', 'ge', 'eq', 'ne']) {
        const module = new WebAssembly.Module(wasmTextToBinary(`(module
            (memory (export "memory") 1 1)
            (func (export "test") (result i32)
                i32.const 128
                i32.load8_u
                i32.const 129
                i32.load8_u
                i32.const 0
                ${ty}.load
                i32.const ${ty == 'f32' ? 4 : 8}
                ${ty}.load
                ${ty}.${op}
                select
            )
            (data (i32.const 128) "\\00\\01"))`));
        const instance = new WebAssembly.Instance(module);
        // Wasm memory is little-endian; use a DataView with explicit byte
        // order so the test is endian-neutral.
        const view = new DataView(instance.exports.memory.buffer);
        const size = ty == 'f32' ? 4 : 8;
        const get = ty == 'f32' ? view.getFloat32.bind(view) : view.getFloat64.bind(view);
        const set = ty == 'f32' ? view.setFloat32.bind(view) : view.setFloat64.bind(view);
        for (let [a, b] of cross(
            [0, 1, -1e100, Infinity, -Infinity, 1e100, -1e-10, 1/-Infinity, NaN]
        )) {
            set(0, a, true); set(size, b, true);
            assertEq(instance.exports.test(),
                     floatOps[op](get(0, true), get(size, true)))
        }
    }
}

const intOps = {
    lt(a, b) { return a < b ? 0 : 1; },
    le(a, b) { return a <= b ? 0 : 1; },
    gt(a, b) { return a > b ? 0 : 1; },
    ge(a, b) { return a >= b ? 0 : 1; },
    eq(a, b) { return a === b ? 0 : 1; },
    ne(a, b) { return a !== b ? 0 : 1; },
}

for (let [ty, signed] of [['i32', true], ['i32', false], ['i64', true], ['i64', false]]) {
    for (let op of ['lt', 'le', 'gt', 'ge', 'eq', 'ne']) {
        const module = new WebAssembly.Module(wasmTextToBinary(`(module
            (memory (export "memory") 1 1)
            (func (export "test") (result i32)
                i32.const 128
                i32.load8_u
                i32.const 129
                i32.load8_u
                i32.const 0
                ${ty}.load
                i32.const ${ty == 'i32' ? 4 : 8}
                ${ty}.load
                ${ty}.${op}${op[0] == 'l' || op[0] == 'g' ? (signed ? '_s' : '_u') : ''}
                select
            )
            (data (i32.const 128) "\\00\\01"))`));
        const instance = new WebAssembly.Instance(module);
        // Wasm memory is little-endian; use a DataView with explicit byte
        // order so the test is endian-neutral.
        const view = new DataView(instance.exports.memory.buffer);
        const size = ty == 'i32' ? 4 : 8;
        const get = ty == 'i32' ? (signed ? view.getInt32 : view.getUint32).bind(view)
                                : (signed ? view.getBigInt64 : view.getBigUint64).bind(view);
        const set = ty == 'i32' ? (signed ? view.setInt32 : view.setUint32).bind(view)
                                : (signed ? view.setBigInt64 : view.setBigUint64).bind(view);
        const c = ty == 'i32' ? (a => a|0) : BigInt;
        for (let [a, b] of cross(
            [c(0), ~c(0), c(1), ~c(1), c(1) << c(8), ~c(1) << c(12)]
        )) {
            set(0, a, true); set(size, b, true);
            assertEq(instance.exports.test(),
                     intOps[op](get(0, true), get(size, true)))
        }
    }
}
