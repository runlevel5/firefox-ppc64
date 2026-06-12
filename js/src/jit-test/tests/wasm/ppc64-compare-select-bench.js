// |jit-test| skip-if: true
//
// Benchmark only, not a correctness test. Invoke manually as shown below.
//
// Microbenchmark for wasm compare+select fusion on PPC64.
//
// Run with:
//   $JS --wasm-compiler=optimizing \
//       js/src/jit-test/tests/wasm/ppc64-compare-select-bench.js
//
// Prints timings for four variants (i32, i64, f32, f64) that exercise a
// tight loop of N select-on-compare operations. Used to decide whether
// specializing lowerWasmCompareAndSelect beyond Int32 is worth the code.
//
// The kernel is a 10-stage select chain so the per-op overhead dominates
// the loop frame. Each iteration touches 10 compare+select ops plus
// ~trivial address math.

const N_ITERS = 1_000_000;

function buildModule(kind) {
  const types = {i32: ['i32', 'i32', 'i32.lt_s'],
                 u32: ['i32', 'i32', 'i32.lt_u'],
                 i64: ['i64', 'i64', 'i64.lt_s'],
                 f32: ['f32', 'i32', 'f32.lt'],
                 f64: ['f64', 'i32', 'f64.lt']}[kind];
  const [ty, iterTy, cmpOp] = types;
  // Load a, b; compute chain of (b < a ? b : a) 10 times per iter.
  const stage = `
    (local.set $a
      (select (result ${ty})
        (local.get $b) (local.get $a)
        (${cmpOp} (local.get $b) (local.get $a))))`;
  const body = Array(10).fill(stage).join('\n');
  const text = `
    (module
      (func (export "run") (param $n i32) (result ${ty})
        (local $i i32) (local $a ${ty}) (local $b ${ty})
        (local.set $a (${ty}.const ${kind === 'f32' || kind === 'f64' ? '3.14' : '12345'}))
        (local.set $b (${ty}.const ${kind === 'f32' || kind === 'f64' ? '2.71' : '67890'}))
        (loop $L
          ${body}
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br_if $L (i32.lt_s (local.get $i) (local.get $n))))
        (local.get $a)))`;
  return new WebAssembly.Module(wasmTextToBinary(text));
}

function bench(kind) {
  const inst = new WebAssembly.Instance(buildModule(kind));
  // Warmup — ensure Ion compiles.
  for (let i = 0; i < 3; i++) inst.exports.run(N_ITERS);
  const t0 = dateNow();
  const res = inst.exports.run(N_ITERS);
  const t1 = dateNow();
  return {ms: t1 - t0, result: res};
}

const kinds = ['i32', 'u32', 'i64', 'f32', 'f64'];
const runs = 5;
print(`\nwasm compare+select microbench (${N_ITERS.toLocaleString()} iters, 10 ops/iter):`);
print(`  Each timing is the best of ${runs} runs.\n`);
for (const kind of kinds) {
  const samples = [];
  for (let i = 0; i < runs; i++) samples.push(bench(kind).ms);
  samples.sort((a, b) => a - b);
  const best = samples[0];
  const median = samples[(runs / 2) | 0];
  print(`  ${kind.padEnd(4)} best=${best.toFixed(1)}ms  median=${median.toFixed(1)}ms  (samples: ${samples.map(s => s.toFixed(0)).join(',')})`);
}
