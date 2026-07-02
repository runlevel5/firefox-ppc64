// Regression test for a PPC64-specific wasm Ion crash in
// CodeGenerator::visitWasmReduceAndBranchSimd128 — it called
// LBlock::label() directly on the branch targets without going through
// skipTrivialBlocks(), so a trivial goto-only successor tripped
// LBlock::label()'s !isTrivial() assertion. Reduced from grantkot.com/poly
// with wasm-reduce. Triggers the bug under --wasm-compiler=optimizing.
new WebAssembly.Module(os.file.readFile(scriptdir + "/bug-ppc64-simd-reduce-and-branch.wasm", "binary"));
