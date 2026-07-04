// |jit-test| skip-if: !isAsmJSCompilationAvailable()||getBuildConfiguration("big-endian"); test-also=--ion-regalloc=simple

setIonCheckGraphCoherency(false);
load(libdir + 'bullet.js');
var results = runBullet();
assertEq(results.asmJSValidated, true);
