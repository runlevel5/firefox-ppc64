// |jit-test| skip-if: getBuildConfiguration("big-endian"); test-also=--ion-regalloc=simple

setIonCheckGraphCoherency(false);
load(libdir + 'bullet.js');
runBullet();
