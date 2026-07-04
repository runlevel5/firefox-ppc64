/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

function throwAndCatch() {
  try {
    throw new Error("BigEndian system not supported");
  } catch (e) {}
}

if (typeof ServiceWorkerGlobalScope == "function") {
  self.addEventListener("activate", event => {
    event.waitUntil(self.clients.claim());
  });
  self.addEventListener("message", event => {
    throwAndCatch();
    event.source.postMessage("done");
  });
} else if (typeof SharedWorkerGlobalScope == "function") {
  self.onconnect = event => {
    throwAndCatch();
    event.ports[0].postMessage("done");
  };
} else {
  throwAndCatch();
  postMessage("done");
}
