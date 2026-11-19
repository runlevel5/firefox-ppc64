/* Any copyright is dedicated to the Public Domain.
   http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

const TEST_PAGE =
  getRootDirectory(gTestPath).replace(
    "chrome://mochitests/content",
    "https://example.com"
  ) + "big_endian_warning.html";

const NOTIFICATION_VALUE = "big-endian-unsupported";

const IS_BIG_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] == 0;

async function runPageFunction(name, expectWarning) {
  await BrowserTestUtils.withNewTab(TEST_PAGE, async browser => {
    await SpecialPowers.spawn(browser, [name], async fn => {
      await content.wrappedJSObject[fn]();
    });

    let notificationBox = gBrowser.getNotificationBox(browser);
    if (expectWarning && IS_BIG_ENDIAN) {
      await TestUtils.waitForCondition(
        () => notificationBox.getNotificationWithValue(NOTIFICATION_VALUE),
        `Waiting for the big-endian notification after ${name}`
      );
      ok(true, `${name} shows the big-endian notification`);
      return;
    }

    // Round-trip through the content process so that a notification request
    // sent before this point would already have arrived.
    await SpecialPowers.spawn(browser, [], () => {});
    await TestUtils.waitForTick();
    ok(
      !notificationBox.getNotificationWithValue(NOTIFICATION_VALUE),
      `${name} shows no big-endian notification`
    );
  });
}

add_task(async function test_main_thread() {
  await runPageFunction("endianError", true);
});

add_task(async function test_unrelated_error() {
  await runPageFunction("otherError", false);
});

add_task(async function test_dedicated_worker() {
  await runPageFunction("dedicatedWorker", true);
});

add_task(async function test_shared_worker() {
  await runPageFunction("sharedWorker", true);
});

add_task(async function test_service_worker() {
  await SpecialPowers.pushPrefEnv({
    set: [["dom.serviceWorkers.testing.enabled", true]],
  });
  await runPageFunction("serviceWorker", true);
  await SpecialPowers.popPrefEnv();
});

add_task(async function test_disabled_by_pref() {
  await SpecialPowers.pushPrefEnv({
    set: [["browser.bigEndianWarning.enabled", false]],
  });
  await runPageFunction("endianError", false);
  await SpecialPowers.popPrefEnv();
});
