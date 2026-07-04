/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

const NOTIFICATION_VALUE = "big-endian-unsupported";
const ENABLED_PREF = "browser.bigEndianWarning.enabled";

export class BigEndianWarningParent extends JSWindowActorParent {
  receiveMessage(message) {
    if (message.name == "BigEndianWarning:Unsupported") {
      this.showNotification();
    }
  }

  showNotification() {
    if (!Services.prefs.getBoolPref(ENABLED_PREF, true)) {
      return;
    }
    let browser = this.browsingContext.top.embedderElement;
    let tabbrowser = browser?.getTabBrowser();
    if (!tabbrowser) {
      return;
    }
    let notificationBox = tabbrowser.getNotificationBox(browser);
    if (notificationBox.getNotificationWithValue(NOTIFICATION_VALUE)) {
      return;
    }
    notificationBox
      .appendNotification(
        NOTIFICATION_VALUE,
        {
          label: { "l10n-id": "big-endian-unsupported-infobar-message" },
          priority: notificationBox.PRIORITY_WARNING_MEDIUM,
        },
        [
          {
            "l10n-id": "big-endian-unsupported-infobar-never-show-button",
            callback: () => {
              Services.prefs.setBoolPref(ENABLED_PREF, false);
            },
          },
        ]
      )
      .catch(console.error);
  }
}
