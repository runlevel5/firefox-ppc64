/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { AppConstants } from "resource://gre/modules/AppConstants.sys.mjs";

const lazy = {};
ChromeUtils.defineESModuleGetters(lazy, {
  UpdateUtils: "resource://gre/modules/UpdateUtils.sys.mjs",
});

// GMP IDs
export const OPEN_H264_ID = "gmp-gmpopenh264";
export const WIDEVINE_L1_ID = "gmp-widevinecdm-l1";
export const WIDEVINE_L3_ID = "gmp-widevinecdm";

export const GMP_PLUGIN_IDS = [OPEN_H264_ID, WIDEVINE_L1_ID, WIDEVINE_L3_ID];

export var GMPUtils = {
  /**
   * Checks whether or not a given plugin is hidden. Hidden plugins are neither
   * downloaded nor displayed in the addons manager.
   *
   * @param   aPlugin
   *          The plugin to check.
   */
  isPluginHidden(aPlugin) {
    if (!this._isPluginSupported(aPlugin) || !this._isPluginVisible(aPlugin)) {
      return true;
    }

    if (!aPlugin.isEME) {
      return false;
    }

    if (!GMPPrefs.getBool(GMPPrefs.KEY_EME_ENABLED, true)) {
      return true;
    }

    return false;
  },

  /**
   * Checks whether or not a given plugin is supported by the current OS.
   *
   * @param   aPlugin
   *          The plugin to check.
   */
  _isPluginSupported(aPlugin) {
    if (this._isPluginForceSupported(aPlugin)) {
      return true;
    }
    if (aPlugin.id == WIDEVINE_L1_ID) {
      // The Widevine L1 plugin is currently only available for Windows x64.
      return (
        AppConstants.MOZ_WMF_CDM &&
        AppConstants.platform == "win" &&
        lazy.UpdateUtils.ABI.match(/x64/)
      );
    }
    if (aPlugin.id == WIDEVINE_L1_ID || aPlugin.id == WIDEVINE_L3_ID) {
      // The Widevine plugin is available for Windows versions Vista and later,
      // Mac OSX, and Linux.
      return (
        AppConstants.platform == "win" ||
        AppConstants.platform == "macosx" ||
        AppConstants.platform == "linux"
      );
    }

    return true;
  },

  /**
   * Checks whether or not a given plugin is visible in the addons manager
   * UI and the "enable DRM" notification box. This can be used to test
   * plugins that aren't yet turned on in the mozconfig.
   *
   * @param   aPlugin
   *          The plugin to check.
   */
  _isPluginVisible(aPlugin) {
    return GMPPrefs.getBool(GMPPrefs.KEY_PLUGIN_VISIBLE, false, aPlugin.id);
  },

  /**
   * Checks whether or not a given plugin is forced-supported. This is used
   * in automated tests to override the checks that prevent GMPs running on an
   * unsupported platform.
   *
   * @param   aPlugin
   *          The plugin to check.
   */
  _isPluginForceSupported(aPlugin) {
    return GMPPrefs.getBool(
      GMPPrefs.KEY_PLUGIN_FORCE_SUPPORTED,
      false,
      aPlugin.id
    );
  },

  _isWindowsOnARM64() {
    return (
      AppConstants.platform == "win" && lazy.UpdateUtils.ABI.match(/aarch64/)
    );
  },

  /**
   * Checks whether a given plugin is the x86_64 Widevine CDM run through a
   * proxy CDM and an emulated helper process, as on Linux ppc64le. Such a
   * plugin is not served for the platform by the update servers: it is taken
   * from the local source config, which aliases the x86_64 package, and the
   * proxy is installed beside the downloaded library (see
   * GMPInstallManager.installProxyCdm). The ppc64 ABI covers both byte
   * orders; the emulated helper exists for little-endian only.
   *
   * @param   aPlugin
   *          The plugin to check.
   */
  _usesProxyCdm(aPlugin) {
    return (
      aPlugin.id == WIDEVINE_L3_ID &&
      AppConstants.platform == "linux" &&
      Services.appinfo.XPCOMABI.startsWith("ppc64-") &&
      new Uint8Array(new Uint16Array([1]).buffer)[0] === 1
    );
  },

  /**
   * Names what stops a proxy CDM installed in aDir from starting its helper,
   * or returns null when gmp-helper, wvhost and box64 can all be run.
   *
   * @param   aDir
   *          The plugin's install directory.
   */
  proxyCdmProblem(aDir) {
    for (let name of ["gmp-helper", "wvhost"]) {
      let path = PathUtils.join(aDir, name);
      if (!this._isExecutable(path)) {
        return `${path} is missing or not executable`;
      }
    }
    if (!this.box64Path()) {
      let fromEnv = Services.env.get("WV_BOX64");
      return fromEnv
        ? `box64 not found at ${fromEnv}, named by WV_BOX64`
        : "box64 was not found on PATH; install box64 or set WV_BOX64";
    }
    return null;
  },

  /**
   * The box64 the plugin's gmp-helper will run, or null when there is none.
   * box64 is packaged separately from the proxy, so it is taken from PATH
   * unless WV_BOX64 names one, which is how gmp-helper resolves it too.
   */
  box64Path() {
    let fromEnv = Services.env.get("WV_BOX64");
    if (fromEnv) {
      return this._isExecutable(fromEnv) ? fromEnv : null;
    }
    for (let dir of (Services.env.get("PATH") || "").split(":")) {
      if (!dir.startsWith("/")) {
        continue;
      }
      let path = PathUtils.join(dir, "box64");
      if (this._isExecutable(path)) {
        return path;
      }
    }
    return null;
  },

  _isExecutable(aPath) {
    try {
      let file = Cc["@mozilla.org/file/local;1"].createInstance(Ci.nsIFile);
      file.initWithPath(aPath);
      return file.exists() && file.isExecutable();
    } catch (e) {
      return false;
    }
  },

  _getChromiumUpdateParameters(aPlugin) {
    let params = "";
    if (AppConstants.platform === "win") {
      params += "&os=win";
    } else if (AppConstants.platform === "macosx") {
      params += "&os=mac";
    } else if (AppConstants.platform === "linux") {
      params += "&os=Linux";
    } else {
      throw new Error("Unknown platform " + AppConstants.platform);
    }

    const abi = Services.appinfo.XPCOMABI;
    if (abi.match(/aarch64/)) {
      params += "&arch=arm64&os_arch=arm64";
    } else if (abi.match(/x86_64/) || this._usesProxyCdm(aPlugin)) {
      params += "&arch=x64&os_arch=x64";
    } else if (abi.match(/x86/)) {
      params += "&arch=x86&os_arch=x86";
    } else {
      throw new Error("Unknown ABI " + abi);
    }

    if (
      GMPPrefs.getBool(
        GMPPrefs.KEY_PLUGIN_FORCE_CHROMIUM_BETA,
        false,
        aPlugin.id
      )
    ) {
      params += "&testrequest=1";
    }

    return params;
  },

  _expectedABI(aPlugin) {
    let defaultABI = lazy.UpdateUtils.ABI;
    let expectedABIs = [defaultABI];
    if (aPlugin.id == WIDEVINE_L3_ID && this._isWindowsOnARM64()) {
      // On Windows on aarch64, we may use either the x86 or the ARM64 plugin
      // as we are still shipping the former to release.
      expectedABIs.push(defaultABI.replace(/aarch64/g, "x86"));
    }
    return expectedABIs.join(",");
  },
};

/**
 * Manages preferences for GMP addons
 */
export var GMPPrefs = {
  KEY_EME_ENABLED: "media.eme.enabled",
  KEY_PLUGIN_ENABLED: "media.{0}.enabled",
  KEY_PLUGIN_LAST_DOWNLOAD: "media.{0}.lastDownload",
  KEY_PLUGIN_LAST_DOWNLOAD_FAILED: "media.{0}.lastDownloadFailed",
  KEY_PLUGIN_LAST_DOWNLOAD_FAIL_REASON: "media.{0}.lastDownloadFailReason",
  KEY_PLUGIN_LAST_INSTALL_FAILED: "media.{0}.lastInstallFailed",
  KEY_PLUGIN_LAST_INSTALL_FAIL_REASON: "media.{0}.lastInstallFailReason",
  KEY_PLUGIN_LAST_INSTALL_START: "media.{0}.lastInstallStart",
  KEY_PLUGIN_LAST_UPDATE: "media.{0}.lastUpdate",
  KEY_PLUGIN_HASHVALUE: "media.{0}.hashValue",
  KEY_PLUGIN_VERSION: "media.{0}.version",
  KEY_PLUGIN_AUTOUPDATE: "media.{0}.autoupdate",
  KEY_PLUGIN_VISIBLE: "media.{0}.visible",
  KEY_PLUGIN_ABI: "media.{0}.abi",
  KEY_PLUGIN_FORCE_SUPPORTED: "media.{0}.forceSupported",
  KEY_PLUGIN_FORCE_INSTALL: "media.{0}.forceInstall",
  KEY_PLUGIN_ALLOW_X64_ON_ARM64: "media.{0}.allow-x64-plugin-on-arm64",
  KEY_PLUGIN_CHROMIUM_GUID: "media.{0}.chromium-guid",
  KEY_PLUGIN_ALLOW_CHROMIUM_UPDATE: "media.{0}.allow-chromium-update",
  KEY_PLUGIN_FORCE_CHROMIUM_UPDATE: "media.{0}.force-chromium-update",
  KEY_PLUGIN_FORCE_CHROMIUM_BETA: "media.{0}.force-chromium-beta",
  KEY_PLUGIN_PROXY_DIR: "media.{0}.proxy-dir",
  KEY_ALLOW_LOCAL_SOURCES: "media.gmp-manager.allowLocalSources",
  KEY_URL: "media.gmp-manager.url",
  KEY_URL_OVERRIDE: "media.gmp-manager.url.override",
  KEY_CHROMIUM_UPDATE_URL: "media.gmp-manager.chromium-update-url",
  KEY_UPDATE_LAST_CHECK: "media.gmp-manager.lastCheck",
  KEY_UPDATE_LAST_EMPTY_CHECK: "media.gmp-manager.lastEmptyCheck",
  KEY_SECONDS_BETWEEN_CHECKS: "media.gmp-manager.secondsBetweenChecks",
  KEY_UPDATE_ENABLED: "media.gmp-manager.updateEnabled",
  KEY_APP_DISTRIBUTION: "distribution.id",
  KEY_APP_DISTRIBUTION_VERSION: "distribution.version",
  KEY_BUILDID: "media.gmp-manager.buildID",
  KEY_PROVIDER_ENABLED: "media.gmp-provider.enabled",
  KEY_LOG_BASE: "media.gmp.log.",
  KEY_LOGGING_LEVEL: "media.gmp.log.level",
  KEY_LOGGING_DUMP: "media.gmp.log.dump",

  /**
   * Obtains the specified string preference in relation to the specified plugin.
   *
   * @param aKey The preference key value to use.
   * @param aDefaultValue The default value if no preference exists.
   * @param aPlugin The plugin to scope the preference to.
   * @return The obtained preference value, or the defaultValue if none exists.
   */
  getString(aKey, aDefaultValue, aPlugin) {
    if (
      aKey === this.KEY_APP_DISTRIBUTION ||
      aKey === this.KEY_APP_DISTRIBUTION_VERSION
    ) {
      return Services.prefs.getDefaultBranch(null).getCharPref(aKey, "default");
    }
    return Services.prefs.getStringPref(
      this.getPrefKey(aKey, aPlugin),
      aDefaultValue
    );
  },

  /**
   * Obtains the specified int preference in relation to the specified plugin.
   *
   * @param aKey The preference key value to use.
   * @param aDefaultValue The default value if no preference exists.
   * @param aPlugin The plugin to scope the preference to.
   * @return The obtained preference value, or the defaultValue if none exists.
   */
  getInt(aKey, aDefaultValue, aPlugin) {
    return Services.prefs.getIntPref(
      this.getPrefKey(aKey, aPlugin),
      aDefaultValue
    );
  },

  /**
   * Obtains the specified bool preference in relation to the specified plugin.
   *
   * @param aKey The preference key value to use.
   * @param aDefaultValue The default value if no preference exists.
   * @param aPlugin The plugin to scope the preference to.
   * @return The obtained preference value, or the defaultValue if none exists.
   */
  getBool(aKey, aDefaultValue, aPlugin) {
    return Services.prefs.getBoolPref(
      this.getPrefKey(aKey, aPlugin),
      aDefaultValue
    );
  },

  /**
   * Sets the specified string preference in relation to the specified plugin.
   *
   * @param aKey The preference key value to use.
   * @param aVal The value to set.
   * @param aPlugin The plugin to scope the preference to.
   */
  setString(aKey, aVal, aPlugin) {
    Services.prefs.setStringPref(this.getPrefKey(aKey, aPlugin), aVal);
  },

  /**
   * Sets the specified bool preference in relation to the specified plugin.
   *
   * @param aKey The preference key value to use.
   * @param aVal The value to set.
   * @param aPlugin The plugin to scope the preference to.
   */
  setBool(aKey, aVal, aPlugin) {
    Services.prefs.setBoolPref(this.getPrefKey(aKey, aPlugin), aVal);
  },

  /**
   * Sets the specified int preference in relation to the specified plugin.
   *
   * @param aKey The preference key value to use.
   * @param aVal The value to set.
   * @param aPlugin The plugin to scope the preference to.
   */
  setInt(aKey, aVal, aPlugin) {
    Services.prefs.setIntPref(this.getPrefKey(aKey, aPlugin), aVal);
  },

  /**
   * Checks whether or not the specified preference is set in relation to the
   * specified plugin.
   *
   * @param aKey The preference key value to use.
   * @param aPlugin The plugin to scope the preference to.
   * @return true if the preference is set, false otherwise.
   */
  isSet(aKey, aPlugin) {
    return Services.prefs.prefHasUserValue(this.getPrefKey(aKey, aPlugin));
  },

  /**
   * Resets the specified preference in relation to the specified plugin to its
   * default.
   *
   * @param aKey The preference key value to use.
   * @param aPlugin The plugin to scope the preference to.
   */
  reset(aKey, aPlugin) {
    Services.prefs.clearUserPref(this.getPrefKey(aKey, aPlugin));
  },

  /**
   * Scopes the specified preference key to the specified plugin.
   *
   * @param aKey The preference key value to use.
   * @param aPlugin The plugin to scope the preference to.
   * @return A preference key scoped to the specified plugin.
   */
  getPrefKey(aKey, aPlugin) {
    return aKey.replace("{0}", aPlugin || "");
  },
};
