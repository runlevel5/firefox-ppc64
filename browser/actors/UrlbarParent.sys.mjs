/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

const lazy = {};

ChromeUtils.defineESModuleGetters(lazy, {
  UrlbarParentController:
    "moz-src:///browser/components/urlbar/UrlbarParentController.sys.mjs",
});

/**
 * @import {UrlbarParentController} from "moz-src:///browser/components/urlbar/UrlbarParentController.sys.mjs"
 */

/**
 * Parent-process counterpart of `UrlbarChild`. Owns the
 * `UrlbarParentController` instances created for the `<moz-urlbar>`
 * elements served by this window global.
 *
 * The controllers are cached in a `WeakMap` keyed by the input element, so
 * a controller's lifetime tracks its element rather than the actor's: when
 * an element goes away (e.g. an about:newtab-style document hosting a
 * smartbar is torn down while the top chrome window lives on) its
 * controller becomes collectable, rather than being pinned until the
 * window closes. Reconnecting the same element (e.g. toggling customize
 * mode) reuses the same controller.
 *
 * In the chrome same-process configuration, `UrlbarChild.getOrCreateController`
 * reaches us via `windowGlobalChild.parentActor.getActor("Urlbar")` and we
 * hand the real `UrlbarParentController` back directly. A future
 * content-process consumer (e.g. about:newtab) will instead trade
 * `sendQuery` messages with us, at which point the parent will need to
 * retain controllers explicitly and route by an instance id.
 */
export class UrlbarParent extends JSWindowActorParent {
  /** @type {WeakMap<object, UrlbarParentController>} */
  #controllers = new WeakMap();

  /**
   * @param {object} input
   *   The `UrlbarInput`/`SmartbarInput` owning the controller.
   *   In-process only; for now the parent controller depends on a live
   *   input reference, which is why content-process `<moz-urlbar>`
   *   isn't supported yet.
   * @returns {UrlbarParentController}
   */
  getOrCreateController(input) {
    let controller = this.#controllers.get(input);
    if (!controller) {
      controller = new lazy.UrlbarParentController({
        sapName: input.sapName,
        isPrivate: input.isPrivate,
      });
      this.#controllers.set(input, controller);
    }
    return controller;
  }
}
