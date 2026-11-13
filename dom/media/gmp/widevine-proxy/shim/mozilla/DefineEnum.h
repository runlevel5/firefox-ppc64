/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#pragma once
#define MOZ_WV_UNPAREN(...) __VA_ARGS__
#define MOZ_DEFINE_ENUM_WITH_BASE_AND_TOSTRING(Name, Base, List) \
  enum Name : Base { MOZ_WV_UNPAREN List }
