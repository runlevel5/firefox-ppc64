// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]
#![allow(clippy::identity_op)]

// The backend requires VSX and the POWER8 vector extensions at compile time: they are the
// powerpc64le baseline, and on big-endian they can be enabled with e.g. `-C target-cpu=pwr8`.
// LLVM does not inline the AltiVec intrinsics into functions that only enable VSX through
// `#[target_feature]`, so detecting POWER8 at runtime would leave every vector operation as a
// function call. POWER9 is detected at runtime on little-endian (see `Power9::is_available`).
#[cfg(all(
    feature = "vsx",
    target_feature = "vsx",
    target_feature = "power8-vector"
))]
pub(super) mod vsx;

#[macro_export]
macro_rules! simd_function {
    (
        $dname:ident,
        $descr:ident: $descr_ty:ident,
        $(#[$($attr:meta)*])*
        $pub:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block
    ) => {
        #[inline(always)]
        $(#[$($attr)*])*
        $pub fn $name<$descr_ty: $crate::SimdDescriptor>($descr: $descr_ty, $($arg: $ty),*) $(-> $ret)? $body
        $(#[$($attr)*])*
        $pub fn $dname($($arg: $ty),*) $(-> $ret)? {
            #[allow(unused)]
            use $crate::SimdDescriptor;
            $crate::simd_function_body_vsx!($name($($arg: $ty),*) $(-> $ret)?; ($($arg),*));
            $name($crate::ScalarDescriptor::new().unwrap(), $($arg),*)
        }
    };
}

#[cfg(all(
    feature = "vsx",
    target_feature = "vsx",
    target_feature = "power8-vector"
))]
#[doc(hidden)]
#[macro_export]
macro_rules! simd_function_body_vsx {
    ($name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )?; ($($val:expr),* $(,)?)) => {
        if let Some(d) = $crate::VsxDescriptor::<$crate::Power9>::new() {
            return d.call(
                #[inline(always)]
                |d| $name(d, $($val),*),
            );
        }
        if let Some(d) = $crate::VsxDescriptor::<$crate::Power8>::new() {
            return $name(d, $($val),*);
        }
    };
}

#[cfg(not(all(
    feature = "vsx",
    target_feature = "vsx",
    target_feature = "power8-vector"
)))]
#[doc(hidden)]
#[macro_export]
macro_rules! simd_function_body_vsx {
    ($($ignore:tt)*) => {};
}

#[macro_export]
macro_rules! test_all_instruction_sets {
    (
        $name:ident
    ) => {
        paste::paste! {
            #[test]
            fn [<$name _scalar>]() {
                use $crate::SimdDescriptor;
                $name($crate::ScalarDescriptor::new().unwrap())
            }
        }

        $crate::test_vsx!($name);
    };
}

#[cfg(all(
    feature = "vsx",
    target_feature = "vsx",
    target_feature = "power8-vector"
))]
#[doc(hidden)]
#[macro_export]
macro_rules! test_vsx {
    ($name:ident) => {
        paste::paste! {
            #[test]
            fn [<$name _power8>]() {
                use $crate::SimdDescriptor;
                let Some(d) = $crate::VsxDescriptor::<$crate::Power8>::new() else { return; };
                d.call($name)
            }

            #[test]
            fn [<$name _power9>]() {
                use $crate::SimdDescriptor;
                let Some(d) = $crate::VsxDescriptor::<$crate::Power9>::new() else { return; };
                d.call($name)
            }
        }
    };
}

#[cfg(not(all(
    feature = "vsx",
    target_feature = "vsx",
    target_feature = "power8-vector"
)))]
#[doc(hidden)]
#[macro_export]
macro_rules! test_vsx {
    ($name:ident) => {};
}

#[macro_export]
macro_rules! bench_all_instruction_sets {
    (
        $name:ident,
        $criterion:ident
    ) => {
        use $crate::SimdDescriptor;
        // `simd_function_body_*` does early return; wrap it with an immediately-invoked closure
        (|| {
            $crate::simd_function_body_vsx!(
                $name($criterion: &mut ::criterion::BenchmarkGroup<'_, impl ::criterion::measurement::Measurement>);
                ($criterion, "vsx")
            );
        })();
        $name(
            $crate::ScalarDescriptor::new().unwrap(),
            $criterion,
            "scalar",
        );
    };
}
