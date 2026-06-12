/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef jit_ppc64_MacroAssembler_ppc64_inl_h
#define jit_ppc64_MacroAssembler_ppc64_inl_h

#include "jit/ppc64/MacroAssembler-ppc64.h"

namespace js {
namespace jit {

//{{{ check_macroassembler_style

// ===============================================================
// Move instructions

void MacroAssembler::move64(Register64 src, Register64 dest) {
  movePtr(src.reg, dest.reg);
}

void MacroAssembler::move64(Imm64 imm, Register64 dest) {
  movePtr(ImmWord(imm.value), dest.reg);
}

void MacroAssembler::moveDoubleToGPR64(FloatRegister src, Register64 dest) {
  as_mfvsrd(dest.reg, src);
}

void MacroAssembler::moveGPR64ToDouble(Register64 src, FloatRegister dest) {
  as_mtvsrd(dest, src.reg);
}

void MacroAssembler::moveLowDoubleToGPR(FloatRegister src, Register dest) {
  MOZ_CRASH("Not supported for this target");
}

void MacroAssembler::move64To32(Register64 src, Register dest) {
  as_extsw(dest, src.reg);
}

void MacroAssembler::move32To64ZeroExtend(Register src, Register64 dest) {
  // clrldi dest, src, 32 — clear upper 32 bits.
  as_rldicl(dest.reg, src, 0, 32);
}

void MacroAssembler::move8To64SignExtend(Register src, Register64 dest) {
  as_extsb(dest.reg, src);
}

void MacroAssembler::move16To64SignExtend(Register src, Register64 dest) {
  as_extsh(dest.reg, src);
}

void MacroAssembler::move32To64SignExtend(Register src, Register64 dest) {
  as_extsw(dest.reg, src);
}

void MacroAssembler::moveFloat32ToGPR(FloatRegister src, Register dest) {
  // FPR holds double-format value (PPC convention). Convert to
  // single-precision bits in bits 0:31 of the VSR, then extract.
  as_xscvdpspn(ScratchDoubleReg, src);
  as_mfvsrd(dest, ScratchDoubleReg);
  x_srdi(dest, dest, 32);
}

void MacroAssembler::moveGPRToFloat32(Register src, FloatRegister dest) {
  // Place raw single-precision bits in VSR bits 0:31, then convert
  // to double-precision format (matching PPC's FPR convention, like lfs).
  if (HasPOWER9()) {
    // mtvsrws splats the 32-bit word to both halves of the VSR.
    as_mtvsrws(dest, src);
  } else {
    // POWER8: shift GPR left 32 bits to place float bits in upper word,
    // then move to VSR. xscvspdpn reads from bits 0:31.
    UseScratchRegisterScope temps(*this);
    Register tmp = temps.Acquire();
    x_sldi(tmp, src, 32);
    as_mtvsrd(dest, tmp);
  }
  as_xscvspdpn(dest, dest);
}

void MacroAssembler::moveFloat16ToGPR(FloatRegister src, Register dest) {
  MOZ_ASSERT(HasPOWER9());
  // src has FP16 in dw0 bits 48:63; rest of dw0 is 0 (per xscvdphp /
  // lxsihzx / mtvsrwz contract). mfvsrd reads dw0 → dest = 0x...0000_HHHH.
  // Mask defensively in case a future caller hands us a non-canonical FP16.
  as_mfvsrd(dest, src);
  as_rldicl(dest, dest, 0, 48);  // clrldi 48: keep low 16 bits
}

void MacroAssembler::moveGPRToFloat16(Register src, FloatRegister dest) {
  MOZ_ASSERT(HasPOWER9());
  // mtvsrwz zeros dw0 word 0 and copies src's low 32 to dw0 word 1; mask
  // src to its low 16 first so dw0 bits 32:47 stay zero (canonical FP16).
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  as_rldicl(scratch, src, 0, 48);  // clrldi 48: keep only low 16
  as_mtvsrwz(dest, scratch);
}

void MacroAssembler::move8ZeroExtend(Register src, Register dest) {
  // rlwinm dest, src, 0, 24, 31 — mask to low 8 bits.
  as_rlwinm(dest, src, 0, 24, 31);
}

void MacroAssembler::move8SignExtend(Register src, Register dest) {
  as_extsb(dest, src);
}

void MacroAssembler::move16SignExtend(Register src, Register dest) {
  as_extsh(dest, src);
}

void MacroAssembler::move8SignExtendToPtr(Register src, Register dest) {
  as_extsb(dest, src);
}

void MacroAssembler::move16SignExtendToPtr(Register src, Register dest) {
  as_extsh(dest, src);
}

void MacroAssembler::move32SignExtendToPtr(Register src, Register dest) {
  as_extsw(dest, src);
}

void MacroAssembler::move32ZeroExtendToPtr(Register src, Register dest) {
  as_rldicl(dest, src, 0, 32);
}

// ===============================================================
// Load instructions

void MacroAssembler::load32SignExtendToPtr(const Address& src, Register dest) {
  load32(src, dest);
  as_extsw(dest, dest);
}

void MacroAssembler::loadAbiReturnAddress(Register dest) { xs_mflr(dest); }

// ===============================================================
// Logical instructions

void MacroAssembler::not32(Register reg) {
  x_not(reg, reg);
  as_extsw(reg, reg);
}

void MacroAssembler::notPtr(Register reg) { x_not(reg, reg); }

void MacroAssembler::andPtr(Register src, Register dest) {
  as_and_(dest, dest, src);
}

// If `mask` is a non-zero, non-all-ones contiguous run of 1-bits in a
// 32-bit value (LSB-numbering), set MB/ME to the BE bit positions
// (PPC convention: bit 0 = MSB) needed by `rlwinm SH=0` and return true.
// Otherwise return false. Run-time cost is at JIT emit time only.
static inline bool IsContigMask32(uint32_t mask, unsigned& mb, unsigned& me) {
  if (mask == 0 || mask == 0xFFFFFFFFu) return false;
  unsigned tz = (unsigned)__builtin_ctz(mask);
  uint32_t shifted = mask >> tz;
  if ((shifted & (shifted + 1)) != 0) return false;  // Has a 0 between 1s.
  unsigned width = 32 - (unsigned)__builtin_clz(shifted);
  // LSB bits set: [tz, tz+width-1]. BE bits: [32-tz-width, 31-tz].
  mb = 32 - tz - width;
  me = 31 - tz;
  return true;
}

// 64-bit contiguous-mask classification for AND-with-imm via PPC's
// rotate-and-mask family (SH=0). On success, sets `lsb` (LSB-numbering
// of lowest set bit) and `width` (number of contiguous 1-bits).
// Caller picks the encoding:
//   - lsb == 0:                low `width` bits set        → rldicl
//   (mb6=64-width)
//   - lsb + width == 64:       high `width` bits set       → rldicr
//   (me6=width-1)
//   - lsb + width <= 32:       contig mask within low 32   → rlwinm (zeros high
//   32)
//   - otherwise (mid-run mask straddling bit 32 with lsb>0): no SH=0 single
//     insn fits, return false to fall back to scratch+and.
static inline bool IsContigMask64(uint64_t mask, unsigned& lsb,
                                  unsigned& width) {
  if (mask == 0 || mask == ~uint64_t(0)) return false;
  unsigned tz = (unsigned)__builtin_ctzll(mask);
  uint64_t shifted = mask >> tz;
  if ((shifted & (shifted + 1)) != 0) return false;  // Has a 0 between 1s.
  width = 64 - (unsigned)__builtin_clzll(shifted);
  lsb = tz;
  return true;
}

void MacroAssembler::andPtr(Imm32 imm, Register dest) {
  // andi. handles 16-bit unsigned immediates in 1 insn (sets CR0).
  // For wider positive immediates, IsContigMask32 → rlwinm (1 insn,
  // also sets CR0). NOTE: andPtr sign-extends Imm32 to 64-bit before
  // ANDing, so contig-mask is only safe when the immediate is
  // non-negative (high bit clear) — rlwinm always zeros the high 32.
  uint32_t uimm = uint32_t(imm.value);
  if (is_uintN(uimm, 16)) {
    as_andi_rc(dest, dest, uimm);
    return;
  }
  unsigned mb, me;
  if (imm.value >= 0 && IsContigMask32(uimm, mb, me)) {
    as_rlwinm_rc(dest, dest, 0, mb, me);
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord(uintptr_t(intptr_t(imm.value))), scratch);
  as_and_(dest, dest, scratch);
}

void MacroAssembler::andPtr(Imm32 imm, Register src, Register dest) {
  if (src != dest) {
    xs_mr(dest, src);
  }
  andPtr(imm, dest);
}

void MacroAssembler::and64(Imm64 imm, Register64 dest) {
  uint64_t u = imm.value;
  // 16-bit unsigned → andi. (1 insn).
  if (u <= 0xFFFFu) {
    as_andi_rc(dest.reg, dest.reg, uint16_t(u));
    return;
  }
  unsigned lsb, width;
  if (IsContigMask64(u, lsb, width)) {
    if (lsb == 0) {
      // low `width` bits set: rldicl SH=0 MB=64-width.
      as_rldicl_rc(dest.reg, dest.reg, 0, 64 - width);
      return;
    }
    if (lsb + width == 64) {
      // high `width` bits set: rldicr SH=0 ME=width-1.
      as_rldicr_rc(dest.reg, dest.reg, 0, width - 1);
      return;
    }
    if (lsb + width <= 32) {
      // contig mask within low 32: rlwinm SH=0 zeros bits 0..31 too.
      // BE positions: mb = 32 - lsb - width, me = 31 - lsb.
      as_rlwinm_rc(dest.reg, dest.reg, 0, 32 - lsb - width, 31 - lsb);
      return;
    }
    // mid-run mask straddling bit 32 (lsb>0, lsb+width>32, lsb+width<64):
    // not encodable as SH=0 mask. Fall through to scratch+and.
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord(u), scratch);
  as_and_(dest.reg, dest.reg, scratch);
}

void MacroAssembler::and64(Register64 src, Register64 dest) {
  as_and_(dest.reg, dest.reg, src.reg);
}

void MacroAssembler::and32(Register src, Register dest) {
  as_and_(dest, dest, src);
  as_extsw(dest, dest);
}

void MacroAssembler::and32(Imm32 imm, Register dest) {
  uint32_t uimm = uint32_t(imm.value);
  if (is_uintN(uimm, 16)) {
    as_andi_rc(dest, dest, uimm);
  } else {
    unsigned mb, me;
    if (IsContigMask32(uimm, mb, me)) {
      // rlwinm.SH=0 ANDs with the contiguous mask; record form sets CR0
      // to match the side-effect of the andi. fast path above.
      as_rlwinm_rc(dest, dest, 0, mb, me);
    } else {
      UseScratchRegisterScope temps(asMasm());
      Register scratch = temps.Acquire();
      move32(imm, scratch);
      as_and_(dest, dest, scratch);
    }
  }
  as_extsw(dest, dest);
}

void MacroAssembler::and32(Imm32 imm, Register src, Register dest) {
  if (src != dest) {
    xs_mr(dest, src);
  }
  and32(imm, dest);
}

void MacroAssembler::and32(Imm32 imm, const Address& dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(dest, scratch);
  and32(imm, scratch);
  store32(scratch, dest);
}

void MacroAssembler::and32(const Address& src, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(src, scratch);
  as_and_(dest, dest, scratch);
  as_extsw(dest, dest);
}

void MacroAssembler::or64(Imm64 imm, Register64 dest) {
  uint64_t u = imm.value;
  // ori/oris zero-extend their immediates and don't touch other bits, so
  // when imm fits in unsigned 32 (high 32 == 0) the pair handles it in
  // 1-2 insns with no scratch.
  if (u <= 0xFFFFFFFFu) {
    uint16_t lo = uint16_t(u);
    uint16_t hi = uint16_t(u >> 16);
    if (hi == 0) {
      as_ori(dest.reg, dest.reg, lo);
    } else if (lo == 0) {
      as_oris(dest.reg, dest.reg, hi);
    } else {
      as_ori(dest.reg, dest.reg, lo);
      as_oris(dest.reg, dest.reg, hi);
    }
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord(u), scratch);
  as_or_(dest.reg, dest.reg, scratch);
}

void MacroAssembler::or32(Register src, Register dest) {
  as_or_(dest, dest, src);
  as_extsw(dest, dest);
}

void MacroAssembler::or32(Imm32 imm, Register dest) {
  uint32_t uimm = uint32_t(imm.value);
  uint16_t lo = uimm & 0xFFFF;
  uint16_t hi = (uimm >> 16) & 0xFFFF;
  if (hi == 0) {
    as_ori(dest, dest, lo);
  } else if (lo == 0) {
    as_oris(dest, dest, hi);
  } else {
    // ori + oris pair handles arbitrary 32-bit unsigned imm in 2 insns
    // without a scratch GPR. ori/oris are non-record forms (don't touch
    // CR0), matching the behavior of the previous scratch+or_ path
    // (or_ is the record form, but the value-only result is what callers
    // observe through dest).
    as_ori(dest, dest, lo);
    as_oris(dest, dest, hi);
  }
  as_extsw(dest, dest);
}

void MacroAssembler::or32(Imm32 imm, Register src, Register dest) {
  if (src != dest) {
    xs_mr(dest, src);
  }
  or32(imm, dest);
}

void MacroAssembler::or32(Imm32 imm, const Address& dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(dest, scratch);
  or32(imm, scratch);
  store32(scratch, dest);
}

void MacroAssembler::xor64(Imm64 imm, Register64 dest) {
  uint64_t u = imm.value;
  // xori/xoris zero-extend their immediates; for unsigned-32-fit values
  // they replace the scratch+xor sequence with 1-2 insns.
  if (u <= 0xFFFFFFFFu) {
    uint16_t lo = uint16_t(u);
    uint16_t hi = uint16_t(u >> 16);
    if (hi == 0) {
      as_xori(dest.reg, dest.reg, lo);
    } else if (lo == 0) {
      as_xoris(dest.reg, dest.reg, hi);
    } else {
      as_xori(dest.reg, dest.reg, lo);
      as_xoris(dest.reg, dest.reg, hi);
    }
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord(u), scratch);
  as_xor_(dest.reg, dest.reg, scratch);
}

void MacroAssembler::orPtr(Register src, Register dest) {
  as_or_(dest, dest, src);
}

void MacroAssembler::orPtr(Imm32 imm, Register dest) {
  uint32_t uimm = uint32_t(imm.value);
  uint16_t lo = uimm & 0xFFFF;
  uint16_t hi = (uimm >> 16) & 0xFFFF;
  // ori/oris zero-extend their immediates, so for non-negative Imm32 (high
  // 32 of sign-extended value = 0) we can use ori+oris to OR the full
  // 32-bit pattern in 1-2 insns. Negative Imm32 sign-extends to set high
  // bits 32..63 in the OR — those bits would be lost with ori+oris alone.
  if (imm.value >= 0) {
    if (hi == 0) {
      as_ori(dest, dest, lo);
    } else if (lo == 0) {
      as_oris(dest, dest, hi);
    } else {
      as_ori(dest, dest, lo);
      as_oris(dest, dest, hi);
    }
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord(uintptr_t(intptr_t(imm.value))), scratch);
  as_or_(dest, dest, scratch);
}

void MacroAssembler::orPtr(Imm32 imm, Register src, Register dest) {
  if (src != dest) {
    xs_mr(dest, src);
  }
  orPtr(imm, dest);
}

void MacroAssembler::or64(Register64 src, Register64 dest) {
  as_or_(dest.reg, dest.reg, src.reg);
}

void MacroAssembler::xor64(Register64 src, Register64 dest) {
  as_xor_(dest.reg, dest.reg, src.reg);
}

void MacroAssembler::xorPtr(Register src, Register dest) {
  as_xor_(dest, dest, src);
}

void MacroAssembler::xorPtr(Imm32 imm, Register dest) {
  uint32_t uimm = uint32_t(imm.value);
  uint16_t lo = uimm & 0xFFFF;
  uint16_t hi = (uimm >> 16) & 0xFFFF;
  if (imm.value >= 0) {
    if (hi == 0) {
      as_xori(dest, dest, lo);
    } else if (lo == 0) {
      as_xoris(dest, dest, hi);
    } else {
      as_xori(dest, dest, lo);
      as_xoris(dest, dest, hi);
    }
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord(uintptr_t(intptr_t(imm.value))), scratch);
  as_xor_(dest, dest, scratch);
}

void MacroAssembler::xorPtr(Imm32 imm, Register src, Register dest) {
  if (src != dest) {
    xs_mr(dest, src);
  }
  xorPtr(imm, dest);
}

void MacroAssembler::xor32(Register src, Register dest) {
  as_xor_(dest, dest, src);
  as_extsw(dest, dest);
}

void MacroAssembler::xor32(Imm32 imm, Register dest) {
  uint32_t uimm = uint32_t(imm.value);
  uint16_t lo = uimm & 0xFFFF;
  uint16_t hi = (uimm >> 16) & 0xFFFF;
  if (hi == 0) {
    as_xori(dest, dest, lo);
  } else if (lo == 0) {
    as_xoris(dest, dest, hi);
  } else {
    // xori + xoris pair — 2 insns, no scratch GPR.
    as_xori(dest, dest, lo);
    as_xoris(dest, dest, hi);
  }
  as_extsw(dest, dest);
}

void MacroAssembler::xor32(Imm32 imm, Register src, Register dest) {
  if (src != dest) {
    xs_mr(dest, src);
  }
  xor32(imm, dest);
}

void MacroAssembler::xor32(Imm32 imm, const Address& dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(dest, scratch);
  xor32(imm, scratch);
  store32(scratch, dest);
}

void MacroAssembler::xor32(const Address& src, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(src, scratch);
  xor32(scratch, dest);
}

// ===============================================================
// Swap instructions

void MacroAssembler::byteSwap16SignExtend(Register reg) {
  if (HasPOWER10()) {
    // brh byte-reverses every halfword in reg; extsh keeps just the
    // low halfword's byte-reversed value, sign-extended to 64 bits.
    as_brh(reg, reg);
    as_extsh(reg, reg);
    return;
  }
  // POWER8/9: rotate-and-mask synthesis. Swap bytes in low halfword via
  // (reg<<8)&0xFF00 | (reg>>8)&0xFF, then sign-extend.
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  as_rlwinm(scratch, reg, 8, 16, 23);  // scratch = (reg<<8) & 0xFF00
  as_rlwinm(reg, reg, 24, 24, 31);     // reg = (reg>>8) & 0xFF
  as_or_(reg, reg, scratch);
  as_extsh(reg, reg);
}

void MacroAssembler::byteSwap16ZeroExtend(Register reg) {
  if (HasPOWER10()) {
    // brh byte-reverses every halfword; rldicl with sh=0,mb=48 zeroes
    // the upper 48 bits — no CR0 side effect (vs andi.).
    as_brh(reg, reg);
    as_rldicl(reg, reg, 0, 48);
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  // Both rlwinm forms zero-extend the 64-bit destination per ISA v3.0B
  // (mask M = MASK(MB+32, ME+32) is 0 above bit 31), so after the OR the
  // upper 48 bits are already zero — no follow-up clearing needed.
  as_rlwinm(scratch, reg, 8, 16, 23);
  as_rlwinm(reg, reg, 24, 24, 31);
  as_or_(reg, reg, scratch);
}

void MacroAssembler::byteSwap32(Register reg) {
  if (HasPOWER10()) {
    // brw byte-reverses both 32-bit halves; extsw drops the upper half
    // and sign-extends the byte-reversed low word to 64 bits.
    as_brw(reg, reg);
    as_extsw(reg, reg);
    return;
  }
  // POWER8/9: rotate-with-insert synthesis (4 insns).
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  // scratch = rotate reg left 8, mask bytes 0,3
  as_rlwinm(scratch, reg, 8, 0, 31);    // rotl32 by 8
  as_rlwimi(scratch, reg, 24, 0, 7);    // insert byte 0
  as_rlwimi(scratch, reg, 24, 16, 23);  // insert byte 2
  // Sign-extend to 64 bits (as 32-bit value).
  as_extsw(reg, scratch);
}

void MacroAssembler::byteSwap64(Register64 reg64) {
  if (HasPOWER10()) {
    // 1 insn, no FPR round-trip.
    as_brd(reg64.reg, reg64.reg);
  } else if (HasPOWER9()) {
    as_mtvsrd(ScratchDoubleReg, reg64.reg);
    as_xxbrd(ScratchDoubleReg, ScratchDoubleReg);
    as_mfvsrd(reg64.reg, ScratchDoubleReg);
  } else {
    // POWER8: byte-swap via stack using stwbrx (word byte-reverse store).
    // stwbrx RS,RA,RB stores RS byte-reversed at RA+RB.
    // For 64-bit swap: store high word reversed at addr+0, low word at addr+4.
    Register r = reg64.reg;
    UseScratchRegisterScope temps(*this);
    Register tmp = temps.Acquire();
    as_stdu(StackPointer, StackPointer, -16);
    // Store low 32 bits byte-reversed at SP+12.
    as_addi(tmp, StackPointer, 12);
    as_stwbrx(r, r0, tmp);  // r0 as RA = 0, so addr = tmp
    // Store high 32 bits byte-reversed at SP+8.
    x_srdi(r, r, 32);
    as_addi(tmp, StackPointer, 8);
    as_stwbrx(r, r0, tmp);  // addr = tmp
    // Load reversed 64-bit value from SP+8.
    as_ld(r, StackPointer, 8);
    as_addi(StackPointer, StackPointer, 16);
  }
}

// ===============================================================
// Arithmetic functions

void MacroAssembler::addPtr(Register src, Register dest) {
  as_add(dest, dest, src);
}

void MacroAssembler::addPtr(Imm32 imm, Register dest) {
  int32_t val = imm.value;
  if (is_intN(val, 16)) {
    as_addi(dest, dest, val);
    return;
  }
  if (HasPOWER10()) {
    // Imm32 always fits 34-bit signed; paddi does dest = dest + imm in one
    // prefixed instruction with no scratch.
    as_paddi(dest, dest, int64_t(val), /*R=*/false);
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord(int64_t(val)), scratch);
  as_add(dest, dest, scratch);
}

void MacroAssembler::addPtr(ImmWord imm, Register dest) {
  if (is_intN(int64_t(imm.value), 16)) {
    as_addi(dest, dest, int16_t(imm.value));
    return;
  }
  if (HasPOWER10() && is_intN((intptr_t)imm.value, 34)) {
    as_paddi(dest, dest, (int64_t)imm.value, /*R=*/false);
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(imm, scratch);
  as_add(dest, dest, scratch);
}

void MacroAssembler::add64(Register64 src, Register64 dest) {
  as_add(dest.reg, dest.reg, src.reg);
}

void MacroAssembler::add64(Imm32 imm, Register64 dest) {
  addPtr(Imm32(imm.value), dest.reg);
}

void MacroAssembler::add64(Imm64 imm, Register64 dest) {
  if (is_intN(int64_t(imm.value), 16)) {
    as_addi(dest.reg, dest.reg, int16_t(imm.value));
    return;
  }
  if (HasPOWER10() && is_intN((int64_t)imm.value, 34)) {
    as_paddi(dest.reg, dest.reg, (int64_t)imm.value, /*R=*/false);
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  MOZ_ASSERT(dest.reg != scratch);
  movePtr(ImmWord(imm.value), scratch);
  as_add(dest.reg, dest.reg, scratch);
}

void MacroAssembler::add32(Register src, Register dest) {
  as_add(dest, dest, src);
  as_extsw(dest, dest);
}

void MacroAssembler::add32(Imm32 imm, Register dest) {
  if (is_intN(imm.value, 16)) {
    as_addi(dest, dest, imm.value);
  } else {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    move32(imm, scratch);
    as_add(dest, dest, scratch);
  }
  as_extsw(dest, dest);
}

void MacroAssembler::add32(Imm32 imm, Register src, Register dest) {
  move32(src, dest);
  add32(imm, dest);
}

void MacroAssembler::add32(Imm32 imm, const Address& dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(dest, scratch);
  add32(imm, scratch);
  store32(scratch, dest);
}

void MacroAssembler::add32(const Address& src, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(src, scratch);
  as_add(dest, dest, scratch);
  as_extsw(dest, dest);
}

void MacroAssembler::addPtr(Imm32 imm, const Address& dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(dest, scratch);
  addPtr(imm, scratch);
  storePtr(scratch, dest);
}

void MacroAssembler::addPtr(const Address& src, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(src, scratch);
  addPtr(scratch, dest);
}

void MacroAssembler::addDouble(FloatRegister src, FloatRegister dest) {
  as_fadd(dest, dest, src);
}

void MacroAssembler::addFloat32(FloatRegister src, FloatRegister dest) {
  as_fadds(dest, dest, src);
}

CodeOffset MacroAssembler::sub32FromStackPtrWithPatch(Register dest) {
  CodeOffset offset = CodeOffset(currentOffset());
  emitLoad64Stanza(dest, 0);
  as_subf(dest, dest, StackPointer);
  return offset;
}

void MacroAssembler::patchSub32FromStackPtr(CodeOffset offset, Imm32 imm) {
  Instruction* inst = (Instruction*)editSrc(BufferOffset(offset.offset()));
  UpdateLoad64Value(inst, uint64_t(int64_t(imm.value)));
}

void MacroAssembler::subPtr(Register src, Register dest) {
  as_subf(dest, src, dest);
}

void MacroAssembler::subPtr(Imm32 imm, Register dest) {
  if (is_intN(-int64_t(imm.value), 16)) {
    as_addi(dest, dest, -imm.value);
    return;
  }
  if (HasPOWER10()) {
    // -Imm32 fits 34-bit signed (worst case -INT32_MIN = +2^31, well within
    // ±2^33). paddi with the negated immediate does the subtract in 1 prefixed
    // insn with no scratch.
    as_paddi(dest, dest, -int64_t(imm.value), /*R=*/false);
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord(int64_t(imm.value)), scratch);
  as_subf(dest, scratch, dest);
}

void MacroAssembler::sub64(Register64 src, Register64 dest) {
  as_subf(dest.reg, src.reg, dest.reg);
}

void MacroAssembler::sub64(Imm64 imm, Register64 dest) {
  if (is_intN(-int64_t(imm.value), 16)) {
    as_addi(dest.reg, dest.reg, int16_t(-int64_t(imm.value)));
    return;
  }
  if (HasPOWER10() && is_intN(-(int64_t)imm.value, 34)) {
    as_paddi(dest.reg, dest.reg, -(int64_t)imm.value, /*R=*/false);
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  MOZ_ASSERT(dest.reg != scratch);
  movePtr(ImmWord(imm.value), scratch);
  as_subf(dest.reg, scratch, dest.reg);
}

void MacroAssembler::sub32(Register src, Register dest) {
  as_subf(dest, src, dest);
  as_extsw(dest, dest);
}

void MacroAssembler::sub32(Imm32 imm, Register dest) {
  if (is_intN(-int64_t(imm.value), 16)) {
    as_addi(dest, dest, -imm.value);
  } else {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    move32(imm, scratch);
    as_subf(dest, scratch, dest);
  }
  as_extsw(dest, dest);
}

void MacroAssembler::sub32(const Address& src, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(src, scratch);
  as_subf(dest, scratch, dest);
  as_extsw(dest, dest);
}

void MacroAssembler::subPtr(Register src, const Address& dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(dest, scratch);
  as_subf(scratch, src, scratch);
  storePtr(scratch, dest);
}

void MacroAssembler::subPtr(const Address& addr, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(addr, scratch);
  as_subf(dest, scratch, dest);
}

void MacroAssembler::subDouble(FloatRegister src, FloatRegister dest) {
  as_fsub(dest, dest, src);
}

void MacroAssembler::subFloat32(FloatRegister src, FloatRegister dest) {
  as_fsubs(dest, dest, src);
}

void MacroAssembler::mul64(const Register64& rhs, const Register64& srcDest) {
  as_mulld(srcDest.reg, srcDest.reg, rhs.reg);
}

void MacroAssembler::mul64(Imm64 imm, const Register64& dest) {
  if (is_intN(int64_t(imm.value), 16)) {
    as_mulli(dest.reg, dest.reg, int16_t(imm.value));
  } else {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    MOZ_ASSERT(dest.reg != scratch);
    movePtr(ImmWord(imm.value), scratch);
    as_mulld(dest.reg, dest.reg, scratch);
  }
}

void MacroAssembler::mul64(Imm64 imm, const Register64& dest,
                           const Register temp) {
  MOZ_ASSERT(temp == Register::Invalid());
  mul64(imm, dest);
}

void MacroAssembler::mul64(const Register64& src, const Register64& dest,
                           const Register temp) {
  MOZ_ASSERT(temp == Register::Invalid());
  as_mulld(dest.reg, dest.reg, src.reg);
}

void MacroAssembler::mulPtr(Register rhs, Register srcDest) {
  as_mulld(srcDest, srcDest, rhs);
}

void MacroAssembler::mulPtr(ImmWord rhs, Register srcDest) {
  if (is_intN(int64_t(rhs.value), 16)) {
    as_mulli(srcDest, srcDest, int16_t(rhs.value));
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  MOZ_ASSERT(srcDest != scratch);
  movePtr(rhs, scratch);
  mulPtr(scratch, srcDest);
}

void MacroAssembler::mulBy3(Register src, Register dest) {
  // mulli is the 16-bit-immediate form of mulld. 1 insn, no scratch,
  // src==dest aliasing safe (RA read before RT write).
  as_mulli(dest, src, 3);
}

void MacroAssembler::mul32(Register rhs, Register srcDest) {
  as_mullw(srcDest, srcDest, rhs);
  as_extsw(srcDest, srcDest);
}

void MacroAssembler::mul32(Imm32 imm, Register srcDest) {
  if (is_intN(imm.value, 16)) {
    as_mulli(srcDest, srcDest, imm.value);
  } else {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    move32(imm, scratch);
    as_mullw(srcDest, srcDest, scratch);
  }
  as_extsw(srcDest, srcDest);
}

void MacroAssembler::mulHighUnsigned32(Imm32 imm, Register src, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  MOZ_ASSERT(src != scratch);
  move32(imm, scratch);
  as_mulhwu(dest, src, scratch);
  as_extsw(dest, dest);
}

void MacroAssembler::mulFloat32(FloatRegister src, FloatRegister dest) {
  as_fmuls(dest, dest, src);
}

void MacroAssembler::mulDouble(FloatRegister src, FloatRegister dest) {
  as_fmul(dest, dest, src);
}

void MacroAssembler::mulDoublePtr(ImmPtr imm, Register temp,
                                  FloatRegister dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(imm, scratch);
  as_lfd(ScratchDoubleReg, scratch, 0);
  as_fmul(dest, dest, ScratchDoubleReg);
}

void MacroAssembler::inc64(AbsoluteAddress dest) {
  UseScratchRegisterScope temps(asMasm());
  Register addrReg = temps.Acquire();
  movePtr(ImmWord(uintptr_t(dest.addr)), addrReg);
  Register scratch = SecondScratchReg;
  as_ld(scratch, addrReg, 0);
  as_addi(scratch, scratch, 1);
  as_std(scratch, addrReg, 0);
}

void MacroAssembler::divFloat32(FloatRegister src, FloatRegister dest) {
  as_fdivs(dest, dest, src);
}

void MacroAssembler::divDouble(FloatRegister src, FloatRegister dest) {
  as_fdiv(dest, dest, src);
}

void MacroAssembler::quotient32(Register lhs, Register rhs, Register dest,
                                bool isUnsigned) {
  if (isUnsigned) {
    as_divwu(dest, lhs, rhs);
  } else {
    as_divw(dest, lhs, rhs);
  }
  as_extsw(dest, dest);
}

void MacroAssembler::quotient64(Register lhs, Register rhs, Register dest,
                                bool isUnsigned) {
  if (isUnsigned) {
    as_divdu(dest, lhs, rhs);
  } else {
    as_divd(dest, lhs, rhs);
  }
}

void MacroAssembler::remainder32(Register lhs, Register rhs, Register dest,
                                 bool isUnsigned) {
  if (HasPOWER9()) {
    if (isUnsigned) {
      as_moduw(dest, lhs, rhs);
    } else {
      as_modsw(dest, lhs, rhs);
    }
  } else {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    if (isUnsigned) {
      as_divwu(scratch, lhs, rhs);
      as_mullw(scratch, scratch, rhs);
    } else {
      as_divw(scratch, lhs, rhs);
      as_mullw(scratch, scratch, rhs);
    }
    as_subf(dest, scratch, lhs);
  }
  as_extsw(dest, dest);
}

void MacroAssembler::remainder64(Register lhs, Register rhs, Register dest,
                                 bool isUnsigned) {
  if (HasPOWER9()) {
    if (isUnsigned) {
      as_modud(dest, lhs, rhs);
    } else {
      as_modsd(dest, lhs, rhs);
    }
  } else {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    if (isUnsigned) {
      as_divdu(scratch, lhs, rhs);
      as_mulld(scratch, scratch, rhs);
    } else {
      as_divd(scratch, lhs, rhs);
      as_mulld(scratch, scratch, rhs);
    }
    as_subf(dest, scratch, lhs);
  }
}

void MacroAssembler::neg64(Register64 reg) { as_neg(reg.reg, reg.reg); }

void MacroAssembler::negPtr(Register reg) { as_neg(reg, reg); }

void MacroAssembler::neg32(Register reg) {
  as_neg(reg, reg);
  as_extsw(reg, reg);
}

void MacroAssembler::negateDouble(FloatRegister reg) { as_fneg(reg, reg); }

void MacroAssembler::negateFloat(FloatRegister reg) { as_fneg(reg, reg); }

void MacroAssembler::abs32(Register src, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  as_srawi(scratch, src, 31);
  as_xor_(dest, src, scratch);
  as_subf(dest, scratch, dest);
  as_extsw(dest, dest);
}

void MacroAssembler::absFloat32(FloatRegister src, FloatRegister dest) {
  as_fabs(dest, src);
}

void MacroAssembler::absDouble(FloatRegister src, FloatRegister dest) {
  as_fabs(dest, src);
}

void MacroAssembler::sqrtFloat32(FloatRegister src, FloatRegister dest) {
  as_fsqrts(dest, src);
}

void MacroAssembler::sqrtDouble(FloatRegister src, FloatRegister dest) {
  as_fsqrt(dest, src);
}

void MacroAssembler::min32(Register lhs, Register rhs, Register dest) {
  as_cmpw(lhs, rhs);
  // isel rt, ra, rb, cond: rt = (CR[cond] set) ? ra : rb
  // LessThan set if lhs < rhs (signed), so pick lhs; else rhs = min.
  as_isel(dest, lhs, rhs, LessThan, cr0);
}

void MacroAssembler::min32(Register lhs, Imm32 rhs, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  move32(rhs, scratch);
  min32(lhs, scratch, dest);
}

void MacroAssembler::max32(Register lhs, Register rhs, Register dest) {
  as_cmpw(lhs, rhs);
  // GT set if lhs > rhs (signed), so pick lhs; else rhs = max.
  as_isel(dest, lhs, rhs, GreaterThan, cr0);
}

void MacroAssembler::max32(Register lhs, Imm32 rhs, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  move32(rhs, scratch);
  max32(lhs, scratch, dest);
}

void MacroAssembler::minPtr(Register lhs, Register rhs, Register dest) {
  as_cmpd(lhs, rhs);
  as_isel(dest, lhs, rhs, LessThan, cr0);
}

void MacroAssembler::minPtr(Register lhs, ImmWord rhs, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(rhs, scratch);
  minPtr(lhs, scratch, dest);
}

void MacroAssembler::maxPtr(Register lhs, Register rhs, Register dest) {
  as_cmpd(lhs, rhs);
  as_isel(dest, lhs, rhs, GreaterThan, cr0);
}

void MacroAssembler::maxPtr(Register lhs, ImmWord rhs, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(rhs, scratch);
  maxPtr(lhs, scratch, dest);
}

void MacroAssembler::minFloat32(FloatRegister other, FloatRegister srcDest,
                                bool handleNaN) {
  if (HasPOWER9()) {
    // xsminjdp matches ECMA-262 Math.min semantics for ±0 and NaN.
    // Float32 values are stored as doubles in PPC FPRs; the J-form
    // result is bit-exact for values representable in float32 (which
    // includes every NaN/±0/±Inf corner case JS observes). 1 insn.
    as_xsminjdp(srcDest, srcDest, other);
    return;
  }
  Label done, nan, equal;
  as_fcmpu(srcDest, other);
  if (handleNaN) {
    ma_b(Assembler::DoubleUnordered, &nan);
  }
  // Handle +0 vs -0.
  ma_b(Assembler::DoubleEqual, &equal);
  ma_b(Assembler::DoubleLessThan, &done);
  as_fmr(srcDest, other);
  jump(&done);

  bind(&equal);
  // Both operands are equal. Check if they're zero.
  loadConstantFloat32(0.0f, ScratchFloat32Reg);
  as_fcmpu(srcDest, ScratchFloat32Reg);
  // If not zero, they're identical; keep srcDest.
  ma_b(Assembler::DoubleNotEqual, &done);
  // Both are some combination of +0/-0. For min, result should be -0
  // if either is -0: -((-srcDest) - other) gives -0 when either is -0.
  as_fneg(ScratchFloat32Reg, srcDest);
  as_fsubs(ScratchFloat32Reg, ScratchFloat32Reg, other);
  as_fneg(srcDest, ScratchFloat32Reg);
  jump(&done);

  if (handleNaN) {
    bind(&nan);
    as_fadds(srcDest, srcDest, other);
  }
  bind(&done);
}

void MacroAssembler::minDouble(FloatRegister other, FloatRegister srcDest,
                               bool handleNaN) {
  if (HasPOWER9()) {
    // xsminjdp matches ECMA-262 Math.min semantics exactly (covers
    // 19 corner cases including ±0 and NaN). 1 insn vs ~12 for the
    // fcmpu/branch fallback. POWER8 fallback follows.
    as_xsminjdp(srcDest, srcDest, other);
    return;
  }
  Label done, nan, equal;
  as_fcmpu(srcDest, other);
  if (handleNaN) {
    ma_b(Assembler::DoubleUnordered, &nan);
  }
  // Handle +0 vs -0.
  ma_b(Assembler::DoubleEqual, &equal);
  ma_b(Assembler::DoubleLessThan, &done);
  as_fmr(srcDest, other);
  jump(&done);

  bind(&equal);
  loadConstantDouble(0.0, ScratchDoubleReg);
  as_fcmpu(srcDest, ScratchDoubleReg);
  ma_b(Assembler::DoubleNotEqual, &done);
  // -((-srcDest) - other) gives -0 when either is -0.
  as_fneg(ScratchDoubleReg, srcDest);
  as_fsub(ScratchDoubleReg, ScratchDoubleReg, other);
  as_fneg(srcDest, ScratchDoubleReg);
  jump(&done);

  if (handleNaN) {
    bind(&nan);
    as_fadd(srcDest, srcDest, other);
  }
  bind(&done);
}

void MacroAssembler::maxFloat32(FloatRegister other, FloatRegister srcDest,
                                bool handleNaN) {
  if (HasPOWER9()) {
    // See minFloat32 above for the float32 ↔ J-form bit-exactness note.
    as_xsmaxjdp(srcDest, srcDest, other);
    return;
  }
  Label done, nan, equal;
  as_fcmpu(srcDest, other);
  if (handleNaN) {
    ma_b(Assembler::DoubleUnordered, &nan);
  }
  // Handle +0 vs -0.
  ma_b(Assembler::DoubleEqual, &equal);
  ma_b(Assembler::DoubleGreaterThan, &done);
  as_fmr(srcDest, other);
  jump(&done);

  bind(&equal);
  loadConstantFloat32(0.0f, ScratchFloat32Reg);
  as_fcmpu(srcDest, ScratchFloat32Reg);
  ma_b(Assembler::DoubleNotEqual, &done);
  // -0 + -0 = -0 and -0 + 0 = +0.
  as_fadds(srcDest, srcDest, other);
  jump(&done);

  if (handleNaN) {
    bind(&nan);
    as_fadds(srcDest, srcDest, other);
  }
  bind(&done);
}

void MacroAssembler::maxDouble(FloatRegister other, FloatRegister srcDest,
                               bool handleNaN) {
  if (HasPOWER9()) {
    // See minDouble above for the J-form semantics note.
    as_xsmaxjdp(srcDest, srcDest, other);
    return;
  }
  Label done, nan, equal;
  as_fcmpu(srcDest, other);
  if (handleNaN) {
    ma_b(Assembler::DoubleUnordered, &nan);
  }
  // Handle +0 vs -0.
  ma_b(Assembler::DoubleEqual, &equal);
  ma_b(Assembler::DoubleGreaterThan, &done);
  as_fmr(srcDest, other);
  jump(&done);

  bind(&equal);
  loadConstantDouble(0.0, ScratchDoubleReg);
  as_fcmpu(srcDest, ScratchDoubleReg);
  ma_b(Assembler::DoubleNotEqual, &done);
  // -0 + -0 = -0 and -0 + 0 = +0.
  as_fadd(srcDest, srcDest, other);
  jump(&done);

  if (handleNaN) {
    bind(&nan);
    as_fadd(srcDest, srcDest, other);
  }
  bind(&done);
}

// ===============================================================
// Shift functions

void MacroAssembler::lshift32(Register src, Register dest) {
  UseScratchRegisterScope temps(*this);
  Register masked = temps.Acquire();
  as_rlwinm(masked, src, 0, 27, 31);
  as_slw(dest, dest, masked);
  as_extsw(dest, dest);
}

void MacroAssembler::lshift32(Imm32 imm, Register dest) {
  lshift32(imm, dest, dest);
}

void MacroAssembler::lshift32(Imm32 imm, Register src, Register dest) {
  x_slwi(dest, src, imm.value & 0x1f);
  as_extsw(dest, dest);
}

void MacroAssembler::flexibleLshift32(Register src, Register dest) {
  lshift32(src, dest);
}

void MacroAssembler::lshift64(Register shift, Register64 dest) {
  // PPC64 sld uses a 7-bit shift field; shifts >= 64 produce 0.
  // Wasm i64.shl requires shift count modulo 64, so mask to 6 bits.
  UseScratchRegisterScope temps(asMasm());
  Register masked = temps.Acquire();
  as_rldicl(masked, shift, 0, 58);  // clrldi: keep low 6 bits
  as_sld(dest.reg, dest.reg, masked);
}

void MacroAssembler::lshift64(Imm32 imm, Register64 dest) {
  MOZ_ASSERT(0 <= imm.value && imm.value < 64);
  x_sldi(dest.reg, dest.reg, imm.value);
}

void MacroAssembler::lshiftPtr(Register shift, Register dest) {
  as_sld(dest, dest, shift);
}

void MacroAssembler::lshiftPtr(Imm32 imm, Register dest) {
  lshiftPtr(imm, dest, dest);
}

void MacroAssembler::lshiftPtr(Imm32 imm, Register src, Register dest) {
  MOZ_ASSERT(0 <= imm.value && imm.value < 64);
  x_sldi(dest, src, imm.value);
}

void MacroAssembler::flexibleLshiftPtr(Register shift, Register srcDest) {
  lshiftPtr(shift, srcDest);
}

void MacroAssembler::rshift32(Register src, Register dest) {
  UseScratchRegisterScope temps(*this);
  Register masked = temps.Acquire();
  as_rlwinm(masked, src, 0, 27, 31);
  as_srw(dest, dest, masked);
  as_extsw(dest, dest);
}

void MacroAssembler::rshift32(Imm32 imm, Register dest) {
  rshift32(imm, dest, dest);
}

void MacroAssembler::rshift32(Imm32 imm, Register src, Register dest) {
  x_srwi(dest, src, imm.value & 0x1f);
  as_extsw(dest, dest);
}

void MacroAssembler::flexibleRshift32(Register src, Register dest) {
  rshift32(src, dest);
}

void MacroAssembler::rshift32Arithmetic(Register src, Register dest) {
  UseScratchRegisterScope temps(*this);
  Register masked = temps.Acquire();
  as_rlwinm(masked, src, 0, 27, 31);
  as_sraw(dest, dest, masked);
}

void MacroAssembler::rshift32Arithmetic(Imm32 imm, Register dest) {
  rshift32Arithmetic(imm, dest, dest);
}

void MacroAssembler::rshift32Arithmetic(Imm32 imm, Register src,
                                        Register dest) {
  as_srawi(dest, src, imm.value & 0x1f);
}

void MacroAssembler::flexibleRshift32Arithmetic(Register src, Register dest) {
  rshift32Arithmetic(src, dest);
}

void MacroAssembler::rshift64(Register shift, Register64 dest) {
  UseScratchRegisterScope temps(asMasm());
  Register masked = temps.Acquire();
  as_rldicl(masked, shift, 0, 58);
  as_srd(dest.reg, dest.reg, masked);
}

void MacroAssembler::rshift64(Imm32 imm, Register64 dest) {
  MOZ_ASSERT(0 <= imm.value && imm.value < 64);
  x_srdi(dest.reg, dest.reg, imm.value);
}

void MacroAssembler::rshift64Arithmetic(Imm32 imm, Register64 dest) {
  MOZ_ASSERT(0 <= imm.value && imm.value < 64);
  as_sradi(dest.reg, dest.reg, imm.value);
}

void MacroAssembler::rshift64Arithmetic(Register shift, Register64 dest) {
  UseScratchRegisterScope temps(asMasm());
  Register masked = temps.Acquire();
  as_rldicl(masked, shift, 0, 58);
  as_srad(dest.reg, dest.reg, masked);
}

void MacroAssembler::rshiftPtr(Register shift, Register dest) {
  as_srd(dest, dest, shift);
}

void MacroAssembler::rshiftPtr(Imm32 imm, Register dest) {
  rshiftPtr(imm, dest, dest);
}

void MacroAssembler::rshiftPtr(Imm32 imm, Register src, Register dest) {
  MOZ_ASSERT(0 <= imm.value && imm.value < 64);
  x_srdi(dest, src, imm.value);
}

void MacroAssembler::flexibleRshiftPtr(Register shift, Register srcDest) {
  rshiftPtr(shift, srcDest);
}

void MacroAssembler::rshiftPtrArithmetic(Imm32 imm, Register dest) {
  rshiftPtrArithmetic(imm, dest, dest);
}

void MacroAssembler::rshiftPtrArithmetic(Imm32 imm, Register src,
                                         Register dest) {
  MOZ_ASSERT(0 <= imm.value && imm.value < 64);
  as_sradi(dest, src, imm.value);
}

void MacroAssembler::rshiftPtrArithmetic(Register shift, Register dest) {
  as_srad(dest, dest, shift);
}

void MacroAssembler::flexibleRshiftPtrArithmetic(Register shift,
                                                 Register srcDest) {
  rshiftPtrArithmetic(shift, srcDest);
}

// ===============================================================
// Rotation functions

void MacroAssembler::rotateLeft(Register count, Register input, Register dest) {
  // PPC rotlw is rlwnm with full mask: rlwnm dest, input, count, 0, 31
  as_rlwnm(dest, input, count, 0, 31);
  as_extsw(dest, dest);
}

void MacroAssembler::rotateLeft(Imm32 count, Register input, Register dest) {
  as_rlwinm(dest, input, count.value & 31, 0, 31);
  as_extsw(dest, dest);
}

void MacroAssembler::rotateLeft64(Register count, Register64 src,
                                  Register64 dest, Register temp) {
  MOZ_ASSERT(temp == Register::Invalid());
  // rldcl dest, src, count, 0 — rotate left doubleword then clear left 0 bits.
  as_rldcl(dest.reg, src.reg, count, 0);
}

void MacroAssembler::rotateLeft64(Imm32 count, Register64 src, Register64 dest,
                                  Register temp) {
  MOZ_ASSERT(temp == Register::Invalid());
  // rldicl dest, src, count, 0 — rotate left doubleword immediate then clear.
  as_rldicl(dest.reg, src.reg, count.value & 63, 0);
}

void MacroAssembler::rotateRight(Register count, Register input,
                                 Register dest) {
  // rotateRight(n) = rotateLeft(32-n). When dest != input, the negated
  // count can land directly in dest, dropping the GPR scratch. dest may
  // alias count harmlessly (subfic reads count, then writes dest, then
  // rlwnm consumes the new dest as its rotate-count).
  if (dest != input) {
    as_subfic(dest, count, 32);
    as_rlwnm(dest, input, dest, 0, 31);
    as_extsw(dest, dest);
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  as_subfic(scratch, count, 32);
  as_rlwnm(dest, input, scratch, 0, 31);
  as_extsw(dest, dest);
}

void MacroAssembler::rotateRight(Imm32 count, Register input, Register dest) {
  rotateLeft(Imm32((32 - count.value) & 31), input, dest);
}

void MacroAssembler::rotateRight64(Register count, Register64 src,
                                   Register64 dest, Register temp) {
  MOZ_ASSERT(temp == Register::Invalid());
  // Same shape as rotateRight32: when dest != src, the negated count
  // can land directly in dest, dropping the GPR scratch.
  if (dest.reg != src.reg) {
    as_subfic(dest.reg, count, 64);
    as_rldcl(dest.reg, src.reg, dest.reg, 0);
    return;
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  as_subfic(scratch, count, 64);
  as_rldcl(dest.reg, src.reg, scratch, 0);
}

void MacroAssembler::rotateRight64(Imm32 count, Register64 src, Register64 dest,
                                   Register temp) {
  MOZ_ASSERT(temp == Register::Invalid());
  rotateLeft64(Imm32((64 - count.value) & 63), src, dest, temp);
}

// ===============================================================
// Bit counting functions

void MacroAssembler::clz64(Register64 src, Register64 dest) {
  as_cntlzd(dest.reg, src.reg);
}

void MacroAssembler::ctz64(Register64 src, Register64 dest) {
  if (HasPOWER9()) {
    as_cnttzd(dest.reg, src.reg);
  } else {
    UseScratchRegisterScope temps(*this);
    Register tmp = temps.Acquire();
    as_neg(tmp, src.reg);
    // and. (record form) sets CR0[eq] based on result; result is 0 iff src==0,
    // so this folds the explicit zero-check that would otherwise need cmpdi.
    as_and__rc(tmp, src.reg, tmp);  // tmp = x & -x; CR0[eq] = (src == 0)
    as_cntlzd(tmp, tmp);            // tmp = clz(isolated bit)
    as_subfic(dest.reg, tmp, 63);   // dest = 63 - clz = ctz (for nonzero)
    xs_li(tmp, 64);
    as_isel(dest.reg, tmp, dest.reg, Equal);  // CR0[eq] → 64 if src==0
  }
}

void MacroAssembler::popcnt64(Register64 input, Register64 output,
                              Register tmp) {
  as_popcntd(output.reg, input.reg);
}

void MacroAssembler::clz32(Register src, Register dest, bool knownNotZero) {
  as_cntlzw(dest, src);
}

void MacroAssembler::ctz32(Register src, Register dest, bool knownNotZero) {
  if (HasPOWER9()) {
    as_cnttzw(dest, src);
  } else {
    UseScratchRegisterScope temps(*this);
    Register tmp = temps.Acquire();
    as_neg(tmp, src);
    // and. record form folds the cmpwi src,0 that would otherwise be needed
    // to drive the isel below: tmp == 0 iff src == 0.
    if (knownNotZero) {
      as_and_(tmp, src, tmp);
    } else {
      as_and__rc(tmp, src, tmp);  // CR0[eq] = (src == 0)
    }
    as_cntlzw(tmp, tmp);
    as_subfic(dest, tmp, 31);
    if (!knownNotZero) {
      xs_li(tmp, 32);
      as_isel(dest, tmp, dest, Equal);  // CR0[eq] → 32 if src==0
    }
  }
}

void MacroAssembler::popcnt32(Register input, Register output, Register tmp) {
  as_popcntw(output, input);
  // popcntw gives per-word results; on 64-bit the low word count is in
  // bits 32:63, so just mask to 32 bits.
  as_rlwinm(output, output, 0, 0, 31);
}

// ===============================================================
// Condition functions

void MacroAssembler::cmp8Set(Condition cond, Address lhs, Imm32 rhs,
                             Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  MOZ_ASSERT(scratch != lhs.base);
  bool isUnsigned = (cond & Assembler::ConditionUnsigned) != 0;
  if (isUnsigned) {
    load8ZeroExtend(lhs, scratch);
    Condition c = ma_cmp(scratch, Imm32(uint8_t(rhs.value)), cond, true);
    ma_cmp_set(dest, c);
  } else {
    load8SignExtend(lhs, scratch);
    Condition c = ma_cmp(scratch, Imm32(int8_t(rhs.value)), cond, true);
    ma_cmp_set(dest, c);
  }
}

void MacroAssembler::cmp16Set(Condition cond, Address lhs, Imm32 rhs,
                              Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  MOZ_ASSERT(scratch != lhs.base);
  bool isUnsigned = (cond & Assembler::ConditionUnsigned) != 0;
  if (isUnsigned) {
    load16ZeroExtend(lhs, scratch);
    Condition c = ma_cmp(scratch, Imm32(uint16_t(rhs.value)), cond, true);
    ma_cmp_set(dest, c);
  } else {
    load16SignExtend(lhs, scratch);
    Condition c = ma_cmp(scratch, Imm32(int16_t(rhs.value)), cond, true);
    ma_cmp_set(dest, c);
  }
}

template <typename T1, typename T2>
void MacroAssembler::cmp32Set(Condition cond, T1 lhs, T2 rhs, Register dest) {
  Condition c = ma_cmp(lhs, rhs, cond, true);
  ma_cmp_set(dest, c);
}

void MacroAssembler::cmp64Set(Condition cond, Register64 lhs, Register64 rhs,
                              Register dest) {
  Condition c = ma_cmp(lhs.reg, rhs.reg, cond);
  ma_cmp_set(dest, c);
}

void MacroAssembler::cmp64Set(Condition cond, Register64 lhs, Imm64 rhs,
                              Register dest) {
  Condition c = ma_cmp(lhs.reg, ImmWord(uint64_t(rhs.value)), cond);
  ma_cmp_set(dest, c);
}

void MacroAssembler::cmp64Set(Condition cond, Address lhs, Register64 rhs,
                              Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs.reg, cond);
  ma_cmp_set(dest, c);
}

void MacroAssembler::cmp64Set(Condition cond, Address lhs, Imm64 rhs,
                              Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Condition c = ma_cmp(scratch, ImmWord(uint64_t(rhs.value)), cond);
  ma_cmp_set(dest, c);
}

template <typename T1, typename T2>
void MacroAssembler::cmpPtrSet(Condition cond, T1 lhs, T2 rhs, Register dest) {
  Condition c = ma_cmp(lhs, rhs, cond);
  ma_cmp_set(dest, c);
}

// ===============================================================
// Branch functions

void MacroAssembler::branch8(Condition cond, const Address& lhs, Imm32 rhs,
                             Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  // Mirror ARM64/LoongArch64/RISC-V: narrow the immediate to the 8-bit
  // memory operand width so both sides of the compare have matching bit
  // patterns regardless of how move32(Imm32) materializes the imm. Use
  // uint8 cast for equality / unsigned, int8 cast for signed relational.
  bool isEqOrNe = (cond == Assembler::Equal) || (cond == Assembler::NotEqual);
  bool isUnsigned = (cond & Assembler::ConditionUnsigned) != 0;
  Imm32 narrowed(0);
  if (isEqOrNe || isUnsigned) {
    load8ZeroExtend(lhs, scratch);
    narrowed = Imm32(uint8_t(rhs.value));
  } else {
    load8SignExtend(lhs, scratch);
    narrowed = Imm32(int8_t(rhs.value));
  }
  Condition c = ma_cmp(scratch, narrowed, cond, true);
  ma_b(c, label);
}

void MacroAssembler::branch8(Condition cond, const BaseIndex& lhs, Register rhs,
                             Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load8ZeroExtend(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs, cond, true);
  ma_b(c, label);
}

void MacroAssembler::branch16(Condition cond, const Address& lhs, Imm32 rhs,
                              Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  // See branch8: narrow the immediate to 16 bits so both sides have matching
  // bit patterns. uint16 for equality / unsigned, int16 for signed relational.
  bool isEqOrNe = (cond == Assembler::Equal) || (cond == Assembler::NotEqual);
  bool isUnsigned = (cond & Assembler::ConditionUnsigned) != 0;
  Imm32 narrowed(0);
  if (isEqOrNe || isUnsigned) {
    load16ZeroExtend(lhs, scratch);
    narrowed = Imm32(uint16_t(rhs.value));
  } else {
    load16SignExtend(lhs, scratch);
    narrowed = Imm32(int16_t(rhs.value));
  }
  Condition c = ma_cmp(scratch, narrowed, cond, true);
  ma_b(c, label);
}

void MacroAssembler::branch32(Condition cond, Register lhs, Register rhs,
                              Label* label) {
  Condition c = ma_cmp(lhs, rhs, cond, true);
  ma_b(c, label);
}

void MacroAssembler::branch32(Condition cond, Register lhs, Imm32 imm,
                              Label* label) {
  Condition c = ma_cmp(lhs, imm, cond, true);
  ma_b(c, label);
}

void MacroAssembler::branch32(Condition cond, const Address& lhs, Register rhs,
                              Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs, cond, true);
  ma_b(c, label);
}

void MacroAssembler::branch32(Condition cond, const Address& lhs, Imm32 rhs,
                              Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs, cond, true);
  ma_b(c, label);
}

void MacroAssembler::branch32(Condition cond, const AbsoluteAddress& lhs,
                              Register rhs, Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord((uintptr_t)lhs.addr), scratch);
  load32(Address(scratch, 0), scratch);
  Condition c = ma_cmp(scratch, rhs, cond, true);
  ma_b(c, label);
}

void MacroAssembler::branch32(Condition cond, const AbsoluteAddress& lhs,
                              Imm32 rhs, Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord((uintptr_t)lhs.addr), scratch);
  load32(Address(scratch, 0), scratch);
  Condition c = ma_cmp(scratch, rhs, cond, true);
  ma_b(c, label);
}

void MacroAssembler::branch32(Condition cond, const BaseIndex& lhs, Imm32 rhs,
                              Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs, cond, true);
  ma_b(c, label);
}

void MacroAssembler::branch32(Condition cond, wasm::SymbolicAddress addr,
                              Imm32 imm, Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(addr, scratch);
  load32(Address(scratch, 0), scratch);
  Condition c = ma_cmp(scratch, imm, cond, true);
  ma_b(c, label);
}

void MacroAssembler::branch64(Condition cond, Register64 lhs, Imm64 val,
                              Label* success, Label* fail) {
  Condition c = ma_cmp(lhs.reg, ImmWord(uint64_t(val.value)), cond);
  if (fail) {
    ma_b(c, success);
    jump(fail);
  } else {
    ma_b(c, success);
  }
}

void MacroAssembler::branch64(Condition cond, Register64 lhs, Register64 rhs,
                              Label* success, Label* fail) {
  Condition c = ma_cmp(lhs.reg, rhs.reg, cond);
  if (fail) {
    ma_b(c, success);
    jump(fail);
  } else {
    ma_b(c, success);
  }
}

void MacroAssembler::branch64(Condition cond, const Address& lhs, Imm64 val,
                              Label* success, Label* fail) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Condition c = ma_cmp(scratch, ImmWord(uint64_t(val.value)), cond);
  if (fail) {
    ma_b(c, success);
    jump(fail);
  } else {
    ma_b(c, success);
  }
}

void MacroAssembler::branch64(Condition cond, const Address& lhs,
                              Register64 rhs, Label* success, Label* fail) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs.reg, cond);
  if (fail) {
    ma_b(c, success);
    jump(fail);
  } else {
    ma_b(c, success);
  }
}

void MacroAssembler::branch64(Condition cond, const Address& lhs,
                              const Address& rhs, Register scratch,
                              Label* label) {
  loadPtr(rhs, scratch);
  branch64(cond, lhs, Register64(scratch), label, nullptr);
}

void MacroAssembler::branchPtr(Condition cond, Register lhs, Register rhs,
                               Label* label) {
  Condition c = ma_cmp(lhs, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, Register lhs, Imm32 rhs,
                               Label* label) {
  Condition c = ma_cmp(lhs, ImmWord(int64_t(rhs.value)), cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, Register lhs, ImmPtr rhs,
                               Label* label) {
  Condition c = ma_cmp(lhs, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, Register lhs, ImmGCPtr rhs,
                               Label* label) {
  Condition c = ma_cmp(lhs, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, Register lhs, ImmWord rhs,
                               Label* label) {
  Condition c = ma_cmp(lhs, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, const Address& lhs, Register rhs,
                               Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, const Address& lhs, ImmPtr rhs,
                               Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, const Address& lhs, ImmGCPtr rhs,
                               Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, const Address& lhs, ImmWord rhs,
                               Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, const AbsoluteAddress& lhs,
                               Register rhs, Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord((uintptr_t)lhs.addr), scratch);
  loadPtr(Address(scratch, 0), scratch);
  Condition c = ma_cmp(scratch, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, const AbsoluteAddress& lhs,
                               ImmWord rhs, Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord((uintptr_t)lhs.addr), scratch);
  loadPtr(Address(scratch, 0), scratch);
  Condition c = ma_cmp(scratch, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, wasm::SymbolicAddress lhs,
                               Register rhs, Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(lhs, scratch);
  loadPtr(Address(scratch, 0), scratch);
  Condition c = ma_cmp(scratch, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, const BaseIndex& lhs,
                               Register rhs, Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPtr(Condition cond, const BaseIndex& lhs,
                               ImmWord rhs, Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs, cond);
  ma_b(c, label);
}

void MacroAssembler::branchPrivatePtr(Condition cond, const Address& lhs,
                                      Register rhs, Label* label) {
  branchPtr(cond, lhs, rhs, label);
}

void MacroAssembler::branchFloat(DoubleCondition cond, FloatRegister lhs,
                                 FloatRegister rhs, Label* label) {
  as_fcmpu(lhs, rhs);
  ma_b(cond, label);
}

void MacroAssembler::branchTruncateFloat32MaybeModUint32(FloatRegister src,
                                                         Register dest,
                                                         Label* fail) {
  // Convert float32 to int64 (truncating toward zero), fail on NaN/overflow.
  as_fctidz(ScratchDoubleReg, src);
  as_mfvsrd(dest, ScratchDoubleReg);
  // PPC64 fctidz saturates to INT64_MIN on negative overflow/NaN,
  // and to INT64_MAX on positive overflow. Check both.
  asMasm().branchPtr(Assembler::Equal, dest, ImmWord(int64_t(INT64_MIN)), fail);
  asMasm().branchPtr(Assembler::Equal, dest, ImmWord(int64_t(INT64_MAX)), fail);
  // Truncate to uint32 (keep low 32 bits).
  as_rldicl(dest, dest, 0, 32);
}

void MacroAssembler::branchTruncateFloat32ToInt32(FloatRegister src,
                                                  Register dest, Label* fail) {
  convertFloat32ToInt32(src, dest, fail, false);
}

void MacroAssembler::branchDouble(DoubleCondition cond, FloatRegister lhs,
                                  FloatRegister rhs, Label* label) {
  as_fcmpu(lhs, rhs);
  ma_b(cond, label);
}

void MacroAssembler::branchTruncateDoubleMaybeModUint32(FloatRegister src,
                                                        Register dest,
                                                        Label* fail) {
  // Convert double to int64 (truncating toward zero), fail on NaN/overflow.
  as_fctidz(ScratchDoubleReg, src);
  as_mfvsrd(dest, ScratchDoubleReg);
  // PPC64 fctidz saturates to INT64_MIN on negative overflow/NaN,
  // and to INT64_MAX on positive overflow. Check both.
  asMasm().branchPtr(Assembler::Equal, dest, ImmWord(int64_t(INT64_MIN)), fail);
  asMasm().branchPtr(Assembler::Equal, dest, ImmWord(int64_t(INT64_MAX)), fail);
  // Truncate to uint32 (keep low 32 bits).
  as_rldicl(dest, dest, 0, 32);
}

void MacroAssembler::branchTruncateDoubleToInt32(FloatRegister src,
                                                 Register dest, Label* fail) {
  convertDoubleToInt32(src, dest, fail, false);
}

void MacroAssembler::branchInt64NotInPtrRange(Register64 src, Label* label) {
  // No-op on 64-bit.
}

void MacroAssembler::branchUInt64NotInPtrRange(Register64 src, Label* label) {
  // Branch if src >= 2^63 (sign bit set = out of signed ptr range).
  as_cmpdi(src.reg, 0);
  ma_b(Assembler::LessThan, label);
}

template <typename T>
void MacroAssembler::branchAdd32(Condition cond, T src, Register dest,
                                 Label* overflow) {
  switch (cond) {
    case Overflow: {
      // Do raw 64-bit add (no sign extension) so we can detect 32-bit overflow.
      // Both inputs should already be sign-extended 32-bit values, so the
      // 64-bit result is mathematically correct. If extsw(result) != result,
      // the 32-bit add overflowed.
      UseScratchRegisterScope temps(asMasm());
      Register scratch = temps.Acquire();
      addPtr(src, dest);
      as_extsw(scratch, dest);
      as_cmpd(dest, scratch);
      as_extsw(dest, dest);
      ma_b(NotEqual, overflow);
      break;
    }
    case NonZero:
    case Zero:
      add32(src, dest);
      as_cmpdi(dest, 0);
      ma_b(cond == NonZero ? NotEqual : Equal, overflow);
      break;
    case Signed:
    case NotSigned:
      add32(src, dest);
      as_cmpdi(dest, 0);
      ma_b(cond == Signed ? LessThan : GreaterThanOrEqual, overflow);
      break;
    case CarryClear:
    case CarrySet: {
      // Unsigned 32-bit carry detection: save dest, do 32-bit add,
      // then unsigned-compare result with original. If result < original
      // (unsigned), a carry occurred.
      UseScratchRegisterScope temps(asMasm());
      Register scratch = temps.Acquire();
      move32(dest, scratch);
      add32(src, dest);
      as_cmplw(dest, scratch);
      ma_b(cond == CarrySet ? LessThan : GreaterThanOrEqual, overflow);
      break;
    }
    default:
      MOZ_CRASH("NYI");
  }
}

template <typename T>
void MacroAssembler::branchSub32(Condition cond, T src, Register dest,
                                 Label* overflow) {
  switch (cond) {
    case Overflow: {
      // Do raw 64-bit sub (no sign extension) so we can detect 32-bit overflow.
      UseScratchRegisterScope temps(asMasm());
      Register scratch = temps.Acquire();
      subPtr(src, dest);
      as_extsw(scratch, dest);
      as_cmpd(dest, scratch);
      as_extsw(dest, dest);
      ma_b(NotEqual, overflow);
      break;
    }
    case NonZero:
    case Zero:
      sub32(src, dest);
      as_cmpdi(dest, 0);
      ma_b(cond == NonZero ? NotEqual : Equal, overflow);
      break;
    case Signed:
    case NotSigned:
      sub32(src, dest);
      as_cmpdi(dest, 0);
      ma_b(cond == Signed ? LessThan : GreaterThanOrEqual, overflow);
      break;
    default:
      MOZ_CRASH("NYI");
  }
}

template <typename T>
void MacroAssembler::branchMul32(Condition cond, T src, Register dest,
                                 Label* overflow) {
  MOZ_ASSERT(cond == Overflow);
  // Do raw 64-bit multiply (no sign extension) so we can detect 32-bit
  // overflow. as_mulld gives full 64-bit low result; if extsw(result) !=
  // result, overflow. scratch is dead after the mulld (consumed as RB),
  // so the sign-extension round-trip reuses it instead of acquiring a
  // second scratch.
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  move32(src, scratch);
  as_mulld(dest, dest, scratch);
  as_extsw(scratch, dest);
  as_cmpd(dest, scratch);
  as_extsw(dest, dest);
  ma_b(NotEqual, overflow);
}

template <typename T>
void MacroAssembler::branchRshift32(Condition cond, T src, Register dest,
                                    Label* label) {
  MOZ_ASSERT(cond == Zero || cond == NonZero);
  rshift32(src, dest);
  branch32(cond == Zero ? Equal : NotEqual, dest, Imm32(0), label);
}

void MacroAssembler::branchNeg32(Condition cond, Register reg, Label* label) {
  MOZ_ASSERT(cond == Overflow);
  neg32(reg);
  branch32(Equal, reg, Imm32(INT32_MIN), label);
}

template <typename T>
void MacroAssembler::branchAddPtr(Condition cond, T src, Register dest,
                                  Label* label) {
  switch (cond) {
    case Overflow: {
      UseScratchRegisterScope temps(asMasm());
      Register scratch = temps.Acquire();
      movePtr(dest, scratch);  // scratch = old_dest
      addPtr(src, dest);       // dest = result = old_dest + src
      as_xor_(SecondScratchReg, dest,
              scratch);  // SecondScratch = result ^ old_dest
      as_subf(scratch, scratch,
              dest);  // scratch = result - old_dest = src_value
      as_xor_(scratch, scratch, dest);  // scratch = src_value ^ result
      // (old_dest ^ result) & (src_value ^ result): bit 63 set iff overflow.
      // and. record form sets CR0[lt]=(bit 63 set), folding the cmpdi.
      as_and__rc(scratch, scratch, SecondScratchReg);
      ma_b(LessThan, label);
      break;
    }
    case NonZero:
    case Zero:
      addPtr(src, dest);
      as_cmpdi(dest, 0);
      ma_b(cond == NonZero ? NotEqual : Equal, label);
      break;
    case Signed:
    case NotSigned:
      addPtr(src, dest);
      as_cmpdi(dest, 0);
      ma_b(cond == Signed ? LessThan : GreaterThanOrEqual, label);
      break;
    case CarryClear:
    case CarrySet: {
      // Unsigned 64-bit carry detection: save dest, do 64-bit add,
      // then unsigned-compare result with original. If result < original
      // (unsigned), a carry occurred.
      UseScratchRegisterScope temps(asMasm());
      Register scratch = temps.Acquire();
      movePtr(dest, scratch);
      addPtr(src, dest);
      as_cmpld(dest, scratch);
      ma_b(cond == CarrySet ? LessThan : GreaterThanOrEqual, label);
      break;
    }
    default:
      MOZ_CRASH("NYI");
  }
}

template <typename T>
void MacroAssembler::branchSubPtr(Condition cond, T src, Register dest,
                                  Label* label) {
  switch (cond) {
    case Overflow: {
      UseScratchRegisterScope temps(asMasm());
      Register scratch = temps.Acquire();
      movePtr(dest, scratch);  // scratch = old_dest
      subPtr(src, dest);       // dest = result = old_dest - src
      // Overflow if (old_dest ^ src_value) & (old_dest ^ result) has bit 63
      // set.
      as_subf(SecondScratchReg, dest,
              scratch);  // SecondScratch = old_dest - result = src_value
      as_xor_(SecondScratchReg, scratch,
              SecondScratchReg);        // old_dest ^ src_value
      as_xor_(scratch, scratch, dest);  // old_dest ^ result
      // Record-form AND sets CR0 to the signed compare of the result vs 0,
      // so a separate cmpdi is unnecessary; LessThan reads CR0.LT.
      as_and__rc(scratch, scratch, SecondScratchReg);
      ma_b(LessThan, label);
      break;
    }
    case NonZero:
    case Zero:
      subPtr(src, dest);
      as_cmpdi(dest, 0);
      ma_b(cond == NonZero ? NotEqual : Equal, label);
      break;
    case Signed:
    case NotSigned:
      subPtr(src, dest);
      as_cmpdi(dest, 0);
      ma_b(cond == Signed ? LessThan : GreaterThanOrEqual, label);
      break;
    default:
      MOZ_CRASH("NYI");
  }
}

void MacroAssembler::branchMulPtr(Condition cond, Register src, Register dest,
                                  Label* label) {
  MOZ_ASSERT(cond == Assembler::Overflow);
  as_mulldo(dest, dest, src);
  ma_b(Overflow, label);
}

void MacroAssembler::branchNegPtr(Condition cond, Register reg, Label* label) {
  MOZ_ASSERT(cond == Overflow);
  negPtr(reg);
  branchPtr(Assembler::Equal, reg, ImmWord(intptr_t(INTPTR_MIN)), label);
}

void MacroAssembler::decBranchPtr(Condition cond, Register lhs, Imm32 rhs,
                                  Label* label) {
  subPtr(rhs, lhs);
  branchPtr(cond, lhs, Imm32(0), label);
}

void MacroAssembler::branchTest32(Condition cond, Register lhs, Register rhs,
                                  Label* label) {
  MOZ_ASSERT(cond == Zero || cond == NonZero || cond == Signed ||
             cond == NotSigned);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  if (lhs != rhs) {
    as_and_(scratch, lhs, rhs);
    as_extsw_rc(scratch, scratch);  // CR0 set on sign-extended i32; folds cmpdi
  } else {
    as_extsw_rc(scratch, lhs);
  }
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  ma_b(base, label);
}

void MacroAssembler::branchTest32(Condition cond, Register lhs, Imm32 rhs,
                                  Label* label) {
  MOZ_ASSERT(cond == Zero || cond == NonZero || cond == Signed ||
             cond == NotSigned);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  if (is_uintN(rhs.value, 16)) {
    as_andi_rc(scratch, lhs, rhs.value);
    // andi_rc sets CR0 on the masked value, but only the low 16 bits matter
    // since rhs is a 16-bit unsigned mask — sign of the i32 result is always
    // 0, so CR0[lt] is always 0. For Signed/NotSigned conditions the answer
    // is fixed; for Zero/NonZero CR0[eq] is correct.
  } else {
    move32(rhs, scratch);
    as_and_(scratch, lhs, scratch);
    as_extsw_rc(scratch, scratch);  // CR0 set on sign-extended i32; folds cmpdi
  }
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  ma_b(base, label);
}

void MacroAssembler::branchTest32(Condition cond, const Address& lhs, Imm32 rhs,
                                  Label* label) {
  MOZ_ASSERT(cond == Zero || cond == NonZero || cond == Signed ||
             cond == NotSigned);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(lhs, scratch);
  // and32 picks up the rlwinm contig-mask fast path for non-16-bit-fit
  // immediates that are a contiguous run of 1-bits (common: tag masks,
  // header bit-fields). It also emits the trailing extsw.
  and32(rhs, scratch);
  as_cmpdi(scratch, 0);
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  ma_b(base, label);
}

void MacroAssembler::branchTest32(Condition cond, const AbsoluteAddress& lhs,
                                  Imm32 rhs, Label* label) {
  MOZ_ASSERT(cond == Zero || cond == NonZero || cond == Signed ||
             cond == NotSigned);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord((uintptr_t)lhs.addr), scratch);
  load32(Address(scratch, 0), scratch);
  and32(rhs, scratch);
  as_cmpdi(scratch, 0);
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  ma_b(base, label);
}

void MacroAssembler::branchTestPtr(Condition cond, Register lhs, Register rhs,
                                   Label* label) {
  MOZ_ASSERT(cond == Zero || cond == NonZero || cond == Signed ||
             cond == NotSigned);
  if (lhs == rhs) {
    as_cmpdi(lhs, 0);
  } else {
    UseScratchRegisterScope temps(asMasm());
    Register scratch = temps.Acquire();
    // Record-form AND sets CR0; no follow-up cmpdi needed.
    as_and__rc(scratch, lhs, rhs);
  }
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  ma_b(base, label);
}

void MacroAssembler::branchTestPtr(Condition cond, Register lhs, Imm32 rhs,
                                   Label* label) {
  MOZ_ASSERT(cond == Zero || cond == NonZero || cond == Signed ||
             cond == NotSigned);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  if (is_uintN(rhs.value, 16)) {
    as_andi_rc(scratch, lhs, rhs.value);
  } else {
    move32(rhs, scratch);
    as_and__rc(scratch, lhs, scratch);  // record form folds the cmpdi
  }
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  ma_b(base, label);
}

void MacroAssembler::branchTestPtr(Condition cond, Register lhs, ImmWord rhs,
                                   Label* label) {
  MOZ_ASSERT(cond == Zero || cond == NonZero || cond == Signed ||
             cond == NotSigned);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(rhs, scratch);
  as_and__rc(scratch, lhs, scratch);
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  ma_b(base, label);
}

void MacroAssembler::branchTestPtr(Condition cond, const Address& lhs,
                                   Imm32 rhs, Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  // andPtr picks up the rlwinm contig-mask fast path for non-16-bit-fit
  // immediates that are a contiguous run of 1-bits.
  andPtr(rhs, scratch);
  as_cmpdi(scratch, 0);
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  ma_b(base, label);
}

void MacroAssembler::branchTest64(Condition cond, Register64 lhs,
                                  Register64 rhs, Register temp, Label* success,
                                  Label* fail) {
  MOZ_ASSERT(cond == Zero || cond == NonZero || cond == Signed ||
             cond == NotSigned);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  as_and__rc(scratch, lhs.reg, rhs.reg);
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  if (fail) {
    ma_b(base, success);
    jump(fail);
  } else {
    ma_b(base, success);
  }
}

void MacroAssembler::branchTest64(Condition cond, Register64 lhs, Imm64 rhs,
                                  Label* success, Label* fail) {
  MOZ_ASSERT(cond == Zero || cond == NonZero || cond == Signed ||
             cond == NotSigned);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord(uint64_t(rhs.value)), scratch);
  as_and__rc(scratch, lhs.reg, scratch);
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  if (fail) {
    ma_b(base, success);
    jump(fail);
  } else {
    ma_b(base, success);
  }
}

// ===============================================================
// Value-type branch functions

void MacroAssembler::branchTestUndefined(Condition cond, Register tag,
                                         Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_UNDEFINED), cond);
  ma_b(c, label);
}

void MacroAssembler::branchTestUndefined(Condition cond,
                                         const ValueOperand& value,
                                         Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(value, scratch);
  branchTestTag(cond, scratch, ImmTag(JSVAL_TAG_UNDEFINED), label);
}

void MacroAssembler::branchTestUndefined(Condition cond, const Address& address,
                                         Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_UNDEFINED), label);
}

void MacroAssembler::branchTestUndefined(Condition cond,
                                         const BaseIndex& address,
                                         Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_UNDEFINED), label);
}

void MacroAssembler::branchTestInt32(Condition cond, Register tag,
                                     Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_INT32), cond);
  ma_b(c, label);
}

void MacroAssembler::branchTestInt32(Condition cond, const ValueOperand& value,
                                     Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(value, scratch);
  branchTestTag(cond, scratch, ImmTag(JSVAL_TAG_INT32), label);
}

void MacroAssembler::branchTestInt32(Condition cond, const Address& address,
                                     Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_INT32), label);
}

void MacroAssembler::branchTestInt32(Condition cond, const BaseIndex& address,
                                     Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_INT32), label);
}

void MacroAssembler::branchTestInt32Truthy(bool b, const ValueOperand& value,
                                           Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  unboxInt32(value, scratch);
  as_cmpwi(scratch, 0);
  ma_b(b ? NotEqual : Equal, label);
}

void MacroAssembler::branchTestDouble(Condition cond, Register tag,
                                      Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  Condition actual = (cond == Equal) ? BelowOrEqual : Above;
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_MAX_DOUBLE), actual);
  ma_b(c, label);
}

void MacroAssembler::branchTestDouble(Condition cond, const ValueOperand& value,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(value, scratch);
  branchTestDouble(cond, scratch, label);
}

void MacroAssembler::branchTestDouble(Condition cond, const Address& address,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestDouble(cond, tag, label);
}

void MacroAssembler::branchTestDouble(Condition cond, const BaseIndex& address,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestDouble(cond, tag, label);
}

void MacroAssembler::branchTestDoubleTruthy(bool b, FloatRegister value,
                                            Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  xs_li(scratch, 0);
  as_mtvsrd(ScratchDoubleReg, scratch);
  as_fcmpu(value, ScratchDoubleReg);
  DoubleCondition cond = b ? DoubleNotEqual : DoubleEqualOrUnordered;
  ma_b(cond, label);
}

void MacroAssembler::branchTestNumber(Condition cond, Register tag,
                                      Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  Condition actual = (cond == Equal) ? BelowOrEqual : Above;
  Condition c = ma_cmp(tag, Imm32(JS::detail::ValueUpperInclNumberTag), actual);
  ma_b(c, label);
}

void MacroAssembler::branchTestNumber(Condition cond, const ValueOperand& value,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(value, scratch);
  branchTestNumber(cond, scratch, label);
}

void MacroAssembler::branchTestBoolean(Condition cond, Register tag,
                                       Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_BOOLEAN), cond);
  ma_b(c, label);
}

void MacroAssembler::branchTestBoolean(Condition cond,
                                       const ValueOperand& value,
                                       Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(value, scratch);
  branchTestTag(cond, scratch, ImmTag(JSVAL_TAG_BOOLEAN), label);
}

void MacroAssembler::branchTestBoolean(Condition cond, const Address& address,
                                       Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_BOOLEAN), label);
}

void MacroAssembler::branchTestBoolean(Condition cond, const BaseIndex& address,
                                       Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_BOOLEAN), label);
}

void MacroAssembler::branchTestBooleanTruthy(bool b, const ValueOperand& value,
                                             Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  unboxBoolean(value, scratch);
  as_cmpwi(scratch, 0);
  ma_b(b ? NotEqual : Equal, label);
}

void MacroAssembler::branchTestString(Condition cond, Register tag,
                                      Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_STRING), cond);
  ma_b(c, label);
}

void MacroAssembler::branchTestString(Condition cond, const ValueOperand& value,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(value, scratch);
  branchTestTag(cond, scratch, ImmTag(JSVAL_TAG_STRING), label);
}

void MacroAssembler::branchTestString(Condition cond, const Address& address,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_STRING), label);
}

void MacroAssembler::branchTestString(Condition cond, const BaseIndex& address,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_STRING), label);
}

void MacroAssembler::branchTestStringTruthy(bool b, const ValueOperand& value,
                                            Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  unboxString(value, scratch);
  load32(Address(scratch, JSString::offsetOfLength()), scratch);
  as_cmpwi(scratch, 0);
  ma_b(b ? NotEqual : Equal, label);
}

void MacroAssembler::branchTestSymbol(Condition cond, Register tag,
                                      Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_SYMBOL), cond);
  ma_b(c, label);
}

void MacroAssembler::branchTestSymbol(Condition cond, const ValueOperand& value,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(value, scratch);
  branchTestTag(cond, scratch, ImmTag(JSVAL_TAG_SYMBOL), label);
}

void MacroAssembler::branchTestSymbol(Condition cond, const BaseIndex& address,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_SYMBOL), label);
}

void MacroAssembler::branchTestSymbol(Condition cond, const Address& address,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_SYMBOL), label);
}

void MacroAssembler::branchTestBigInt(Condition cond, Register tag,
                                      Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_BIGINT), cond);
  ma_b(c, label);
}

void MacroAssembler::branchTestBigInt(Condition cond, const ValueOperand& value,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(value, scratch);
  branchTestTag(cond, scratch, ImmTag(JSVAL_TAG_BIGINT), label);
}

void MacroAssembler::branchTestBigInt(Condition cond, const Address& address,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_BIGINT), label);
}

void MacroAssembler::branchTestBigInt(Condition cond, const BaseIndex& address,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_BIGINT), label);
}

void MacroAssembler::branchTestBigIntTruthy(bool b, const ValueOperand& value,
                                            Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  unboxBigInt(value, scratch);
  load32(Address(scratch, BigInt::offsetOfDigitLength()), scratch);
  as_cmpwi(scratch, 0);
  ma_b(b ? NotEqual : Equal, label);
}

void MacroAssembler::branchTestNull(Condition cond, Register tag,
                                    Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_NULL), cond);
  ma_b(c, label);
}

void MacroAssembler::branchTestNull(Condition cond, const ValueOperand& value,
                                    Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(value, scratch);
  branchTestTag(cond, scratch, ImmTag(JSVAL_TAG_NULL), label);
}

void MacroAssembler::branchTestNull(Condition cond, const Address& address,
                                    Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_NULL), label);
}

void MacroAssembler::branchTestNull(Condition cond, const BaseIndex& address,
                                    Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_NULL), label);
}

void MacroAssembler::branchTestObject(Condition cond, Register tag,
                                      Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_OBJECT), cond);
  ma_b(c, label);
}

void MacroAssembler::branchTestObject(Condition cond, const ValueOperand& value,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(value, scratch);
  branchTestTag(cond, scratch, ImmTag(JSVAL_TAG_OBJECT), label);
}

void MacroAssembler::branchTestObject(Condition cond, const Address& address,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_OBJECT), label);
}

void MacroAssembler::branchTestObject(Condition cond, const BaseIndex& address,
                                      Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_OBJECT), label);
}

void MacroAssembler::branchTestPrimitive(Condition cond,
                                         const ValueOperand& value,
                                         Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(value, scratch);
  branchTestPrimitive(cond, scratch, label);
}

void MacroAssembler::branchTestGCThing(Condition cond, const Address& address,
                                       Label* label) {
  branchTestGCThingImpl(cond, address, label);
}

void MacroAssembler::branchTestGCThing(Condition cond, const BaseIndex& address,
                                       Label* label) {
  branchTestGCThingImpl(cond, address, label);
}

void MacroAssembler::branchTestGCThing(Condition cond,
                                       const ValueOperand& address,
                                       Label* label) {
  branchTestGCThingImpl(cond, address, label);
}

template <typename T>
void MacroAssembler::branchTestGCThingImpl(Condition cond, const T& address,
                                           Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  Condition actual = (cond == Equal) ? AboveOrEqual : Below;
  Condition c =
      ma_cmp(tag, Imm32(JS::detail::ValueLowerInclGCThingTag), actual);
  ma_b(c, label);
}

void MacroAssembler::branchTestPrimitive(Condition cond, Register tag,
                                         Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  Condition actual = (cond == Equal) ? Below : AboveOrEqual;
  Condition c =
      ma_cmp(tag, Imm32(JS::detail::ValueUpperExclPrimitiveTag), actual);
  ma_b(c, label);
}

void MacroAssembler::branchTestMagic(Condition cond, Register tag,
                                     Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_MAGIC), cond);
  ma_b(c, label);
}

void MacroAssembler::branchTestMagic(Condition cond, const Address& address,
                                     Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_MAGIC), label);
}

void MacroAssembler::branchTestMagic(Condition cond, const BaseIndex& address,
                                     Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(address, scratch);
  branchTestTag(cond, tag, ImmTag(JSVAL_TAG_MAGIC), label);
}

void MacroAssembler::branchTestMagic(Condition cond, const ValueOperand& value,
                                     Label* label) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(value, scratch);
  branchTestTag(cond, scratch, ImmTag(JSVAL_TAG_MAGIC), label);
}

void MacroAssembler::branchTestMagic(Condition cond, const Address& valaddr,
                                     JSWhyMagic why, Label* label) {
  uint64_t magic = MagicValue(why).asRawBits();
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(valaddr, scratch);
  Condition c = ma_cmp(scratch, ImmWord(magic), cond);
  ma_b(c, label);
}

void MacroAssembler::branchTestMagic(Condition cond, const BaseIndex& valaddr,
                                     JSWhyMagic why, Label* label) {
  uint64_t magic = MagicValue(why).asRawBits();
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(valaddr, scratch);
  Condition c = ma_cmp(scratch, ImmWord(magic), cond);
  ma_b(c, label);
}

template <typename T>
void MacroAssembler::branchTestValue(Condition cond, const T& lhs,
                                     const ValueOperand& rhs, Label* label) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs.valueReg(), cond);
  ma_b(c, label);
}

// ===============================================================
// Test-set functions

template <typename T>
void MacroAssembler::testNumberSet(Condition cond, const T& src,
                                   Register dest) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(src, scratch);
  Condition actual = (cond == Equal) ? BelowOrEqual : Above;
  Condition c = ma_cmp(tag, Imm32(JS::detail::ValueUpperInclNumberTag), actual);
  ma_cmp_set(dest, c);
}

template <typename T>
void MacroAssembler::testBooleanSet(Condition cond, const T& src,
                                    Register dest) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(src, scratch);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_BOOLEAN), cond);
  ma_cmp_set(dest, c);
}

template <typename T>
void MacroAssembler::testStringSet(Condition cond, const T& src,
                                   Register dest) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(src, scratch);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_STRING), cond);
  ma_cmp_set(dest, c);
}

template <typename T>
void MacroAssembler::testSymbolSet(Condition cond, const T& src,
                                   Register dest) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(src, scratch);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_SYMBOL), cond);
  ma_cmp_set(dest, c);
}

template <typename T>
void MacroAssembler::testBigIntSet(Condition cond, const T& src,
                                   Register dest) {
  MOZ_ASSERT(cond == Equal || cond == NotEqual);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  Register tag = extractTag(src, scratch);
  Condition c = ma_cmp(tag, ImmTag(JSVAL_TAG_BIGINT), cond);
  ma_cmp_set(dest, c);
}

// ===============================================================
// Computed address / conditional move / conditional load

void MacroAssembler::branchToComputedAddress(const BaseIndex& addr) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(addr, scratch);
  branch(scratch);
}

void MacroAssembler::cmp32Move32(Condition cond, Register lhs, Imm32 rhs,
                                 Register src, Register dest) {
  Condition c = ma_cmp(lhs, rhs, cond, true);
  ma_cmp_move(dest, src, c);
}

void MacroAssembler::cmp32Move32(Condition cond, Register lhs, Register rhs,
                                 Register src, Register dest) {
  Condition c = ma_cmp(lhs, rhs, cond, true);
  ma_cmp_move(dest, src, c);
}

void MacroAssembler::cmp32Move32(Condition cond, Register lhs,
                                 const Address& rhs, Register src,
                                 Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(rhs, scratch);
  Condition c = ma_cmp(lhs, scratch, cond, true);
  ma_cmp_move(dest, src, c);
}

void MacroAssembler::cmp32MovePtr(Condition cond, Register lhs, Imm32 rhs,
                                  Register src, Register dest) {
  Condition c = ma_cmp(lhs, rhs, cond, true);
  ma_cmp_move(dest, src, c);
}

void MacroAssembler::cmpPtrMovePtr(Condition cond, Register lhs, Imm32 rhs,
                                   Register src, Register dest) {
  Condition c = ma_cmp(lhs, ImmWord(int64_t(rhs.value)), cond);
  ma_cmp_move(dest, src, c);
}

void MacroAssembler::cmpPtrMovePtr(Condition cond, Register lhs, Register rhs,
                                   Register src, Register dest) {
  Condition c = ma_cmp(lhs, rhs, cond);
  ma_cmp_move(dest, src, c);
}

void MacroAssembler::cmpPtrMovePtr(Condition cond, Register lhs,
                                   const Address& rhs, Register src,
                                   Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(rhs, scratch);
  Condition c = ma_cmp(lhs, scratch, cond);
  ma_cmp_move(dest, src, c);
}

void MacroAssembler::cmp32Load32(Condition cond, Register lhs,
                                 const Address& rhs, const Address& src,
                                 Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(rhs, scratch);
  Condition c = ma_cmp(lhs, scratch, cond, true);
  // Conditional load: load into scratch, then isel.
  load32(src, scratch);
  ma_cmp_move(dest, scratch, c);
}

void MacroAssembler::cmp32Load32(Condition cond, Register lhs, Register rhs,
                                 const Address& src, Register dest) {
  Condition c = ma_cmp(lhs, rhs, cond, true);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(src, scratch);
  ma_cmp_move(dest, scratch, c);
}

void MacroAssembler::cmp32Load32(Condition cond, Register lhs, Imm32 rhs,
                                 const Address& src, Register dest) {
  Condition c = ma_cmp(lhs, rhs, cond, true);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(src, scratch);
  ma_cmp_move(dest, scratch, c);
}

void MacroAssembler::cmp32LoadPtr(Condition cond, const Address& lhs, Imm32 rhs,
                                  const Address& src, Register dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(lhs, scratch);
  Condition c = ma_cmp(scratch, rhs, cond, true);
  loadPtr(src, scratch);
  ma_cmp_move(dest, scratch, c);
}

void MacroAssembler::test32LoadPtr(Condition cond, const Address& addr,
                                   Imm32 mask, const Address& src,
                                   Register dest) {
  MOZ_ASSERT(cond == Zero || cond == NonZero);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(addr, scratch);
  if (is_uintN(mask.value, 16)) {
    as_andi_rc(scratch, scratch, mask.value);
  } else {
    // Use a nested scope so scratch2 is released before loadPtr below.
    UseScratchRegisterScope temps2(asMasm());
    Register scratch2 = temps2.Acquire();
    move32(mask, scratch2);
    as_and__rc(scratch, scratch, scratch2);  // record form folds the cmpdi
  }
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  loadPtr(src, scratch);
  ma_cmp_move(dest, scratch, base);
}

void MacroAssembler::test32MovePtr(Condition cond, Register operand, Imm32 mask,
                                   Register src, Register dest) {
  MOZ_ASSERT(cond == Zero || cond == NonZero);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  if (is_uintN(mask.value, 16)) {
    as_andi_rc(scratch, operand, mask.value);
  } else {
    move32(mask, scratch);
    as_and__rc(scratch, operand, scratch);  // record form folds the cmpdi
  }
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  ma_cmp_move(dest, src, base);
}

void MacroAssembler::test32MovePtr(Condition cond, const Address& addr,
                                   Imm32 mask, Register src, Register dest) {
  MOZ_ASSERT(cond == Zero || cond == NonZero);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(addr, scratch);
  and32(mask, scratch);
  as_cmpdi(scratch, 0);
  Condition base = static_cast<Condition>(cond & ~Assembler::ConditionZero);
  ma_cmp_move(dest, src, base);
}

// ===============================================================
// Spectre mitigations

void MacroAssembler::spectreMovePtr(Condition cond, Register src,
                                    Register dest) {
  // Assumes compare already issued.
  Condition base = static_cast<Condition>(
      cond & ~(Assembler::ConditionUnsigned | Assembler::ConditionZero));
  ma_cmp_move(dest, src, base);
}

void MacroAssembler::spectreZeroRegister(Condition cond, Register scratch,
                                         Register dest) {
  // Assumes compare already issued. Zero dest if condition is true.
  Condition origBase = static_cast<Condition>(
      cond & ~(Assembler::ConditionUnsigned | Assembler::ConditionZero));
  // If original condition is true, we want dest=0.
  // isel: if condition true, select zero; else keep dest.
  xs_li(scratch, 0);
  ma_cmp_move(dest, scratch, origBase);
}

void MacroAssembler::spectreBoundsCheck32(Register index, Register length,
                                          Register maybeScratch,
                                          Label* failure) {
  Condition c = ma_cmp(index, length, Below, true);
  if (failure) {
    ma_b(InvertCondition(c), failure);
  }
  if (maybeScratch != InvalidReg) {
    xs_li(maybeScratch, 0);
    ma_cmp_move(index, maybeScratch, InvertCondition(c));
  }
}

void MacroAssembler::spectreBoundsCheck32(Register index, const Address& length,
                                          Register maybeScratch,
                                          Label* failure) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  load32(length, scratch);
  spectreBoundsCheck32(index, scratch, maybeScratch, failure);
}

void MacroAssembler::spectreBoundsCheckPtr(Register index, Register length,
                                           Register maybeScratch,
                                           Label* failure) {
  Condition c = ma_cmp(index, length, Below);
  if (failure) {
    ma_b(InvertCondition(c), failure);
  }
  if (maybeScratch != InvalidReg) {
    xs_li(maybeScratch, 0);
    ma_cmp_move(index, maybeScratch, InvertCondition(c));
  }
}

void MacroAssembler::spectreBoundsCheckPtr(Register index,
                                           const Address& length,
                                           Register maybeScratch,
                                           Label* failure) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  loadPtr(length, scratch);
  spectreBoundsCheckPtr(index, scratch, maybeScratch, failure);
}

// ===============================================================
// Memory access primitives

FaultingCodeOffset MacroAssembler::storeFloat32(FloatRegister src,
                                                const Address& addr) {
  MOZ_ASSERT(addr.base != r0);
  if (is_intN(addr.offset, 16)) {
    return FaultingCodeOffset(as_stfs(src, addr.base, addr.offset).getOffset());
  }
  if (HasPOWER10() && is_intN((intptr_t)addr.offset, 34)) {
    return FaultingCodeOffset(
        as_pstfs(src, addr.base, (int64_t)addr.offset, /*R=*/false)
            .getOffset());
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord(addr.offset), scratch);
  return FaultingCodeOffset(as_stfsx(src, addr.base, scratch).getOffset());
}

FaultingCodeOffset MacroAssembler::storeFloat32(FloatRegister src,
                                                const BaseIndex& addr) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  computeEffectiveAddress(addr, scratch);
  return FaultingCodeOffset(as_stfs(src, scratch, 0).getOffset());
}

FaultingCodeOffset MacroAssembler::storeDouble(FloatRegister src,
                                               const Address& addr) {
  MOZ_ASSERT(addr.base != r0);
  if (is_intN(addr.offset, 16)) {
    return FaultingCodeOffset(as_stfd(src, addr.base, addr.offset).getOffset());
  }
  if (HasPOWER10() && is_intN((intptr_t)addr.offset, 34)) {
    return FaultingCodeOffset(
        as_pstfd(src, addr.base, (int64_t)addr.offset, /*R=*/false)
            .getOffset());
  }
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  movePtr(ImmWord(addr.offset), scratch);
  return FaultingCodeOffset(as_stfdx(src, addr.base, scratch).getOffset());
}

FaultingCodeOffset MacroAssembler::storeDouble(FloatRegister src,
                                               const BaseIndex& addr) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  computeEffectiveAddress(addr, scratch);
  return FaultingCodeOffset(as_stfd(src, scratch, 0).getOffset());
}

FaultingCodeOffset MacroAssembler::storeFloat16(FloatRegister src,
                                                const Address& dest,
                                                Register temp) {
  MOZ_ASSERT(HasPOWER9());
  if (dest.offset == 0) {
    return FaultingCodeOffset(as_stxsihx(src, r0, dest.base).getOffset());
  }
  if (is_intN(dest.offset, 16)) {
    as_addi(temp, dest.base, dest.offset);
    return FaultingCodeOffset(as_stxsihx(src, r0, temp).getOffset());
  }
  movePtr(ImmWord(dest.offset), temp);
  return FaultingCodeOffset(as_stxsihx(src, dest.base, temp).getOffset());
}

FaultingCodeOffset MacroAssembler::storeFloat16(FloatRegister src,
                                                const BaseIndex& dest,
                                                Register temp) {
  MOZ_ASSERT(HasPOWER9());
  computeEffectiveAddress(dest, temp);
  return FaultingCodeOffset(as_stxsihx(src, r0, temp).getOffset());
}

void MacroAssembler::memoryBarrier(MemoryBarrier barrier) {
  if (barrier.isNone()) {
    return;
  }
  if (barrier.hasStoreLoad() || barrier.hasSync()) {
    as_sync();
  } else {
    as_lwsync();
  }
}

// ===============================================================
// Clamping functions

void MacroAssembler::clampIntToUint8(Register reg) {
  // Clamp to [0, 255].
  Label done;
  as_cmpwi(reg, 255);
  ma_b(LessThanOrEqual, &done);
  move32(Imm32(255), reg);
  bind(&done);
  Label positive;
  as_cmpwi(reg, 0);
  ma_b(GreaterThanOrEqual, &positive);
  move32(Imm32(0), reg);
  bind(&positive);
}

// ===============================================================
// Unboxing

void MacroAssembler::fallibleUnboxPtr(const ValueOperand& src, Register dest,
                                      JSValueType type, Label* fail) {
  MOZ_ASSERT(type == JSVAL_TYPE_OBJECT || type == JSVAL_TYPE_STRING ||
             type == JSVAL_TYPE_SYMBOL || type == JSVAL_TYPE_BIGINT);
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  splitTag(src, scratch);
  Condition c = ma_cmp(scratch, ImmTag(JSVAL_TYPE_TO_TAG(type)), NotEqual);
  ma_b(c, fail);
  unboxNonDouble(src, dest, type);
}

void MacroAssembler::fallibleUnboxPtr(const Address& src, Register dest,
                                      JSValueType type, Label* fail) {
  loadValue(src, ValueOperand(dest));
  fallibleUnboxPtr(ValueOperand(dest), dest, type, fail);
}

void MacroAssembler::fallibleUnboxPtr(const BaseIndex& src, Register dest,
                                      JSValueType type, Label* fail) {
  loadValue(src, ValueOperand(dest));
  fallibleUnboxPtr(ValueOperand(dest), dest, type, fail);
}

void MacroAssembler::wasmAddSubI128HI64(Register lhsLo, Register lhsHi,
                                        Register rhsLo, Register rhsHi,
                                        Register output, bool isAdd) {
  MOZ_RELEASE_ASSERT(output != lhsLo && output != lhsHi && output != rhsLo &&
                     output != rhsHi);
  if (isAdd) {
    // addc sets CA (carry), adde uses it.
    as_addc(output, lhsLo, rhsLo);  // output = lhsLo + rhsLo, CA = carry
    as_adde(output, lhsHi, rhsHi);  // output = lhsHi + rhsHi + CA
  } else {
    // subfc: rd = rb - ra, sets CA (borrow complement).
    // subfe: rd = rb + ~ra + CA.
    as_subfc(output, rhsLo, lhsLo);  // output = lhsLo - rhsLo, CA = ~borrow
    as_subfe(output, rhsHi, lhsHi);  // output = lhsHi - rhsHi - borrow
  }
}

void MacroAssembler::wasmMulI64WideHI64(Register lhs, Register rhs,
                                        Register output, bool isSigned) {
  if (isSigned) {
    as_mulhd(output, lhs, rhs);
  } else {
    as_mulhdu(output, lhs, rhs);
  }
}

//}}} check_macroassembler_style

void MacroAssemblerPPC64Compat::incrementInt32Value(const Address& addr) {
  asMasm().add32(Imm32(1), addr);
}

void MacroAssemblerPPC64Compat::retn(Imm32 n) {
  // Load return address from [SP,0] first, then adjust SP, then return.
  // Must load RA before adjusting SP (like loong64), since the RA is at
  // the current top of stack, not at SP+n.
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  as_ld(scratch, StackPointer, 0);
  if (n.value != 0) {
    asMasm().addPtr(Imm32(n.value), StackPointer);
  }
  xs_mtlr(scratch);
  as_blr();
}

// ===============================================================
// Template specializations (outside check_macroassembler_style)

template <>
inline void MacroAssembler::cmpPtrSet(Assembler::Condition cond, Address lhs,
                                      ImmPtr rhs, Register dest) {
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Assembler::Condition c = ma_cmp(scratch, rhs, cond);
  ma_cmp_set(dest, c);
}

template <>
inline void MacroAssembler::cmpPtrSet(Assembler::Condition cond, Register lhs,
                                      Address rhs, Register dest) {
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  loadPtr(rhs, scratch);
  Assembler::Condition c = ma_cmp(lhs, scratch, cond);
  ma_cmp_set(dest, c);
}

template <>
inline void MacroAssembler::cmpPtrSet(Assembler::Condition cond, Address lhs,
                                      Register rhs, Register dest) {
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  loadPtr(lhs, scratch);
  Assembler::Condition c = ma_cmp(scratch, rhs, cond);
  ma_cmp_set(dest, c);
}

template <>
inline void MacroAssembler::cmp32Set(Assembler::Condition cond, Register lhs,
                                     Address rhs, Register dest) {
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  load32(rhs, scratch);
  Assembler::Condition c = ma_cmp(lhs, scratch, cond, true);
  ma_cmp_set(dest, c);
}

template <>
inline void MacroAssembler::cmp32Set(Assembler::Condition cond, Address lhs,
                                     Register rhs, Register dest) {
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  load32(lhs, scratch);
  Assembler::Condition c = ma_cmp(scratch, rhs, cond, true);
  ma_cmp_set(dest, c);
}

template <>
inline void MacroAssembler::cmp32Set(Assembler::Condition cond, Address lhs,
                                     Imm32 rhs, Register dest) {
  UseScratchRegisterScope temps(*this);
  Register scratch = temps.Acquire();
  load32(lhs, scratch);
  Assembler::Condition c = ma_cmp(scratch, rhs, cond, true);
  ma_cmp_set(dest, c);
}

//{{{ check_macroassembler_style
// ===============================================================
// SIMD load/store (128-bit)

FaultingCodeOffset MacroAssembler::loadUnalignedSimd128(const Address& src,
                                                        FloatRegister dest) {
  UseScratchRegisterScope temps(asMasm());
  if (HasPOWER10() && is_intN((intptr_t)src.offset, 34)) {
    // POWER10 prefixed load — natural-LE byte order, no GPR scratch.
    return FaultingCodeOffset(
        as_plxv(dest.encoding(), src.base, (int64_t)src.offset, /*R=*/false)
            .getOffset());
  }
  if (HasPOWER9()) {
    // POWER9: lxvx (X-form, indexed) loads 128 bits in correct LE order.
    Register scratch = temps.Acquire();
    if (src.offset == 0) {
      // RA=0 means "use 0 as base" in indexed forms, so use r0 encoding.
      return FaultingCodeOffset(as_lxvx(dest, r0, src.base).getOffset());
    }
    movePtr(ImmWord(src.offset), scratch);
    return FaultingCodeOffset(as_lxvx(dest, src.base, scratch).getOffset());
  }
  // POWER8: lxvd2x loads with doubleword swap on LE. Fix with xxpermdi.
  Register scratch = temps.Acquire();
  FaultingCodeOffset fco;
  if (src.offset == 0) {
    fco = FaultingCodeOffset(as_lxvd2x(dest, r0, src.base).getOffset());
  } else {
    movePtr(ImmWord(src.offset), scratch);
    fco = FaultingCodeOffset(as_lxvd2x(dest, src.base, scratch).getOffset());
  }
  as_xxpermdi(dest, dest, dest, 2);
  return fco;
}

FaultingCodeOffset MacroAssembler::loadUnalignedSimd128(const BaseIndex& src,
                                                        FloatRegister dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  computeScaledAddress(src, scratch);
  if (src.offset != 0) {
    // addPtr picks up POWER10 paddi (1 prefixed insn) when available;
    // falls back to movePtr+add on P9/P8. Drops the explicit scratch2.
    addPtr(ImmWord(src.offset), scratch);
  }
  if (HasPOWER9()) {
    return FaultingCodeOffset(as_lxvx(dest, r0, scratch).getOffset());
  }
  FaultingCodeOffset fco(as_lxvd2x(dest, r0, scratch).getOffset());
  as_xxpermdi(dest, dest, dest, 2);
  return fco;
}

FaultingCodeOffset MacroAssembler::storeUnalignedSimd128(FloatRegister src,
                                                         const Address& dest) {
  UseScratchRegisterScope temps(asMasm());
  if (HasPOWER10() && is_intN((intptr_t)dest.offset, 34)) {
    // POWER10 prefixed store — natural-LE byte order, no GPR scratch.
    return FaultingCodeOffset(
        as_pstxv(src.encoding(), dest.base, (int64_t)dest.offset, /*R=*/false)
            .getOffset());
  }
  if (HasPOWER9()) {
    Register scratch = temps.Acquire();
    if (dest.offset == 0) {
      return FaultingCodeOffset(as_stxvx(src, r0, dest.base).getOffset());
    }
    movePtr(ImmWord(dest.offset), scratch);
    return FaultingCodeOffset(as_stxvx(src, dest.base, scratch).getOffset());
  }
  // POWER8: stxvd2x stores with doubleword swap on LE.
  // Swap before store, then swap back to restore the register.
  ScratchSimd128Scope scratch128(*this);
  as_xxpermdi(scratch128, src, src, 2);
  Register scratch = temps.Acquire();
  FaultingCodeOffset fco;
  if (dest.offset == 0) {
    fco = FaultingCodeOffset(as_stxvd2x(scratch128, r0, dest.base).getOffset());
  } else {
    movePtr(ImmWord(dest.offset), scratch);
    fco = FaultingCodeOffset(
        as_stxvd2x(scratch128, dest.base, scratch).getOffset());
  }
  return fco;
}

FaultingCodeOffset MacroAssembler::storeUnalignedSimd128(
    FloatRegister src, const BaseIndex& dest) {
  UseScratchRegisterScope temps(asMasm());
  Register scratch = temps.Acquire();
  computeScaledAddress(dest, scratch);
  if (dest.offset != 0) {
    addPtr(ImmWord(dest.offset), scratch);
  }
  if (HasPOWER9()) {
    return FaultingCodeOffset(as_stxvx(src, r0, scratch).getOffset());
  }
  ScratchSimd128Scope scratch128(*this);
  as_xxpermdi(scratch128, src, src, 2);
  return FaultingCodeOffset(as_stxvd2x(scratch128, r0, scratch).getOffset());
}

// ===============================================================
// SIMD operations
//
// Scratch register conventions for SIMD helpers (read this before writing
// a new one):
//
// 1. `ScratchSimd128Scope scratch(*this)` — acquires v0 (= VR0 = VSR32,
//    non-allocatable). Constructed as {FloatRegisters::f0, Simd128} so
//    encoding() = 0 + 32 = 32 (per Architecture-ppc64.h). Default temp.
//    One scope at a time per helper. Safe to pass to any VMX/VSX
//    instruction; the allocator never places a live v128 in v0.
//
// 2. **Do NOT** write to VR1..VR31 (= VSR33..VSR63) without a Lowering
//    temp. VR1..VR31 are allocatable; a live wasm v128 may be sitting in
//    any of them. Use `ScratchSimd128Scope` (rule 1) or a Lowering temp.
//
// 3. **Red-zone stash** — use `RedZoneStashSimd128` / `RedZoneRestoreSimd128`
//    (declared just below) when a helper genuinely needs >1 SIMD scratch
//    AND adding a Lowering temp would require LIR + MIR + CodeGen changes.
//    ELFv2 reserves 288 bytes below SP; we use at most 32 (two 16-byte
//    slots). Live users: `extAddPairwiseInt*` (2 slots), `swizzleInt8x16`
//    (1 slot), `dotInt8x16Int7x16ThenAdd` 4-arg (1 slot). If you find
//    yourself wanting a 3rd slot or nested save/restore, prefer a Lowering
//    temp instead — the red-zone approach is tolerable because it's
//    self-contained to a single helper. The `MOZ_ASSERT(slot < 2)` inside
//    the helpers enforces this at test time.
//
// Simd128 lives in VR-namespace (VSR32-63), so VMX ops address Simd128
// FloatRegisters directly with no staging. Encoding is 32-63; the VMX
// VR field is 5-bit (0-31), so we mask with `& 31`.

// Two 16-byte Simd128 slots available in the ELFv2 red zone for short-lived
// SIMD spills (see point 3 of the SIMD conventions preamble above).
static constexpr int kRedZoneSimd128MaxSlots = 2;

static inline void RedZoneStashSimd128(MacroAssembler& masm, FloatRegister src,
                                       int slot) {
  MOZ_ASSERT(slot >= 0 && slot < kRedZoneSimd128MaxSlots);
  masm.storeUnalignedSimd128(src, Address(StackPointer, -16 * (slot + 1)));
}

static inline void RedZoneRestoreSimd128(MacroAssembler& masm, int slot,
                                         FloatRegister dest) {
  MOZ_ASSERT(slot >= 0 && slot < kRedZoneSimd128MaxSlots);
  masm.loadUnalignedSimd128(Address(StackPointer, -16 * (slot + 1)), dest);
}

typedef void (*VmxBinaryFn)(Assembler&, uint8_t, uint8_t, uint8_t);

static void EmitVmxBinary(MacroAssembler& masm, VmxBinaryFn vmxOp,
                          FloatRegister lhs, FloatRegister rhs,
                          FloatRegister dest) {
  vmxOp(static_cast<Assembler&>(masm), dest.encoding() & 31,
        lhs.encoding() & 31, rhs.encoding() & 31);
}

// Macro for defining VMX binary wrappers.
#define VMX_BINARY_WRAPPER(vmxInst)                         \
  [](Assembler& a, uint8_t vrt, uint8_t vra, uint8_t vrb) { \
    a.as_##vmxInst(vrt, vra, vrb);                          \
  }

// Emit op directly on Simd128 dest, then xxlnor in place.
template <typename VmxBinaryFnT>
static void EmitVmxBinaryNot(MacroAssembler& masm, VmxBinaryFnT vmxOp,
                             FloatRegister lhs, FloatRegister rhs,
                             FloatRegister dest) {
  vmxOp(static_cast<Assembler&>(masm), dest.encoding() & 31,
        lhs.encoding() & 31, rhs.encoding() & 31);
  masm.as_xxlnor(dest, dest, dest);
}

// Integer SIMD compare helper. VMX compare instructions produce all-ones
// for true, all-zeros for false per element.
// Available VMX compares: vcmpequ* (eq), vcmpgts* (signed gt), vcmpgtu*
// (unsigned gt). Other conditions derived by swapping operands or
// complementing.
template <typename EqFn, typename GtsFn, typename GtuFn>
static void EmitVmxCompare(MacroAssembler& masm, Assembler::Condition cond,
                           FloatRegister lhs, FloatRegister rhs,
                           FloatRegister dest, EqFn eqFn, GtsFn gtsFn,
                           GtuFn gtuFn) {
  switch (cond) {
    case Assembler::Equal:
      EmitVmxBinary(masm, eqFn, lhs, rhs, dest);
      break;
    case Assembler::NotEqual:
      EmitVmxBinaryNot(masm, eqFn, lhs, rhs, dest);
      break;
    case Assembler::GreaterThan:
      EmitVmxBinary(masm, gtsFn, lhs, rhs, dest);
      break;
    case Assembler::GreaterThanOrEqual:
      // !(rhs > lhs)
      EmitVmxBinaryNot(masm, gtsFn, rhs, lhs, dest);
      break;
    case Assembler::LessThan:
      // rhs > lhs (swap)
      EmitVmxBinary(masm, gtsFn, rhs, lhs, dest);
      break;
    case Assembler::LessThanOrEqual:
      // !(lhs > rhs)
      EmitVmxBinaryNot(masm, gtsFn, lhs, rhs, dest);
      break;
    case Assembler::Above:
      EmitVmxBinary(masm, gtuFn, lhs, rhs, dest);
      break;
    case Assembler::AboveOrEqual:
      EmitVmxBinaryNot(masm, gtuFn, rhs, lhs, dest);
      break;
    case Assembler::Below:
      EmitVmxBinary(masm, gtuFn, rhs, lhs, dest);
      break;
    case Assembler::BelowOrEqual:
      EmitVmxBinaryNot(masm, gtuFn, lhs, rhs, dest);
      break;
    default:
      MOZ_CRASH("Unexpected SIMD integer condition");
  }
}

// Emit ternary VMX op directly on Simd128 regs, no staging.
typedef void (*VmxTernaryFn)(Assembler&, uint8_t, uint8_t, uint8_t, uint8_t);

static void EmitVmxTernary(MacroAssembler& masm, VmxTernaryFn vmxOp,
                           FloatRegister a, FloatRegister b, FloatRegister c,
                           FloatRegister dest) {
  vmxOp(static_cast<Assembler&>(masm), dest.encoding() & 31, a.encoding() & 31,
        b.encoding() & 31, c.encoding() & 31);
}

// Emit unary VMX op directly on Simd128 regs, no staging.
typedef void (*VmxUnaryFn)(Assembler&, uint8_t, uint8_t);

static void EmitVmxUnary(MacroAssembler& masm, VmxUnaryFn vmxOp,
                         FloatRegister src, FloatRegister dest) {
  vmxOp(static_cast<Assembler&>(masm), dest.encoding() & 31,
        src.encoding() & 31);
}

// Helper: create a zero SIMD register using xxlxor.
static void ZeroSimd128(MacroAssembler& masm, FloatRegister dest) {
  masm.as_xxlxor(dest, dest, dest);
}

void MacroAssembler::moveSimd128(FloatRegister src, FloatRegister dest) {
  if (src != dest) {
    as_xxlor(dest, src, src);
  }
}

void MacroAssembler::loadConstantSimd128(const SimdConstant& v,
                                         FloatRegister dest) {
  // Load 128-bit constant from inline constant pool.
  // Clobbers SecondScratchReg (r12).
  loadFromPoolSimd128(dest, v);
}

// PPC64 LE lane mapping:
// Wasm lane K = memory byte K = register byte (15-K).
// mfvsrd extracts register bits[0:63] = BE dword 0 = Wasm lanes 8-15 (bytes).
// For VMX byte ops, BE byte index = 15 - wasm_lane.
// For VMX halfword ops, BE halfword index = 7 - wasm_halfword.
// For VSX word ops (xxspltw), BE word index = 3 - wasm_word.
// For doubleword ops, BE dword index = 1 - wasm_dword.

void MacroAssembler::splatX16(Register src, FloatRegister dest) {
  // mtvsrd writes src into BE 0..63 of dest (low byte at BE byte 7);
  // vspltb then splats that byte over all 16 lanes. dest aliases as
  // both source and destination — vspltb tolerates this. No extra
  // scratch register required, so callers that already hold a
  // ScratchSimd128Scope (extAddPairwise*, var-shift narrow forms) do
  // not see a nested-acquire collision.
  as_mtvsrd(dest, src);
  as_vspltb(dest, dest, 7);
}

void MacroAssembler::splatX8(Register src, FloatRegister dest) {
  // Same shape as splatX16 with halfword granularity. mtvsrd places
  // the low 16 bits at BE halfword 3 (= BE bytes 6..7); vsplth picks
  // it up and splats across 8 lanes. vsplth reads only the chosen
  // halfword, so negative i32 inputs do not need a 16-bit pre-mask
  // (which the previous GPR-replicate path required).
  as_mtvsrd(dest, src);
  as_vsplth(dest, dest, 3);
}

void MacroAssembler::splatX4(Register src, FloatRegister dest) {
  if (HasPOWER9()) {
    as_mtvsrws(dest, src);
  } else {
    as_mtvsrd(dest, src);
    as_xxspltw(dest, dest, 1);
  }
}

void MacroAssembler::splatX4(FloatRegister src, FloatRegister dest) {
  // src is a double-precision FPR holding a float value (the JIT keeps
  // FP32 in DP-equivalent form on PPC64). Convert DP→SP into BE word 0
  // (xscvdpspn lays the single at bits[0:31] / BE word 0), then splat
  // word 0 to all four lanes.
  as_xscvdpspn(dest, src);
  as_xxspltw(dest, dest, 0);
}

void MacroAssembler::splatX2(FloatRegister src, FloatRegister dest) {
  // Splat scalar double to both doubleword lanes.
  // Scalar value is in register bits[0:63] (BE dword 0).
  // xxpermdi dm=0: dest = [src.dw0, src.dw0]
  as_xxpermdi(dest, src, src, 0);
}

// Helpers: splat Imm32 into SIMD register at various element widths.
// VMX shift instructions read the shift count from EACH element independently,
// so the count must be replicated to every byte/halfword/word as appropriate.
//
// Fast path for small constants: vspltis{b,h,w} (POWER7+) splats a 5-bit
// signed immediate to all lanes in 1 insn with no pool entry. For values
// outside [-16, 15] we fall back to the inline-pool path.
static void SplatImm8(MacroAssembler& masm, Imm32 imm, FloatRegister dest) {
  int8_t val = (int8_t)imm.value;
  if (val >= -16 && val <= 15) {
    masm.as_vspltisb(dest.encoding() & 31, val);
    return;
  }
  if (HasPOWER9()) {
    // P9 xxspltib handles the full 8-bit range in 1 insn.
    masm.as_xxspltib(dest, (uint8_t)val);
    return;
  }
  int8_t bytes[16];
  for (int i = 0; i < 16; i++) bytes[i] = val;
  masm.loadConstantSimd128(SimdConstant::CreateX16(bytes), dest);
}

static void SplatImm16(MacroAssembler& masm, Imm32 imm, FloatRegister dest) {
  int16_t val = (int16_t)imm.value;
  if (val >= -16 && val <= 15) {
    masm.as_vspltish(dest.encoding() & 31, (int8_t)val);
    return;
  }
  int16_t halfs[8];
  for (int i = 0; i < 8; i++) halfs[i] = val;
  masm.loadConstantSimd128(SimdConstant::CreateX8(halfs), dest);
}

static void SplatImm32(MacroAssembler& masm, Imm32 imm, FloatRegister dest) {
  int32_t val = imm.value;
  if (val >= -16 && val <= 15) {
    masm.as_vspltisw(dest.encoding() & 31, (int8_t)val);
    return;
  }
  int32_t words[4] = {val, val, val, val};
  masm.loadConstantSimd128(SimdConstant::CreateX4(words), dest);
}

// ===============================================================
// Extract lane

static void ExtractLaneToGPR(MacroAssembler& masm, uint32_t lane,
                             FloatRegister src, Register dest,
                             unsigned laneWidthBytes, unsigned laneWidthBits) {
  // Extract Wasm lane from vector register to GPR.
  // Wasm lane K → register byte offset (15 - K*laneWidthBytes).
  //
  // Strategy: use mfvsrd to get one 64-bit half of the register, then shift
  // and mask to isolate the lane.
  //
  // mfvsrd gets register bits[0:63] (BE dword 0) = Wasm lanes in the high
  // half of the register (high-numbered lanes in LE memory order).
  // For an N-bit lane at Wasm index L:
  //   If L is in the high dword (L >= 8/laneWidthBytes):
  //     use mfvsrd; lane is at GPR bit offset laneWidthBits*(L -
  //     8/laneWidthBytes) from LSB
  //   Else (L in low dword):
  //     swap dwords, then mfvsrd; lane is at GPR bit offset laneWidthBits*L
  //     from LSB

  unsigned lanesPerDword = 8 / laneWidthBytes;

  if (lane >= lanesPerDword) {
    masm.as_mfvsrd(dest, src);
    unsigned shift = laneWidthBits * (lane - lanesPerDword);
    if (shift) {
      masm.x_srdi(dest, dest, shift);
    }
  } else {
    if (HasPOWER9()) {
      masm.as_mfvsrld(dest, src);
    } else {
      // POWER8: swap dwords to get dw1 into scalar position.
      // Avoid ScratchSimd128Scope — callers may already hold it.
      // Use xxpermdi directly on ScratchSimd128Reg (v0/VSR32, non-allocatable).
      masm.as_xxpermdi(ScratchSimd128Reg, src, src, 2);
      masm.as_mfvsrd(dest, ScratchSimd128Reg);
    }
    unsigned shift = laneWidthBits * lane;
    if (shift) {
      masm.x_srdi(dest, dest, shift);
    }
  }
}

void MacroAssembler::unsignedExtractLaneInt8x16(uint32_t lane,
                                                FloatRegister src,
                                                Register dest) {
  MOZ_ASSERT(lane < 16);
  if (HasPOWER9()) {
    // vextractub puts VRB.BE_byte[UIM] at VRT.BE_byte[7] with the rest
    // zeroed; mfvsrd then reads BE bytes 0..7 → low byte of dest, high
    // bytes already 0. No mask needed.
    as_vextractub(ScratchSimd128Reg, src, 15 - lane);
    as_mfvsrd(dest, ScratchSimd128Reg);
    return;
  }
  ExtractLaneToGPR(*this, lane, src, dest, 1, 8);
  as_rldicl(dest, dest, 0, 56);
}

void MacroAssembler::unsignedExtractLaneInt16x8(uint32_t lane,
                                                FloatRegister src,
                                                Register dest) {
  MOZ_ASSERT(lane < 8);
  if (HasPOWER9()) {
    as_vextractuh(ScratchSimd128Reg, src, 14 - 2 * lane);
    as_mfvsrd(dest, ScratchSimd128Reg);
    return;
  }
  ExtractLaneToGPR(*this, lane, src, dest, 2, 16);
  as_rldicl(dest, dest, 0, 48);
}

void MacroAssembler::extractLaneFloat32x4(uint32_t lane, FloatRegister src,
                                          FloatRegister dest) {
  MOZ_ASSERT(lane < 4);
  // BE word index = 3 - lane. xxextractuw extracts a word by BE byte offset.
  // BE byte offset of BE word W = W*4. So offset = (3-lane)*4.
  // xxextractuw puts the extracted word into bits[32:63] of dest (the low
  // word of the scalar doubleword), then xscvspdpn converts SP→DP.
  // xxspltw replicates a word into all 4 positions. The scalar SP value
  // is then at bits[0:31] where xscvspdpn expects it.
  as_xxspltw(dest, src, 3 - lane);
  as_xscvspdpn(dest, dest);
}

void MacroAssembler::extractLaneFloat64x2(uint32_t lane, FloatRegister src,
                                          FloatRegister dest) {
  MOZ_ASSERT(lane < 2);
  if (lane == 0) {
    // Lane 0 = LE low dword = BE dword 1. Need to swap to scalar position.
    as_xxpermdi(dest, src, src, 2);
  } else {
    // Lane 1 = LE high dword = BE dword 0 = scalar position.
    if (src != dest) {
      as_xxlor(dest, src, src);
    }
  }
}

// ===============================================================
// Replace lane

void MacroAssembler::replaceLaneInt8x16(unsigned lane, Register rhs,
                                        FloatRegister lhsDest) {
  MOZ_ASSERT(lane < 16);
  if (HasPOWER10()) {
    // 2 insns + 1 GPR scratch: load lane index, vinsbrx (right-indexed
    // = LE-natural). vinsbrx masks RA & 0xF, so the immediate fits.
    UseScratchRegisterScope temps(asMasm());
    Register idx = temps.Acquire();
    xs_li(idx, int16_t(lane));
    as_vinsbrx(lhsDest, idx, rhs);
    return;
  }
  if (HasPOWER9()) {
    // 2 insns + 1 VSR scratch: stage rhs in BE 0..63 of a scratch VSR
    // (low byte of rhs lands at BE byte 7), then vinsertb copies that
    // BE byte 7 into lhsDest's BE byte (15 - lane) = wasm lane L.
    ScratchSimd128Scope scratch(*this);
    as_mtvsrd(scratch, rhs);
    as_vinsertb(lhsDest, scratch, 15 - lane);
    return;
  }
  {
    // POWER8: extract dword, use rldimi to insert byte, write back.
    // Only needs 1 GPR scratch.
    UseScratchRegisterScope temps(asMasm());
    ScratchSimd128Scope scratch128(*this);
    Register tmp = temps.Acquire();
    unsigned dword = lane / 8;
    unsigned byteInDword = lane % 8;
    if (dword == 1) {
      as_mfvsrd(tmp, lhsDest);
    } else {
      as_xxpermdi(scratch128, lhsDest, lhsDest, 2);
      as_mfvsrd(tmp, scratch128);
    }
    // rldimi RT,RS,SH,MB: insert rotated RS bits into RT at positions
    // MB..63-SH. Insert rhs byte at bit offset 8*byteInDword from LSB:
    //   SH = 8*byteInDword, MB = 56 - 8*byteInDword
    as_rldimi(tmp, rhs, 8 * byteInDword, 56 - 8 * byteInDword);
    as_mtvsrd(scratch128, tmp);
    // mtvsrd writes scratch128.dw0 from `tmp` and leaves scratch128.dw1
    // undefined. Both xxpermdi forms below select scratch128.dw0 only:
    //   DM=0b01 → [scratch.dw0, lhsDest.dw1]
    //   DM=0b00 → [lhsDest.dw0, scratch.dw0]
    // So the undefined dw1 is never read. INVARIANT: any future change
    // to either DM literal MUST first zero scratch128.dw1 via xxlxor or
    // adopt a different staging scheme; otherwise reads of dw1 produce
    // POWER9-zero / POWER8-undefined garbage in the output.
    if (dword == 1) {
      as_xxpermdi(lhsDest, scratch128, lhsDest, 1);
    } else {
      as_xxpermdi(lhsDest, lhsDest, scratch128, 0);
    }
  }
}

void MacroAssembler::replaceLaneInt16x8(unsigned lane, Register rhs,
                                        FloatRegister lhsDest) {
  MOZ_ASSERT(lane < 8);
  if (HasPOWER10()) {
    // 2 insns + 1 GPR scratch: lane*2 → byte position, then vinshrx.
    UseScratchRegisterScope temps(asMasm());
    Register idx = temps.Acquire();
    xs_li(idx, int16_t(lane * 2));
    as_vinshrx(lhsDest, idx, rhs);
    return;
  }
  if (HasPOWER9()) {
    // 2 insns + 1 VSR scratch: stage rhs in BE 0..63 (low 16 of rhs
    // lands at BE bytes 6..7), then vinserth copies those two bytes
    // into lhsDest's BE bytes (14 - 2L)..(15 - 2L) = wasm lane L.
    ScratchSimd128Scope scratch(*this);
    as_mtvsrd(scratch, rhs);
    as_vinserth(lhsDest, scratch, 14 - 2 * lane);
    return;
  }
  {
    // POWER8: extract dword, rldimi to insert halfword, write back.
    // Same dw1-undef invariant as replaceLaneInt8x16 above.
    UseScratchRegisterScope temps(asMasm());
    ScratchSimd128Scope scratch128(*this);
    Register tmp = temps.Acquire();
    unsigned dword = lane / 4;
    unsigned hwInDword = lane % 4;
    if (dword == 1) {
      as_mfvsrd(tmp, lhsDest);
    } else {
      as_xxpermdi(scratch128, lhsDest, lhsDest, 2);
      as_mfvsrd(tmp, scratch128);
    }
    as_rldimi(tmp, rhs, 16 * hwInDword, 48 - 16 * hwInDword);
    as_mtvsrd(scratch128, tmp);
    if (dword == 1) {
      as_xxpermdi(lhsDest, scratch128, lhsDest, 1);
    } else {
      as_xxpermdi(lhsDest, lhsDest, scratch128, 0);
    }
  }
}

void MacroAssembler::replaceLaneInt32x4(unsigned lane, Register rhs,
                                        FloatRegister lhsDest) {
  MOZ_ASSERT(lane < 4);
  if (HasPOWER10()) {
    // 1 insn, no scratch VSR. UIM is the BE byte offset.
    as_vinsw(lhsDest, rhs, (3 - lane) * 4);
    return;
  }
  if (HasPOWER9()) {
    // POWER9: xxinsertw inserts word from bits[32:63] of XB at BE byte
    // offset UIM in XT. mtvsrd puts GPR into bits[0:63]; low 32 bits
    // land at bits[32:63]. BE byte offset of Wasm word lane = (3-lane)*4.
    ScratchSimd128Scope scratch(*this);
    as_mtvsrd(scratch, rhs);
    as_xxinsertw(lhsDest, scratch, (3 - lane) * 4);
    return;
  }
  // POWER8: extract dword, rldimi to insert word, write back.
  // Modeled on replaceLaneInt16x8 below.
  UseScratchRegisterScope temps(asMasm());
  ScratchSimd128Scope scratch128(*this);
  Register tmp = temps.Acquire();
  unsigned dword = lane / 2;        // 0 = lanes 0,1; 1 = lanes 2,3.
  unsigned wordInDword = lane % 2;  // 0 = low LE word; 1 = high LE word.
  if (dword == 1) {
    as_mfvsrd(tmp, lhsDest);
  } else {
    as_xxpermdi(scratch128, lhsDest, lhsDest, 2);
    as_mfvsrd(tmp, scratch128);
  }
  as_rldimi(tmp, rhs, 32 * wordInDword, 32 - 32 * wordInDword);
  as_mtvsrd(scratch128, tmp);
  if (dword == 1) {
    as_xxpermdi(lhsDest, scratch128, lhsDest, 1);
  } else {
    as_xxpermdi(lhsDest, lhsDest, scratch128, 0);
  }
}

void MacroAssembler::replaceLaneFloat32x4(unsigned lane, FloatRegister rhs,
                                          FloatRegister lhsDest) {
  MOZ_ASSERT(lane < 4);
  if (HasPOWER9()) {
    ScratchSimd128Scope scratch(*this);
    as_xscvdpspn(scratch, rhs);
    as_xxinsertw(lhsDest, scratch, (3 - lane) * 4);
    return;
  }
  // POWER8: convert double rhs to single (lands in BE bits 0..31 of FPR),
  // extract bits to a GPR, then route through the integer insert path.
  UseScratchRegisterScope temps(asMasm());
  Register rhsBits = temps.Acquire();
  {
    ScratchSimd128Scope scratch(*this);
    as_xscvdpspn(scratch, rhs);
    as_mfvsrd(rhsBits, scratch);   // single is in high 32 bits of GPR
    x_srdi(rhsBits, rhsBits, 32);  // single → low 32 bits
  }
  // Inline the int-insert sequence (can't call replaceLaneInt32x4 from
  // here because we're already inside a UseScratchRegisterScope and
  // need to acquire a separate tmp).
  ScratchSimd128Scope scratch128(*this);
  Register tmp = temps.Acquire();
  unsigned dword = lane / 2;
  unsigned wordInDword = lane % 2;
  if (dword == 1) {
    as_mfvsrd(tmp, lhsDest);
  } else {
    as_xxpermdi(scratch128, lhsDest, lhsDest, 2);
    as_mfvsrd(tmp, scratch128);
  }
  as_rldimi(tmp, rhsBits, 32 * wordInDword, 32 - 32 * wordInDword);
  as_mtvsrd(scratch128, tmp);
  if (dword == 1) {
    as_xxpermdi(lhsDest, scratch128, lhsDest, 1);
  } else {
    as_xxpermdi(lhsDest, lhsDest, scratch128, 0);
  }
}

void MacroAssembler::replaceLaneFloat64x2(unsigned lane, FloatRegister rhs,
                                          FloatRegister lhsDest) {
  MOZ_ASSERT(lane < 2);
  // xxpermdi to place the scalar double into the correct lane.
  if (lane == 0) {
    // Replace LE low dword (= dw1). Keep lhsDest dw0 (lane 1).
    // rhs scalar is in dw0. dm=0b00: [lhsDest.dw0, rhs.dw0]
    as_xxpermdi(lhsDest, lhsDest, rhs, 0);
  } else {
    // Replace LE high dword (= dw0). Keep lhsDest dw1 (lane 0).
    // rhs scalar is in dw0. dm=0b01: [rhs.dw0, lhsDest.dw1]
    as_xxpermdi(lhsDest, rhs, lhsDest, 1);
  }
}

void MacroAssembler::shuffleInt8x16(const uint8_t lanes[16], FloatRegister rhs,
                                    FloatRegister lhsDest) {
  shuffleInt8x16(lanes, lhsDest, rhs, lhsDest);
}

void MacroAssembler::shuffleInt8x16(const uint8_t lanes[16], FloatRegister lhs,
                                    FloatRegister rhs, FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  // PPC64 vperm uses BE byte indices: VRA[0]=MSB, VRA[15]=LSB, VRB[16..31].
  // Convert Wasm LE lane indices to vperm control: lhs lane N = BE index
  // (15-N), rhs lane N = BE index (31-N) = (47 - (N+16)).
  int8_t ctrl[16];
  for (unsigned i = 0; i < 16; i++) {
    uint8_t src = lanes[i];
    if (src < 16) {
      ctrl[i] = 15 - src;
    } else {
      ctrl[i] = 47 - src;
    }
  }
  loadConstantSimd128(SimdConstant::CreateX16(ctrl), scratch);
  // vperm directly on Simd128 regs.
  as_vperm(dest.encoding() & 31, lhs.encoding() & 31, rhs.encoding() & 31,
           scratch.encoding() & 31);
}

void MacroAssembler::laneSelectSimd128(FloatRegister mask, FloatRegister lhs,
                                       FloatRegister rhs, FloatRegister dest) {
  // xxsel: XC=0→XA, XC=1→XB → XT = (XA & ~XC) | (XB & XC)
  // laneSelect: dest = (lhs & mask) | (rhs & ~mask)
  // Need XA=rhs, XB=lhs, XC=mask.
  as_xxsel(dest, rhs, lhs, mask);
}

void MacroAssembler::interleaveHighInt8x16(FloatRegister lhs, FloatRegister rhs,
                                           FloatRegister dest) {
  // On LE, vmrghb(rhs, lhs) gives Wasm interleave_high.
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmrghb), rhs, lhs, dest);
}

void MacroAssembler::interleaveHighInt16x8(FloatRegister lhs, FloatRegister rhs,
                                           FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmrghh), rhs, lhs, dest);
}

void MacroAssembler::interleaveHighInt32x4(FloatRegister lhs, FloatRegister rhs,
                                           FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmrghw), rhs, lhs, dest);
}

void MacroAssembler::interleaveHighInt64x2(FloatRegister lhs, FloatRegister rhs,
                                           FloatRegister dest) {
  // xxpermdi DM=0: [XA.dw0, XB.dw0] = merge high dwords.
  // On LE: dw0 = high Wasm lane (lane 1).
  as_xxpermdi(dest, rhs, lhs, 0);
}

void MacroAssembler::interleaveLowInt8x16(FloatRegister lhs, FloatRegister rhs,
                                          FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmrglb), rhs, lhs, dest);
}

void MacroAssembler::interleaveLowInt16x8(FloatRegister lhs, FloatRegister rhs,
                                          FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmrglh), rhs, lhs, dest);
}

void MacroAssembler::interleaveLowInt32x4(FloatRegister lhs, FloatRegister rhs,
                                          FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmrglw), rhs, lhs, dest);
}

void MacroAssembler::interleaveLowInt64x2(FloatRegister lhs, FloatRegister rhs,
                                          FloatRegister dest) {
  // xxpermdi DM=3: [XA.dw1, XB.dw1] = merge low dwords.
  as_xxpermdi(dest, rhs, lhs, 3);
}

void MacroAssembler::concatAndRightShiftSimd128(FloatRegister lhs,
                                                FloatRegister rhs,
                                                FloatRegister dest,
                                                uint32_t shift) {
  // vsldoi(VRA, VRB, SH) extracts 16 bytes starting at byte SH of the
  // big-endian concatenation VRA||VRB. Endianness mapping for the Wasm
  // `v128.shuffle` right-shift-concat semantic:
  //   Wasm:  result[i] = (i + shift < 16) ? rhs[i + shift]
  //                                       : lhs[i + shift - 16]
  //   PPC LE: vsldoi(rhs, lhs, shift) produces exactly that — the LE byte
  //   layout reverses from BE, so passing (rhs, lhs, shift) here is the LE
  //   equivalent of (lhs, rhs, 16 - shift) on BE.
  MOZ_ASSERT(shift < 16);
  if (shift == 0) {
    moveSimd128(rhs, dest);
    return;
  }
  // vsldoi VRT,VRA,VRB,SH: result[i] = (VRA||VRB)[SH+i]
  // Emit vsldoi directly on Simd128 regs (VRA = lhs = high part, VRB =
  // rhs = low part). The VMX emitter masks `& 31` internally to extract
  // the 5-bit VR field from the Simd128 encoding.
  as_vsldoi(dest, lhs, rhs, shift);
}

void MacroAssembler::leftShiftSimd128(Imm32 count, FloatRegister src,
                                      FloatRegister dest) {
  MOZ_ASSERT(count.value < 16);
  if (count.value == 0) {
    moveSimd128(src, dest);
    return;
  }
  // vslo shifts left by bytes (count in bits 121-124 of VRB, i.e. byte 15 bits
  // 1-4). vsl shifts left by bits (count in bits 125-127 of VRB, i.e. byte 15
  // bits 5-7). For byte shift: splatX4(count*8, scratch), then vslo.
  ScratchSimd128Scope scratch(*this);
  SplatImm32(*this, Imm32(count.value * 8), scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vslo), src, scratch, dest);
}

void MacroAssembler::rightShiftSimd128(Imm32 count, FloatRegister src,
                                       FloatRegister dest) {
  MOZ_ASSERT(count.value < 16);
  if (count.value == 0) {
    moveSimd128(src, dest);
    return;
  }
  ScratchSimd128Scope scratch(*this);
  SplatImm32(*this, Imm32(count.value * 8), scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsro), src, scratch, dest);
}

void MacroAssembler::zeroExtend8x16To16x8(FloatRegister src,
                                          FloatRegister dest) {
  // Unsigned widen low: interleave low bytes with zero bytes.
  // On LE, vmrglb(zero, src) interleaves the low 8 bytes of src with zeros.
  // Use ScratchSimd128Reg as the zero. Order matters: read src into the
  // merge BEFORE writing dest (which might alias src). vmrglb reads
  // vra+vrb, writes vrt — single-cycle issue.
  ScratchSimd128Scope zero(*this);
  as_xxlxor(zero, zero, zero);
  as_vmrglb(dest.encoding() & 31, zero.encoding() & 31, src.encoding() & 31);
}

void MacroAssembler::zeroExtend8x16To32x4(FloatRegister src,
                                          FloatRegister dest) {
  zeroExtend8x16To16x8(src, dest);
  zeroExtend16x8To32x4(dest, dest);
}

void MacroAssembler::zeroExtend8x16To64x2(FloatRegister src,
                                          FloatRegister dest) {
  zeroExtend8x16To32x4(src, dest);
  zeroExtend32x4To64x2(dest, dest);
}

void MacroAssembler::zeroExtend16x8To32x4(FloatRegister src,
                                          FloatRegister dest) {
  // Unsigned widen low: interleave low halfwords with zero halfwords.
  ScratchSimd128Scope zero(*this);
  as_xxlxor(zero, zero, zero);
  as_vmrglh(dest.encoding() & 31, zero.encoding() & 31, src.encoding() & 31);
}

void MacroAssembler::zeroExtend16x8To64x2(FloatRegister src,
                                          FloatRegister dest) {
  zeroExtend16x8To32x4(src, dest);
  zeroExtend32x4To64x2(dest, dest);
}

void MacroAssembler::zeroExtend32x4To64x2(FloatRegister src,
                                          FloatRegister dest) {
  // Unsigned widen low: interleave low words with zero words.
  ScratchSimd128Scope zero(*this);
  as_xxlxor(zero, zero, zero);
  as_vmrglw(dest.encoding() & 31, zero.encoding() & 31, src.encoding() & 31);
}

void MacroAssembler::reverseInt16x8(FloatRegister src, FloatRegister dest) {
  const uint8_t lanes[] = {14, 15, 12, 13, 10, 11, 8, 9,
                           6,  7,  4,  5,  2,  3,  0, 1};
  shuffleInt8x16(lanes, src, src, dest);
}

void MacroAssembler::reverseInt32x4(FloatRegister src, FloatRegister dest) {
  const uint8_t lanes[] = {12, 13, 14, 15, 8, 9, 10, 11,
                           4,  5,  6,  7,  0, 1, 2,  3};
  shuffleInt8x16(lanes, src, src, dest);
}

void MacroAssembler::reverseInt64x2(FloatRegister src, FloatRegister dest) {
  as_xxpermdi(dest, src, src, 2);
}

void MacroAssembler::swizzleInt8x16Relaxed(FloatRegister lhs, FloatRegister rhs,
                                           FloatRegister dest) {
  swizzleInt8x16(lhs, rhs, dest);
}

// extMul{Low,High}Int{8x16,16x8} use POWER8+ widening multiplies
// (vmul{e,o}{s,u}{b,h}) plus a halfword/word merge to map BE-indexed
// even/odd products into Wasm lane order on PPC64 LE.
//
// Lane mapping:
//   For Low (Wasm lanes from LE bytes/HW 0..N/2-1 = BE 15..N/2):
//     vmrgl{h,w}(even_products, odd_products) places the right products
//     at BE result indices, which on LE map to Wasm lanes 0..N/2-1.
//   For High (Wasm lanes from LE indices N/2..N-1 = BE N/2-1..0):
//     vmrgh{h,w} takes the upper-half BE indices instead.
//
// Aliasing safety: vmul* reads both operands before writing, so
// `dest = vmulo* lhs, rhs` is safe even when dest aliases lhs/rhs.
// We use one scratch for the even-product half because vmrgl{h,w}
// reads dest after the odd multiply.

void MacroAssembler::extMulLowInt8x16(FloatRegister lhs, FloatRegister rhs,
                                      FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  uint8_t l = lhs.encoding() & 31, r = rhs.encoding() & 31;
  uint8_t d = dest.encoding() & 31, s = scratch.encoding() & 31;
  as_vmulesb(s, l, r);
  as_vmulosb(d, l, r);
  as_vmrglh(d, s, d);
}

void MacroAssembler::extMulHighInt8x16(FloatRegister lhs, FloatRegister rhs,
                                       FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  uint8_t l = lhs.encoding() & 31, r = rhs.encoding() & 31;
  uint8_t d = dest.encoding() & 31, s = scratch.encoding() & 31;
  as_vmulesb(s, l, r);
  as_vmulosb(d, l, r);
  as_vmrghh(d, s, d);
}

void MacroAssembler::unsignedExtMulLowInt8x16(FloatRegister lhs,
                                              FloatRegister rhs,
                                              FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  uint8_t l = lhs.encoding() & 31, r = rhs.encoding() & 31;
  uint8_t d = dest.encoding() & 31, s = scratch.encoding() & 31;
  as_vmuleub(s, l, r);
  as_vmuloub(d, l, r);
  as_vmrglh(d, s, d);
}

void MacroAssembler::unsignedExtMulHighInt8x16(FloatRegister lhs,
                                               FloatRegister rhs,
                                               FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  uint8_t l = lhs.encoding() & 31, r = rhs.encoding() & 31;
  uint8_t d = dest.encoding() & 31, s = scratch.encoding() & 31;
  as_vmuleub(s, l, r);
  as_vmuloub(d, l, r);
  as_vmrghh(d, s, d);
}

void MacroAssembler::extMulLowInt16x8(FloatRegister lhs, FloatRegister rhs,
                                      FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  uint8_t l = lhs.encoding() & 31, r = rhs.encoding() & 31;
  uint8_t d = dest.encoding() & 31, s = scratch.encoding() & 31;
  as_vmulesh(s, l, r);
  as_vmulosh(d, l, r);
  as_vmrglw(d, s, d);
}

void MacroAssembler::extMulHighInt16x8(FloatRegister lhs, FloatRegister rhs,
                                       FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  uint8_t l = lhs.encoding() & 31, r = rhs.encoding() & 31;
  uint8_t d = dest.encoding() & 31, s = scratch.encoding() & 31;
  as_vmulesh(s, l, r);
  as_vmulosh(d, l, r);
  as_vmrghw(d, s, d);
}

void MacroAssembler::unsignedExtMulLowInt16x8(FloatRegister lhs,
                                              FloatRegister rhs,
                                              FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  uint8_t l = lhs.encoding() & 31, r = rhs.encoding() & 31;
  uint8_t d = dest.encoding() & 31, s = scratch.encoding() & 31;
  as_vmuleuh(s, l, r);
  as_vmulouh(d, l, r);
  as_vmrglw(d, s, d);
}

void MacroAssembler::unsignedExtMulHighInt16x8(FloatRegister lhs,
                                               FloatRegister rhs,
                                               FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  uint8_t l = lhs.encoding() & 31, r = rhs.encoding() & 31;
  uint8_t d = dest.encoding() & 31, s = scratch.encoding() & 31;
  as_vmuleuh(s, l, r);
  as_vmulouh(d, l, r);
  as_vmrghw(d, s, d);
}

// ExtMul{Low,High}Int32x4 use vmul{e,o}{s,u}w (POWER8+) plus xxpermdi
// to combine the two i64 partial products into Wasm lane order on PPC64
// LE. xxpermdi accepts the full 6-bit VSR encoding so it works directly
// on Simd128 regs (encoding 32-63) without any VR staging.
//
// Aliasing safe: both vmul* reads complete before the second one writes
// dest, and xxpermdi reads both inputs before writing.

static void EmitExtMulInt32x4(
    MacroAssembler& masm, FloatRegister lhs, FloatRegister rhs,
    FloatRegister dest, void (*mulEven)(Assembler&, uint8_t, uint8_t, uint8_t),
    void (*mulOdd)(Assembler&, uint8_t, uint8_t, uint8_t), uint8_t dm) {
  ScratchSimd128Scope scratch(masm);
  uint8_t l = lhs.encoding() & 31;
  uint8_t r = rhs.encoding() & 31;
  uint8_t d = dest.encoding() & 31;
  uint8_t s = scratch.encoding() & 31;
  mulEven(static_cast<Assembler&>(masm), s, l, r);
  mulOdd(static_cast<Assembler&>(masm), d, l, r);
  masm.as_xxpermdi(dest, scratch, dest, dm);
}

void MacroAssembler::extMulLowInt32x4(FloatRegister lhs, FloatRegister rhs,
                                      FloatRegister dest) {
  EmitExtMulInt32x4(
      *this, lhs, rhs, dest,
      [](Assembler& a, uint8_t t, uint8_t x, uint8_t y) {
        a.as_vmulesw(t, x, y);
      },
      [](Assembler& a, uint8_t t, uint8_t x, uint8_t y) {
        a.as_vmulosw(t, x, y);
      },
      3);
}

void MacroAssembler::extMulHighInt32x4(FloatRegister lhs, FloatRegister rhs,
                                       FloatRegister dest) {
  EmitExtMulInt32x4(
      *this, lhs, rhs, dest,
      [](Assembler& a, uint8_t t, uint8_t x, uint8_t y) {
        a.as_vmulesw(t, x, y);
      },
      [](Assembler& a, uint8_t t, uint8_t x, uint8_t y) {
        a.as_vmulosw(t, x, y);
      },
      0);
}

void MacroAssembler::unsignedExtMulLowInt32x4(FloatRegister lhs,
                                              FloatRegister rhs,
                                              FloatRegister dest) {
  EmitExtMulInt32x4(
      *this, lhs, rhs, dest,
      [](Assembler& a, uint8_t t, uint8_t x, uint8_t y) {
        a.as_vmuleuw(t, x, y);
      },
      [](Assembler& a, uint8_t t, uint8_t x, uint8_t y) {
        a.as_vmulouw(t, x, y);
      },
      3);
}

void MacroAssembler::unsignedExtMulHighInt32x4(FloatRegister lhs,
                                               FloatRegister rhs,
                                               FloatRegister dest) {
  EmitExtMulInt32x4(
      *this, lhs, rhs, dest,
      [](Assembler& a, uint8_t t, uint8_t x, uint8_t y) {
        a.as_vmuleuw(t, x, y);
      },
      [](Assembler& a, uint8_t t, uint8_t x, uint8_t y) {
        a.as_vmulouw(t, x, y);
      },
      0);
}

void MacroAssembler::q15MulrSatInt16x8(FloatRegister lhs, FloatRegister rhs,
                                       FloatRegister dest) {
  // Q15 multiply-round-saturate: vmhraddshs(a, b, zero) computes
  // saturate((a[i]*b[i] + 0x4000) >> 15) for each halfword.
  ScratchSimd128Scope scratch(*this);
  ZeroSimd128(*this, scratch);
  EmitVmxTernary(
      *this,
      [](Assembler& a, uint8_t vrt, uint8_t vra, uint8_t vrb, uint8_t vrc) {
        a.as_vmhraddshs(vrt, vra, vrb, vrc);
      },
      lhs, rhs, scratch, dest);
}

// neg = 0 - src. Use ScratchSimd128Reg (= VR0, non-allocatable) as the
// zero source so the register allocator sees no clobbered VRs.
// 2 insns: xxlxor scratch + vsubuXm dest, scratch, src. vneg{b,h}
// doesn't exist in any POWER ISA, hence the subtract.
void MacroAssembler::negInt8x16(FloatRegister src, FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  ZeroSimd128(*this, scratch);
  as_vsububm(dest.encoding() & 31, scratch.encoding() & 31,
             src.encoding() & 31);
}

void MacroAssembler::negInt16x8(FloatRegister src, FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  ZeroSimd128(*this, scratch);
  as_vsubuhm(dest.encoding() & 31, scratch.encoding() & 31,
             src.encoding() & 31);
}

void MacroAssembler::negInt32x4(FloatRegister src, FloatRegister dest) {
  if (HasPOWER9()) {
    EmitVmxUnary(
        *this,
        [](Assembler& a, uint8_t vrt, uint8_t vrb) { a.as_vnegw(vrt, vrb); },
        src, dest);
    return;
  }
  // POWER8 fallback: 0 - src via ScratchSimd128Reg (VR0).
  ScratchSimd128Scope scratch(*this);
  ZeroSimd128(*this, scratch);
  as_vsubuwm(dest.encoding() & 31, scratch.encoding() & 31,
             src.encoding() & 31);
}

void MacroAssembler::negInt64x2(FloatRegister src, FloatRegister dest) {
  if (HasPOWER9()) {
    EmitVmxUnary(
        *this,
        [](Assembler& a, uint8_t vrt, uint8_t vrb) { a.as_vnegd(vrt, vrb); },
        src, dest);
    return;
  }
  // POWER8 fallback: 0 - src via ScratchSimd128Reg (VR0).
  ScratchSimd128Scope scratch(*this);
  ZeroSimd128(*this, scratch);
  as_vsubudm(dest.encoding() & 31, scratch.encoding() & 31,
             src.encoding() & 31);
}
#undef DEF_NEG_INTNxM_VSUB

void MacroAssembler::unsignedAddSatInt8x16(FloatRegister lhs, FloatRegister rhs,
                                           FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vaddubs), lhs, rhs, dest);
}

void MacroAssembler::unsignedAddSatInt16x8(FloatRegister lhs, FloatRegister rhs,
                                           FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vadduhs), lhs, rhs, dest);
}

void MacroAssembler::unsignedSubSatInt8x16(FloatRegister lhs, FloatRegister rhs,
                                           FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsububs), lhs, rhs, dest);
}

void MacroAssembler::unsignedSubSatInt16x8(FloatRegister lhs, FloatRegister rhs,
                                           FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsubuhs), lhs, rhs, dest);
}

void MacroAssembler::unsignedMinInt8x16(FloatRegister lhs, FloatRegister rhs,
                                        FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vminub), lhs, rhs, dest);
}

void MacroAssembler::unsignedMinInt16x8(FloatRegister lhs, FloatRegister rhs,
                                        FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vminuh), lhs, rhs, dest);
}

void MacroAssembler::unsignedMinInt32x4(FloatRegister lhs, FloatRegister rhs,
                                        FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vminuw), lhs, rhs, dest);
}

void MacroAssembler::unsignedMaxInt8x16(FloatRegister lhs, FloatRegister rhs,
                                        FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmaxub), lhs, rhs, dest);
}

void MacroAssembler::unsignedMaxInt16x8(FloatRegister lhs, FloatRegister rhs,
                                        FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmaxuh), lhs, rhs, dest);
}

void MacroAssembler::unsignedMaxInt32x4(FloatRegister lhs, FloatRegister rhs,
                                        FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmaxuw), lhs, rhs, dest);
}

void MacroAssembler::unsignedAverageInt8x16(FloatRegister lhs,
                                            FloatRegister rhs,
                                            FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vavgub), lhs, rhs, dest);
}

void MacroAssembler::unsignedAverageInt16x8(FloatRegister lhs,
                                            FloatRegister rhs,
                                            FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vavguh), lhs, rhs, dest);
}

// abs(x) = max(x, -x) per signed lane. No vabs{b,h,w,d} exists in any ISA.
// vneg{w,d} exists only on POWER9.
// We use ScratchSimd128Reg as a temp for -src. Order matters: compute
// -src into temp first (reads src), then max(src, temp) into dest (reads
// src + temp, writes dest). Safe even when dest == src because src is
// read before dest is written by vmaxsX.

void MacroAssembler::absInt8x16(FloatRegister src, FloatRegister dest) {
  ScratchSimd128Scope tmp(*this);
  as_xxlxor(tmp, tmp, tmp);  // tmp = 0
  as_vsububm(tmp.encoding() & 31, tmp.encoding() & 31,
             src.encoding() & 31);  // tmp = -src
  as_vmaxsb(dest.encoding() & 31, src.encoding() & 31,
            tmp.encoding() & 31);  // dest = max(src, -src)
}

void MacroAssembler::absInt16x8(FloatRegister src, FloatRegister dest) {
  ScratchSimd128Scope tmp(*this);
  as_xxlxor(tmp, tmp, tmp);
  as_vsubuhm(tmp.encoding() & 31, tmp.encoding() & 31, src.encoding() & 31);
  as_vmaxsh(dest.encoding() & 31, src.encoding() & 31, tmp.encoding() & 31);
}

void MacroAssembler::absInt32x4(FloatRegister src, FloatRegister dest) {
  ScratchSimd128Scope tmp(*this);
  if (HasPOWER9()) {
    as_vnegw(tmp.encoding() & 31, src.encoding() & 31);  // tmp = -src
  } else {
    as_xxlxor(tmp, tmp, tmp);
    as_vsubuwm(tmp.encoding() & 31, tmp.encoding() & 31, src.encoding() & 31);
  }
  as_vmaxsw(dest.encoding() & 31, src.encoding() & 31, tmp.encoding() & 31);
}

void MacroAssembler::absInt64x2(FloatRegister src, FloatRegister dest) {
  ScratchSimd128Scope tmp(*this);
  if (HasPOWER9()) {
    as_vnegd(tmp.encoding() & 31, src.encoding() & 31);  // tmp = -src
  } else {
    as_xxlxor(tmp, tmp, tmp);
    as_vsubudm(tmp.encoding() & 31, tmp.encoding() & 31, src.encoding() & 31);
  }
  as_vmaxsd(dest.encoding() & 31, src.encoding() & 31, tmp.encoding() & 31);
}

void MacroAssembler::leftShiftInt8x16(Imm32 count, FloatRegister src,
                                      FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  SplatImm8(*this, count, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vslb), src, scratch, dest);
}

void MacroAssembler::leftShiftInt16x8(Imm32 count, FloatRegister src,
                                      FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  SplatImm16(*this, count, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vslh), src, scratch, dest);
}

void MacroAssembler::leftShiftInt32x4(Imm32 count, FloatRegister src,
                                      FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  SplatImm32(*this, count, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vslw), src, scratch, dest);
}

void MacroAssembler::leftShiftInt64x2(Imm32 count, FloatRegister src,
                                      FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  SplatImm32(*this, count, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsld), src, scratch, dest);
}

void MacroAssembler::rightShiftInt8x16(Imm32 count, FloatRegister src,
                                       FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  SplatImm8(*this, count, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrab), src, scratch, dest);
}

void MacroAssembler::unsignedRightShiftInt8x16(Imm32 count, FloatRegister src,
                                               FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  SplatImm8(*this, count, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrb), src, scratch, dest);
}

void MacroAssembler::rightShiftInt16x8(Imm32 count, FloatRegister src,
                                       FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  SplatImm16(*this, count, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrah), src, scratch, dest);
}

void MacroAssembler::unsignedRightShiftInt16x8(Imm32 count, FloatRegister src,
                                               FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  SplatImm16(*this, count, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrh), src, scratch, dest);
}

void MacroAssembler::rightShiftInt32x4(Imm32 count, FloatRegister src,
                                       FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  SplatImm32(*this, count, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsraw), src, scratch, dest);
}

void MacroAssembler::unsignedRightShiftInt32x4(Imm32 count, FloatRegister src,
                                               FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  SplatImm32(*this, count, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrw), src, scratch, dest);
}

void MacroAssembler::rightShiftInt64x2(Imm32 count, FloatRegister src,
                                       FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  SplatImm32(*this, count, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrad), src, scratch, dest);
}

void MacroAssembler::unsignedRightShiftInt64x2(Imm32 count, FloatRegister src,
                                               FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  SplatImm32(*this, count, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrd), src, scratch, dest);
}

void MacroAssembler::bitwiseAndSimd128(FloatRegister rhs,
                                       FloatRegister lhsDest) {
  as_xxland(lhsDest, lhsDest, rhs);
}

void MacroAssembler::bitwiseAndSimd128(FloatRegister lhs, FloatRegister rhs,
                                       FloatRegister dest) {
  as_xxland(dest, lhs, rhs);
}

void MacroAssembler::bitwiseOrSimd128(FloatRegister rhs,
                                      FloatRegister lhsDest) {
  as_xxlor(lhsDest, lhsDest, rhs);
}

void MacroAssembler::bitwiseOrSimd128(FloatRegister lhs, FloatRegister rhs,
                                      FloatRegister dest) {
  as_xxlor(dest, lhs, rhs);
}

void MacroAssembler::bitwiseXorSimd128(FloatRegister rhs,
                                       FloatRegister lhsDest) {
  as_xxlxor(lhsDest, lhsDest, rhs);
}

void MacroAssembler::bitwiseXorSimd128(FloatRegister lhs, FloatRegister rhs,
                                       FloatRegister dest) {
  as_xxlxor(dest, lhs, rhs);
}

void MacroAssembler::bitwiseNotSimd128(FloatRegister src, FloatRegister dest) {
  as_xxlnor(dest, src, src);
}

void MacroAssembler::bitwiseNotAndSimd128(FloatRegister rhs,
                                          FloatRegister lhsDest) {
  // notand(lhs, rhs) = ~lhs & rhs = xxlandc(rhs, lhs)
  as_xxlandc(lhsDest, rhs, lhsDest);
}

void MacroAssembler::anyTrueSimd128(FloatRegister src, Register dest) {
  // vcmpequd. (POWER8+) against zero sets CR6:
  //   - CR6.LT (BE bit 24) = 1 iff the per-lane result is all-1s, i.e.
  //     every doubleword of src equals zero (= src is all-zero).
  //   - CR6.EQ (BE bit 26) = 1 iff no lane was equal (= any nonzero).
  // any-true = !all-zero = !CR6.LT.
  ScratchSimd128Scope scratch(*this);
  uint8_t s = scratch.encoding() & 31;
  as_xxlxor(scratch, scratch, scratch);
  as_vcmpequd_rc(s, src.encoding() & 31, s);
  if (HasPOWER10()) {
    // setbcr materialises (CR[BI] == 0) ? 1 : 0 directly into dest.
    // dest = (CR6.LT == 0) = "not all-zero" = any-true.
    as_setbcr(dest, Assembler::LessThan, cr6);
    return;
  }
  as_mfocrf(dest, cr6);
  // CR6.LT is at BE bit 24 of the GPR. rlwinm sh=25 rotates left 25:
  // bit (24 - 25) mod 32 = 31 (LSB). Mask 31..31 keeps just bit 31.
  as_rlwinm(dest, dest, 25, 31, 31);
  as_xori(dest, dest, 1);
}

// vcmpequX. against zero sets CR6: LT = all input lanes were zero,
// EQ = no input lane was zero. The latter is exactly "all-true".
// mfocrf places CR6 at bits 24-27 of the low 32-bit half (LT=24, EQ=26).
// rlwinm rd,rd,27,31,31 extracts bit 26 (CR6.EQ) right-justified.
template <typename VmxCmpRcFn>
static void EmitAllTrueInt(MacroAssembler& masm, FloatRegister src,
                           Register dest, VmxCmpRcFn vmxCmpRc) {
  ScratchSimd128Scope scratch(masm);
  ZeroSimd128(masm, scratch);
  uint8_t s = scratch.encoding() & 31;
  vmxCmpRc(static_cast<Assembler&>(masm), s, src.encoding() & 31, s);
  if (HasPOWER10()) {
    // setbc materialises CR6.EQ directly into dest (1 insn vs the 2-insn
    // mfocrf + rlwinm extract). Already wired in ma_cmp_set.
    masm.as_setbc(dest, Assembler::Equal, cr6);
    return;
  }
  masm.as_mfocrf(dest, cr6);
  masm.as_rlwinm(dest, dest, 27, 31, 31);
}

void MacroAssembler::allTrueInt8x16(FloatRegister src, Register dest) {
  EmitAllTrueInt(*this, src, dest,
                 [](Assembler& a, uint8_t t, uint8_t r, uint8_t b) {
                   a.as_vcmpequb_rc(t, r, b);
                 });
}

void MacroAssembler::allTrueInt16x8(FloatRegister src, Register dest) {
  EmitAllTrueInt(*this, src, dest,
                 [](Assembler& a, uint8_t t, uint8_t r, uint8_t b) {
                   a.as_vcmpequh_rc(t, r, b);
                 });
}

void MacroAssembler::allTrueInt32x4(FloatRegister src, Register dest) {
  EmitAllTrueInt(*this, src, dest,
                 [](Assembler& a, uint8_t t, uint8_t r, uint8_t b) {
                   a.as_vcmpequw_rc(t, r, b);
                 });
}

void MacroAssembler::allTrueInt64x2(FloatRegister src, Register dest) {
  EmitAllTrueInt(*this, src, dest,
                 [](Assembler& a, uint8_t t, uint8_t r, uint8_t b) {
                   a.as_vcmpequd_rc(t, r, b);
                 });
}

void MacroAssembler::compareInt8x16(Assembler::Condition cond,
                                    FloatRegister rhs, FloatRegister lhsDest) {
  compareInt8x16(cond, lhsDest, rhs, lhsDest);
}

void MacroAssembler::compareInt8x16(Assembler::Condition cond,
                                    FloatRegister lhs, FloatRegister rhs,
                                    FloatRegister dest) {
  if (cond == Assembler::NotEqual && HasPOWER9()) {
    EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vcmpneb), lhs, rhs, dest);
    return;
  }
  EmitVmxCompare(*this, cond, lhs, rhs, dest, VMX_BINARY_WRAPPER(vcmpequb),
                 VMX_BINARY_WRAPPER(vcmpgtsb), VMX_BINARY_WRAPPER(vcmpgtub));
}

void MacroAssembler::compareInt16x8(Assembler::Condition cond,
                                    FloatRegister rhs, FloatRegister lhsDest) {
  compareInt16x8(cond, lhsDest, rhs, lhsDest);
}

void MacroAssembler::compareInt16x8(Assembler::Condition cond,
                                    FloatRegister lhs, FloatRegister rhs,
                                    FloatRegister dest) {
  if (cond == Assembler::NotEqual && HasPOWER9()) {
    EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vcmpneh), lhs, rhs, dest);
    return;
  }
  EmitVmxCompare(*this, cond, lhs, rhs, dest, VMX_BINARY_WRAPPER(vcmpequh),
                 VMX_BINARY_WRAPPER(vcmpgtsh), VMX_BINARY_WRAPPER(vcmpgtuh));
}

void MacroAssembler::compareInt32x4(Assembler::Condition cond,
                                    FloatRegister rhs, FloatRegister lhsDest) {
  compareInt32x4(cond, lhsDest, rhs, lhsDest);
}

void MacroAssembler::compareInt32x4(Assembler::Condition cond,
                                    FloatRegister lhs, FloatRegister rhs,
                                    FloatRegister dest) {
  if (cond == Assembler::NotEqual && HasPOWER9()) {
    EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vcmpnew), lhs, rhs, dest);
    return;
  }
  EmitVmxCompare(*this, cond, lhs, rhs, dest, VMX_BINARY_WRAPPER(vcmpequw),
                 VMX_BINARY_WRAPPER(vcmpgtsw), VMX_BINARY_WRAPPER(vcmpgtuw));
}

void MacroAssembler::compareFloat32x4(Assembler::Condition cond,
                                      FloatRegister rhs,
                                      FloatRegister lhsDest) {
  compareFloat32x4(cond, lhsDest, rhs, lhsDest);
}

void MacroAssembler::compareFloat32x4(Assembler::Condition cond,
                                      FloatRegister lhs, FloatRegister rhs,
                                      FloatRegister dest) {
  switch (cond) {
    case Assembler::Equal:
      as_xvcmpeqsp(dest, lhs, rhs);
      break;
    case Assembler::NotEqual:
      as_xvcmpeqsp(dest, lhs, rhs);
      bitwiseNotSimd128(dest, dest);
      break;
    case Assembler::GreaterThan:
      as_xvcmpgtsp(dest, lhs, rhs);
      break;
    case Assembler::GreaterThanOrEqual:
      as_xvcmpgesp(dest, lhs, rhs);
      break;
    case Assembler::LessThan:
      as_xvcmpgtsp(dest, rhs, lhs);
      break;
    case Assembler::LessThanOrEqual:
      as_xvcmpgesp(dest, rhs, lhs);
      break;
    default:
      MOZ_CRASH("Unexpected SIMD float condition");
  }
}

void MacroAssembler::compareFloat64x2(Assembler::Condition cond,
                                      FloatRegister rhs,
                                      FloatRegister lhsDest) {
  compareFloat64x2(cond, lhsDest, rhs, lhsDest);
}

void MacroAssembler::compareFloat64x2(Assembler::Condition cond,
                                      FloatRegister lhs, FloatRegister rhs,
                                      FloatRegister dest) {
  switch (cond) {
    case Assembler::Equal:
      as_xvcmpeqdp(dest, lhs, rhs);
      break;
    case Assembler::NotEqual:
      as_xvcmpeqdp(dest, lhs, rhs);
      bitwiseNotSimd128(dest, dest);
      break;
    case Assembler::GreaterThan:
      as_xvcmpgtdp(dest, lhs, rhs);
      break;
    case Assembler::GreaterThanOrEqual:
      as_xvcmpgedp(dest, lhs, rhs);
      break;
    case Assembler::LessThan:
      as_xvcmpgtdp(dest, rhs, lhs);
      break;
    case Assembler::LessThanOrEqual:
      as_xvcmpgedp(dest, rhs, lhs);
      break;
    default:
      MOZ_CRASH("Unexpected SIMD float condition");
  }
}

void MacroAssembler::negFloat32x4(FloatRegister src, FloatRegister dest) {
  as_xvnegsp(dest, src);
}

void MacroAssembler::negFloat64x2(FloatRegister src, FloatRegister dest) {
  as_xvnegdp(dest, src);
}

void MacroAssembler::absFloat32x4(FloatRegister src, FloatRegister dest) {
  as_xvabssp(dest, src);
}

void MacroAssembler::absFloat64x2(FloatRegister src, FloatRegister dest) {
  as_xvabsdp(dest, src);
}

// Per spec:
//   result[k] = (s|u)ext_widen(src[2k]) + (s|u)ext_widen(src[2k+1])
// POWER lacks pairwise multiply-add. Emulate via vmulX{e,o}X(src, splat(1))
// + vadd. Both vmuls need `src` AND `splat(1)` available simultaneously.
//
// Available SIMD slots without involving Lowering:
//   - ScratchSimd128Reg (VR0, non-allocatable)
//   - dest, src
// That's 3 regs when dest != src — enough for {src, splat, intermediate}.
// When dest == src we stash src and the even product to the 288-byte ELFv2
// red zone and rebuild splat(1).
//
// (Earlier implementations of these helpers routed through hardcoded
// VR1/VR2/VR3 via xxlor_vsr — faster but stomped allocator-managed VRs
// and silently corrupted any live wasm v128 the allocator had placed
// there. ScratchSimd128Reg + red-zone stash is the safe contract.)
// Always-safe pattern: stash src to red zone so dest can be freely overwritten,
// stash even to red zone after first vmul so we can rebuild splat(1) for the
// second vmul. The splat-of-1 is now `vspltis{b,h}` (5-bit signed immediate
// splat) — 1 insn vs the 3-insn movePtr+mtvsrd+vsplt sequence the previous
// path used.
// Pattern: stash src to red zone slot 0 so dest can be freely overwritten;
// vmul-even (signed/unsigned) of src with splat(1) produces sign/zero-extended
// even-lane products into dest; stash that to slot 1 and rebuild scratch=src
// (slot 0) and dest=splat(1); vmul-odd produces the odd products; restore
// even from slot 1 and pairwise-add.
void MacroAssembler::extAddPairwiseInt8x16(FloatRegister src,
                                           FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  uint8_t s = scratch.encoding() & 31;
  uint8_t srcEnc = src.encoding() & 31;
  uint8_t destEnc = dest.encoding() & 31;
  RedZoneStashSimd128(*this, src, 0);
  as_vspltisb(s, 1);
  as_vmulesb(destEnc, srcEnc, s);
  RedZoneStashSimd128(*this, dest, 1);
  RedZoneRestoreSimd128(*this, 0, scratch);
  as_vspltisb(destEnc, 1);
  as_vmulosb(destEnc, s, destEnc);
  RedZoneRestoreSimd128(*this, 1, scratch);
  as_vadduhm(destEnc, destEnc, s);
}

void MacroAssembler::unsignedExtAddPairwiseInt8x16(FloatRegister src,
                                                   FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  uint8_t s = scratch.encoding() & 31;
  uint8_t srcEnc = src.encoding() & 31;
  uint8_t destEnc = dest.encoding() & 31;
  RedZoneStashSimd128(*this, src, 0);
  as_vspltisb(s, 1);
  as_vmuleub(destEnc, srcEnc, s);
  RedZoneStashSimd128(*this, dest, 1);
  RedZoneRestoreSimd128(*this, 0, scratch);
  as_vspltisb(destEnc, 1);
  as_vmuloub(destEnc, s, destEnc);
  RedZoneRestoreSimd128(*this, 1, scratch);
  as_vadduhm(destEnc, destEnc, s);
}

// vmsumshm/vmsumuhm collapse the i16x8 → i32x4 pairwise-add into a single
// multiply-sum: VT.i32[k] = VRA.i16[2k]*VRB.i16[2k] +
// VRA.i16[2k+1]*VRB.i16[2k+1]
// + VRC.i32[k]. With VRB = splat(1) and VRC = 0 this is exactly the wasm
// i32x4.extadd_pairwise_i16x8_{s,u} contract. 3 insns when dest != src;
// LWasmUnarySimd128 uses useRegisterAtStart so dest may alias src — in that
// case we put splat(1) into scratch (preserving src in dest) and use a
// red-zone slot for the zero VRC operand.
void MacroAssembler::extAddPairwiseInt16x8(FloatRegister src,
                                           FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  if (dest != src) {
    as_xxlxor(scratch, scratch, scratch);  // scratch = 0 (VRC addend)
    as_vspltish(dest.encoding() & 31, 1);  // dest = splat(1) (VRB multiplier)
    as_vmsumshm(dest.encoding() & 31, src.encoding() & 31, dest.encoding() & 31,
                scratch.encoding() & 31);
    return;
  }
  // dest == src: load splat(1) into scratch instead, stash zero to the red
  // zone, restore zero into scratch after the splat is consumed... actually
  // simpler: use vmule/vmulo + vadd trio with red zone. Same shape as the
  // pre-vmsumshm fallback for i8x16.
  uint8_t s = scratch.encoding() & 31;
  uint8_t srcEnc = src.encoding() & 31;
  uint8_t destEnc = dest.encoding() & 31;
  RedZoneStashSimd128(*this, src, 0);
  as_vspltish(s, 1);
  as_vmulesh(destEnc, srcEnc, s);
  RedZoneStashSimd128(*this, dest, 1);
  RedZoneRestoreSimd128(*this, 0, scratch);
  as_vspltish(destEnc, 1);
  as_vmulosh(destEnc, s, destEnc);
  RedZoneRestoreSimd128(*this, 1, scratch);
  as_vadduwm(destEnc, destEnc, s);
}

void MacroAssembler::unsignedExtAddPairwiseInt16x8(FloatRegister src,
                                                   FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  if (dest != src) {
    as_xxlxor(scratch, scratch, scratch);
    as_vspltish(dest.encoding() & 31, 1);
    as_vmsumuhm(dest.encoding() & 31, src.encoding() & 31, dest.encoding() & 31,
                scratch.encoding() & 31);
    return;
  }
  uint8_t s = scratch.encoding() & 31;
  uint8_t srcEnc = src.encoding() & 31;
  uint8_t destEnc = dest.encoding() & 31;
  RedZoneStashSimd128(*this, src, 0);
  as_vspltish(s, 1);
  as_vmuleuh(destEnc, srcEnc, s);
  RedZoneStashSimd128(*this, dest, 1);
  RedZoneRestoreSimd128(*this, 0, scratch);
  as_vspltish(destEnc, 1);
  as_vmulouh(destEnc, s, destEnc);
  RedZoneRestoreSimd128(*this, 1, scratch);
  as_vadduwm(destEnc, destEnc, s);
}

void MacroAssembler::sqrtFloat32x4(FloatRegister src, FloatRegister dest) {
  as_xvsqrtsp(dest, src);
}

void MacroAssembler::sqrtFloat64x2(FloatRegister src, FloatRegister dest) {
  as_xvsqrtdp(dest, src);
}

void MacroAssembler::convertInt32x4ToFloat32x4(FloatRegister src,
                                               FloatRegister dest) {
  as_xvcvsxwsp(dest, src);
}

void MacroAssembler::unsignedConvertInt32x4ToFloat32x4(FloatRegister src,
                                                       FloatRegister dest) {
  as_xvcvuxwsp(dest, src);
}

// i32x4 (low 2 lanes) → f64x2. Wasm `f64x2.convert_low_i32x4_{s,u}`.
// xvcv{s,u}xwdp converts BE word 0 and BE word 2 of source to doubles in
// BE dwords 0 and 1. vmrglw places src.word_BE[2,3] at the read positions,
// matching the f32→f64 promote shape:
//   vmrglw    scratch, src, src    ; BE words 2,3 of src → BE words 0,2 of
//   scratch xvcv*xwdp dest, scratch        ; convert both, place in BE dwords
//   0,1
// Output BE dwords land as [convert(input lane 1), convert(input lane 0)],
// which on PPC64LE storage IS the wasm output layout.
//
// 2 insns each, single ScratchSimd128 scope, no GPR or FPR scratch.
// All ops POWER7+. dest==src aliasing safe (vmrglw consumes src into
// scratch before dest is written).
void MacroAssembler::convertInt32x4ToFloat64x2(FloatRegister src,
                                               FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  as_vmrglw(scratch.encoding() & 31, src.encoding() & 31, src.encoding() & 31);
  as_xvcvsxwdp(dest, scratch);
}

void MacroAssembler::unsignedConvertInt32x4ToFloat64x2(FloatRegister src,
                                                       FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  as_vmrglw(scratch.encoding() & 31, src.encoding() & 31, src.encoding() & 31);
  as_xvcvuxwdp(dest, scratch);
}

void MacroAssembler::truncSatFloat32x4ToInt32x4(FloatRegister src,
                                                FloatRegister dest) {
  // xvcvspsxws gives INT32_MIN for NaN, but Wasm requires 0.
  ScratchSimd128Scope scratch(*this);
  as_xvcmpeqsp(scratch, src, src);  // ~0 for non-NaN, 0 for NaN
  as_xvcvspsxws(dest, src);
  as_xxland(dest, dest, scratch);  // zero NaN lanes
}

// Pack the two "interesting" 32-bit results that xvcv*xws / xvcvdpsp leaves
// at scratch.word_BE[0] (= A) and scratch.word_BE[2] (= B) into a zeroed dest
// as dest.word_BE = [0, 0, A, B]. This is the layout wasm requires for
// f64x2 → {i32x4 trunc_sat, f32x4 demote}. Writes dest, consumes scratch.
//
// POWER9 path (4 insns) uses xxinsertw/xxextractuw. POWER8 path (7 insns)
// goes via two GPR round-trips: extract A and B with mfvsrd, splice them
// into a single dword with rldimi, mtvsrd back into a SIMD reg, and
// xxpermdi the result into dest.dw1 while keeping dest.dw0 zero.
static inline void PackTwoWordsToLowHalf(MacroAssembler& masm,
                                         FloatRegister scratch,
                                         FloatRegister dest) {
  if (HasPOWER9()) {
    masm.as_xxinsertw(dest, scratch,
                      8);  // dest.word_BE[2] ← scratch.word_BE[1] (= A)
    masm.as_xxextractuw(scratch, scratch,
                        8);  // scratch.word_BE[1] ← scratch.word_BE[2] (= B)
    masm.as_xxinsertw(dest, scratch,
                      12);  // dest.word_BE[3] ← scratch.word_BE[1] (= B)
    return;
  }
  // POWER8: xxinsertw/xxextractuw are ISA 3.0. Take a GPR detour instead.
  // scratch.dw_BE[0] = (A << 32) | A, scratch.dw_BE[1] = (B << 32) | B.
  UseScratchRegisterScope temps(masm);
  Register tmpA = temps.Acquire();
  Register tmpB = temps.Acquire();
  masm.as_mfvsrd(tmpA, scratch);  // tmpA = (A << 32) | A
  masm.as_xxpermdi(scratch, scratch, scratch,
                   2);            // swap dwords: now dw0 = (B<<32)|B
  masm.as_mfvsrd(tmpB, scratch);  // tmpB = (B << 32) | B
  masm.x_srdi(tmpA, tmpA, 32);    // tmpA = 0x00000000_AAAAAAAA
  masm.as_rldimi(tmpB, tmpA, 32,
                 0);              // tmpB[0..31] = A; tmpB[32..63] = B (kept)
  masm.as_mtvsrd(scratch, tmpB);  // scratch.dw_BE[0] = (A << 32) | B; dw1 = 0
  masm.as_xxpermdi(dest, dest, scratch,
                   0);  // dest = {dest.dw0=0, scratch.dw0} = [0, 0, A, B]
}

// fctiwz / fcmpu / fctiduz are X-form scalar FP instructions that only
// encode 5-bit FRT/FRB fields, so emitting them on a Simd128 reg
// (encoding 32+) would corrupt the opcode. Bridge through
// ScratchDoubleReg (FPR f0) for the conversion. Extract both lanes' GPR
// results before writing dest so that dest == src is safe.
//
// Avoid replaceLaneInt32x4 on the tail: on POWER8 it needs an extra
// GPR scratch, but r11 and r12 are already held as a/b here. Pack both
// int32s into `a` with rldimi, transfer via mtvsrd, then xxpermdi the
// DWs into the low half so wasm lane 0 (BE W3) holds a, lane 1 (W2) b.
void MacroAssembler::truncSatFloat64x2ToInt32x4(FloatRegister src,
                                                FloatRegister dest,
                                                FloatRegister temp) {
  // Wasm `i32x4.trunc_sat_f64x2_s_zero`. xvcvdpsxws saturates to INT32_MIN
  // on overflow/NaN (per ISA); wasm requires NaN → 0, so a per-dword NaN
  // mask via xvcmpeqdp clamps NaN lanes to 0 before laying out the result.
  // Output BE word positions need wasm lane order: lane 1 → BE word 2,
  // lane 0 → BE word 3. xvcvdpsxws lands its results at BE words 0 and 2
  // (with replication into 1/3); PackTwoWordsToLowHalf moves them into
  // the right positions while zeroing the rest.
  // dest==src safe: src is consumed by xvcvdpsxws and xvcmpeqdp before
  // dest is zeroed.
  ScratchSimd128Scope scratch(*this);
  as_xvcvdpsxws(scratch, src);
  as_xvcmpeqdp(dest, src,
               src);  // NaN-mask: 0xFF...F per dword for non-NaN, 0 for NaN
  as_xxland(scratch, scratch, dest);
  as_xxlxor(dest, dest, dest);
  PackTwoWordsToLowHalf(*this, scratch, dest);
}

void MacroAssembler::unsignedTruncSatFloat64x2ToInt32x4(FloatRegister src,
                                                        FloatRegister dest,
                                                        FloatRegister temp) {
  // Wasm `i32x4.trunc_sat_f64x2_u_zero`. xvcvdpuxws semantics already
  // match the wasm spec without any masking: NaN → 0, negative → 0,
  // positive overflow → UINT32_MAX. So no NaN mask needed; just position
  // the saturated results into BE words 2,3 with zeros at words 0,1.
  // dest==src safe: src consumed by xvcvdpuxws before dest is zeroed.
  ScratchSimd128Scope scratch(*this);
  as_xvcvdpuxws(scratch, src);
  as_xxlxor(dest, dest, dest);
  PackTwoWordsToLowHalf(*this, scratch, dest);
}

void MacroAssembler::truncFloat32x4ToInt32x4Relaxed(FloatRegister src,
                                                    FloatRegister dest) {
  truncSatFloat32x4ToInt32x4(src, dest);
}

void MacroAssembler::unsignedTruncFloat32x4ToInt32x4Relaxed(
    FloatRegister src, FloatRegister dest) {
  unsignedTruncSatFloat32x4ToInt32x4(src, dest);
}

void MacroAssembler::truncFloat64x2ToInt32x4Relaxed(FloatRegister src,
                                                    FloatRegister dest) {
  truncSatFloat64x2ToInt32x4(src, dest, ScratchSimd128Reg);
}

void MacroAssembler::unsignedTruncFloat64x2ToInt32x4Relaxed(
    FloatRegister src, FloatRegister dest) {
  unsignedTruncSatFloat64x2ToInt32x4(src, dest, ScratchSimd128Reg);
}

// f64x2 → f32x4 (low 2 lanes; high lanes zero). Wasm `f32x4.demote_f64x2_zero`.
// xvcvdpsp converts both doubles in one shot, replicating each result across
// its dword: BE word lanes = [s(in.dw0), s(in.dw0), s(in.dw1), s(in.dw1)].
// On PPC64LE wasm storage (lxvx-loaded), input.dw_BE[0] = wasm lane 1 and
// input.dw_BE[1] = wasm lane 0, so we get [s(l1), s(l1), s(l0), s(l0)] in
// BE word order. We then zero dest and pack s(l1) into BE word 2 (wasm
// output lane 1) and s(l0) into BE word 3 (wasm output lane 0) via the
// shared PackTwoWordsToLowHalf helper, which has POWER9 and POWER8 paths.
//
// dest==src aliasing safe: src is consumed by xvcvdpsp before dest is zeroed.
void MacroAssembler::convertFloat64x2ToFloat32x4(FloatRegister src,
                                                 FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  as_xvcvdpsp(scratch, src);
  ZeroSimd128(*this, dest);
  PackTwoWordsToLowHalf(*this, scratch, dest);
}

// f32x4 (low 2 lanes) → f64x2. Wasm `f64x2.promote_low_f32x4`. xvcvspdp
// converts both BE word 0 and BE word 2 of its source to doubles in BE
// dwords 0 and 1 respectively. To get wasm lanes 0 and 1 (= input BE
// words 3 and 2) into those source positions, vmrglw merges low words:
// VRT.word[0] = VRA.word[2] = wasm lane 1, VRT.word[2] = VRA.word[3] =
// wasm lane 0 (with replicated copies in odd word slots that xvcvspdp
// ignores). Output BE dwords land as [double(lane1), double(lane0)],
// which on PPC64LE storage is exactly the wasm f64x2 output layout.
//
// dest==src aliasing safe: vmrglw consumes src into a separate scratch
// before dest is written.
//
// 2 insns, single ScratchSimd128 scope. All ops POWER7+.
void MacroAssembler::convertFloat32x4ToFloat64x2(FloatRegister src,
                                                 FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  as_vmrglw(scratch.encoding() & 31, src.encoding() & 31, src.encoding() & 31);
  as_xvcvspdp(dest, scratch);
}

void MacroAssembler::unsignedNarrowInt16x8(FloatRegister lhs, FloatRegister rhs,
                                           FloatRegister dest) {
  // On LE, VMX pack swaps operand order vs Wasm convention.
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vpkshus), rhs, lhs, dest);
}

void MacroAssembler::unsignedNarrowInt32x4(FloatRegister lhs, FloatRegister rhs,
                                           FloatRegister dest) {
  // On LE, VMX pack swaps operand order vs Wasm convention.
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vpkswus), rhs, lhs, dest);
}

void MacroAssembler::widenLowInt8x16(FloatRegister src, FloatRegister dest) {
  // On PPC64 LE, raw vupklsb unpacks the LOW Wasm lanes (not vupkhsb).
  // GCC vec_unpackh maps to vupklsb on LE (swapped from BE naming).
  // Raw vupklsb([1..8,-1..-8]) = [1,2,3,4,5,6,7,8].
  EmitVmxUnary(
      *this,
      [](Assembler& a, uint8_t vrt, uint8_t vrb) { a.as_vupklsb(vrt, vrb); },
      src, dest);
}

void MacroAssembler::widenHighInt8x16(FloatRegister src, FloatRegister dest) {
  // On PPC64 LE, raw vupkhsb unpacks the HIGH Wasm lanes.
  EmitVmxUnary(
      *this,
      [](Assembler& a, uint8_t vrt, uint8_t vrb) { a.as_vupkhsb(vrt, vrb); },
      src, dest);
}

void MacroAssembler::unsignedWidenLowInt8x16(FloatRegister src,
                                             FloatRegister dest) {
  zeroExtend8x16To16x8(src, dest);
}

void MacroAssembler::unsignedWidenHighInt8x16(FloatRegister src,
                                              FloatRegister dest) {
  // vmrghb(zero, src) interleaves zero bytes with the BE-high half of src,
  // producing zero-extended halfwords of the LE-high (Wasm-high) lanes.
  ScratchSimd128Scope scratch(*this);
  as_xxlxor(scratch, scratch, scratch);
  as_vmrghb(dest.encoding() & 31, scratch.encoding() & 31, src.encoding() & 31);
}

void MacroAssembler::widenLowInt16x8(FloatRegister src, FloatRegister dest) {
  // On PPC64 LE, raw vupklsh unpacks LOW Wasm lanes (GCC swaps h/l on LE).
  EmitVmxUnary(
      *this,
      [](Assembler& a, uint8_t vrt, uint8_t vrb) { a.as_vupklsh(vrt, vrb); },
      src, dest);
}

void MacroAssembler::widenHighInt16x8(FloatRegister src, FloatRegister dest) {
  EmitVmxUnary(
      *this,
      [](Assembler& a, uint8_t vrt, uint8_t vrb) { a.as_vupkhsh(vrt, vrb); },
      src, dest);
}

void MacroAssembler::unsignedWidenLowInt16x8(FloatRegister src,
                                             FloatRegister dest) {
  zeroExtend16x8To32x4(src, dest);
}

void MacroAssembler::unsignedWidenHighInt16x8(FloatRegister src,
                                              FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  as_xxlxor(scratch, scratch, scratch);
  as_vmrghh(dest.encoding() & 31, scratch.encoding() & 31, src.encoding() & 31);
}

void MacroAssembler::widenLowInt32x4(FloatRegister src, FloatRegister dest) {
  // On PPC64 LE, raw vupklsw unpacks LOW Wasm lanes.
  EmitVmxUnary(
      *this,
      [](Assembler& a, uint8_t vrt, uint8_t vrb) { a.as_vupklsw(vrt, vrb); },
      src, dest);
}

void MacroAssembler::unsignedWidenLowInt32x4(FloatRegister src,
                                             FloatRegister dest) {
  zeroExtend32x4To64x2(src, dest);
}

void MacroAssembler::widenHighInt32x4(FloatRegister src, FloatRegister dest) {
  EmitVmxUnary(
      *this,
      [](Assembler& a, uint8_t vrt, uint8_t vrb) { a.as_vupkhsw(vrt, vrb); },
      src, dest);
}

void MacroAssembler::unsignedWidenHighInt32x4(FloatRegister src,
                                              FloatRegister dest) {
  // i64x2.extend_high_i32x4_u: take high 2 i32 lanes of src, zero-extend
  // to i64 each. Use vmrghw to interleave a zero VR with src — same shape
  // as the (already-correct) unsignedWidenHighInt16x8 sibling above.
  ScratchSimd128Scope scratch(*this);
  ZeroSimd128(*this, scratch);
  as_vmrghw(dest.encoding() & 31, scratch.encoding() & 31, src.encoding() & 31);
}

void MacroAssembler::pseudoMinFloat32x4(FloatRegister rhsOrRhsDest,
                                        FloatRegister lhsOrLhsDest) {
  // pmin: result[i] = rhs[i] < lhs[i] ? rhs[i] : lhs[i]
  // xvcmpgtsp(mask, lhs, rhs) → 1 where lhs > rhs (i.e., rhs < lhs)
  // xxsel: mask=1 → XB=rhs. mask=0 → XA=lhs.
  // Result goes to lhsOrLhsDest (second param).
  ScratchSimd128Scope scratch(*this);
  as_xvcmpgtsp(scratch, lhsOrLhsDest, rhsOrRhsDest);
  as_xxsel(lhsOrLhsDest, lhsOrLhsDest, rhsOrRhsDest, scratch);
}

void MacroAssembler::pseudoMinFloat32x4(FloatRegister lhs, FloatRegister rhs,
                                        FloatRegister dest) {
  // pmin(lhs, rhs) = rhs < lhs ? rhs : lhs
  // Inline to handle dest aliasing with either operand.
  ScratchSimd128Scope scratch(*this);
  as_xvcmpgtsp(scratch, lhs, rhs);
  // mask=1 where lhs > rhs. XC=1 → select XB=rhs. XC=0 → select XA=lhs.
  as_xxsel(dest, lhs, rhs, scratch);
}

void MacroAssembler::pseudoMinFloat64x2(FloatRegister rhsOrRhsDest,
                                        FloatRegister lhsOrLhsDest) {
  ScratchSimd128Scope scratch(*this);
  as_xvcmpgtdp(scratch, lhsOrLhsDest, rhsOrRhsDest);
  as_xxsel(lhsOrLhsDest, lhsOrLhsDest, rhsOrRhsDest, scratch);
}

void MacroAssembler::pseudoMinFloat64x2(FloatRegister lhs, FloatRegister rhs,
                                        FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  as_xvcmpgtdp(scratch, lhs, rhs);
  as_xxsel(dest, lhs, rhs, scratch);
}

void MacroAssembler::pseudoMaxFloat32x4(FloatRegister rhsOrRhsDest,
                                        FloatRegister lhsOrLhsDest) {
  ScratchSimd128Scope scratch(*this);
  as_xvcmpgtsp(scratch, rhsOrRhsDest, lhsOrLhsDest);
  as_xxsel(lhsOrLhsDest, lhsOrLhsDest, rhsOrRhsDest, scratch);
}

void MacroAssembler::pseudoMaxFloat32x4(FloatRegister lhs, FloatRegister rhs,
                                        FloatRegister dest) {
  // pmax(lhs, rhs) = lhs < rhs ? rhs : lhs
  ScratchSimd128Scope scratch(*this);
  as_xvcmpgtsp(scratch, rhs, lhs);
  // mask=1 where rhs > lhs (lhs < rhs). XC=1 → select XB=rhs. XC=0 → select
  // XA=lhs.
  as_xxsel(dest, lhs, rhs, scratch);
}

void MacroAssembler::pseudoMaxFloat64x2(FloatRegister rhsOrRhsDest,
                                        FloatRegister lhsOrLhsDest) {
  ScratchSimd128Scope scratch(*this);
  as_xvcmpgtdp(scratch, rhsOrRhsDest, lhsOrLhsDest);
  as_xxsel(lhsOrLhsDest, lhsOrLhsDest, rhsOrRhsDest, scratch);
}

void MacroAssembler::pseudoMaxFloat64x2(FloatRegister lhs, FloatRegister rhs,
                                        FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  as_xvcmpgtdp(scratch, rhs, lhs);
  as_xxsel(dest, lhs, rhs, scratch);
}

void MacroAssembler::dotInt8x16Int7x16(FloatRegister lhs, FloatRegister rhs,
                                       FloatRegister dest) {
  // result[k] = lhs[2k]*rhs[2k] + lhs[2k+1]*rhs[2k+1] for k=0..7.
  // vmulesb/vmulosb produce even/odd byte products as i16 in matching
  // halfword lanes; vadduhm sums them pairwise.
  ScratchSimd128Scope scratch(*this);
  uint8_t l = lhs.encoding() & 31;
  uint8_t r = rhs.encoding() & 31;
  uint8_t d = dest.encoding() & 31;
  uint8_t s = scratch.encoding() & 31;
  as_vmulesb(s, l, r);
  as_vmulosb(d, l, r);
  as_vadduhm(d, s, d);
}

void MacroAssembler::ceilFloat32x4(FloatRegister src, FloatRegister dest) {
  as_xvrspip(dest, src);
}

void MacroAssembler::ceilFloat64x2(FloatRegister src, FloatRegister dest) {
  as_xvrdpip(dest, src);
}

void MacroAssembler::floorFloat32x4(FloatRegister src, FloatRegister dest) {
  as_xvrspim(dest, src);
}

void MacroAssembler::floorFloat64x2(FloatRegister src, FloatRegister dest) {
  as_xvrdpim(dest, src);
}

void MacroAssembler::truncFloat32x4(FloatRegister src, FloatRegister dest) {
  as_xvrspiz(dest, src);
}

void MacroAssembler::truncFloat64x2(FloatRegister src, FloatRegister dest) {
  as_xvrdpiz(dest, src);
}

void MacroAssembler::nearestFloat32x4(FloatRegister src, FloatRegister dest) {
  as_xvrspic(dest, src);
}

void MacroAssembler::nearestFloat64x2(FloatRegister src, FloatRegister dest) {
  as_xvrdpic(dest, src);
}

void MacroAssembler::fnmaFloat32x4(FloatRegister src1, FloatRegister src2,
                                   FloatRegister srcDest) {
  as_xvnmsubasp(srcDest, src1, src2);
}

void MacroAssembler::fnmaFloat64x2(FloatRegister src1, FloatRegister src2,
                                   FloatRegister srcDest) {
  as_xvnmsubadp(srcDest, src1, src2);
}

void MacroAssembler::minFloat32x4Relaxed(FloatRegister src,
                                         FloatRegister srcDest) {
  as_xvminsp(srcDest, srcDest, src);
}

void MacroAssembler::minFloat32x4Relaxed(FloatRegister lhs, FloatRegister rhs,
                                         FloatRegister dest) {
  as_xvminsp(dest, lhs, rhs);
}

void MacroAssembler::maxFloat32x4Relaxed(FloatRegister src,
                                         FloatRegister srcDest) {
  as_xvmaxsp(srcDest, srcDest, src);
}

void MacroAssembler::maxFloat32x4Relaxed(FloatRegister lhs, FloatRegister rhs,
                                         FloatRegister dest) {
  as_xvmaxsp(dest, lhs, rhs);
}

void MacroAssembler::minFloat64x2Relaxed(FloatRegister src,
                                         FloatRegister srcDest) {
  as_xvmindp(srcDest, srcDest, src);
}

void MacroAssembler::minFloat64x2Relaxed(FloatRegister lhs, FloatRegister rhs,
                                         FloatRegister dest) {
  as_xvmindp(dest, lhs, rhs);
}

void MacroAssembler::maxFloat64x2Relaxed(FloatRegister src,
                                         FloatRegister srcDest) {
  as_xvmaxdp(srcDest, srcDest, src);
}

void MacroAssembler::maxFloat64x2Relaxed(FloatRegister lhs, FloatRegister rhs,
                                         FloatRegister dest) {
  as_xvmaxdp(dest, lhs, rhs);
}

void MacroAssembler::q15MulrInt16x8Relaxed(FloatRegister lhs, FloatRegister rhs,
                                           FloatRegister dest) {
  q15MulrSatInt16x8(lhs, rhs, dest);
}

// SIMD overloads accepting an extra FloatRegister temp (shared-header signature
// used by x86; on PPC64 the temp is unused for most of these).
void MacroAssembler::popcntInt8x16(FloatRegister src, FloatRegister dest,
                                   FloatRegister temp) {
  popcntInt8x16(src, dest);
}

void MacroAssembler::unsignedTruncSatFloat32x4ToInt32x4(FloatRegister src,
                                                        FloatRegister dest,
                                                        FloatRegister temp) {
  unsignedTruncSatFloat32x4ToInt32x4(src, dest);
}

void MacroAssembler::dotInt8x16Int7x16ThenAdd(FloatRegister lhs,
                                              FloatRegister rhs,
                                              FloatRegister dest,
                                              FloatRegister temp) {
  // dest += pairwise_widen_i16_to_i32(dot_i8x16(lhs, rhs)).
  //
  // Step 1: i16x8 dot of i8 byte pairs (vmulesb/vmulosb/vadduhm). Keeps
  // the existing signed-byte multiply semantics that match ARM64 sdot
  // and x86 vpdpbssd (vmsummbm would be signed×unsigned and diverge for
  // i7 lanes that bit-pattern as negative).
  //
  // Step 2: vmsumshm dest, dot, splat_hw(1), dest computes
  //   dest.i32[k] = dest.i32[k] + dot.i16[2k]*1 + dot.i16[2k+1]*1
  // which is exactly pairwise widen + accumulate in a single insn.
  // splat_hw(1) is a single vspltish (5-bit SIMM splat to all 8 halfwords).
  ScratchSimd128Scope scratch(*this);
  uint8_t l = lhs.encoding() & 31;
  uint8_t r = rhs.encoding() & 31;
  uint8_t d = dest.encoding() & 31;
  uint8_t s = scratch.encoding() & 31;
  uint8_t t = temp.encoding() & 31;

  as_vmulesb(s, l, r);
  as_vmulosb(t, l, r);
  as_vadduhm(t, s, t);
  as_vspltish(s, 1);
  as_vmsumshm(d, t, s, d);
}

// SIMD ops ported from arm64- and x86/x64-shaped signatures.

void MacroAssembler::permuteInt16x8(const uint16_t lanes[8], FloatRegister src,
                                    FloatRegister dest) {
  uint8_t shuffleLanes[16];
  for (unsigned i = 0; i < 8; i++) {
    shuffleLanes[i * 2] = lanes[i] * 2;
    shuffleLanes[i * 2 + 1] = lanes[i] * 2 + 1;
  }
  shuffleInt8x16(shuffleLanes, src, src, dest);
}

void MacroAssembler::rotateRightSimd128(FloatRegister src, FloatRegister dest,
                                        uint32_t shift) {
  MOZ_ASSERT(shift < 16);
  if (shift == 0) {
    moveSimd128(src, dest);
    return;
  }
  // vsldoi VRT,VRA,VRB,SH: concatenate VRA||VRB, take bytes [SH..SH+15].
  // Rotate right by N = vsldoi(src, src, 16-N).
  as_vsldoi(dest, src, src, 16 - shift);
}

void MacroAssembler::mulInt64x2(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest, FloatRegister temp1,
                                FloatRegister temp2) {
  // POWER10 collapses the entire i64x2 multiply to a single vmulld.
  // POWER9/POWER8 fall back to the GPR round-trip path: extract each
  // lane pair into GPRs (mfvsrld for LE-dw0/Wasm-lane-0, mfvsrd for
  // LE-dw1/lane-1), multiply, and reassemble via mtvsrd + xxpermdi.
  if (HasPOWER10()) {
    as_vmulld(dest.encoding() & 31, lhs.encoding() & 31, rhs.encoding() & 31);
    return;
  }
  // Aliasing safety: stash the lane-0 product in ScratchSimd128 (which
  // is non-allocatable, so cannot alias lhs/rhs) and only write dest at
  // the very end, after both lhs and rhs have been fully consumed.
  ScratchSimd128Scope scratch(*this);
  UseScratchRegisterScope temps(asMasm());
  Register a = temps.Acquire();
  Register b = temps.Acquire();

  if (HasPOWER9()) {
    as_mfvsrld(a, lhs);
    as_mfvsrld(b, rhs);
  } else {
    as_xxpermdi(scratch, lhs, lhs, 2);
    as_mfvsrd(a, scratch);
    as_xxpermdi(scratch, rhs, rhs, 2);
    as_mfvsrd(b, scratch);
  }
  as_mulld(a, a, b);
  as_mtvsrd(scratch, a);

  as_mfvsrd(a, lhs);
  as_mfvsrd(b, rhs);
  as_mulld(a, a, b);
  as_mtvsrd(dest, a);
  as_xxpermdi(dest, dest, scratch, 0);
}

void MacroAssembler::bitwiseAndNotSimd128(FloatRegister lhs, FloatRegister rhs,
                                          FloatRegister dest) {
  // andnot(lhs, rhs) = lhs & ~rhs = xxlandc(lhs, rhs)
  as_xxlandc(dest, lhs, rhs);
}

void MacroAssembler::bitwiseSelectSimd128(FloatRegister onTrue,
                                          FloatRegister onFalse,
                                          FloatRegister maskDest) {
  // result = (onTrue & mask) | (onFalse & ~mask)
  // xxsel: XC=0→XA, XC=1→XB → XT = (XA & ~XC) | (XB & XC)
  // Need XA=onFalse, XB=onTrue, XC=mask.
  as_xxsel(maskDest, onFalse, onTrue, maskDest);
}

void MacroAssembler::popcntInt8x16(FloatRegister src, FloatRegister dest) {
  EmitVmxUnary(
      *this,
      [](Assembler& a, uint8_t vrt, uint8_t vrb) { a.as_vpopcntb(vrt, vrb); },
      src, dest);
}

void MacroAssembler::bitmaskInt8x16(FloatRegister src, Register dest,
                                    FloatRegister temp) {
  if (HasPOWER10()) {
    // Single-instruction collapse on POWER10.
    as_vextractbm(dest, src);
    return;
  }
  // POWER8+ vbpermq-based bitmask: ctl[i] = (15-i)*8 produces the wasm-spec
  // bitmap (bit i = MSB of LE lane i) in dw0 low 16 bits.
  int8_t ctl[16] = {120, 112, 104, 96, 88, 80, 72, 64,
                    56,  48,  40,  32, 24, 16, 8,  0};
  loadConstantSimd128(SimdConstant::CreateX16(ctl), temp);
  as_vbpermq(temp.encoding() & 31, src.encoding() & 31, temp.encoding() & 31);
  as_mfvsrd(dest, temp);
}

void MacroAssembler::bitmaskInt16x8(FloatRegister src, Register dest,
                                    FloatRegister temp) {
  if (HasPOWER10()) {
    as_vextracthm(dest, src);
    return;
  }
  // Same recipe as bitmaskInt8x16 but ctl picks halfword MSBs:
  // BE bit (14-2i)*8 for lane i, plus 8 ignore-bytes (high bit set).
  int8_t ctl[16] = {112,  96,   80,   64,   48,   32,   16,   0,
                    -128, -128, -128, -128, -128, -128, -128, -128};
  loadConstantSimd128(SimdConstant::CreateX16(ctl), temp);
  as_vbpermq(temp.encoding() & 31, src.encoding() & 31, temp.encoding() & 31);
  as_mfvsrd(dest, temp);
}

void MacroAssembler::bitmaskInt32x4(FloatRegister src, Register dest,
                                    FloatRegister temp) {
  if (HasPOWER10()) {
    as_vextractwm(dest, src);
    return;
  }
  // Same recipe as bitmaskInt8x16 but ctl picks word MSBs:
  // BE bit (12-4i)*8 for lane i, plus 12 ignore-bytes (high bit set).
  int8_t ctl[16] = {96,   64,   32,   0,    -128, -128, -128, -128,
                    -128, -128, -128, -128, -128, -128, -128, -128};
  loadConstantSimd128(SimdConstant::CreateX16(ctl), temp);
  as_vbpermq(temp.encoding() & 31, src.encoding() & 31, temp.encoding() & 31);
  as_mfvsrd(dest, temp);
}

void MacroAssembler::bitmaskInt64x2(FloatRegister src, Register dest,
                                    FloatRegister temp) {
  if (HasPOWER10()) {
    as_vextractdm(dest, src);
    return;
  }
  // Same recipe as the other bitmask variants. ctl picks dword MSBs:
  // BE bit 64 for lane 0, BE bit 0 for lane 1, plus 14 ignore-bytes.
  int8_t ctl[16] = {64,   0,    -128, -128, -128, -128, -128, -128,
                    -128, -128, -128, -128, -128, -128, -128, -128};
  loadConstantSimd128(SimdConstant::CreateX16(ctl), temp);
  as_vbpermq(temp.encoding() & 31, src.encoding() & 31, temp.encoding() & 31);
  as_mfvsrd(dest, temp);
}

void MacroAssembler::compareInt64x2(Assembler::Condition cond,
                                    FloatRegister rhs, FloatRegister lhsDest) {
  compareInt64x2(cond, lhsDest, rhs, lhsDest);
}

void MacroAssembler::compareInt64x2(Assembler::Condition cond,
                                    FloatRegister lhs, FloatRegister rhs,
                                    FloatRegister dest) {
  EmitVmxCompare(*this, cond, lhs, rhs, dest, VMX_BINARY_WRAPPER(vcmpequd),
                 VMX_BINARY_WRAPPER(vcmpgtsd), VMX_BINARY_WRAPPER(vcmpgtud));
}

void MacroAssembler::minFloat32x4(FloatRegister rhs, FloatRegister lhsDest) {
  minFloat32x4(lhsDest, rhs, lhsDest);
}

void MacroAssembler::minFloat32x4(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest) {
  as_xvminsp(dest, lhs, rhs);
}

void MacroAssembler::minFloat32x4(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest, FloatRegister temp1,
                                  FloatRegister temp2) {
  // Wasm min with NaN propagation.
  // Detect NaN in either operand (not via add which falsely flags inf+(-inf)).
  // Compute mask and add BEFORE min (min may clobber lhs via dest aliasing).
  as_xvcmpeqsp(temp1, lhs, lhs);
  as_xvcmpeqsp(temp2, rhs, rhs);
  as_xxland(temp1, temp1, temp2);
  as_xvaddsp(temp2, lhs, rhs);
  as_xvminsp(dest, lhs, rhs);
  as_xxsel(dest, temp2, dest, temp1);
}

void MacroAssembler::minFloat64x2(FloatRegister rhs, FloatRegister lhsDest) {
  minFloat64x2(lhsDest, rhs, lhsDest);
}

void MacroAssembler::minFloat64x2(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest) {
  as_xvmindp(dest, lhs, rhs);
}

void MacroAssembler::minFloat64x2(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest, FloatRegister temp1,
                                  FloatRegister temp2) {
  // NaN mask and add must be computed BEFORE min (which may clobber lhs via
  // dest).
  as_xvcmpeqdp(temp1, lhs, lhs);
  as_xvcmpeqdp(temp2, rhs, rhs);
  as_xxland(temp1, temp1, temp2);  // temp1 = ~0 when both non-NaN
  as_xvadddp(temp2, lhs, rhs);     // temp2 = add (NaN source)
  as_xvmindp(dest, lhs, rhs);      // dest = min (may clobber lhs)
  as_xxsel(dest, temp2, dest, temp1);
}

void MacroAssembler::maxFloat32x4(FloatRegister rhs, FloatRegister lhsDest) {
  maxFloat32x4(lhsDest, rhs, lhsDest);
}

void MacroAssembler::maxFloat32x4(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest) {
  as_xvmaxsp(dest, lhs, rhs);
}

void MacroAssembler::maxFloat32x4(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest, FloatRegister temp1,
                                  FloatRegister temp2) {
  // Wasm max with NaN propagation, using temp registers.
  as_xvcmpeqsp(temp1, lhs, lhs);
  as_xvcmpeqsp(temp2, rhs, rhs);
  as_xxland(temp1, temp1, temp2);
  as_xvaddsp(temp2, lhs, rhs);
  as_xvmaxsp(dest, lhs, rhs);
  as_xxsel(dest, temp2, dest, temp1);
}

void MacroAssembler::maxFloat64x2(FloatRegister rhs, FloatRegister lhsDest) {
  maxFloat64x2(lhsDest, rhs, lhsDest);
}

void MacroAssembler::maxFloat64x2(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest) {
  as_xvmaxdp(dest, lhs, rhs);
}

void MacroAssembler::maxFloat64x2(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest, FloatRegister temp1,
                                  FloatRegister temp2) {
  as_xvcmpeqdp(temp1, lhs, lhs);
  as_xvcmpeqdp(temp2, rhs, rhs);
  as_xxland(temp1, temp1, temp2);
  as_xvadddp(temp2, lhs, rhs);
  as_xvmaxdp(dest, lhs, rhs);
  as_xxsel(dest, temp2, dest, temp1);
}

void MacroAssembler::unsignedTruncSatFloat32x4ToInt32x4(FloatRegister src,
                                                        FloatRegister dest) {
  as_xvcvspuxws(dest, src);
}

void MacroAssembler::extractLaneInt64x2(uint32_t lane, FloatRegister src,
                                        Register64 dest) {
  MOZ_ASSERT(lane < 2);
  if (lane == 1) {
    // Lane 1 = BE dword 0 = register bits[0:63].
    as_mfvsrd(dest.reg, src);
  } else {
    // Lane 0 = BE dword 1.
    if (HasPOWER9()) {
      as_mfvsrld(dest.reg, src);
    } else {
      ScratchSimd128Scope scratch(*this);
      as_xxpermdi(scratch, src, src, 2);
      as_mfvsrd(dest.reg, scratch);
    }
  }
}

void MacroAssembler::replaceLaneInt64x2(unsigned lane, Register64 rhs,
                                        FloatRegister lhsDest) {
  MOZ_ASSERT(lane < 2);
  if (HasPOWER10()) {
    // 1 insn, no scratch VSR. UIM byte offset: lane 0 → 8, lane 1 → 0.
    as_vinsd(lhsDest, rhs.reg, (1 - lane) * 8);
    return;
  }
  ScratchSimd128Scope scratch(*this);
  as_mtvsrd(scratch, rhs.reg);
  if (lane == 0) {
    // Replace dw1 (LE low = lane 0). Keep dw0 (lane 1).
    // dm=0b00: [lhsDest.dw0, scratch.dw0]
    as_xxpermdi(lhsDest, lhsDest, scratch, 0);
  } else {
    // Replace dw0 (LE high = lane 1). Keep dw1 (lane 0).
    // dm=0b01: [scratch.dw0, lhsDest.dw1]
    as_xxpermdi(lhsDest, scratch, lhsDest, 1);
  }
}

// SIMD 3-operand arithmetic (x86_shared-style signatures).

void MacroAssembler::addFloat32x4(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest) {
  as_xvaddsp(dest, lhs, rhs);
}

void MacroAssembler::addFloat64x2(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest) {
  as_xvadddp(dest, lhs, rhs);
}

void MacroAssembler::addInt16x8(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vadduhm), lhs, rhs, dest);
}

void MacroAssembler::addInt8x16(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vaddubm), lhs, rhs, dest);
}

void MacroAssembler::divFloat32x4(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest) {
  as_xvdivsp(dest, lhs, rhs);
}

void MacroAssembler::extractLaneInt16x8(uint32_t lane, FloatRegister src,
                                        Register dest) {
  MOZ_ASSERT(lane < 8);
  if (HasPOWER9()) {
    as_vextractuh(ScratchSimd128Reg, src, 14 - 2 * lane);
    as_mfvsrd(dest, ScratchSimd128Reg);
    as_extsh(dest, dest);
    return;
  }
  ExtractLaneToGPR(*this, lane, src, dest, 2, 16);
  as_extsh(dest, dest);
}

void MacroAssembler::extractLaneInt32x4(uint32_t lane, FloatRegister src,
                                        Register dest) {
  MOZ_ASSERT(lane < 4);
  ExtractLaneToGPR(*this, lane, src, dest, 4, 32);
  // ExtractLaneToGPR leaves the adjacent lane in the high 32 bits for the
  // unshifted lanes (0 and 2); canonicalize to a sign-extended i32, as the
  // i8x16/i16x8 extracts do with extsb/extsh. A consumer that reads the full
  // 64-bit register -- e.g. the POWER8 i32.ctz emulation, whose 64-bit neg/and.
  // with a 32-bit cntlzw otherwise mis-handles a zero low word over nonzero
  // high garbage and returns -1 -- requires this.
  as_extsw(dest, dest);
}

void MacroAssembler::extractLaneInt8x16(uint32_t lane, FloatRegister src,
                                        Register dest) {
  MOZ_ASSERT(lane < 16);
  if (HasPOWER9()) {
    as_vextractub(ScratchSimd128Reg, src, 15 - lane);
    as_mfvsrd(dest, ScratchSimd128Reg);
    as_extsb(dest, dest);
    return;
  }
  ExtractLaneToGPR(*this, lane, src, dest, 1, 8);
  as_extsb(dest, dest);
}

void MacroAssembler::maxInt16x8(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmaxsh), lhs, rhs, dest);
}

void MacroAssembler::maxInt32x4(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmaxsw), lhs, rhs, dest);
}

void MacroAssembler::maxInt8x16(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmaxsb), lhs, rhs, dest);
}

void MacroAssembler::minInt8x16(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vminsb), lhs, rhs, dest);
}

void MacroAssembler::mulInt32x4(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vmuluwm), lhs, rhs, dest);
}

void MacroAssembler::narrowInt16x8(FloatRegister lhs, FloatRegister rhs,
                                   FloatRegister dest) {
  // On LE, VMX pack swaps operand order vs Wasm convention.
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vpkshss), rhs, lhs, dest);
}

void MacroAssembler::splatX2(Register64 src, FloatRegister dest) {
  if (HasPOWER9()) {
    as_mtvsrdd(dest, src.reg, src.reg);
  } else {
    as_mtvsrd(dest, src.reg);
    as_xxpermdi(dest, dest, dest, 0);
  }
}

void MacroAssembler::subInt32x4(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsubuwm), lhs, rhs, dest);
}

void MacroAssembler::swizzleInt8x16(FloatRegister lhs, FloatRegister rhs,
                                    FloatRegister dest) {
  // Wasm i8x16.swizzle: result[i] = (rhs[i] < 16) ? lhs[rhs[i]] : 0.
  //
  // Strategy: build ctrl in ScratchSimd128 (which can't alias inputs
  // because v0 is non-allocatable). Use vsububs(splat(15), rhs) to
  // produce ctrl = max(0, 15 - rhs); the saturation clamps out-of-range
  // indices to 0, and those positions get masked off below.
  //
  // The mask is computed via vcmpgtub(rhs, splat(15)) + xxlnor — 0xFF
  // where rhs <= 15. Reformulating "rhs < 16" as "!(rhs > 15)" lets us
  // use vspltisb with a 5-bit signed immediate (P7+, 1 insn, no GPR
  // scratch) for both splat-of-15 sites, replacing the previous
  // movePtr(0x0F0F0F0F)/movePtr(0x10101010) + splatX4 dance.
  //
  // Aliasing: dest may equal lhs (wasm baseline calls swizzleInt8x16(
  // rsd, rs, rsd); Ion's useRegisterAtStart permits the same). When
  // dest != rhs, ctrl can be built in scratch and the mask computed
  // after the permute (rhs is still alive). When dest == rhs, the
  // permute would clobber rhs before we could compute the mask, so the
  // mask goes to the red zone first.
  ScratchSimd128Scope scratch(*this);
  uint8_t s = scratch.encoding() & 31;
  uint8_t l = lhs.encoding() & 31;
  uint8_t r = rhs.encoding() & 31;
  uint8_t d = dest.encoding() & 31;

  if (dest != rhs) {
    as_vspltisb(s, 15);
    as_vsububs(s, s, r);   // scratch = ctrl
    as_vperm(d, l, l, s);  // dest = vperm(lhs, lhs, ctrl)
    as_vspltisb(s, 15);
    as_vcmpgtub(s, r, s);             // scratch = 0xFF where rhs > 15
    as_xxlandc(dest, dest, scratch);  // dest &= ~scratch (= bytes-to-keep)
    return;
  }

  // dest == rhs: vperm clobbers rhs, so build the bytes-to-zero mask first
  // and stash it. The xxlandc at the end consumes the un-inverted form.
  as_vspltisb(s, 15);
  as_vcmpgtub(s, r, s);  // scratch = 0xFF where rhs > 15
  RedZoneStashSimd128(*this, scratch, 0);
  as_vspltisb(s, 15);
  as_vsububs(s, s, r);   // scratch = ctrl
  as_vperm(d, l, l, s);  // dest = vperm(lhs, lhs, ctrl)
  RedZoneRestoreSimd128(*this, 0, scratch);
  as_xxlandc(dest, dest, scratch);  // dest &= ~scratch (= bytes-to-keep)
}
// SIMD 3-operand arithmetic (continued).

void MacroAssembler::addInt32x4(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vadduwm), lhs, rhs, dest);
}

void MacroAssembler::addInt64x2(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vaddudm), lhs, rhs, dest);
}

void MacroAssembler::addSatInt16x8(FloatRegister lhs, FloatRegister rhs,
                                   FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vaddshs), lhs, rhs, dest);
}

void MacroAssembler::addSatInt8x16(FloatRegister lhs, FloatRegister rhs,
                                   FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vaddsbs), lhs, rhs, dest);
}

void MacroAssembler::divFloat64x2(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest) {
  as_xvdivdp(dest, lhs, rhs);
}

void MacroAssembler::minInt16x8(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vminsh), lhs, rhs, dest);
}

void MacroAssembler::minInt32x4(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vminsw), lhs, rhs, dest);
}

void MacroAssembler::mulFloat32x4(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest) {
  as_xvmulsp(dest, lhs, rhs);
}

void MacroAssembler::mulFloat64x2(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest) {
  as_xvmuldp(dest, lhs, rhs);
}

void MacroAssembler::mulInt16x8(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  ZeroSimd128(*this, scratch);
  EmitVmxTernary(
      *this,
      [](Assembler& a, uint8_t vrt, uint8_t vra, uint8_t vrb, uint8_t vrc) {
        a.as_vmladduhm(vrt, vra, vrb, vrc);
      },
      lhs, rhs, scratch, dest);
}

void MacroAssembler::narrowInt32x4(FloatRegister lhs, FloatRegister rhs,
                                   FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vpkswss), rhs, lhs, dest);
}

void MacroAssembler::subFloat32x4(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest) {
  as_xvsubsp(dest, lhs, rhs);
}

void MacroAssembler::subFloat64x2(FloatRegister lhs, FloatRegister rhs,
                                  FloatRegister dest) {
  as_xvsubdp(dest, lhs, rhs);
}

void MacroAssembler::subInt16x8(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsubuhm), lhs, rhs, dest);
}

void MacroAssembler::subInt64x2(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsubudm), lhs, rhs, dest);
}

void MacroAssembler::subInt8x16(FloatRegister lhs, FloatRegister rhs,
                                FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsububm), lhs, rhs, dest);
}

void MacroAssembler::subSatInt16x8(FloatRegister lhs, FloatRegister rhs,
                                   FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsubshs), lhs, rhs, dest);
}

void MacroAssembler::subSatInt8x16(FloatRegister lhs, FloatRegister rhs,
                                   FloatRegister dest) {
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsubsbs), lhs, rhs, dest);
}

void MacroAssembler::widenDotInt16x8(FloatRegister lhs, FloatRegister rhs,
                                     FloatRegister dest) {
  // i32x4.dot_i16x8_s: result[k] = lhs[2k]*rhs[2k] + lhs[2k+1]*rhs[2k+1].
  // vmsumshm computes exactly that for each i32 lane plus an addend (VRC).
  // With VRC = 0, the addend disappears and we get the wasm spec result in
  // a single instruction. xxlxor zeros the scratch in 1 insn, so total is
  // 2 insns vs the old vmulesh/vmulosh/vadduwm trio.
  ScratchSimd128Scope scratch(*this);
  as_xxlxor(scratch, scratch, scratch);
  as_vmsumshm(dest.encoding() & 31, lhs.encoding() & 31, rhs.encoding() & 31,
              scratch.encoding() & 31);
}

// SIMD variable-shift and FMA helpers.
// Pattern: splat the GPR shift count across all lanes of a scratch VSR,
// then issue a vector-shift on lhs and the splat. vsl{b,h} / vsr{b,h} /
// vsra{b,h} use the low 3-or-4 bits of each lane's shift count, exactly
// matching wasm modulo-N shift semantics.

void MacroAssembler::leftShiftInt8x16(FloatRegister lhs, Register rhs,
                                      FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  splatX16(rhs, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vslb), lhs, scratch, dest);
}

void MacroAssembler::rightShiftInt8x16(FloatRegister lhs, Register rhs,
                                       FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  splatX16(rhs, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrab), lhs, scratch, dest);
}

void MacroAssembler::unsignedRightShiftInt8x16(FloatRegister lhs, Register rhs,
                                               FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  splatX16(rhs, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrb), lhs, scratch, dest);
}

void MacroAssembler::leftShiftInt16x8(FloatRegister lhs, Register rhs,
                                      FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  splatX8(rhs, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vslh), lhs, scratch, dest);
}

void MacroAssembler::rightShiftInt16x8(FloatRegister lhs, Register rhs,
                                       FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  splatX8(rhs, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrah), lhs, scratch, dest);
}

void MacroAssembler::unsignedRightShiftInt16x8(FloatRegister lhs, Register rhs,
                                               FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  splatX8(rhs, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrh), lhs, scratch, dest);
}

void MacroAssembler::leftShiftInt32x4(FloatRegister lhs, Register rhs,
                                      FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  splatX4(rhs, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vslw), lhs, scratch, dest);
}

void MacroAssembler::leftShiftInt64x2(FloatRegister lhs, Register rhs,
                                      FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  splatX4(rhs, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsld), lhs, scratch, dest);
}

void MacroAssembler::rightShiftInt32x4(FloatRegister lhs, Register rhs,
                                       FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  splatX4(rhs, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsraw), lhs, scratch, dest);
}

void MacroAssembler::rightShiftInt64x2(FloatRegister lhs, Register rhs,
                                       FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  splatX4(rhs, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrad), lhs, scratch, dest);
}

void MacroAssembler::unsignedRightShiftInt32x4(FloatRegister lhs, Register rhs,
                                               FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  splatX4(rhs, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrw), lhs, scratch, dest);
}

void MacroAssembler::unsignedRightShiftInt64x2(FloatRegister lhs, Register rhs,
                                               FloatRegister dest) {
  ScratchSimd128Scope scratch(*this);
  splatX4(rhs, scratch);
  EmitVmxBinary(*this, VMX_BINARY_WRAPPER(vsrd), lhs, scratch, dest);
}

void MacroAssembler::fmaFloat32x4(FloatRegister src1, FloatRegister src2,
                                  FloatRegister srcDest) {
  as_xvmaddasp(srcDest, src1, src2);
}

void MacroAssembler::fmaFloat64x2(FloatRegister src1, FloatRegister src2,
                                  FloatRegister srcDest) {
  as_xvmaddadp(srcDest, src1, src2);
}

//}}} check_macroassembler_style

}  // namespace jit
}  // namespace js

#endif /* jit_ppc64_MacroAssembler_ppc64_inl_h */
