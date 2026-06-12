/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef jit_ppc64_MacroAssembler_ppc64_h
#define jit_ppc64_MacroAssembler_ppc64_h

#include "jit/MoveResolver.h"
#include "jit/ppc64/Assembler-ppc64.h"
#include "wasm/WasmBuiltins.h"

namespace js {
namespace jit {

inline bool is_intN(int64_t x, unsigned n) {
  MOZ_ASSERT((0 < n) && (n < 64));
  int64_t limit = static_cast<int64_t>(1) << (n - 1);
  return (-limit <= x) && (x < limit);
}

inline bool is_uintN(uint64_t x, unsigned n) {
  MOZ_ASSERT((0 < n) && (n < 64));
  return !(x >> n);
}

// enterNoPool() guard sizes. Inhibiting the constant pool keeps these
// stanzas at a fixed instruction count, which patchers and long-branch
// resolvers rely on. Each constant names a distinct stanza shape; see
// the emitting call site for the exact layout.
//
// kNoPoolLoad64StanzaInsns (8): emitLoad64Stanza body — 8 NOPs that
//   WriteLoad64Instructions later overwrites in place. Two shapes share
//   the same 8-slot footprint with the .quad fixed at slots [6..7]:
//     - POWER9+ (HasPOWER9()): addpcis + ld + b + 3 NOPs (2 dynamic insns,
//       no LR clobber). Preferred path.
//     - POWER8 fallback: mflr/bcl/mflr/mtlr/ld/b LR-bouncing sequence
//       (6 dynamic insns, RAS-thrashing — kept only because P8 has no
//       addpcis).
//
// kNoPoolPatchableBranchInsns (10): patchable far call / jump /
//   unconditional branch. Three alternative shapes, all fitting the
//   same budget:
//     - load64 stanza (8) + mtctr + bctr[l]  = 10  (bound call/jump)
//     - 9 NOPs + bl                          = 10  (short bound call)
//     - xs_trap_tagged(TAG) + chain + 8 NOPs = 10  (fwd-ref stanza)
//
// kNoPoolCondLongBranchInsnsP8Max (14): conditional long branch, POWER8
//   Overflow worst case. POWER8 has no mcrxrx so overflow/carry test is
//   mfxer+rlwinm+mtcrf (3 insns) on top of the base shape. Budget =
//   3 (XER inspection) + 1 (bc) + 8 (load64 stanza) + 2 (mtctr+bctr) = 14.
static constexpr size_t kNoPoolLoad64StanzaInsns = 8;
static constexpr size_t kNoPoolPatchableBranchInsns = 10;
static constexpr size_t kNoPoolCondLongBranchInsnsP8Max = 14;

enum LoadStoreSize {
  SizeByte = 8,
  SizeHalfWord = 16,
  SizeWord = 32,
  SizeDouble = 64
};

enum LoadStoreExtension { ZeroExtend = 0, SignExtend = 1 };

static Register CallReg = r12;

struct ImmShiftedTag : public ImmWord {
  explicit ImmShiftedTag(JSValueShiftedTag shtag) : ImmWord((uintptr_t)shtag) {}
  explicit ImmShiftedTag(JSValueType type)
      : ImmWord(((uintptr_t)JSVAL_TYPE_TO_SHIFTED_TAG(type))) {}
};

struct ImmTag : public Imm32 {
  explicit ImmTag(JSValueTag tag) : Imm32(tag) {}
};

class ScratchTagScope {
  UseScratchRegisterScope temps_;
  Register scratch_;
  bool owned_;
  mozilla::DebugOnly<bool> released_;

 public:
  ScratchTagScope(Assembler& masm, const ValueOperand&)
      : temps_(masm), owned_(true), released_(false) {
    scratch_ = temps_.Acquire();
  }

  operator Register() {
    MOZ_ASSERT(!released_);
    return scratch_;
  }

  void release() {
    MOZ_ASSERT(!released_);
    released_ = true;
    if (owned_) {
      temps_.Release(scratch_);
      owned_ = false;
    }
  }

  void reacquire() {
    MOZ_ASSERT(released_);
    released_ = false;
    if (!owned_) {
      scratch_ = temps_.Acquire();
      owned_ = true;
    }
  }
};

class ScratchTagScopeRelease {
  ScratchTagScope* ts_;

 public:
  explicit ScratchTagScopeRelease(ScratchTagScope* ts) : ts_(ts) {
    ts_->release();
  }
  ~ScratchTagScopeRelease() { ts_->reacquire(); }
};

class MacroAssemblerPPC64 : public Assembler {
 protected:
  MacroAssembler& asMasm();
  const MacroAssembler& asMasm() const;
};

class MacroAssemblerPPC64Compat : public MacroAssemblerPPC64 {
 public:
  using MacroAssemblerPPC64::MacroAssemblerPPC64;

  MacroAssemblerPPC64Compat() {}

  bool buildOOLFakeExitFrame(void* fakeReturnAddr);

  // ===============================================================
  // Conversion functions

  void convertBoolToInt32(Register src, Register dest) {
    as_rlwinm(dest, src, 0, 31, 31);
  }
  void convertInt32ToDouble(Register src, FloatRegister dest) {
    // mtvsrwa: VSR[dest].dw0 = sign_ext_64(src[32:63]); P8+ (ISA 2.07).
    // Replaces extsw + mtvsrd (2 insns + scratch GPR) with 1 insn.
    as_mtvsrwa(dest, src);
    as_fcfid(dest, dest);
  }
  void convertInt32ToDouble(const Address& src, FloatRegister dest) {
    // lfiwax (P7+): FPR.dw[0] = sign_ext_64(MEM[addr, 4]). X-form indexed
    // — no immediate offset, so when offset != 0 we add it into a scratch
    // first. Replaces lwz + extsw + mtvsrd with lfiwax (one insn) plus
    // optional address add.
    if (src.offset == 0) {
      as_lfiwax(dest, r0, src.base);
    } else {
      UseScratchRegisterScope temps(*this);
      Register scratch = temps.Acquire();
      if (is_intN(src.offset, 16)) {
        as_addi(scratch, src.base, src.offset);
        as_lfiwax(dest, r0, scratch);
      } else {
        // X-form indexed: lfiwax computes base + scratch directly, no add.
        movePtr(ImmWord(src.offset), scratch);
        as_lfiwax(dest, src.base, scratch);
      }
    }
    as_fcfid(dest, dest);
  }
  void convertInt32ToDouble(const BaseIndex& src, FloatRegister dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(src, scratch);
    convertInt32ToDouble(Address(scratch, src.offset), dest);
  }
  void convertUInt32ToDouble(Register src, FloatRegister dest);
  void convertUInt32ToFloat32(Register src, FloatRegister dest);
  void convertDoubleToFloat32(FloatRegister src, FloatRegister dest) {
    as_frsp(dest, src);
  }
  // POWER9 FP16 conversions (1 insn each). Caller must have verified
  // HasPOWER9() — SupportsFloat{64,32}To16 gates that. PPC64 FPRs hold
  // doubles internally; an "FP32-in-FPR" is just the FP32 value stored
  // as exact FP64, so xscvdphp/xscvhpdp work for both FP32↔FP16 and
  // FP64↔FP16 (FP16 fits exactly in FP32 which fits exactly in FP64).
  void convertDoubleToFloat16(FloatRegister src, FloatRegister dest) {
    MOZ_ASSERT(HasPOWER9());
    as_xscvdphp(dest, src);
  }
  void convertFloat16ToDouble(FloatRegister src, FloatRegister dest) {
    MOZ_ASSERT(HasPOWER9());
    as_xscvhpdp(dest, src);
  }
  void convertFloat32ToFloat16(FloatRegister src, FloatRegister dest) {
    MOZ_ASSERT(HasPOWER9());
    as_xscvdphp(dest, src);
  }
  void convertFloat16ToFloat32(FloatRegister src, FloatRegister dest) {
    MOZ_ASSERT(HasPOWER9());
    as_xscvhpdp(dest, src);
  }
  void convertInt32ToFloat16(Register src, FloatRegister dest) {
    MOZ_ASSERT(HasPOWER9());
    convertInt32ToFloat32(src, dest);
    convertFloat32ToFloat16(dest, dest);
  }
  void convertDoubleToInt32(FloatRegister src, Register dest, Label* fail,
                            bool negativeZeroCheck = true);
  void convertDoubleToPtr(FloatRegister src, Register dest, Label* fail,
                          bool negativeZeroCheck = true);
  void convertFloat32ToInt32(FloatRegister src, Register dest, Label* fail,
                             bool negativeZeroCheck = true);
  void convertFloat32ToDouble(FloatRegister src, FloatRegister dest) {
    // PPC64 FPRs hold every FP32 value in its FP64-equivalent representation,
    // so f64.promote_f32 is conceptually a no-op except that wasm requires
    // sNaN inputs to be quieted. frsp (Round to Single-Precision) is the
    // identity for SP-representable inputs but applies IEEE NaN-quieting as
    // a side effect, replacing the prior fmr + fcmpu + branch + canonical-
    // NaN-load (5+ insns + scratch GPR) with a single instruction. Result
    // matches what x86 vcvtss2sd / ARM fcvt produce.
    as_frsp(dest, src);
  }
  void convertInt32ToFloat32(Register src, FloatRegister dest) {
    // mtvsrwa + fcfids; same recipe as convertInt32ToDouble(Register).
    as_mtvsrwa(dest, src);
    as_fcfids(dest, dest);
  }
  void convertInt32ToFloat32(const Address& src, FloatRegister dest) {
    // lfiwax + fcfids; same recipe as convertInt32ToDouble(Address).
    if (src.offset == 0) {
      as_lfiwax(dest, r0, src.base);
    } else {
      UseScratchRegisterScope temps(*this);
      Register scratch = temps.Acquire();
      if (is_intN(src.offset, 16)) {
        as_addi(scratch, src.base, src.offset);
        as_lfiwax(dest, r0, scratch);
      } else {
        movePtr(ImmWord(src.offset), scratch);
        as_lfiwax(dest, src.base, scratch);
      }
    }
    as_fcfids(dest, dest);
  }

  // POWER9 FP16 load: lxsihzx writes the 2 memory bytes directly into
  // dw[0] low 16 bits with the rest zeroed — matching the layout that
  // xscvhpdp expects, in a single instruction.
  FaultingCodeOffset loadFloat16(const Address& addr, FloatRegister dest,
                                 Register temp) {
    MOZ_ASSERT(HasPOWER9());
    if (addr.offset == 0) {
      return FaultingCodeOffset(as_lxsihzx(dest, r0, addr.base).getOffset());
    }
    if (is_intN(addr.offset, 16)) {
      as_addi(temp, addr.base, addr.offset);
      return FaultingCodeOffset(as_lxsihzx(dest, r0, temp).getOffset());
    }
    movePtr(ImmWord(addr.offset), temp);
    return FaultingCodeOffset(as_lxsihzx(dest, addr.base, temp).getOffset());
  }
  FaultingCodeOffset loadFloat16(const BaseIndex& src, FloatRegister dest,
                                 Register temp) {
    MOZ_ASSERT(HasPOWER9());
    computeEffectiveAddress(src, temp);
    return FaultingCodeOffset(as_lxsihzx(dest, r0, temp).getOffset());
  }

  // ===============================================================
  // Effective address computation

  void computeScaledAddress(const BaseIndex& address, Register dest) {
    if (address.scale == TimesOne) {
      as_add(dest, address.base, address.index);
    } else if (dest != address.base && dest != address.index) {
      x_sldi(dest, address.index, address.scale);
      as_add(dest, address.base, dest);
    } else {
      UseScratchRegisterScope temps(*this);
      Register scratch = temps.Acquire();
      x_sldi(scratch, address.index, address.scale);
      as_add(dest, address.base, scratch);
    }
  }

  void computeEffectiveAddress(const Address& address, Register dest) {
    if (address.offset == 0) {
      if (dest != address.base) {
        xs_mr(dest, address.base);
      }
    } else if (is_intN(address.offset, 16)) {
      as_addi(dest, address.base, address.offset);
    } else if (HasPOWER10() && is_intN(address.offset, 34)) {
      // Single-insn 34-bit-signed reg+imm add. Avoids the scratch GPR.
      as_paddi(dest, address.base, address.offset, /*R=*/false);
    } else {
      UseScratchRegisterScope temps(*this);
      Register scratch = temps.Acquire();
      MOZ_ASSERT(scratch != dest);
      movePtr(ImmWord(address.offset), scratch);
      as_add(dest, address.base, scratch);
    }
  }
  void computeEffectiveAddress(const BaseIndex& address, Register dest) {
    computeScaledAddress(address, dest);
    if (address.offset) {
      if (is_intN(address.offset, 16)) {
        as_addi(dest, dest, address.offset);
      } else if (HasPOWER10() && is_intN(address.offset, 34)) {
        as_paddi(dest, dest, address.offset, /*R=*/false);
      } else {
        UseScratchRegisterScope temps(*this);
        Register scratch = temps.Acquire();
        MOZ_ASSERT(scratch != dest);
        movePtr(ImmWord(address.offset), scratch);
        as_add(dest, dest, scratch);
      }
    }
  }

  // ===============================================================
  // Move instructions

  void mov(Register src, Register dest) { xs_mr(dest, src); }
  void mov(ImmWord imm, Register dest) { movePtr(imm, dest); }
  void mov(ImmPtr imm, Register dest) {
    mov(ImmWord(uintptr_t(imm.value)), dest);
  }
  // Emit an 8-instruction NOP stanza for a patchable 64-bit load.
  // Pool flushes are inhibited during emission to prevent pool data
  // from being inserted mid-stanza.
  BufferOffset emitLoad64Stanza(Register dest, uint64_t value) {
    m_buffer.enterNoPool(kNoPoolLoad64StanzaInsns);
    BufferOffset bo = writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    m_buffer.leaveNoPool();
    // If any of the 8 writeInst calls hit OOM, only some of the stanza
    // was reserved in the buffer. WriteLoad64Instructions writes 32 bytes
    // unconditionally, so calling it here would overflow the Vector's
    // backing store and corrupt the next heap chunk's metadata, surfacing
    // later as a malloc-detected free-time crash.
    if (m_buffer.oom()) {
      return bo;
    }
    WriteLoad64Instructions((Instruction*)editSrc(bo), dest, value);
    return bo;
  }

  void mov(CodeLabel* label, Register dest) {
    BufferOffset bo = emitLoad64Stanza(dest, LabelBase::INVALID_OFFSET);
    label->patchAt()->bind(bo.getOffset());
    label->setLinkMode(CodeLabel::MoveImmediate);
  }
  void mov(Register src, Address dest) { storePtr(src, dest); }
  void mov(Address src, Register dest) { loadPtr(src, dest); }

  void move32(Imm32 imm, Register dest) {
    if (is_intN(imm.value, 16)) {
      xs_li(dest, (int16_t)imm.value);
    } else if (is_uintN((uint32_t)imm.value, 16)) {
      xs_li(dest, 0);
      as_ori(dest, dest, (uint16_t)imm.value);
    } else {
      xs_lis(dest, (int16_t)((uint32_t)imm.value >> 16));
      if (imm.value & 0xffff) {
        as_ori(dest, dest, (uint16_t)imm.value);
      }
    }
  }
  void move32(Register src, Register dest) { as_extsw(dest, src); }

  void movePtr(Register src, Register dest) {
    if (src != dest) {
      xs_mr(dest, src);
    }
  }
  void movePtr(ImmWord imm, Register dest) {
    if (imm.value == 0) {
      xs_li(dest, 0);
    } else if (is_intN((intptr_t)imm.value, 16)) {
      xs_li(dest, (int16_t)imm.value);
    } else if (is_uintN(imm.value, 16)) {
      xs_li(dest, 0);
      as_ori(dest, dest, (uint16_t)imm.value);
    } else if (is_intN((intptr_t)imm.value, 32)) {
      // 32-bit signed: lis + ori (2 instructions).
      xs_lis(dest, (int16_t)((uint32_t)imm.value >> 16));
      if (imm.value & 0xFFFF) {
        as_ori(dest, dest, (uint16_t)imm.value);
      }
    } else if (HasPOWER10() && is_intN((intptr_t)imm.value, 34)) {
      // POWER10 single-instruction 34-bit signed immediate. Replaces the
      // 5-insn fallback for values in (33-34)-bit signed range.
      // 8 bytes vs 20 bytes; one slot temp register is no longer needed.
      as_paddi(dest, r0, (int64_t)imm.value, /*R=*/false);
    } else {
      // Full 64-bit: GCC-style lis+ori+lis+ori+rldimi (5 instructions).
      // No LR clobber, no embedded data — pure instruction sequence.
      uint32_t lo32 = (uint32_t)(imm.value);
      uint32_t hi32 = (uint32_t)(imm.value >> 32);
      Register temp = (dest != SecondScratchReg) ? SecondScratchReg
                                                 : SavedScratchRegister;
      m_buffer.ensureSpace(5 * sizeof(uint32_t));
      xs_lis(dest, (int16_t)(lo32 >> 16));
      as_ori(dest, dest, lo32 & 0xFFFF);
      xs_lis(temp, (int16_t)(hi32 >> 16));
      as_ori(temp, temp, hi32 & 0xFFFF);
      as_rldimi(dest, temp, 32, 0);
    }
  }
  void movePtr(ImmPtr imm, Register dest) {
    movePtr(ImmWord(uintptr_t(imm.value)), dest);
  }

  // Load a 64-bit FPR constant from the inline constant pool.
  // POWER9: 2 instructions (addpcis + lfd) -- no alignment constraint.
  // POWER10: 1 prefixed instruction (plfd, 2 slots), or 3 slots in the
  //   (loadAddr & 63) == 60 alignment-leading-nop case. Reserve 3 to
  //   cover both cases conservatively.
  // POWER8: not used -- loadConstantDouble inlines the constant.
  BufferOffset loadFromPoolFloat64(FloatRegister dest, double value) {
    size_t slots = HasPOWER10() ? 3 : 2;
    uint32_t hint = (uint32_t(dest.encoding()) << 16) |
                    (uint32_t(PoolLoadFPR64) << 21) | 0xF0000000;
    uint32_t inst[3] = {hint, NopInst, NopInst};
    return m_buffer.allocEntry(slots, 2, (uint8_t*)inst, (uint8_t*)&value);
  }
  // Load a 32-bit FPR constant from the inline constant pool.
  // Same shape as loadFromPoolFloat64 (above). lfs/plfs auto-expand the
  // 32-bit single-precision value to double in the FPR, so no follow-up
  // xscvspdpn is needed.
  BufferOffset loadFromPoolFloat32(FloatRegister dest, float value) {
    size_t slots = HasPOWER10() ? 3 : 2;
    uint32_t hint = (uint32_t(dest.encoding()) << 16) |
                    (uint32_t(PoolLoadFPR32) << 21) | 0xF0000000;
    uint32_t inst[3] = {hint, NopInst, NopInst};
    return m_buffer.allocEntry(slots, 1, (uint8_t*)inst, (uint8_t*)&value);
  }
  // Load a 128-bit SIMD constant from the inline constant pool.
  // Per-arch slot reservation -- the patcher writes only the slots
  // each micro-arch actually needs:
  //   P8: 5 (bcl + mflr + addi + lxvd2x + xxpermdi)
  //   P9: 3 (addpcis + addi + lxvx) -- no LR touch, no RAS hazard
  //   P10: 3 (alignment-safe: prefix + suffix + 1 reserve for the
  //          (loadAddr & 63) == 60 leading-nop case)
  // Pool entry is 4 × 4-byte words = 16 bytes. P9 uses
  // SavedScratchRegister (r16) as the PC base; P10 emits a single
  // PC-relative plxv with no scratch and no LR touch. Only P8 still
  // clobbers LR (correctness-only fallback; live by design).
  BufferOffset loadFromPoolSimd128(FloatRegister dest,
                                   const SimdConstant& v) {
    size_t slots;
    if (HasPOWER10()) {
      slots = 3;
    } else if (HasPOWER9()) {
      slots = 3;
    } else {
      slots = 5;
    }
    // Simd128 encoding is 32-63; mask to 5 bits for hint.
    // PatchConstantPoolLoad sets TX bit unconditionally for Simd128.
    uint32_t hint = ((uint32_t(dest.encoding()) & 0x1F) << 16) |
                    (uint32_t(PoolLoadSimd128) << 21) | 0xF0000000;
    uint32_t inst[5] = {hint, NopInst, NopInst, NopInst, NopInst};
    return m_buffer.allocEntry(slots, 4, (uint8_t*)inst, (uint8_t*)v.bytes());
  }
  void movePtr(wasm::SymbolicAddress imm, Register dest) {
    BufferOffset bo = emitLoad64Stanza(dest, (uint64_t)-1);
    append(wasm::SymbolicAccess(CodeOffset(bo.getOffset()), imm));
  }
  void movePtr(ImmGCPtr imm, Register dest) {
    BufferOffset bo = emitLoad64Stanza(dest,
                                       (uint64_t)uintptr_t(imm.value));
    Assembler::writeDataRelocation(bo, imm);
  }

  void moveFloat32(FloatRegister src, FloatRegister dest) {
    if (src != dest) {
      as_fmr(dest, src);
    }
  }
  void moveDouble(FloatRegister src, FloatRegister dest) {
    if (src != dest) {
      as_fmr(dest, src);
    }
  }

  // ===============================================================
  // Branch functions

  void branch(JitCode* c) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    BufferOffset bo = emitLoad64Stanza(scratch, (uint64_t)uintptr_t(c->raw()));
    addPendingJump(bo, ImmPtr(c->raw()), RelocationKind::JITCODE);
    xs_mtctr(scratch);
    as_bctr();
  }
  void branch(const Register reg) {
    xs_mtctr(reg);
    as_bctr();
  }

  void jump(Label* label) {
    if (label->bound()) {
      // Open the no-pool window BEFORE computing the displacement. The
      // enterNoPool() call itself can trigger a pool flush, which advances
      // currentOffset(). Computing the displacement against the pre-flush
      // offset and then emitting the b at the post-flush offset would land
      // the branch (poolSize) bytes past the intended target.
      m_buffer.enterNoPool(2);
      int32_t offset = label->offset() - currentOffset();
      if (JOffImm26::IsInRange(offset)) {
        as_b(offset);
        writeInst(NopInst);
        m_buffer.leaveNoPool();
        return;
      }
      m_buffer.leaveNoPool();
      // Long jump to bound label.
      BufferOffset bo =
          emitLoad64Stanza(SecondScratchReg, LabelBase::INVALID_OFFSET);
      xs_mtctr(SecondScratchReg);
      as_bctr();
      addLongJump(bo, BufferOffset(label->offset()));
      return;
    }
    // Unbound label: emit trap-tagged stanza (10 slots).
    m_buffer.enterNoPool(kNoPoolPatchableBranchInsns);
    BufferOffset bo = xs_trap_tagged(BTag);
    writeInst(label->used() ? label->offset() : LabelBase::INVALID_OFFSET);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    m_buffer.leaveNoPool();
    if (!oom()) {
      label->use(bo.getOffset());
    }
  }
  void jump(Register reg) {
    xs_mtctr(reg);
    as_bctr();
  }
  void jump(const Address& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    loadPtr(address, scratch);
    xs_mtctr(scratch);
    as_bctr();
  }
  void jump(JitCode* code) { branch(code); }
  void jump(ImmPtr ptr) {
    BufferOffset bo =
        emitLoad64Stanza(SecondScratchReg, (uint64_t)uintptr_t(ptr.value));
    addPendingJump(bo, ptr, RelocationKind::HARDCODED);
    xs_mtctr(SecondScratchReg);
    as_bctr();
  }
  void jump(TrampolinePtr code) { jump(ImmPtr(code.value)); }

  // Conditional branch to label. Assumes a compare instruction has already
  // been emitted that sets CR0.
  template <typename CondT>
  void ma_b(CondT cond, Label* label) {
    if constexpr (std::is_same_v<CondT, Condition>) {
      if (cond == Always) {
        jump(label);
        return;
      }
    }
    if (label->bound()) {
      // Open the no-pool window BEFORE computing the displacement. Same
      // hazard as jump(): enterNoPool may itself flush a pending pool,
      // advancing currentOffset(); the bc must emit with a displacement
      // computed against the post-flush offset. Budget covers max 6
      // instructions: POWER8 Overflow XER ops (3) + cror (1) + bc (1) +
      // nop (1) for the worst-case DoubleCondition+Overflow short path.
      m_buffer.enterNoPool(6);
      // For DoubleCondition, as_bc emits cror/crandc before the bc
      // instruction, advancing currentOffset() by 4. Account for this
      // in the offset calculation.
      int32_t crAdjust = 0;
      if constexpr (std::is_same_v<CondT, DoubleCondition>) {
        crAdjust = -(int32_t)sizeof(uint32_t);
      }
      int32_t offset = label->offset() - currentOffset() + crAdjust;
      if (BOffImm16::IsInRange(offset)) {
        as_bc((int16_t)offset, cond);
        writeInst(NopInst);
        m_buffer.leaveNoPool();
        return;
      }
      m_buffer.leaveNoPool();
      // Long conditional branch for bound label.
      // XER ops(0-3) + cror(0-1) + bc(1) + stanza(8) + mtctr(1) + bctr(1).
      // P8 Overflow: mfxer+rlwinm+mtcrf+bc+stanza+mtctr+bctr = 14 max.
      m_buffer.enterNoPool(kNoPoolCondLongBranchInsnsP8Max);
      as_bc((int16_t)44, InvertCondition(cond));
      BufferOffset boLoad =
          emitLoad64Stanza(SecondScratchReg, LabelBase::INVALID_OFFSET);
      xs_mtctr(SecondScratchReg);
      as_bctr();
      m_buffer.leaveNoPool();
      addLongJump(boLoad, BufferOffset(label->offset()));
      return;
    }
    // Forward reference: emit BCTag stanza.
    // XER ops(0-3) + cror(0-1) + bc(1) + trap_tagged(1) + chain(1) + 8 NOPs.
    // P8 Overflow: mfxer+rlwinm+mtcrf+bc+trap+chain+8NOPs = 14 max.
    m_buffer.enterNoPool(kNoPoolCondLongBranchInsnsP8Max);
    as_bc((int16_t)44, InvertCondition(cond));
    BufferOffset bo = xs_trap_tagged(BCTag);
    writeInst(label->used() ? label->offset() : LabelBase::INVALID_OFFSET);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    writeInst(NopInst);
    m_buffer.leaveNoPool();
    if (!oom()) {
      label->use(bo.getOffset());
    }
  }

  // Set dest = 1 if CR0 satisfies cond, else dest = 0.
  // POWER10: setbc/setbcr (1 insn). P8/P9: isel-based path with the
  // r0-as-zero trick on the BranchOnClear half.
  void ma_cmp_set(Register dest, Condition cond) {
    uint32_t base = uint32_t(cond) & 0xff;
    uint32_t setbase = (base & ~BranchOptionMask) | BranchOnSet;
    if (HasPOWER10()) {
      if ((base & BranchOptionMask) == BranchOnSet) {
        as_setbc(dest, setbase, cr0);
      } else {
        as_setbcr(dest, setbase, cr0);
      }
      return;
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    xs_li(scratch, 1);
    if ((base & BranchOptionMask) == BranchOnSet) {
      xs_li(dest, 0);
      as_isel(dest, scratch, dest, setbase, cr0);
    } else {
      as_isel0(dest, r0, scratch, setbase, cr0);
    }
  }

  void ma_cmp_set_dbl(Register dest, DoubleCondition cond) {
    uint32_t base = uint32_t(cond) & 0xff;
    bool hasUnorderedFlag = uint32_t(cond) & DoubleConditionUnordered;
    uint32_t setbase = (base & ~BranchOptionMask) | BranchOnSet;
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    if (HasPOWER10()) {
      if ((base & BranchOptionMask) == BranchOnSet) {
        as_setbc(dest, setbase, cr0);
      } else {
        as_setbcr(dest, setbase, cr0);
      }
      // Fixup paths below still need scratch=1 for the SO-isel.
      if (hasUnorderedFlag || ((base & BranchOptionMask) != BranchOnSet &&
                               cond != DoubleOrdered)) {
        xs_li(scratch, 1);
      }
    } else {
      xs_li(scratch, 1);
      if ((base & BranchOptionMask) == BranchOnSet) {
        xs_li(dest, 0);
        as_isel(dest, scratch, dest, setbase, cr0);
      } else {
        as_isel0(dest, r0, scratch, setbase, cr0);
      }
    }
    if (hasUnorderedFlag) {
      // Condition includes unordered (NaN): force dest=1 when SO is set.
      // isel dest, scratch(=1), dest, SO
      as_isel(dest, scratch, dest, uint16_t(SOBit), cr0);
    } else if ((base & BranchOptionMask) != BranchOnSet &&
               cond != DoubleOrdered) {
      // Ordered comparison that negates a CR bit (BranchOnClear): NaN
      // produces all-zero LT/GT/EQ bits which makes the negation return
      // true.  Fix by forcing dest=0 when SO is set.
      as_isel0(dest, r0, dest, uint16_t(SOBit), cr0);
    }
  }

  // Conditional move: if CR0 satisfies cond, dest = src.
  void ma_cmp_move(Register dest, Register src, Condition cond) {
    uint32_t base = uint32_t(cond) & 0xff;
    uint32_t setbase = (base & ~BranchOptionMask) | BranchOnSet;
    if ((base & BranchOptionMask) == BranchOnSet) {
      as_isel(dest, src, dest, setbase, cr0);
    } else {
      as_isel(dest, dest, src, setbase, cr0);
    }
  }

  // If cond == 0, move src to dst; otherwise dst is unchanged. The only
  // callers are wasm select, whose condition is a 32-bit value: test its
  // 32-bit sign with cmpwi so high-bit garbage (e.g. under register pressure)
  // does not make a zero condition read as non-zero.
  void moveIfZero(Register dst, Register src, Register cond) {
    as_cmpwi(cond, 0);
    as_isel(dst, src, dst, Equal, cr0);
  }

  void ma_add32TestCarry(Condition cond, Register rd, Register rs, Imm32 imm,
                         Label* overflow);
  void ma_addPtrTestCarry(Condition cond, Register rd, Register rs, ImmWord imm,
                          Label* overflow);

  // Issue the correct compare instruction for the given condition and
  // operand sizes. Returns the condition to use with ma_b or ma_cmp_set
  // (usually the same, but unsigned conditions use cmpl* variants).
  Condition ma_cmp(Register lhs, Register rhs, Condition cond,
                   bool is32bit = false) {
    Condition base =
        static_cast<Condition>(cond & ~(ConditionUnsigned | ConditionZero));
    bool isUnsigned = (cond & ConditionUnsigned) != 0;
    // ConditionZero-flagged conditions (Signed, NotSigned, Zero, NonZero)
    // test a single register against zero, not two registers against each
    // other. Compare against immediate 0.
    if ((cond & ConditionZero) != 0) {
      if (is32bit) {
        as_cmpwi(lhs, 0);
      } else {
        as_cmpdi(lhs, 0);
      }
      return base;
    }
    if (is32bit) {
      if (isUnsigned) {
        as_cmplw(lhs, rhs);
      } else {
        as_cmpw(lhs, rhs);
      }
    } else {
      if (isUnsigned) {
        as_cmpld(lhs, rhs);
      } else {
        as_cmpd(lhs, rhs);
      }
    }
    return base;
  }

  Condition ma_cmp(Register lhs, Imm32 rhs, Condition cond,
                   bool is32bit = false) {
    Condition base =
        static_cast<Condition>(cond & ~(ConditionUnsigned | ConditionZero));
    bool isUnsigned = (cond & ConditionUnsigned) != 0;
    if (isUnsigned) {
      if (is_uintN(rhs.value, 16)) {
        if (is32bit) {
          as_cmplwi(lhs, rhs.value);
        } else {
          as_cmpldi(lhs, rhs.value);
        }
        return base;
      }
    } else {
      if (is_intN(rhs.value, 16)) {
        if (is32bit) {
          as_cmpwi(lhs, rhs.value);
        } else {
          as_cmpdi(lhs, rhs.value);
        }
        return base;
      }
    }
    // Immediate doesn't fit — materialize into scratch and compare.
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    move32(rhs, scratch);
    return ma_cmp(lhs, scratch, cond, is32bit);
  }

  Condition ma_cmp(Register lhs, ImmWord rhs, Condition cond) {
    Condition base =
        static_cast<Condition>(cond & ~(ConditionUnsigned | ConditionZero));
    bool isUnsigned = (cond & ConditionUnsigned) != 0;
    if (isUnsigned) {
      if (is_uintN(rhs.value, 16)) {
        as_cmpldi(lhs, rhs.value);
        return base;
      }
    } else {
      if (is_intN(rhs.value, 16)) {
        as_cmpdi(lhs, rhs.value);
        return base;
      }
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(rhs, scratch);
    return ma_cmp(lhs, scratch, cond);
  }

  Condition ma_cmp(Register lhs, ImmPtr rhs, Condition cond) {
    return ma_cmp(lhs, ImmWord(uintptr_t(rhs.value)), cond);
  }

  Condition ma_cmp(Register lhs, ImmGCPtr rhs, Condition cond) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(rhs, scratch);
    return ma_cmp(lhs, scratch, cond);
  }

  Condition ma_cmp(Register lhs, ImmTag rhs, Condition cond) {
    // Tag values on PUNBOX64 are 17-bit (0x1FFF0+), too large for 16-bit
    // signed or unsigned immediates.
    Condition base =
        static_cast<Condition>(cond & ~(ConditionUnsigned | ConditionZero));
    bool isUnsigned = (cond & ConditionUnsigned) != 0;
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    move32(Imm32(rhs.value), scratch);
    if (isUnsigned) {
      as_cmpld(lhs, scratch);
    } else {
      as_cmpd(lhs, scratch);
    }
    return base;
  }

  // Compare a tag register against an ImmTag constant and branch, WITHOUT
  // acquiring a scratch register.  Uses xoris+cmplwi which MODIFIES tagReg.
  // Only safe when tagReg is a scratch register owned by the caller.
  void branchTestTag(Condition cond, Register tagReg, ImmTag tag, Label* label) {
    MOZ_ASSERT(cond == Equal || cond == NotEqual);
    uint32_t t = tag.value;
    as_xoris(tagReg, tagReg, t >> 16);
    as_cmplwi(tagReg, t & 0xFFFF);
    Condition c = (cond == Equal) ? Equal : NotEqual;
    ma_b(c, label);
  }

  void ma_mod_mask(Register src, Register dest, Register hold, Register remain,
                   int32_t shift, Label* negZero = nullptr);

  void nop() { writeInst(NopInst); }
  void breakpoint(uint32_t value = 0) { xs_trap(); }

  inline void retn(Imm32 n);

  // ===============================================================
  // Stack operations

  void push(Imm32 imm) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    move32(imm, scratch);
    push(scratch);
  }
  void push(ImmWord imm) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(imm, scratch);
    push(scratch);
  }
  void push(ImmGCPtr imm) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(imm, scratch);
    push(scratch);
  }
  void push(const Address& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    loadPtr(address, scratch);
    push(scratch);
  }
  void push(Register reg) { as_stdu(reg, StackPointer, -8); }
  void push(FloatRegister reg) {
    // stfdu/stfsu fuses the SP decrement and the FP store: EA=SP-8,
    // MEM[EA]=reg, SP=EA. 1 insn instead of addi+stfd/stfs.
    if (reg.isSingle()) {
      as_stfsu(reg, StackPointer, -8);
    } else {
      as_stfdu(reg, StackPointer, -8);
    }
  }
  void pop(Register reg) {
    as_ld(reg, StackPointer, 0);
    as_addi(StackPointer, StackPointer, 8);
  }
  void pop(FloatRegister reg) {
    if (reg.isSingle()) {
      as_lfs(reg, StackPointer, 0);
    } else {
      as_lfd(reg, StackPointer, 0);
    }
    as_addi(StackPointer, StackPointer, 8);
  }

  CodeOffset pushWithPatch(ImmWord imm) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    CodeOffset offset = movWithPatch(imm, scratch);
    push(scratch);
    return offset;
  }
  CodeOffset movWithPatch(ImmWord imm, Register dest) {
    BufferOffset bo = emitLoad64Stanza(dest, (uint64_t)imm.value);
    return CodeOffset(bo.getOffset());
  }
  CodeOffset movWithPatch(ImmPtr imm, Register dest) {
    return movWithPatch(ImmWord(uintptr_t(imm.value)), dest);
  }

  // ===============================================================
  // Tag/unbox operations

  void splitTag(Register src, Register dest) {
    x_srdi(dest, src, JSVAL_TAG_SHIFT);
  }
  void splitTag(const ValueOperand& operand, Register dest) {
    splitTag(operand.valueReg(), dest);
  }
  void splitTagForTest(const ValueOperand& value, ScratchTagScope& tag) {
    splitTag(value, tag);
  }

  void unboxNonDouble(const ValueOperand& operand, Register dest,
                      JSValueType type) {
    unboxNonDouble(operand.valueReg(), dest, type);
  }
  template <typename T>
  void unboxNonDouble(T src, Register dest, JSValueType type) {
    MOZ_ASSERT(type != JSVAL_TYPE_DOUBLE);
    if (type == JSVAL_TYPE_INT32 || type == JSVAL_TYPE_BOOLEAN) {
      load32(src, dest);
      return;
    }
    loadPtr(src, dest);
    unboxNonDouble(dest, dest, type);
  }
  void unboxNonDouble(Register src, Register dest, JSValueType type) {
    MOZ_ASSERT(type != JSVAL_TYPE_DOUBLE);
    if (type == JSVAL_TYPE_INT32 || type == JSVAL_TYPE_BOOLEAN) {
      as_extsw(dest, src);
      return;
    }
    // Extract the payload (lower 47 bits) by clearing the tag.
    // This avoids acquiring a scratch register, preventing pool exhaustion
    // when called from nested scratch scopes (e.g., ScratchTagScope →
    // branchTestStringTruthy → unboxString → here).
    // rldicl dest, src, 0, 17 — clear upper 17 bits (tag), keep lower 47.
    as_rldicl(dest, src, 0, 17);
  }
  void unboxGCThingForGCBarrier(const Address& src, Register dest) {
    loadPtr(src, dest);
    // Clear tag bits (top 17 bits on 64-bit).
    as_rldicl(dest, dest, 0, 64 - JSVAL_TAG_SHIFT);
  }
  void unboxGCThingForGCBarrier(const ValueOperand& src, Register dest) {
    as_rldicl(dest, src.valueReg(), 0, 64 - JSVAL_TAG_SHIFT);
  }
  void unboxWasmAnyRefGCThingForGCBarrier(const Address& src, Register dest) {
    static_assert(wasm::AnyRef::TagShift == 2);
    loadPtr(src, dest);
    as_rldicr(dest, dest, 0, 61);
  }
  void getGCThingValueChunk(const Address& src, Register dest) {
    loadPtr(src, dest);
    as_rldicl(dest, dest, 0, 17);
    as_rldicr(dest, dest, 0, 43);
  }
  void getGCThingValueChunk(const ValueOperand& src, Register dest) {
    as_rldicl(dest, src.valueReg(), 0, 17);
    as_rldicr(dest, dest, 0, 43);
  }

  void boxDouble(FloatRegister src, const ValueOperand& dest, FloatRegister) {
    as_mfvsrd(dest.valueReg(), src);
  }
  void boxNonDouble(JSValueType type, Register src, const ValueOperand& dest) {
    boxValue(type, src, dest.valueReg());
  }
  void boxNonDouble(Register type, Register src, const ValueOperand& dest) {
    boxValue(type, src, dest.valueReg());
  }
  void unboxInt32(const ValueOperand& operand, Register dest) {
    as_extsw(dest, operand.valueReg());
  }
  void unboxInt32(const Address& src, Register dest) { load32(src, dest); }
  void unboxInt32(const BaseIndex& src, Register dest) { load32(src, dest); }
  void unboxBoolean(const ValueOperand& operand, Register dest) {
    as_extsw(dest, operand.valueReg());
  }
  void unboxBoolean(const Address& src, Register dest) { load32(src, dest); }
  void unboxBoolean(const BaseIndex& src, Register dest) { load32(src, dest); }
  void unboxDouble(const ValueOperand& operand, FloatRegister dest) {
    as_mtvsrd(dest, operand.valueReg());
  }
  void unboxDouble(const Address& src, FloatRegister dest) {
    loadDouble(src, dest);
  }
  void unboxDouble(const BaseIndex& src, FloatRegister dest) {
    loadDouble(src, dest);
  }
  void unboxString(const ValueOperand& operand, Register dest) {
    unboxNonDouble(operand, dest, JSVAL_TYPE_STRING);
  }
  void unboxString(const Address& src, Register dest) {
    unboxNonDouble(src, dest, JSVAL_TYPE_STRING);
  }
  void unboxSymbol(const ValueOperand& operand, Register dest) {
    unboxNonDouble(operand, dest, JSVAL_TYPE_SYMBOL);
  }
  void unboxSymbol(const Address& src, Register dest) {
    unboxNonDouble(src, dest, JSVAL_TYPE_SYMBOL);
  }
  void unboxBigInt(const ValueOperand& operand, Register dest) {
    unboxNonDouble(operand, dest, JSVAL_TYPE_BIGINT);
  }
  void unboxBigInt(const Address& src, Register dest) {
    unboxNonDouble(src, dest, JSVAL_TYPE_BIGINT);
  }
  void unboxObject(const ValueOperand& src, Register dest) {
    unboxNonDouble(src, dest, JSVAL_TYPE_OBJECT);
  }
  void unboxObject(const Address& src, Register dest) {
    unboxNonDouble(src, dest, JSVAL_TYPE_OBJECT);
  }
  void unboxObject(const BaseIndex& src, Register dest) {
    unboxNonDouble(src, dest, JSVAL_TYPE_OBJECT);
  }
  void unboxValue(const ValueOperand& src, AnyRegister dest, JSValueType type) {
    if (dest.isFloat()) {
      unboxDouble(src, dest.fpu());
    } else {
      unboxNonDouble(src, dest.gpr(), type);
    }
  }
  void unboxObjectOrNull(const Address& src, Register dest) {
    loadPtr(src, dest);
    // Object pointers have the object tag in high bits; null has a different
    // tag. Clear the top bits to get either a valid pointer or zero.
    as_rldicl(dest, dest, 0, 64 - JSVAL_TAG_SHIFT);
  }

  void tagValue(JSValueType type, Register payload, ValueOperand dest) {
    MOZ_ASSERT(type != JSVAL_TYPE_UNDEFINED && type != JSVAL_TYPE_NULL);
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    MOZ_ASSERT(scratch != payload && scratch != dest.valueReg());
    tagValueWithScratch(type, payload, dest, scratch);
  }
  void tagValueWithScratch(JSValueType type, Register payload,
                           ValueOperand dest, Register scratch) {
    movePtr(ImmShiftedTag(type), scratch);
    if (type == JSVAL_TYPE_INT32 || type == JSVAL_TYPE_BOOLEAN ||
        type == JSVAL_TYPE_MAGIC) {
      if (payload != dest.valueReg()) {
        movePtr(payload, dest.valueReg());
      }
      as_rldicl(dest.valueReg(), dest.valueReg(), 0, 32);
      as_or_(dest.valueReg(), dest.valueReg(), scratch);
    } else {
      if (payload != dest.valueReg()) {
        movePtr(payload, dest.valueReg());
      }
      as_or_(dest.valueReg(), dest.valueReg(), scratch);
    }
  }
  void boxValue(JSValueType type, Register src, Register dest) {
    MOZ_ASSERT(src != dest);
    MOZ_ASSERT(type != JSVAL_TYPE_UNDEFINED && type != JSVAL_TYPE_NULL);
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    boxValueWithScratch(type, src, dest, scratch);
  }
  void boxValueWithScratch(JSValueType type, Register src, Register dest,
                           Register scratch) {
    if (type == JSVAL_TYPE_INT32 || type == JSVAL_TYPE_BOOLEAN ||
        type == JSVAL_TYPE_MAGIC) {
      as_rldicl(dest, src, 0, 32);
      movePtr(ImmShiftedTag(type), scratch);
      as_or_(dest, dest, scratch);
    } else {
      movePtr(ImmShiftedTag(type), scratch);
      xs_mr(dest, src);
      as_or_(dest, dest, scratch);
    }
  }
  void boxValue(Register type, Register src, Register dest) {
    MOZ_ASSERT(src != dest);

#ifdef DEBUG
    Label done, isNullOrUndefined, isBoolean, isInt32OrMagic;

    // Use ma_cmp + ma_b instead of asMasm().branch32() because
    // MacroAssembler is not yet fully defined at this point.
    Condition cond;
    cond = ma_cmp(type, Imm32(JSVAL_TYPE_NULL), Equal, true);
    ma_b(cond, &isNullOrUndefined);
    cond = ma_cmp(type, Imm32(JSVAL_TYPE_UNDEFINED), Equal, true);
    ma_b(cond, &isNullOrUndefined);
    cond = ma_cmp(type, Imm32(JSVAL_TYPE_BOOLEAN), Equal, true);
    ma_b(cond, &isBoolean);
    cond = ma_cmp(type, Imm32(JSVAL_TYPE_INT32), Equal, true);
    ma_b(cond, &isInt32OrMagic);
    cond = ma_cmp(type, Imm32(JSVAL_TYPE_MAGIC), Equal, true);
    ma_b(cond, &isInt32OrMagic);
    // GCThing types aren't supported, because as_rldicl truncates
    // payloads above UINT32_MAX.
    breakpoint();
    {
      bind(&isNullOrUndefined);

      // Ensure no payload for null and undefined.
      cond = ma_cmp(src, ImmWord(0), Equal);
      ma_b(cond, &done);
      breakpoint();
    }
    {
      bind(&isBoolean);

      // Ensure boolean values are either 0 or 1.
      cond = ma_cmp(src, Imm32(1), BelowOrEqual, true);
      ma_b(cond, &done);
      breakpoint();
    }
    {
      bind(&isInt32OrMagic);

      // Ensure |src| is sign-extended.
      UseScratchRegisterScope debugTemps(*this);
      Register debugScratch = debugTemps.Acquire();
      as_extsw(debugScratch, src);
      cond = ma_cmp(src, debugScratch, Equal);
      ma_b(cond, &done);
      breakpoint();
    }
    bind(&done);
#endif

    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    MOZ_ASSERT(scratch != dest && scratch != src && scratch != type);
    // Build tag: (type | JSVAL_TAG_MAX_DOUBLE) << JSVAL_TAG_SHIFT
    move32(Imm32(JSVAL_TAG_MAX_DOUBLE), scratch);
    as_or_(scratch, scratch, type);
    x_sldi(scratch, scratch, JSVAL_TAG_SHIFT);
    // Insert 32-bit payload.
    as_rldicl(dest, src, 0, 32);
    as_or_(dest, dest, scratch);
  }

  // ===============================================================
  // Value store/load/push/pop

  void storeValue(ValueOperand val, const Address& dest) {
    storePtr(val.valueReg(), dest);
  }
  void storeValue(ValueOperand val, const BaseIndex& dest) {
    storePtr(val.valueReg(), dest);
  }
  void storeValue(JSValueType type, Register reg, Address dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    MOZ_ASSERT(dest.base != scratch);
    boxValue(type, reg, scratch);
    storePtr(scratch, dest);
  }
  void storeValue(JSValueType type, Register reg, BaseIndex dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    MOZ_ASSERT(dest.base != scratch);
    boxValue(type, reg, scratch);
    storePtr(scratch, dest);
  }
  void storeValue(const Value& val, Address dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    MOZ_ASSERT(dest.base != scratch);
    if (val.isGCThing()) {
      CodeOffset off = movWithPatch(ImmWord(val.asRawBits()), scratch);
      writeDataRelocation(off, val);
    } else {
      movePtr(ImmWord(val.asRawBits()), scratch);
    }
    storePtr(scratch, dest);
  }
  void storeValue(const Value& val, BaseIndex dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    MOZ_ASSERT(dest.base != scratch);
    if (val.isGCThing()) {
      CodeOffset off = movWithPatch(ImmWord(val.asRawBits()), scratch);
      writeDataRelocation(off, val);
    } else {
      movePtr(ImmWord(val.asRawBits()), scratch);
    }
    storePtr(scratch, dest);
  }
  void storeValue(const Address& src, const Address& dest, Register temp) {
    loadPtr(src, temp);
    storePtr(temp, dest);
  }

  void storePrivateValue(Register src, const Address& dest) {
    storePtr(src, dest);
  }
  void storePrivateValue(ImmGCPtr imm, const Address& dest) {
    storePtr(imm, dest);
  }

  void loadValue(Address src, ValueOperand val) {
    loadPtr(src, val.valueReg());
  }
  void loadValue(const BaseIndex& src, ValueOperand val) {
    loadPtr(src, val.valueReg());
  }
  void loadUnalignedValue(const Address& src, ValueOperand dest) {
    loadPtr(src, dest.valueReg());
  }

  void pushValue(ValueOperand val) { push(val.valueReg()); }
  void popValue(ValueOperand val) { pop(val.valueReg()); }
  void pushValue(const Value& val) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    if (val.isGCThing()) {
      CodeOffset off = movWithPatch(ImmWord(val.asRawBits()), scratch);
      writeDataRelocation(off, val);
    } else {
      movePtr(ImmWord(val.asRawBits()), scratch);
    }
    push(scratch);
  }
  void pushValue(JSValueType type, Register reg) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    boxValue(type, reg, scratch);
    push(scratch);
  }
  void pushValue(const Address& addr) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    loadPtr(addr, scratch);
    push(scratch);
  }
  void pushValue(const BaseIndex& addr, Register scratch) {
    loadPtr(addr, scratch);
    push(scratch);
  }

  // ===============================================================
  // Load instructions

  FaultingCodeOffset load8SignExtend(const Address& address, Register dest) {
    FaultingCodeOffset fco;
    if (is_intN(address.offset, 16)) {
      fco = FaultingCodeOffset(
          as_lbz(dest, address.base, address.offset).getOffset());
    } else {
      UseScratchRegisterScope temps(*this);
      Register scratch = temps.Acquire();
      MOZ_ASSERT(scratch != dest);
      movePtr(ImmWord(address.offset), scratch);
      fco =
          FaultingCodeOffset(as_lbzx(dest, address.base, scratch).getOffset());
    }
    as_extsb(dest, dest);
    return fco;
  }
  FaultingCodeOffset load8SignExtend(const BaseIndex& src, Register dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(src, scratch);
    FaultingCodeOffset fco;
    if (is_intN(src.offset, 16)) {
      fco = FaultingCodeOffset(as_lbz(dest, scratch, src.offset).getOffset());
    } else {
      MOZ_ASSERT(scratch != dest);
      movePtr(ImmWord(src.offset), dest);
      fco = FaultingCodeOffset(as_lbzx(dest, scratch, dest).getOffset());
    }
    as_extsb(dest, dest);
    return fco;
  }
  FaultingCodeOffset load8ZeroExtend(const Address& address, Register dest) {
    if (is_intN(address.offset, 16)) {
      return FaultingCodeOffset(
          as_lbz(dest, address.base, address.offset).getOffset());
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    MOZ_ASSERT(scratch != dest);
    movePtr(ImmWord(address.offset), scratch);
    return FaultingCodeOffset(as_lbzx(dest, address.base, scratch).getOffset());
  }
  FaultingCodeOffset load8ZeroExtend(const BaseIndex& src, Register dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(src, scratch);
    if (is_intN(src.offset, 16)) {
      return FaultingCodeOffset(as_lbz(dest, scratch, src.offset).getOffset());
    }
    MOZ_ASSERT(scratch != dest);
    movePtr(ImmWord(src.offset), dest);
    return FaultingCodeOffset(as_lbzx(dest, scratch, dest).getOffset());
  }
  FaultingCodeOffset load16SignExtend(const Address& address, Register dest) {
    if (is_intN(address.offset, 16)) {
      return FaultingCodeOffset(
          as_lha(dest, address.base, address.offset).getOffset());
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    MOZ_ASSERT(scratch != dest);
    movePtr(ImmWord(address.offset), scratch);
    return FaultingCodeOffset(as_lhax(dest, address.base, scratch).getOffset());
  }
  FaultingCodeOffset load16SignExtend(const BaseIndex& src, Register dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(src, scratch);
    if (is_intN(src.offset, 16)) {
      return FaultingCodeOffset(as_lha(dest, scratch, src.offset).getOffset());
    }
    MOZ_ASSERT(scratch != dest);
    movePtr(ImmWord(src.offset), dest);
    return FaultingCodeOffset(as_lhax(dest, scratch, dest).getOffset());
  }
  template <typename S>
  void load16UnalignedSignExtend(const S& src, Register dest) {
    load16SignExtend(src, dest);
  }
  FaultingCodeOffset load16ZeroExtend(const Address& address, Register dest) {
    if (is_intN(address.offset, 16)) {
      return FaultingCodeOffset(
          as_lhz(dest, address.base, address.offset).getOffset());
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    MOZ_ASSERT(scratch != dest);
    movePtr(ImmWord(address.offset), scratch);
    return FaultingCodeOffset(as_lhzx(dest, address.base, scratch).getOffset());
  }
  FaultingCodeOffset load16ZeroExtend(const BaseIndex& src, Register dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(src, scratch);
    if (is_intN(src.offset, 16)) {
      return FaultingCodeOffset(as_lhz(dest, scratch, src.offset).getOffset());
    }
    MOZ_ASSERT(scratch != dest);
    movePtr(ImmWord(src.offset), dest);
    return FaultingCodeOffset(as_lhzx(dest, scratch, dest).getOffset());
  }
  template <typename S>
  void load16UnalignedZeroExtend(const S& src, Register dest) {
    load16ZeroExtend(src, dest);
  }

  FaultingCodeOffset load32(const Address& address, Register dest) {
    // lwa is DS-form (14-bit displacement × 4 = 16-bit-signed effective
    // range, 4-byte alignment required). lwax is X-form indexed, no
    // alignment constraint. Both sign-extend in one instruction; only
    // the misaligned 16-bit-fitting case still needs lwz + extsw.
    if (is_intN(address.offset, 16) && (address.offset & 3) == 0) {
      return FaultingCodeOffset(
          as_lwa(dest, address.base, address.offset).getOffset());
    }
    if (is_intN(address.offset, 16)) {
      FaultingCodeOffset fco(
          as_lwz(dest, address.base, address.offset).getOffset());
      as_extsw(dest, dest);
      return fco;
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    MOZ_ASSERT(scratch != dest);
    movePtr(ImmWord(address.offset), scratch);
    return FaultingCodeOffset(as_lwax(dest, address.base, scratch).getOffset());
  }
  FaultingCodeOffset load32(const BaseIndex& address, Register dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(address, scratch);
    if (is_intN(address.offset, 16) && (address.offset & 3) == 0) {
      return FaultingCodeOffset(
          as_lwa(dest, scratch, address.offset).getOffset());
    }
    if (is_intN(address.offset, 16)) {
      FaultingCodeOffset fco(as_lwz(dest, scratch, address.offset).getOffset());
      as_extsw(dest, dest);
      return fco;
    }
    MOZ_ASSERT(scratch != dest);
    movePtr(ImmWord(address.offset), dest);
    return FaultingCodeOffset(as_lwax(dest, scratch, dest).getOffset());
  }
  void load32(AbsoluteAddress address, Register dest) {
    movePtr(ImmWord((uintptr_t)address.addr), dest);
    as_lwa(dest, dest, 0);
  }
  void load32(wasm::SymbolicAddress address, Register dest) {
    movePtr(address, dest);
    as_lwa(dest, dest, 0);
  }
  template <typename S>
  void load32Unaligned(const S& src, Register dest) {
    load32(src, dest);
  }

  FaultingCodeOffset load64(const Address& address, Register64 dest) {
    return loadPtr(address, dest.reg);
  }
  FaultingCodeOffset load64(const BaseIndex& address, Register64 dest) {
    return loadPtr(address, dest.reg);
  }
  template <typename S>
  void load64Unaligned(const S& src, Register64 dest) {
    load64(src, dest);
  }

  FaultingCodeOffset loadPtr(const Address& address, Register dest) {
    // as_ld (DS-form) requires 4-byte aligned offset.
    if (is_intN(address.offset, 16) && !(address.offset & 0x3)) {
      return FaultingCodeOffset(
          as_ld(dest, address.base, address.offset).getOffset());
    }
    if (HasPOWER10() && is_intN((intptr_t)address.offset, 34)) {
      return FaultingCodeOffset(
          as_pld(dest, address.base, (int64_t)address.offset, /*R=*/false)
              .getOffset());
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    MOZ_ASSERT(scratch != dest);
    movePtr(ImmWord(address.offset), scratch);
    return FaultingCodeOffset(as_ldx(dest, address.base, scratch).getOffset());
  }
  FaultingCodeOffset loadPtr(const BaseIndex& src, Register dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(src, scratch);
    if (is_intN(src.offset, 16) && !(src.offset & 0x3)) {
      return FaultingCodeOffset(as_ld(dest, scratch, src.offset).getOffset());
    }
    MOZ_ASSERT(scratch != dest);
    movePtr(ImmWord(src.offset), dest);
    return FaultingCodeOffset(as_ldx(dest, scratch, dest).getOffset());
  }
  void loadPtr(AbsoluteAddress address, Register dest) {
    movePtr(ImmWord((uintptr_t)address.addr), dest);
    as_ld(dest, dest, 0);
  }
  void loadPtr(wasm::SymbolicAddress address, Register dest) {
    movePtr(address, dest);
    as_ld(dest, dest, 0);
  }

  void loadPrivate(const Address& address, Register dest) {
    loadPtr(address, dest);
  }

  FaultingCodeOffset loadDouble(const Address& addr, FloatRegister dest) {
    if (is_intN(addr.offset, 16)) {
      return FaultingCodeOffset(
          as_lfd(dest, addr.base, addr.offset).getOffset());
    }
    if (HasPOWER10() && is_intN((intptr_t)addr.offset, 34)) {
      return FaultingCodeOffset(
          as_plfd(dest, addr.base, (int64_t)addr.offset, /*R=*/false)
              .getOffset());
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(ImmWord(addr.offset), scratch);
    return FaultingCodeOffset(as_lfdx(dest, addr.base, scratch).getOffset());
  }
  FaultingCodeOffset loadDouble(const BaseIndex& src, FloatRegister dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(src, scratch);
    if (is_intN(src.offset, 16)) {
      return FaultingCodeOffset(as_lfd(dest, scratch, src.offset).getOffset());
    }
    Register scratch2 = temps.Acquire();
    movePtr(ImmWord(src.offset), scratch2);
    return FaultingCodeOffset(as_lfdx(dest, scratch, scratch2).getOffset());
  }
  FaultingCodeOffset loadFloat32(const Address& addr, FloatRegister dest) {
    if (is_intN(addr.offset, 16)) {
      return FaultingCodeOffset(
          as_lfs(dest, addr.base, addr.offset).getOffset());
    }
    if (HasPOWER10() && is_intN((intptr_t)addr.offset, 34)) {
      return FaultingCodeOffset(
          as_plfs(dest, addr.base, (int64_t)addr.offset, /*R=*/false)
              .getOffset());
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(ImmWord(addr.offset), scratch);
    return FaultingCodeOffset(as_lfsx(dest, addr.base, scratch).getOffset());
  }
  FaultingCodeOffset loadFloat32(const BaseIndex& src, FloatRegister dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(src, scratch);
    if (is_intN(src.offset, 16)) {
      return FaultingCodeOffset(as_lfs(dest, scratch, src.offset).getOffset());
    }
    Register scratch2 = temps.Acquire();
    movePtr(ImmWord(src.offset), scratch2);
    return FaultingCodeOffset(as_lfsx(dest, scratch, scratch2).getOffset());
  }
  // Load a FP constant into `dest`.
  //
  // +0.0 / +0.0f: `xxlxor dest, dest, dest` (1 insn). No register clobbers.
  //
  // POWER9 non-zero: constant pool load via `addpcis r16, hi; lfd/lfs fD,
  // lo(r16); nop`. 2 real insns + nop, no LR clobber, no Return Address
  // Stack corruption. lfs auto-expands single-precision to double, so no
  // separate xscvspdpn step. Clobbers r16 (SavedScratchRegister). Pool
  // entries are shared across duplicate constants.
  //
  // POWER8 non-zero: inline `movePtr + mtvsrd(+xscvspdpn)` path. We do NOT
  // use the bcl-based pool path on POWER8: bcl clobbers LR and corrupts
  // the Return Address Stack, which causes catastrophic mispredicts in
  // hot FP-constant loops (~200x slowdown observed on cmp-bitselect.js).
  //
  // Precondition: must not be called inside an `enterNoPool` region when
  // HasPOWER9() is true (the pool path calls `allocEntry` which asserts
  // `inhibitPools_ == 0`). Audit-verified that no such call site exists
  // today; the POWER8 inline path is unaffected.
  void loadConstantDouble(double dp, FloatRegister dest) {
    if (mozilla::IsPositiveZero(dp)) {
      as_xxlxor(dest, dest, dest);
      return;
    }
    if (HasPOWER9()) {
      loadFromPoolFloat64(dest, dp);
      return;
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    union {
      double d;
      uint64_t u;
    } u;
    u.d = dp;
    movePtr(ImmWord(u.u), scratch);
    as_mtvsrd(dest, scratch);
  }
  void loadConstantFloat32(float f, FloatRegister dest) {
    if (mozilla::IsPositiveZero(f)) {
      as_xxlxor(dest, dest, dest);
      return;
    }
    if (HasPOWER9()) {
      loadFromPoolFloat32(dest, f);
      return;
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    union {
      float f;
      uint32_t u;
    } u;
    u.f = f;
    movePtr(ImmWord(u.u), scratch);
    x_sldi(scratch, scratch, 32);
    as_mtvsrd(dest, scratch);
    as_xscvspdpn(dest, dest);
  }

  void notBoolean(const ValueOperand& val) {
    as_xori(val.valueReg(), val.valueReg(), 1);
  }

  [[nodiscard]] Register extractTag(const Address& address, Register scratch) {
    loadPtr(address, scratch);
    x_srdi(scratch, scratch, JSVAL_TAG_SHIFT);
    return scratch;
  }
  [[nodiscard]] Register extractTag(const BaseIndex& address,
                                    Register scratch) {
    if (scratch == r0) {
      // r0 cannot be used as a base register in D-form/X-form loads,
      // so we need a separate temp for the intermediate address.
      UseScratchRegisterScope temps(*this);
      Register base = temps.Acquire();
      computeScaledAddress(address, base);
      loadPtr(Address(base, address.offset), scratch);
    } else {
      // scratch is a pool register (r11/r12) or another GPR that can
      // serve as a base register, so reuse it for the address computation.
      computeScaledAddress(address, scratch);
      loadPtr(Address(scratch, address.offset), scratch);
    }
    x_srdi(scratch, scratch, JSVAL_TAG_SHIFT);
    return scratch;
  }
  [[nodiscard]] Register extractTag(const ValueOperand& value,
                                    Register scratch) {
    splitTag(value, scratch);
    return scratch;
  }

  [[nodiscard]] Register extractObject(const Address& address,
                                       Register scratch) {
    loadPtr(address, scratch);
    as_rldicl(scratch, scratch, 0, 64 - JSVAL_TAG_SHIFT);
    return scratch;
  }
  [[nodiscard]] Register extractObject(const ValueOperand& value,
                                       Register scratch) {
    unboxObject(value, scratch);
    return scratch;
  }
  [[nodiscard]] Register extractInt32(const ValueOperand& value,
                                      Register scratch) {
    unboxInt32(value, scratch);
    return scratch;
  }
  [[nodiscard]] Register extractString(const ValueOperand& value,
                                       Register scratch) {
    unboxString(value, scratch);
    return scratch;
  }
  [[nodiscard]] Register extractSymbol(const ValueOperand& value,
                                       Register scratch) {
    unboxSymbol(value, scratch);
    return scratch;
  }
  [[nodiscard]] Register extractBoolean(const ValueOperand& value,
                                        Register scratch) {
    unboxBoolean(value, scratch);
    return scratch;
  }

  void testObjectSet(Condition cond, const ValueOperand& value, Register dest) {
    MOZ_ASSERT(cond == Equal || cond == NotEqual);
    {
      UseScratchRegisterScope temps(*this);
      Register tag = temps.Acquire();
      splitTag(value, tag);
      uint32_t t = JSVAL_TAG_OBJECT;
      as_xoris(tag, tag, t >> 16);
      as_cmplwi(tag, t & 0xFFFF);
    }
    ma_cmp_set(dest, cond);
  }
  void testUndefinedSet(Condition cond, const ValueOperand& value,
                        Register dest) {
    MOZ_ASSERT(cond == Equal || cond == NotEqual);
    {
      UseScratchRegisterScope temps(*this);
      Register tag = temps.Acquire();
      splitTag(value, tag);
      // Use xoris+cmplwi to compare without a second scratch.
      uint32_t t = JSVAL_TAG_UNDEFINED;
      as_xoris(tag, tag, t >> 16);
      as_cmplwi(tag, t & 0xFFFF);
    }
    ma_cmp_set(dest, cond);
  }
  void testNullSet(Condition cond, const ValueOperand& value, Register dest) {
    MOZ_ASSERT(cond == Equal || cond == NotEqual);
    {
      UseScratchRegisterScope temps(*this);
      Register tag = temps.Acquire();
      splitTag(value, tag);
      uint32_t t = JSVAL_TAG_NULL;
      as_xoris(tag, tag, t >> 16);
      as_cmplwi(tag, t & 0xFFFF);
    }
    ma_cmp_set(dest, cond);
  }

  BufferOffset ret() {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    as_ld(scratch, StackPointer, 0);
    as_addi(StackPointer, StackPointer, 8);
    xs_mtlr(scratch);
    return as_blr();
  }

  void j(Label* dest) { jump(dest); }

  void getWasmAnyRefGCThingChunk(Register anyref, Register dest) {
    static_assert(js::gc::ChunkShift == 20);
    as_rldicr(dest, anyref, 0, 43);
  }

  template <typename T>
  void loadUnboxedValue(const T& address, MIRType type, AnyRegister dest) {
    if (dest.isFloat()) {
      loadInt32OrDouble(address, dest.fpu());
    } else {
      unboxNonDouble(address, dest.gpr(), ValueTypeFromMIRType(type));
    }
  }

  void loadInt32OrDouble(const Address& src, FloatRegister dest);
  void loadInt32OrDouble(const BaseIndex& addr, FloatRegister dest);

  // ===============================================================
  // Store instructions

  FaultingCodeOffset store8(Register src, const Address& address) {
    if (is_intN(address.offset, 16)) {
      return FaultingCodeOffset(
          as_stb(src, address.base, address.offset).getOffset());
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(ImmWord(address.offset), scratch);
    return FaultingCodeOffset(as_stbx(src, address.base, scratch).getOffset());
  }
  FaultingCodeOffset store8(Register src, const BaseIndex& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(address, scratch);
    if (is_intN(address.offset, 16)) {
      return FaultingCodeOffset(
          as_stb(src, scratch, address.offset).getOffset());
    }
    Register scratch2 = temps.Acquire();
    movePtr(ImmWord(address.offset), scratch2);
    return FaultingCodeOffset(as_stbx(src, scratch, scratch2).getOffset());
  }
  void store8(Imm32 imm, const Address& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    move32(imm, scratch);
    store8(scratch, address);
  }
  void store8(Imm32 imm, const BaseIndex& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    move32(imm, scratch);
    store8(scratch, address);
  }

  FaultingCodeOffset store16(Register src, const Address& address) {
    if (is_intN(address.offset, 16)) {
      return FaultingCodeOffset(
          as_sth(src, address.base, address.offset).getOffset());
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(ImmWord(address.offset), scratch);
    return FaultingCodeOffset(as_sthx(src, address.base, scratch).getOffset());
  }
  FaultingCodeOffset store16(Register src, const BaseIndex& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(address, scratch);
    if (is_intN(address.offset, 16)) {
      return FaultingCodeOffset(
          as_sth(src, scratch, address.offset).getOffset());
    }
    Register scratch2 = temps.Acquire();
    movePtr(ImmWord(address.offset), scratch2);
    return FaultingCodeOffset(as_sthx(src, scratch, scratch2).getOffset());
  }
  void store16(Imm32 imm, const Address& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    move32(imm, scratch);
    store16(scratch, address);
  }
  void store16(Imm32 imm, const BaseIndex& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    move32(imm, scratch);
    store16(scratch, address);
  }
  template <typename T>
  void store16Unaligned(Register src, const T& dest) {
    store16(src, dest);
  }

  FaultingCodeOffset store32(Register src, const Address& address) {
    if (is_intN(address.offset, 16)) {
      return FaultingCodeOffset(
          as_stw(src, address.base, address.offset).getOffset());
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(ImmWord(address.offset), scratch);
    return FaultingCodeOffset(as_stwx(src, address.base, scratch).getOffset());
  }
  FaultingCodeOffset store32(Register src, const BaseIndex& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(address, scratch);
    if (is_intN(address.offset, 16)) {
      return FaultingCodeOffset(
          as_stw(src, scratch, address.offset).getOffset());
    }
    Register scratch2 = temps.Acquire();
    movePtr(ImmWord(address.offset), scratch2);
    return FaultingCodeOffset(as_stwx(src, scratch, scratch2).getOffset());
  }
  void store32(Register src, AbsoluteAddress address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(ImmWord((uintptr_t)address.addr), scratch);
    as_stw(src, scratch, 0);
  }
  void store32(Imm32 src, const Address& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    move32(src, scratch);
    store32(scratch, address);
  }
  void store32(Imm32 src, const BaseIndex& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    move32(src, scratch);
    store32(scratch, address);
  }
  template <typename T>
  void store32Unaligned(Register src, const T& dest) {
    store32(src, dest);
  }

  void store64(Imm64 imm, Address address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(ImmWord(imm.value), scratch);
    storePtr(scratch, address);
  }
  void store64(Imm64 imm, const BaseIndex& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(ImmWord(imm.value), scratch);
    storePtr(scratch, address);
  }
  FaultingCodeOffset store64(Register64 src, Address address) {
    return storePtr(src.reg, address);
  }
  FaultingCodeOffset store64(Register64 src, const BaseIndex& address) {
    return storePtr(src.reg, address);
  }
  template <typename T>
  void store64Unaligned(Register64 src, const T& dest) {
    store64(src, dest);
  }

  template <typename T>
  void storePtr(ImmWord imm, T address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(imm, scratch);
    storePtr(scratch, address);
  }
  template <typename T>
  void storePtr(ImmPtr imm, T address) {
    storePtr(ImmWord(uintptr_t(imm.value)), address);
  }
  template <typename T>
  void storePtr(ImmGCPtr imm, T address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(imm, scratch);
    storePtr(scratch, address);
  }
  void storePtr(Register src, AbsoluteAddress dest) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(ImmWord((uintptr_t)dest.addr), scratch);
    as_std(src, scratch, 0);
  }
  FaultingCodeOffset storePtr(Register src, const Address& address) {
    // as_std (DS-form) requires 4-byte aligned offset.
    if (is_intN(address.offset, 16) && !(address.offset & 0x3)) {
      return FaultingCodeOffset(
          as_std(src, address.base, address.offset).getOffset());
    }
    if (HasPOWER10() && is_intN((intptr_t)address.offset, 34)) {
      return FaultingCodeOffset(
          as_pstd(src, address.base, (int64_t)address.offset, /*R=*/false)
              .getOffset());
    }
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    movePtr(ImmWord(address.offset), scratch);
    return FaultingCodeOffset(as_stdx(src, address.base, scratch).getOffset());
  }
  FaultingCodeOffset storePtr(Register src, const BaseIndex& address) {
    UseScratchRegisterScope temps(*this);
    Register scratch = temps.Acquire();
    computeScaledAddress(address, scratch);
    if (is_intN(address.offset, 16) && !(address.offset & 0x3)) {
      return FaultingCodeOffset(
          as_std(src, scratch, address.offset).getOffset());
    }
    Register scratch2 = temps.Acquire();
    movePtr(ImmWord(address.offset), scratch2);
    return FaultingCodeOffset(as_stdx(src, scratch, scratch2).getOffset());
  }

  // ===============================================================
  // Misc

  void handleFailureWithHandlerTail(Label* profilerExitTail, Label* bailoutTail,
                                    uint32_t* returnValueCheckOffset);

  inline void incrementInt32Value(const Address& addr);

  void zeroDouble(FloatRegister reg) { as_xxlxor(reg, reg, reg); }

  void writeCodePointer(CodeLabel* label) {
    label->patchAt()->bind(currentOffset());
    label->setLinkMode(CodeLabel::RawPointer);
    m_buffer.ensureSpace(sizeof(void*));
    writeInst(-1);
    writeInst(-1);
  }
  void writeDataRelocation(const Value& val) {
    if (val.isGCThing()) {
      gc::Cell* cell = val.toGCThing();
      if (cell && gc::IsInsideNursery(cell)) {
        embedsNurseryPointers_ = true;
      }
      dataRelocations_.writeUnsigned(currentOffset());
    }
  }
  void writeDataRelocation(CodeOffset off, const Value& val) {
    if (val.isGCThing()) {
      gc::Cell* cell = val.toGCThing();
      if (cell && gc::IsInsideNursery(cell)) {
        embedsNurseryPointers_ = true;
      }
      dataRelocations_.writeUnsigned(off.offset());
    }
  }

  CodeOffset toggledJump(Label* label) {
    CodeOffset ret(nextOffset().getOffset());
    jump(label);
    return ret;
  }
  CodeOffset toggledCall(JitCode* target, bool enabled);
  // 8 instructions for load64 + mtctr + bctrl = 10 instructions total.
  static size_t ToggledCallSize(uint8_t* code) { return 10 * sizeof(uint32_t); }

  void checkStackAlignment() {}

  static void calculateAlignedStackPointer(void** stackPointer) {
    *stackPointer = reinterpret_cast<void*>((uintptr_t(*stackPointer)) &
                                            ~(ABIStackAlignment - 1));
  }

  void lea(Operand addr, Register dest) {
    // x86-ism; on PPC, compute effective address manually.
    MOZ_CRASH("PPC64: lea not supported; use computeEffectiveAddress");
  }

  void abiret() { as_blr(); }

  void profilerEnterFrame(Register framePtr, Register scratch);
  void profilerExitFrame();

  void outOfLineWasmTruncateToInt32Check(
      FloatRegister input, Register output, MIRType fromType, TruncFlags flags,
      Label* rejoin, const wasm::TrapSiteDesc& trapSiteDesc);
  void outOfLineWasmTruncateToInt64Check(
      FloatRegister input, Register64 output, MIRType fromType,
      TruncFlags flags, Label* rejoin, const wasm::TrapSiteDesc& trapSiteDesc);

  void wasmLoadImpl(const wasm::MemoryAccessDesc& access, Register memoryBase,
                    Register ptr, Register ptrScratch, AnyRegister output);
  void wasmStoreImpl(const wasm::MemoryAccessDesc& access, AnyRegister value,
                     Register memoryBase, Register ptr, Register ptrScratch);
  void wasmLoadI64Impl(const wasm::MemoryAccessDesc& access,
                       Register memoryBase, Register ptr, Register ptrScratch,
                       Register64 output);
  void wasmStoreI64Impl(const wasm::MemoryAccessDesc& access, Register64 value,
                        Register memoryBase, Register ptr, Register ptrScratch);

  // Last-byte probing load to enforce wasm-spec atomicity for multi-byte
  // wasm accesses on POWER ISA. POWER permits unaligned page-spanning
  // accesses to commit one half before the other half takes a DSI; wasm
  // requires atomicity. Touching the last byte of the upcoming access
  // with a 1-byte lbzx triggers SIGSEGV (→ wasm trap via the signal
  // handler) before the actual access executes — POWER's precise-
  // interrupt model guarantees the subsequent access is never
  // architecturally executed if the probe faults.
  //
  // Wasm linear memory is one contiguous mapped region followed by an
  // mprotect'd guard, so last-byte-mapped ⇒ all-bytes-mapped, and a
  // single-byte probe is sufficient regardless of access size.
  //
  // No-op when HasPOWER9() (real POWER9/POWER10 silicon handles page-
  // spanning unaligned stores atomically at the µarch level), and when
  // access size is 1. Never called on the atomic path: atomic ops are
  // naturally aligned per wasm spec + ISA-enforced lwarx alignment, so
  // they cannot span pages; misaligned atomics take a precise SIGBUS
  // before any commit.
  //
  // 2 instructions when emitted (addi + lbzx).
  void wasmProbeLastByte(const wasm::MemoryAccessDesc& access,
                         Register memoryBase, Register ptr);
};

typedef MacroAssemblerPPC64Compat MacroAssemblerSpecific;

}  // namespace jit
}  // namespace js

#endif /* jit_ppc64_MacroAssembler_ppc64_h */
