/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "jit/ppc64/Assembler-ppc64.h"

#include "mozilla/DebugOnly.h"
#include "mozilla/Maybe.h"

#include "gc/Marking.h"
#include "jit/AutoWritableJitCode.h"
#include "jit/ExecutableAllocator.h"
#include "jit/FlushICache.h"

using mozilla::DebugOnly;

using namespace js;
using namespace js::jit;

// ELFv2 ABI: 8 GPRs (r3-r10), 13 FPRs (f1-f13).
// FP arguments also consume a GPR slot per ELFv2 convention.
ABIArg ABIArgGenerator::next(MIRType type) {
  switch (type) {
    case MIRType::Int32:
    case MIRType::Int64:
    case MIRType::Pointer:
    case MIRType::WasmAnyRef:
    case MIRType::WasmArrayData:
    case MIRType::StackResults: {
      if (intRegIndex_ >= NumIntArgRegs) {
        current_ = ABIArg(stackOffset_);
        stackOffset_ += sizeof(uintptr_t);
        break;
      }
      current_ = ABIArg(Register::FromCode(Registers::r3 + intRegIndex_));
      intRegIndex_++;
      break;
    }
    case MIRType::Float32:
    case MIRType::Double: {
      if (floatRegIndex_ == NumFloatArgRegs) {
        current_ = ABIArg(stackOffset_);
        stackOffset_ += sizeof(double);
        break;
      }
      current_ = ABIArg(FloatRegister(
          FloatRegisters::Encoding(FloatRegisters::f1 + floatRegIndex_),
          type == MIRType::Double ? FloatRegisters::Double
                                  : FloatRegisters::Single));
      floatRegIndex_++;
      // ELFv2 ABI: each FP arg also consumes a GPR slot (shadow).
      // Cap at NumIntArgRegs so subsequent int args go to the stack.
      if (intRegIndex_ < NumIntArgRegs) {
        intRegIndex_++;
      }
      break;
    }
    case MIRType::Simd128: {
      // Pass v128 in FP registers (Simd128 kind). On PPC64 ELFv2, SIMD
      // values use the same VSR register file as FP args.
      if (floatRegIndex_ == NumFloatArgRegs) {
        current_ = ABIArg(stackOffset_);
        stackOffset_ += 16;
        break;
      }
      current_ = ABIArg(FloatRegister(
          FloatRegisters::Encoding(FloatRegisters::f1 + floatRegIndex_),
          FloatRegisters::Simd128));
      floatRegIndex_++;
      if (intRegIndex_ < NumIntArgRegs) {
        intRegIndex_++;
      }
      break;
    }
    default:
      MOZ_CRASH("Unexpected argument type");
  }
  return current_;
}

// Condition inversion tables.
Assembler::Condition Assembler::InvertCondition(Condition cond) {
  switch (cond) {
    case Equal:
      return NotEqual;
    case NotEqual:
      return Equal;
    case LessThan:
      return GreaterThanOrEqual;
    case LessThanOrEqual:
      return GreaterThan;
    case GreaterThan:
      return LessThanOrEqual;
    case GreaterThanOrEqual:
      return LessThan;
    case Above:
      return BelowOrEqual;
    case AboveOrEqual:
      return Below;
    case Below:
      return AboveOrEqual;
    case BelowOrEqual:
      return Above;
    case Zero:
      return NonZero;
    case NonZero:
      return Zero;
    case Signed:
      return NotSigned;
    case NotSigned:
      return Signed;
    case SOBit:
      return NSOBit;
    case NSOBit:
      return SOBit;
    case Overflow:
      return NotOverflow;
    case NotOverflow:
      return Overflow;
    case CarrySet:
      return CarryClear;
    case CarryClear:
      return CarrySet;
    default:
      MOZ_CRASH("unexpected condition");
  }
}

Assembler::DoubleCondition Assembler::InvertCondition(DoubleCondition cond) {
  switch (cond) {
    case DoubleOrdered:
      return DoubleUnordered;
    case DoubleEqual:
      return DoubleNotEqualOrUnordered;
    case DoubleNotEqual:
      return DoubleEqualOrUnordered;
    case DoubleGreaterThan:
      return DoubleLessThanOrEqualOrUnordered;
    case DoubleGreaterThanOrEqual:
      return DoubleLessThanOrUnordered;
    case DoubleLessThan:
      return DoubleGreaterThanOrEqualOrUnordered;
    case DoubleLessThanOrEqual:
      return DoubleGreaterThanOrUnordered;
    case DoubleUnordered:
      return DoubleOrdered;
    case DoubleEqualOrUnordered:
      return DoubleNotEqual;
    case DoubleNotEqualOrUnordered:
      return DoubleEqual;
    case DoubleGreaterThanOrUnordered:
      return DoubleLessThanOrEqual;
    case DoubleGreaterThanOrEqualOrUnordered:
      return DoubleLessThan;
    case DoubleLessThanOrUnordered:
      return DoubleGreaterThanOrEqual;
    case DoubleLessThanOrEqualOrUnordered:
      return DoubleGreaterThan;
    default:
      MOZ_CRASH("unexpected condition");
  }
}

// InstImm helper.
uint8_t InstImm::traptag() {
  uint8_t r = ((data & 0x001f0000) >> 16);
  MOZ_ASSERT(isOpcode(PPC_tw));
  MOZ_ASSERT(r == ((data & 0x0000f800) >> 11));
  return r & 0xfe;
}

BOffImm16::BOffImm16(InstImm inst) : data(inst.extractImm16Value() & 0xFFFC) {
  // Sign-extend the 16-bit field.
  if (data & 0x8000) {
    data |= ~0xFFFF;
  }
}

Instruction* BOffImm16::getDest(Instruction* src) const {
  return (Instruction*)((uint8_t*)src + data);
}

Instruction* JOffImm26::getDest(Instruction* src) const {
  return (Instruction*)((uint8_t*)src + data);
}

Imm16::Imm16() : value(0) {}

Imm8::Imm8() : value(0) {}

// Buffer management.
bool Assembler::oom() const {
  return AssemblerShared::oom() || m_buffer.oom() || jumpRelocations_.oom() ||
         dataRelocations_.oom();
}

void Assembler::finish() {
  MOZ_ASSERT(!isFinished);
  isFinished = true;
  m_buffer.flushPool();
}

bool Assembler::appendRawCode(const uint8_t* code, size_t numBytes) {
  return m_buffer.appendRawCode(code, numBytes);
}

bool Assembler::reserve(size_t size) {
  // Fixed-size chunk buffer; no point in reserving now vs. on-demand.
  return !oom();
}

bool Assembler::swapBuffer(wasm::Bytes& bytes) {
  MOZ_ASSERT(bytes.empty());
  if (!bytes.resize(bytesNeeded())) {
    return false;
  }
  m_buffer.executableCopy(bytes.begin());
  return true;
}

void Assembler::copyJumpRelocationTable(uint8_t* dest) {
  if (jumpRelocations_.length()) {
    memcpy(dest, jumpRelocations_.buffer(), jumpRelocations_.length());
  }
}

void Assembler::copyDataRelocationTable(uint8_t* dest) {
  if (dataRelocations_.length()) {
    memcpy(dest, dataRelocations_.buffer(), dataRelocations_.length());
  }
}

void Assembler::executableCopy(void* buffer) {
  MOZ_ASSERT(isFinished);
  m_buffer.executableCopy(static_cast<uint8_t*>(buffer));
}

void Assembler::executableCopy(uint8_t* buffer) {
  MOZ_ASSERT(isFinished);
  m_buffer.executableCopy(buffer);
}

size_t Assembler::size() const {
  // AssemblerBufferWithConstantPools::size() asserts pool is empty.
  // Flush pending pool entries first.
  const_cast<PPCBufferWithExecutableCopy&>(m_buffer).flushPool();
  return m_buffer.size();
}

size_t Assembler::jumpRelocationTableBytes() const {
  return jumpRelocations_.length();
}

size_t Assembler::dataRelocationTableBytes() const {
  return dataRelocations_.length();
}

size_t Assembler::bytesNeeded() const {
  return size() + jumpRelocationTableBytes() + dataRelocationTableBytes();
}

// Write an instruction into the buffer or to an external destination.
BufferOffset Assembler::writeInst(uint32_t x, uint32_t* dest) {
  MOZ_ASSERT(hasCreator());
  if (dest == nullptr) {
    return m_buffer.putInt(x);
  }

  WriteInstStatic(x, dest);
  return BufferOffset();
}

void Assembler::WriteInstStatic(uint32_t x, uint32_t* dest) {
  MOZ_ASSERT(dest != nullptr);
  *dest = x;
}

// Alignment.
BufferOffset Assembler::haltingAlign(int alignment) {
  BufferOffset ret;
  MOZ_ASSERT(m_buffer.isAligned(4));
  if (alignment == 8) {
    if (!m_buffer.isAligned(alignment)) {
      BufferOffset tmp = xs_trap();
      if (!ret.assigned()) {
        ret = tmp;
      }
    }
  } else {
    MOZ_ASSERT((alignment & (alignment - 1)) == 0);
    while (size() & (alignment - 1)) {
      BufferOffset tmp = xs_trap();
      if (!ret.assigned()) {
        ret = tmp;
      }
    }
  }
  return ret;
}

BufferOffset Assembler::nopAlign(int alignment) {
  BufferOffset ret;
  MOZ_ASSERT(m_buffer.isAligned(4));
  if (alignment == 8) {
    if (!m_buffer.isAligned(alignment)) {
      BufferOffset tmp = as_nop();
      if (!ret.assigned()) {
        ret = tmp;
      }
    }
  } else {
    MOZ_ASSERT((alignment & (alignment - 1)) == 0);
    while (size() & (alignment - 1)) {
      BufferOffset tmp = as_nop();
      if (!ret.assigned()) {
        ret = tmp;
      }
    }
  }
  return ret;
}

// Primitive instructions.
BufferOffset Assembler::as_nop() {
  spew("nop");
  return writeInst(PPC_nop);
}

BufferOffset Assembler::as_lwsync() {
  spew("lwsync");
  return writeInst(PPC_lwsync);
}

BufferOffset Assembler::as_sync() {
  spew("sync");
  return writeInst(PPC_sync);
}

BufferOffset Assembler::as_isync() {
  spew("isync");
  return writeInst(PPC_isync);
}

// Branch and jump instructions.
BufferOffset Assembler::as_b(JOffImm26 off, BranchAddressType bat, LinkBit lb) {
  return as_b(off.encode(), bat, lb);
}

BufferOffset Assembler::as_b(int32_t off, BranchAddressType bat, LinkBit lb) {
  spew("b%s%s\t%x", bat == AbsoluteBranch ? "a" : "", lb ? "l" : "", off);
  MOZ_ASSERT(!(off & 0x03));
  return writeInst(PPC_b | ((uint32_t)off & 0x3fffffc) | bat | lb);
}

BufferOffset Assembler::as_blr(LinkBit lb) {
  spew("blr%s", lb ? "l" : "");
  return writeInst(uint32_t(PPC_blr) | uint32_t(lb));
}

BufferOffset Assembler::as_bctr(LinkBit lb) {
  spew("bctr%s", lb ? "l" : "");
  return writeInst(uint32_t(PPC_bctr) | uint32_t(lb));
}

// Conditional branches.
BufferOffset Assembler::as_bc(BOffImm16 off, Condition cond, CRegisterID cr,
                              LikelyBit lkb, LinkBit lb) {
  return as_bc(off.encode(), cond, cr, lkb, lb);
}

BufferOffset Assembler::as_bc(int16_t off, Condition cond, CRegisterID cr,
                              LikelyBit lkb, LinkBit lb) {
  return as_bc(off, computeConditionCode(cond, cr), lkb, lb);
}

BufferOffset Assembler::as_bc(BOffImm16 off, DoubleCondition cond,
                              CRegisterID cr, LikelyBit lkb, LinkBit lb) {
  return as_bc(off.encode(), cond, cr, lkb, lb);
}

BufferOffset Assembler::as_bc(int16_t off, DoubleCondition cond, CRegisterID cr,
                              LikelyBit lkb, LinkBit lb) {
  return as_bc(off, computeConditionCode(cond, cr), lkb, lb);
}

BufferOffset Assembler::as_bcctr(Condition cond, CRegisterID cr, LikelyBit lkb,
                                 LinkBit lb) {
  return as_bcctr(computeConditionCode(cond, cr), lkb, lb);
}

BufferOffset Assembler::as_bcctr(DoubleCondition cond, CRegisterID cr,
                                 LikelyBit lkb, LinkBit lb) {
  return as_bcctr(computeConditionCode(cond, cr), lkb, lb);
}

// Condition code computation: turn DoubleCondition + CR into BO|BI.
// May emit CR logic instructions for synthetic conditions involving FU bit.
uint16_t Assembler::computeConditionCode(DoubleCondition op, CRegisterID cr) {
  const uint8_t condBit = crBit(cr, op);
  const uint8_t fuBit = crBit(cr, DoubleUnordered);
  uint32_t newop = (uint32_t)op & 255;

  if (op & DoubleConditionUnordered) {
    if ((uint32_t(op) & BranchOptionMask) == BranchOnClear) {
      as_crorc(condBit, fuBit, condBit);
      newop |= BranchOnSet;
    } else {
      if (condBit != fuBit) {
        as_cror(condBit, fuBit, condBit);
      }
    }
  } else {
    if ((uint32_t(op) & BranchOptionMask) == BranchOnClear) {
      if (condBit != fuBit) {
        as_cror(condBit, fuBit, condBit);
      }
    } else {
      if (condBit != fuBit) {
        as_crandc(condBit, condBit, fuBit);
      }
    }
  }

  return (newop + ((uint8_t)cr << 6));
}

// Condition code computation: turn Condition + CR into BO|BI.
// May emit mcrxrx for XER-mediated conditions.
uint16_t Assembler::computeConditionCode(Condition op, CRegisterID cr) {
  uint32_t newop = (uint32_t)op & 255;

  if (op & ConditionOnlyXER) {
    MOZ_ASSERT(op == Overflow || op == NotOverflow);
    if (HasPOWER9()) {
      as_mcrxrx(cr);
    } else {
      // POWER8: read XER, place OV into the GT position of the target
      // CR field. Overflow condition (0x1c = GreaterThan) tests GT bit,
      // which mcrxrx populates with OV32. For 64-bit ops OV == OV32.
      // XER layout in GPR low 32 bits (IBM): bit 0=SO, 1=OV, 2=CA.
      // Target: GT position = IBM bit 4*cr+1.
      xs_mfxer(r0);
      int gtBit = 4 * (int)cr + 1;          // GT position in CR field
      int sh = (1 - gtBit) & 31;            // rotate OV from bit 1 to gtBit
      as_rlwinm(r0, r0, sh, gtBit, gtBit);  // isolate OV at GT only
      as_mtcrf(1 << (7 - (int)cr), r0);
    }
    newop = (uint32_t)op & 255;
  }

  return (newop + ((uint8_t)cr << 6));
}

// Given BO|BI in a 16-bit quantity, split into bit fields for instruction.
static uint32_t makeOpMask(uint16_t op) {
  MOZ_ASSERT(!(op & 0xfc00));
  return ((op & 0x0f) << 21) | ((op & 0xfff0) << 12);
}

BufferOffset Assembler::as_bc(int16_t off, uint16_t op, LikelyBit lkb,
                              LinkBit lb) {
  spew("bc%s%s\tBO_BI=0x%04x,%d", lb ? "l" : "", lkb ? "+" : "", op, off);
  MOZ_ASSERT(!(off & 0x03));
  return writeInst(Instruction(PPC_bc | makeOpMask(op) | lkb << 21 |
                               ((uint16_t)off & 0xfffc) | lb)
                       .encode());
}

BufferOffset Assembler::as_bcctr(uint16_t op, LikelyBit lkb, LinkBit lb) {
  spew("bcctr%s%s", lb ? "l" : "", lkb ? "+" : "");
  return writeInst(PPC_bcctr | makeOpMask(op) | lkb << 21 | lb);
}

// SPR operations.
BufferOffset Assembler::as_mtspr(SPRegisterID spr, Register ra) {
  spew("mtspr\t%d,%3s", spr, ra.name());
  return writeInst(PPC_mtspr | ra.code() << 21 | PPC_SPR(spr));
}

BufferOffset Assembler::as_mfspr(Register rd, SPRegisterID spr) {
  spew("mfspr\t%3s,%d", rd.name(), spr);
  return writeInst(PPC_mfspr | rd.code() << 21 | PPC_SPR(spr));
}

// CR operations.
#define DEF_CRCR(op)                                                 \
  BufferOffset Assembler::as_##op(uint8_t t, uint8_t a, uint8_t b) { \
    spew(#op "\t%d,%d,%d", t, a, b);                                 \
    return writeInst(PPC_##op | t << 21 | a << 16 | b << 11);        \
  }
DEF_CRCR(crandc)
DEF_CRCR(cror)
DEF_CRCR(crorc)
#undef DEF_CRCR

BufferOffset Assembler::as_mtcrf(uint32_t mask, Register rs) {
  spew("mtcrf\t%d,%3s", mask, rs.name());
  return writeInst(PPC_mtcrf | rs.code() << 21 | mask << 12);
}

BufferOffset Assembler::as_mfocrf(Register rd, CRegisterID crfs) {
  spew("mfocrf\t%3s,cr%d", rd.name(), crfs);
  // FXM is a one-hot 8-bit mask at bits 12-19. Bit (7-crfs) selects the CR.
  return writeInst(PPC_mfocrf | rd.code() << 21 | (1 << (7 - crfs)) << 12);
}

BufferOffset Assembler::as_mcrxrx(CRegisterID cr) {
  spew("mcrxrx\tcr%d", cr);
  return writeInst(PPC_mcrxrx | cr << 23);
}

// GPR neg.
BufferOffset Assembler::as_neg(Register rd, Register rs) {
  spew("neg\t%3s,%3s", rd.name(), rs.name());
  return writeInst(InstReg(PPC_neg, rd, rs, r0).encode());
}

// Compare instructions.
BufferOffset Assembler::as_cmpd(CRegisterID cr, Register ra, Register rb) {
  spew("cmpd\tcr%d,%3s,%3s", cr, ra.name(), rb.name());
  return writeInst(PPC_cmpd | cr << 23 | ra.code() << 16 | rb.code() << 11);
}

BufferOffset Assembler::as_cmpdi(CRegisterID cr, Register ra, int16_t im) {
  spew("cmpdi\tcr%d,%3s,%d", cr, ra.name(), im);
  return writeInst(PPC_cmpdi | cr << 23 | ra.code() << 16 |
                   ((uint16_t)im & 0xffff));
}

BufferOffset Assembler::as_cmpld(CRegisterID cr, Register ra, Register rb) {
  spew("cmpld\tcr%d,%3s,%3s", cr, ra.name(), rb.name());
  return writeInst(PPC_cmpld | cr << 23 | ra.code() << 16 | rb.code() << 11);
}

BufferOffset Assembler::as_cmpldi(CRegisterID cr, Register ra, int16_t im) {
  spew("cmpldi\tcr%d,%3s,%d", cr, ra.name(), im);
  return writeInst(PPC_cmpldi | cr << 23 | ra.code() << 16 |
                   ((uint16_t)im & 0xffff));
}

BufferOffset Assembler::as_cmpw(CRegisterID cr, Register ra, Register rb) {
  spew("cmpw\tcr%d,%3s,%3s", cr, ra.name(), rb.name());
  return writeInst(PPC_cmpw | cr << 23 | ra.code() << 16 | rb.code() << 11);
}

BufferOffset Assembler::as_cmpwi(CRegisterID cr, Register ra, int16_t im) {
  spew("cmpwi\tcr%d,%3s,%d", cr, ra.name(), im);
  return writeInst(PPC_cmpwi | cr << 23 | ra.code() << 16 |
                   ((uint16_t)im & 0xffff));
}

BufferOffset Assembler::as_cmplw(CRegisterID cr, Register ra, Register rb) {
  spew("cmplw\tcr%d,%3s,%3s", cr, ra.name(), rb.name());
  return writeInst(PPC_cmplw | cr << 23 | ra.code() << 16 | rb.code() << 11);
}

BufferOffset Assembler::as_cmplwi(CRegisterID cr, Register ra, int16_t im) {
  spew("cmplwi\tcr%d,%3s,%d", cr, ra.name(), im);
  return writeInst(PPC_cmplwi | cr << 23 | ra.code() << 16 |
                   ((uint16_t)im & 0xffff));
}

// Compare instructions (cr0 implicit).
BufferOffset Assembler::as_cmpd(Register ra, Register rb) {
  spew("cmpd\t%3s,%3s", ra.name(), rb.name());
  return writeInst(PPC_cmpd | ra.code() << 16 | rb.code() << 11);
}

BufferOffset Assembler::as_cmpdi(Register ra, int16_t im) {
  spew("cmpdi\t%3s,%d", ra.name(), im);
  return writeInst(PPC_cmpdi | ra.code() << 16 | ((uint16_t)im & 0xffff));
}

BufferOffset Assembler::as_cmpld(Register ra, Register rb) {
  spew("cmpld\t%3s,%3s", ra.name(), rb.name());
  return writeInst(PPC_cmpld | ra.code() << 16 | rb.code() << 11);
}

BufferOffset Assembler::as_cmpldi(Register ra, int16_t im) {
  spew("cmpldi\t%3s,%d", ra.name(), im);
  return writeInst(PPC_cmpldi | ra.code() << 16 | ((uint16_t)im & 0xffff));
}

BufferOffset Assembler::as_cmpw(Register ra, Register rb) {
  spew("cmpw\t%3s,%3s", ra.name(), rb.name());
  return writeInst(PPC_cmpw | ra.code() << 16 | rb.code() << 11);
}

BufferOffset Assembler::as_cmpwi(Register ra, int16_t im) {
  spew("cmpwi\t%3s,%d", ra.name(), im);
  return writeInst(PPC_cmpwi | ra.code() << 16 | ((uint16_t)im & 0xffff));
}

BufferOffset Assembler::as_cmplw(Register ra, Register rb) {
  spew("cmplw\t%3s,%3s", ra.name(), rb.name());
  return writeInst(PPC_cmplw | ra.code() << 16 | rb.code() << 11);
}

BufferOffset Assembler::as_cmplwi(Register ra, int16_t im) {
  spew("cmplwi\t%3s,%d", ra.name(), im);
  return writeInst(PPC_cmplwi | ra.code() << 16 | ((uint16_t)im & 0xffff));
}

// FP encoding helpers.
static uint32_t AForm(uint32_t op, FloatRegister frt, FloatRegister fra,
                      FloatRegister frb, FloatRegister frc, bool rc) {
  return (op | (frt.encoding() << 21) | (fra.encoding() << 16) |
          (frb.encoding() << 11) | (frc.encoding() << 6) | rc);
}

static uint32_t XForm(uint32_t op, FloatRegister frt, FloatRegister fra,
                      FloatRegister frb, bool rc) {
  return (op | (frt.encoding() << 21) | (fra.encoding() << 16) |
          (frb.encoding() << 11) | rc);
}

static uint32_t XForm(uint32_t op, FloatRegister frt, Register ra, Register rb,
                      bool rc) {
  return (op | (frt.encoding() << 21) | (ra.code() << 16) | (rb.code() << 11) |
          rc);
}

static uint32_t DForm(uint32_t op, FloatRegister frt, Register ra,
                      int16_t imm) {
  return (op | (frt.encoding() << 21) | (ra.code() << 16) |
          ((uint16_t)imm & 0xffff));
}

// XX-form encoders. Each form has its own X-bit positions.
// All take uint32_t encodings (0-63) so they correctly
// emit the high bit for VSR32-63. FloatRegister.encoding() returns 0-31
// for Single/Double (= VSR0-31 = FPR namespace) and 32-63 for Simd128
// (= VSR32-63 = VR namespace) — so a single XX-form encoder addresses
// the full VSR space.

// XX1-form: T + GPR (RA) + GPR (RB). TX bit at instruction bit 0.
// Used by lxvx, stxvx, lxvd2x, stxvd2x, mtvsrdd, mtvsrd, mtvsrws, mtvsrwz.
static uint32_t XX1Form(uint32_t op, uint32_t xt, uint32_t ra, uint32_t rb) {
  return op | (xt & 31) << 21 | (ra & 31) << 16 | (rb & 31) << 11 |
         ((xt >> 5) & 1);
}

// XX1-form for mfvsrX: GPR (RT) + VSR (XS). TX bit ("SX") at instruction
// bit 0; the X spec calls this SX since the source register is the VSR.
// Used by mfvsrd, mfvsrld.
static uint32_t XX1FormMfvsr(uint32_t op, uint32_t rt, uint32_t xs) {
  return op | (xs & 31) << 21 | (rt & 31) << 16 | ((xs >> 5) & 1);
}

// XX2-form: T + B (no A field; bits 16-20 unused or hold a UIM). BX bit
// at instruction bit 1, TX bit at instruction bit 0. The bits16-20 slot
// is set by callers — for plain XX2 it must be 0, for XX2 with UIM it
// holds the immediate.
// Used by xxbrd, xxbrh, xxbrw, xxbrq, xscvdpsp, xscvspdp, xscvdpspn,
// xscvspdpn, xxspltw (UIM=2 bits), xxinsertw (UIM=4 bits),
// xxextractuw (UIM=4 bits), xvabs*/xvneg*/xvsqrt*/xvr* etc. via
// DEF_VSX_UN.
static uint32_t XX2Form(uint32_t op, uint32_t xt, uint32_t xb,
                        uint32_t bits16to20 = 0) {
  return op | (xt & 31) << 21 | (bits16to20 & 31) << 16 | (xb & 31) << 11 |
         ((xb >> 5) & 1) << 1 | ((xt >> 5) & 1);
}

// XX3-form: T + A + B. AX/BX/TX bits at instruction bits 2/1/0.
// Used by xxlor, xxland, xxlxor, xxlnor, xxlandc, xxpermdi, xsmaxjdp,
// xsminjdp, xvadd*, xvcmp*, etc.
static uint32_t XX3Form(uint32_t op, uint32_t xt, uint32_t xa, uint32_t xb) {
  return op | (xt & 31) << 21 | (xa & 31) << 16 | (xb & 31) << 11 |
         ((xa >> 5) & 1) << 2 | ((xb >> 5) & 1) << 1 | ((xt >> 5) & 1);
}

// XX4-form: T + A + B + C. CX/AX/BX/TX bits at instruction bits 3/2/1/0.
// Used by xxsel.
static uint32_t XX4Form(uint32_t op, uint32_t xt, uint32_t xa, uint32_t xb,
                        uint32_t xc) {
  return op | (xt & 31) << 21 | (xa & 31) << 16 | (xb & 31) << 11 |
         (xc & 31) << 6 | ((xc >> 5) & 1) << 3 | ((xa >> 5) & 1) << 2 |
         ((xb >> 5) & 1) << 1 | ((xt >> 5) & 1);
}

// FloatRegister convenience overload for XX3Form (the most common form).
static uint32_t XX3Form(uint32_t op, FloatRegister xt, FloatRegister xa,
                        FloatRegister xb) {
  return XX3Form(op, uint32_t(xt.encoding()), uint32_t(xa.encoding()),
                 uint32_t(xb.encoding()));
}

// --- Macro-defined instruction emitters ---

// X-form: rd in bits 21-25, ra in 16-20, rb in 11-15.
#define DEF_XFORM(op)                                                      \
  BufferOffset Assembler::as_##op(Register rd, Register ra, Register rb) { \
    spew(#op "\t%3s,%3s,%3s", rd.name(), ra.name(), rb.name());            \
    return writeInst(InstReg(PPC_##op, rd, ra, rb).encode());              \
  }

#define DEF_XFORM_RC(op)                                            \
  BufferOffset Assembler::as_##op##_rc(Register rd, Register ra,    \
                                       Register rb) {               \
    spew(#op ".\t%3s,%3s,%3s", rd.name(), ra.name(), rb.name());    \
    return writeInst(InstReg(PPC_##op, rd, ra, rb).encode() | 0x1); \
  }

// X-form with swapped RS/RA encoding: rs in bits 21-25, ra in 16-20.
#define DEF_XFORMS(op)                                                     \
  BufferOffset Assembler::as_##op(Register rd, Register ra, Register rb) { \
    spew(#op "\t%3s,%3s,%3s", rd.name(), ra.name(), rb.name());            \
    return writeInst(InstReg(PPC_##op, ra, rd, rb).encode());              \
  }

#define DEF_XFORMS_RC(op)                                           \
  BufferOffset Assembler::as_##op##_rc(Register rd, Register ra,    \
                                       Register rb) {               \
    spew(#op ".\t%3s,%3s,%3s", rd.name(), ra.name(), rb.name());    \
    return writeInst(InstReg(PPC_##op, ra, rd, rb).encode() | 0x1); \
  }

// X-form shift immediate with swapped encoding.
#define DEF_XFORMS_I(op)                                                       \
  BufferOffset Assembler::as_##op(Register rd, Register ra, uint8_t sh) {      \
    spew(#op "\t%3s,%3s,%d", rd.name(), ra.name(), sh);                        \
    MOZ_ASSERT(sh < 32);                                                       \
    return writeInst(PPC_##op | ra.code() << 21 | rd.code() << 16 | sh << 11); \
  }

// 2-reg X-form: rd in bits 21-25, ra in 16-20, rb=r0.
#define DEF_XFORM2(op)                                        \
  BufferOffset Assembler::as_##op(Register rd, Register ra) { \
    spew(#op "\t%3s,%3s", rd.name(), ra.name());              \
    return writeInst(InstReg(PPC_##op, rd, ra, r0).encode()); \
  }

#define DEF_XFORM2_RC(op)                                           \
  BufferOffset Assembler::as_##op##_rc(Register rd, Register ra) {  \
    spew(#op ".\t%3s,%3s", rd.name(), ra.name());                   \
    return writeInst(InstReg(PPC_##op, rd, ra, r0).encode() | 0x1); \
  }

// 2-reg X-form swapped: ra in bits 21-25, rd in 16-20.
#define DEF_XFORM2S(op)                                       \
  BufferOffset Assembler::as_##op(Register rd, Register ra) { \
    spew(#op "\t%3s,%3s", rd.name(), ra.name());              \
    return writeInst(InstReg(PPC_##op, ra, rd, r0).encode()); \
  }

#define DEF_XFORM2S_RC(op)                                          \
  BufferOffset Assembler::as_##op##_rc(Register rd, Register ra) {  \
    spew(#op ".\t%3s,%3s", rd.name(), ra.name());                   \
    return writeInst(InstReg(PPC_##op, ra, rd, r0).encode() | 0x1); \
  }

// D-form load/store: rd=RT, rb=RA (base register), off=displacement.
// r0 cannot be used as base register for D-form loads/stores.
#define DEF_DFORM(op)                                                      \
  BufferOffset Assembler::as_##op(Register rd, Register rb, int16_t off) { \
    spew(#op "\t%3s,%d(%3s)", rd.name(), off, rb.name());                  \
    MOZ_ASSERT(rb != r0);                                                  \
    return writeInst(InstImm(PPC_##op, rd, rb, off).encode());             \
  }

// D-form with swapped RS/RA encoding for logical immediates.
#define DEF_DFORMS(op)                                                     \
  BufferOffset Assembler::as_##op(Register rd, Register ra, uint16_t im) { \
    spew(#op "\t%3s,%3s,%d", rd.name(), ra.name(), im);                    \
    return writeInst(InstImm(PPC_##op, ra, rd, im).encode());              \
  }

// M-form: rotate with 3 registers + mb + me.
#define DEF_MFORM(op)                                                         \
  BufferOffset Assembler::as_##op(Register rd, Register rs, Register rb,      \
                                  uint8_t mb, uint8_t me) {                   \
    spew(#op "\t%3s,%3s,%3s,%d,%d", rd.name(), rs.name(), rb.name(), mb, me); \
    MOZ_ASSERT(mb < 32);                                                      \
    MOZ_ASSERT(me < 32);                                                      \
    return writeInst(PPC_##op | rs.code() << 21 | rd.code() << 16 |           \
                     rb.code() << 11 | mb << 6 | me << 1);                    \
  }

// M-form with immediate shift.
#define DEF_MFORM_I(op)                                                        \
  BufferOffset Assembler::as_##op(Register rd, Register rs, uint8_t sh,        \
                                  uint8_t mb, uint8_t me) {                    \
    spew(#op "\t%3s,%3s,%d,%d,%d", rd.name(), rs.name(), sh, mb, me);          \
    MOZ_ASSERT(sh < 32);                                                       \
    MOZ_ASSERT(mb < 32);                                                       \
    MOZ_ASSERT(me < 32);                                                       \
    return writeInst(PPC_##op | rs.code() << 21 | rd.code() << 16 | sh << 11 | \
                     mb << 6 | me << 1);                                       \
  }

#define DEF_MFORM_I_RC(op)                                                     \
  BufferOffset Assembler::as_##op##_rc(Register rd, Register rs, uint8_t sh,   \
                                       uint8_t mb, uint8_t me) {               \
    spew(#op ".\t%3s,%3s,%d,%d,%d", rd.name(), rs.name(), sh, mb, me);         \
    MOZ_ASSERT(sh < 32);                                                       \
    MOZ_ASSERT(mb < 32);                                                       \
    MOZ_ASSERT(me < 32);                                                       \
    return writeInst(PPC_##op | rs.code() << 21 | rd.code() << 16 | sh << 11 | \
                     mb << 6 | me << 1 | 1);                                   \
  }

// MDS-form: rotate with register + mb (64-bit).
#define DEF_MDSFORM(op)                                                   \
  BufferOffset Assembler::as_##op(Register ra, Register rs, Register rb,  \
                                  uint8_t mb) {                           \
    spew(#op "\t%3s,%3s,%3s,%d", ra.name(), rs.name(), rb.name(), mb);    \
    MOZ_ASSERT(mb < 64);                                                  \
    return writeInst(PPC_##op | rs.code() << 21 | ra.code() << 16 |       \
                     rb.code() << 11 | ((mb & 0x1f) << 6) | (mb & 0x20)); \
  }

#define DEF_MDSFORM_RC(op)                                                    \
  BufferOffset Assembler::as_##op##_rc(Register ra, Register rs, Register rb, \
                                       uint8_t mb) {                          \
    spew(#op ".\t%3s,%3s,%3s,%d", ra.name(), rs.name(), rb.name(), mb);       \
    MOZ_ASSERT(mb < 64);                                                      \
    return writeInst(PPC_##op | rs.code() << 21 | ra.code() << 16 |           \
                     rb.code() << 11 | ((mb & 0x1f) << 6) | (mb & 0x20) | 1); \
  }

// MD-form: rotate/shift with immediate sh + mb (64-bit).
// sh and mb are 6-bit fields split across the instruction word.
#define DEF_MDFORM(op)                                                        \
  BufferOffset Assembler::as_##op(Register ra, Register rs, uint8_t sh,       \
                                  uint8_t mb) {                               \
    spew(#op "\t%3s,%3s,%d,%d", ra.name(), rs.name(), sh, mb);                \
    MOZ_ASSERT(sh < 64);                                                      \
    MOZ_ASSERT(mb < 64);                                                      \
    return writeInst(PPC_##op | rs.code() << 21 | ra.code() << 16 |           \
                     ((sh & 0x1f) << 11) | ((mb & 0x1f) << 6) | (mb & 0x20) | \
                     ((sh & 0x20) >> 4));                                     \
  }

#define DEF_MDFORM_RC(op)                                                     \
  BufferOffset Assembler::as_##op##_rc(Register ra, Register rs, uint8_t sh,  \
                                       uint8_t mb) {                          \
    spew(#op ".\t%3s,%3s,%d,%d", ra.name(), rs.name(), sh, mb);               \
    MOZ_ASSERT(sh < 64);                                                      \
    MOZ_ASSERT(mb < 64);                                                      \
    return writeInst(PPC_##op | rs.code() << 21 | ra.code() << 16 |           \
                     ((sh & 0x1f) << 11) | ((mb & 0x1f) << 6) | (mb & 0x20) | \
                     ((sh & 0x20) >> 4) | 0x01);                              \
  }

// FP 2-reg X-form: frt in bits 21-25, fra=f0, frb in 11-15.
#define DEF_XFORM2_F(op)                                                \
  BufferOffset Assembler::as_##op(FloatRegister rd, FloatRegister ra) { \
    spew(#op "\t%3s,%3s", rd.name(), ra.name());                        \
    return writeInst(XForm(PPC_##op, rd, f0, ra, false));               \
  }

#define DEF_XFORM2_F_RC(op)                                                  \
  BufferOffset Assembler::as_##op##_rc(FloatRegister rd, FloatRegister ra) { \
    spew(#op ".\t%3s,%3s", rd.name(), ra.name());                            \
    return writeInst(XForm(PPC_##op, rd, f0, ra, true));                     \
  }

// FP A-form with frc (fmul-type): frt, fra, frc; frb=f0.
#define DEF_AFORM_C(op)                                               \
  BufferOffset Assembler::as_##op(FloatRegister rd, FloatRegister ra, \
                                  FloatRegister rc) {                 \
    spew(#op "\t%3s,%3s,%3s", rd.name(), ra.name(), rc.name());       \
    return writeInst(AForm(PPC_##op, rd, ra, f0, rc, false));         \
  }

#define DEF_AFORM_C_RC(op)                                                 \
  BufferOffset Assembler::as_##op##_rc(FloatRegister rd, FloatRegister ra, \
                                       FloatRegister rc) {                 \
    spew(#op ".\t%3s,%3s,%3s", rd.name(), ra.name(), rc.name());           \
    return writeInst(AForm(PPC_##op, rd, ra, f0, rc, true));               \
  }

// FP A-form with frb (fadd-type): frt, fra, frb; frc=f0.
#define DEF_AFORM_B(op)                                               \
  BufferOffset Assembler::as_##op(FloatRegister rd, FloatRegister ra, \
                                  FloatRegister rb) {                 \
    spew(#op "\t%3s,%3s,%3s", rd.name(), ra.name(), rb.name());       \
    return writeInst(AForm(PPC_##op, rd, ra, rb, f0, false));         \
  }

#define DEF_AFORM_B_RC(op)                                                 \
  BufferOffset Assembler::as_##op##_rc(FloatRegister rd, FloatRegister ra, \
                                       FloatRegister rb) {                 \
    spew(#op ".\t%3s,%3s,%3s", rd.name(), ra.name(), rb.name());           \
    return writeInst(AForm(PPC_##op, rd, ra, rb, f0, true));               \
  }

// Full FP A-form: frt, fra, frc, frb (fmadd-type).
#define DEF_AFORM(op)                                                          \
  BufferOffset Assembler::as_##op(FloatRegister rd, FloatRegister ra,          \
                                  FloatRegister rc, FloatRegister rb) {        \
    spew(#op "\t%3s,%3s,%3s,%3s", rd.name(), ra.name(), rc.name(), rb.name()); \
    return writeInst(AForm(PPC_##op, rd, ra, rb, rc, false));                  \
  }

#define DEF_AFORM_RC(op)                                                     \
  BufferOffset Assembler::as_##op##_rc(FloatRegister rd, FloatRegister ra,   \
                                       FloatRegister rc, FloatRegister rb) { \
    spew(#op ".\t%3s,%3s,%3s,%3s", rd.name(), ra.name(), rc.name(),          \
         rb.name());                                                         \
    return writeInst(AForm(PPC_##op, rd, ra, rb, rc, true));                 \
  }

// FP D-form load/store.
#define DEF_DFORM_F(op)                                          \
  BufferOffset Assembler::as_##op(FloatRegister rd, Register rb, \
                                  int16_t off) {                 \
    spew(#op "\t%3s,%d(%3s)", rd.name(), off, rb.name());        \
    MOZ_ASSERT(rb != r0);                                        \
    return writeInst(DForm(PPC_##op, rd, rb, off));              \
  }

// FP X-form indexed load/store.
#define DEF_FMEMx(op)                                            \
  BufferOffset Assembler::as_##op(FloatRegister rd, Register ra, \
                                  Register rb) {                 \
    spew(#op "\t%3s,%3s,%3s", rd.name(), ra.name(), rb.name());  \
    return writeInst(XForm(PPC_##op, rd, ra, rb, false));        \
  }

// --- Rotate/shift instructions ---

DEF_MFORM(rlwnm)
DEF_MFORM_I(rlwinm)
DEF_MFORM_I_RC(rlwinm)
DEF_MFORM_I(rlwimi)
DEF_XFORMS_I(srawi)

DEF_MDSFORM(rldcl)
DEF_MDFORM(rldicl)
DEF_MDFORM_RC(rldicl)
DEF_MDFORM(rldicr)
DEF_MDFORM_RC(rldicr)
DEF_MDFORM(rldimi)

BufferOffset Assembler::as_sradi(Register rd, Register rs, int sh) {
  spew("sradi\t%3s,%3s,%d", rd.name(), rs.name(), sh);
  MOZ_ASSERT(sh >= 0 && sh < 64);
  return writeInst(PPC_sradi | rd.code() << 16 | rs.code() << 21 |
                   (sh & 0x1f) << 11 | (sh & 0x20) >> 4);
}

// --- ALU three-register ---

#define DEF_ALU2(op) DEF_XFORM(op)

DEF_ALU2(add)
DEF_ALU2(addc)
DEF_ALU2(adde)
DEF_ALU2(subf)
DEF_ALU2(subfc)
DEF_ALU2(subfe)
DEF_ALU2(divd)
DEF_ALU2(divdu)
DEF_ALU2(divw)
DEF_ALU2(divwu)
// POWER9 modulo (XO-form, same encoding pattern as div).
DEF_XFORM(modsd)
DEF_XFORM(modsw)
DEF_XFORM(modud)
DEF_XFORM(moduw)
DEF_ALU2(mulld)
DEF_ALU2(mulhd)
DEF_ALU2(mulhdu)
DEF_ALU2(mulldo)
DEF_ALU2(mullw)
DEF_ALU2(mulhwu)
#undef DEF_ALU2

// --- ALU immediate ---

// D-form ALU-immediate ops have no Rc bit at instruction LSB (that bit
// is part of the 16-bit immediate). The only valid record-form variant
// in this group is `addic.`, which is a separate primary opcode (13)
// hand-written below; subfic and mulli have no record form at all.
#define DEF_ALUI(op)                                                      \
  BufferOffset Assembler::as_##op(Register rd, Register ra, int16_t im) { \
    spew(#op "\t%3s,%3s,%d", rd.name(), ra.name(), im);                   \
    return writeInst(InstImm(PPC_##op, rd, ra, im).encode());             \
  }

BufferOffset Assembler::as_addi(Register rd, Register ra, int16_t im,
                                bool actually_li) {
#ifdef DEBUG
  if (actually_li) {
    spew("li\t%3s,%d", rd.name(), im);
  } else {
    MOZ_ASSERT(ra != r0);
    spew("addi\t%3s,%3s,%d", rd.name(), ra.name(), im);
  }
#endif
  return writeInst(InstImm(PPC_addi, rd, ra, im).encode());
}

BufferOffset Assembler::as_addis(Register rd, Register ra, int16_t im,
                                 bool actually_lis) {
#ifdef DEBUG
  if (actually_lis) {
    spew("lis\t%3s,%d", rd.name(), im);
  } else {
    MOZ_ASSERT(ra != r0);
    spew("addis\t%3s,%3s,%d", rd.name(), ra.name(), im);
  }
#endif
  return writeInst(InstImm(PPC_addis, rd, ra, im).encode());
}

DEF_ALUI(mulli)
DEF_ALUI(subfic)
#undef DEF_ALUI

// --- ALU unary/extended ---


#define DEF_ALUE_S(op) DEF_XFORM2S(op)
DEF_ALUE_S(cntlzw)
DEF_ALUE_S(cntlzd)
DEF_ALUE_S(cnttzd)
DEF_ALUE_S(cnttzw)
#undef DEF_ALUE_S

DEF_XFORM2S(popcntd)
DEF_XFORM2S(popcntw)
DEF_XFORM2S(brd)  // POWER10
DEF_XFORM2S(brh)  // POWER10
DEF_XFORM2S(brw)  // POWER10

// --- Bitwise logical (three-register) ---

#define DEF_BITALU2(op) DEF_XFORMS(op)
DEF_BITALU2(nor)
DEF_BITALU2(slw)
DEF_BITALU2(srw)
DEF_BITALU2(sraw)
DEF_BITALU2(sld)
DEF_BITALU2(srd)
DEF_BITALU2(srad)
#undef DEF_BITALU2

// and_, or_, xor_ are manually defined (trailing underscore to avoid C++
// keyword conflicts). xs_mr delegates to as_or_ so we must not assert
// rd==rs==rb in as_or_ (which would be a valid mr).
BufferOffset Assembler::as_or_(Register rd, Register rs, Register rb) {
  spew("or\t%3s,%3s,%3s", rd.name(), rs.name(), rb.name());
  return writeInst(InstReg(PPC_or_, rs, rd, rb).encode());
}

BufferOffset Assembler::as_xor_(Register rd, Register rs, Register rb) {
  spew("xor\t%3s,%3s,%3s", rd.name(), rs.name(), rb.name());
  return writeInst(InstReg(PPC_xor_, rs, rd, rb).encode());
}

BufferOffset Assembler::as_and_(Register rd, Register rs, Register rb) {
  spew("and\t%3s,%3s,%3s", rd.name(), rs.name(), rb.name());
  return writeInst(InstReg(PPC_and_, rs, rd, rb).encode());
}

BufferOffset Assembler::as_and__rc(Register rd, Register rs, Register rb) {
  spew("and.\t%3s,%3s,%3s", rd.name(), rs.name(), rb.name());
  return writeInst(InstReg(PPC_and_, rs, rd, rb).encode() | 0x1);
}

// --- Bitwise logical (immediate) ---

DEF_DFORMS(ori)
DEF_DFORMS(oris)
DEF_DFORMS(xori)
DEF_DFORMS(xoris)

BufferOffset Assembler::as_andi_rc(Register rd, Register ra, uint16_t im) {
  spew("andi.\t%3s,%3s,%d", rd.name(), ra.name(), im);
  return writeInst(InstImm(PPC_andi_dot, ra, rd, im).encode());
}

// --- Sign extension ---

#define DEF_ALUEXT(op) DEF_XFORM2S(op) DEF_XFORM2S_RC(op)
DEF_XFORM2S(extsb)
DEF_XFORM2S(extsh)
DEF_ALUEXT(extsw)
#undef DEF_ALUEXT

// --- Integer loads (D-form) ---

DEF_DFORM(lbz)
DEF_DFORM(lha)
DEF_DFORM(lhz)

BufferOffset Assembler::as_lwa(Register rd, Register rb, int16_t off) {
  spew("lwa\t%3s,%d(%3s)", rd.name(), off, rb.name());
  MOZ_ASSERT(rb != r0);
  MOZ_ASSERT(!(off & 0x03));
  return writeInst(InstImm(PPC_lwa, rd, rb, off).encode());
}

DEF_DFORM(lwz)

BufferOffset Assembler::as_ld(Register rd, Register rb, int16_t off) {
  spew("ld\t%3s,%d(%3s)", rd.name(), off, rb.name());
  MOZ_ASSERT(rb != r0);
  MOZ_ASSERT(!(off & 0x03));
  return writeInst(InstImm(PPC_ld, rd, rb, off).encode());
}

// --- Integer stores (D-form) ---

DEF_DFORM(stb)
DEF_DFORM(sth)
DEF_DFORM(stw)

BufferOffset Assembler::as_std(Register rd, Register rb, int16_t off) {
  spew("std\t%3s,%d(%3s)", rd.name(), off, rb.name());
  MOZ_ASSERT(rb != r0);
  MOZ_ASSERT(!(off & 0x03));
  return writeInst(InstImm(PPC_std, rd, rb, off).encode());
}

DEF_DFORM(stdu)

#undef DEF_DFORM
#undef DEF_DFORMS

// --- Integer loads/stores (X-form, indexed) ---

#define DEF_MEMx(op) DEF_XFORM(op)
DEF_MEMx(lbzx) DEF_MEMx(lhax) DEF_MEMx(lhzx) DEF_MEMx(lwax)
    DEF_MEMx(lwzx) DEF_MEMx(lwarx) DEF_MEMx(lbarx)
        DEF_MEMx(lharx) DEF_MEMx(ldx) DEF_MEMx(ldarx) DEF_MEMx(stbx)
            DEF_MEMx(stbcx) DEF_MEMx(stwx) DEF_MEMx(stwbrx) DEF_MEMx(sthx)
                DEF_MEMx(sthcx) DEF_MEMx(stdx) DEF_MEMx(stdcx)
                    DEF_MEMx(stwcx)
#undef DEF_MEMx

// --- Integer select ---

BufferOffset Assembler::as_isel(Register rt, Register ra, Register rb,
                                uint16_t bc, CRegisterID cr) {
  MOZ_ASSERT(ra != r0);
  return as_isel0(rt, ra, rb, bc, cr);
}

BufferOffset Assembler::as_isel0(Register rt, Register ra, Register rb,
                                 uint16_t bc, CRegisterID cr) {
  spew("isel\t%3s,%3s,%3s,cr%d:0x%02x", rt.name(), ra.name(), rb.name(), cr,
       bc);
  MOZ_ASSERT((bc < 0x40) && ((bc & 0x0f) == 0x0c));
  uint16_t nbc = (bc >> 4) + (cr << 2);
  return writeInst(PPC_isel | rt.code() << 21 | ra.code() << 16 |
                   rb.code() << 11 | nbc << 6);
}

BufferOffset Assembler::as_setbc(Register rt, uint16_t bc, CRegisterID cr) {
  spew("setbc\t%3s,cr%d:0x%02x", rt.name(), cr, bc);
  MOZ_ASSERT((bc < 0x40) && ((bc & 0x0f) == 0x0c));
  uint16_t nbc = (bc >> 4) + (cr << 2);
  return writeInst(PPC_setbc | (rt.code() << 21) | (nbc << 16));
}

BufferOffset Assembler::as_setbcr(Register rt, uint16_t bc, CRegisterID cr) {
  spew("setbcr\t%3s,cr%d:0x%02x", rt.name(), cr, bc);
  MOZ_ASSERT((bc < 0x40) && ((bc & 0x0f) == 0x0c));
  uint16_t nbc = (bc >> 4) + (cr << 2);
  return writeInst(PPC_setbcr | (rt.code() << 21) | (nbc << 16));
}

// --- FP compare ---

BufferOffset Assembler::as_fcmpu(CRegisterID cr, FloatRegister ra,
                                 FloatRegister rb) {
  spew("fcmpu\tcr%d,%3s,%3s", cr, ra.name(), rb.name());
  return writeInst(PPC_fcmpu | cr << 23 | ra.encoding() << 16 |
                   rb.encoding() << 11);
}

BufferOffset Assembler::as_fcmpu(FloatRegister ra, FloatRegister rb) {
  return as_fcmpu(cr0, ra, rb);
}

// --- FP arithmetic ---

#define DEF_FPUAC(op) DEF_AFORM_C(op)
DEF_FPUAC(fmul)
DEF_FPUAC(fmuls)
#undef DEF_FPUAC

#define DEF_FPUAB(op) DEF_AFORM_B(op)
DEF_FPUAB(fadd)
DEF_FPUAB(fdiv)
DEF_FPUAB(fsub)
DEF_FPUAB(fadds)
DEF_FPUAB(fdivs)
DEF_FPUAB(fsubs)
DEF_FPUAB(fcpsgn)
#undef DEF_FPUAB

// --- FP unary/conversion/rounding ---

#define DEF_FPUDS(op) DEF_XFORM2_F(op)
DEF_FPUDS(fabs)
DEF_FPUDS(fneg)
DEF_FPUDS(fmr)
DEF_FPUDS(fcfid)
DEF_FPUDS(fcfids)
DEF_FPUDS(fcfidu)
DEF_FPUDS(fcfidus)
DEF_FPUDS(fctid)
DEF_FPUDS(fctidz)
DEF_FPUDS(fctiduz)
DEF_FPUDS(fctiwz)
DEF_FPUDS(frim)
DEF_FPUDS(frip)
DEF_FPUDS(friz)
DEF_FPUDS(frsp)
DEF_FPUDS(fsqrt)
DEF_FPUDS(fsqrts)
#undef DEF_FPUDS

// --- FP loads/stores (D-form) ---

DEF_DFORM_F(lfd)
DEF_DFORM_F(lfs)
DEF_DFORM_F(stfd)
DEF_DFORM_F(stfs)
DEF_DFORM_F(stfdu)
DEF_DFORM_F(stfsu)

// --- FP loads/stores (X-form, indexed) ---

DEF_FMEMx(lfdx) DEF_FMEMx(lfsx) DEF_FMEMx(lfiwax)
    DEF_FMEMx(stfdx) DEF_FMEMx(stfsx)
// Clean up macros.
#undef DEF_XFORM
#undef DEF_XFORM_RC
#undef DEF_XFORMS
#undef DEF_XFORMS_RC
#undef DEF_XFORMS_I
#undef DEF_XFORM2
#undef DEF_XFORM2_RC
#undef DEF_XFORM2S
#undef DEF_XFORM2S_RC
#undef DEF_XFORM2_F
#undef DEF_XFORM2_F_RC
#undef DEF_MFORM
#undef DEF_MFORM_I
#undef DEF_MFORM_I_RC
#undef DEF_MDSFORM
#undef DEF_MDSFORM_RC
#undef DEF_MDFORM
#undef DEF_MDFORM_RC
#undef DEF_DFORM_F
#undef DEF_FMEMx
#undef DEF_AFORM_C
#undef DEF_AFORM_C_RC
#undef DEF_AFORM_B
#undef DEF_AFORM_B_RC
#undef DEF_AFORM
#undef DEF_AFORM_RC

    // --- FPSCR operations ---

    BufferOffset Assembler::as_mtfsb0(uint8_t bt) {
  spew("mtfsb0\t%d", bt);
  return writeInst(PPC_mtfsb0 | (uint32_t)bt << 21);
}

BufferOffset Assembler::as_mcrfs(CRegisterID bf, uint8_t bfa) {
  spew("mcrfs\tcr%d,%d", bf, bfa);
  return writeInst(PPC_mcrfs | (uint32_t)bf << 23 | (uint32_t)bfa << 18);
}

// --- VSX (FPR-only subset) ---

BufferOffset Assembler::as_mfvsrd(Register ra, FloatRegister xs) {
  spew("mfvsrd\t%3s,%3s", ra.name(), xs.name());
  return writeInst(XX1FormMfvsr(PPC_mfvsrd, ra.code(), xs.encoding()));
}

BufferOffset Assembler::as_mtvsrd(FloatRegister xt, Register ra) {
  spew("mtvsrd\t%3s,%3s", xt.name(), ra.name());
  return writeInst(XX1Form(PPC_mtvsrd, xt.encoding(), ra.code(), 0));
}

BufferOffset Assembler::as_mtvsrwa(FloatRegister xt, Register ra) {
  spew("mtvsrwa\t%3s,%3s", xt.name(), ra.name());
  return writeInst(XX1Form(PPC_mtvsrwa, xt.encoding(), ra.code(), 0));
}

BufferOffset Assembler::as_mtvsrws(FloatRegister xt, Register ra) {
  spew("mtvsrws\t%3s,%3s", xt.name(), ra.name());
  return writeInst(XX1Form(PPC_mtvsrws, xt.encoding(), ra.code(), 0));
}

BufferOffset Assembler::as_mtvsrwz(FloatRegister xt, Register ra) {
  spew("mtvsrwz\t%3s,%3s", xt.name(), ra.name());
  return writeInst(XX1Form(PPC_mtvsrwz, xt.encoding(), ra.code(), 0));
}

BufferOffset Assembler::as_xxbrd(FloatRegister xt, FloatRegister xb) {
  spew("xxbrd\t%3s,%3s", xt.name(), xb.name());
  return writeInst(XX2Form(PPC_xxbrd, xt.encoding(), xb.encoding()));
}

BufferOffset Assembler::as_xscvdpspn(FloatRegister xt, FloatRegister xb) {
  spew("xscvdpspn\t%3s,%3s", xt.name(), xb.name());
  return writeInst(XX2Form(PPC_xscvdpspn, xt.encoding(), xb.encoding()));
}

BufferOffset Assembler::as_xscvspdpn(FloatRegister xt, FloatRegister xb) {
  spew("xscvspdpn\t%3s,%3s", xt.name(), xb.name());
  return writeInst(XX2Form(PPC_xscvspdpn, xt.encoding(), xb.encoding()));
}

// POWER9 (ISA 3.0) scalar FP16 conversions. The UIM disambiguator is
// already in PPC_xscvdphp / PPC_xscvhpdp; XX2Form's bits16to20 default
// of 0 leaves it intact.
BufferOffset Assembler::as_xscvdphp(FloatRegister xt, FloatRegister xb) {
  spew("xscvdphp\t%3s,%3s", xt.name(), xb.name());
  return writeInst(XX2Form(PPC_xscvdphp, xt.encoding(), xb.encoding()));
}

BufferOffset Assembler::as_xscvhpdp(FloatRegister xt, FloatRegister xb) {
  spew("xscvhpdp\t%3s,%3s", xt.name(), xb.name());
  return writeInst(XX2Form(PPC_xscvhpdp, xt.encoding(), xb.encoding()));
}

BufferOffset Assembler::as_xsxexpdp(FloatRegister xt, FloatRegister xb) {
  spew("xsxexpdp\t%3s,%3s", xt.name(), xb.name());
  return writeInst(XX2Form(PPC_xsxexpdp, xt.encoding(), xb.encoding()));
}

// POWER9 (ISA 3.0) FP16 load/store, X-form indexed. lxsihzx loads
// 16 bits into VSR dword 0 word 1's low halfword (zeroing the rest);
// stxsihx stores from there. The XT[5]/XS[5] bit travels via the
// X-form's TX/SX bit at instruction bit 0.
BufferOffset Assembler::as_lxsihzx(FloatRegister xt, Register ra, Register rb) {
  spew("lxsihzx\t%3s,%3s,%3s", xt.name(), ra.name(), rb.name());
  return writeInst(PPC_lxsihzx | (xt.encoding() & 31) << 21 |
                   ra.code() << 16 | rb.code() << 11 |
                   ((xt.encoding() >> 5) & 1));
}

BufferOffset Assembler::as_stxsihx(FloatRegister xs, Register ra, Register rb) {
  spew("stxsihx\t%3s,%3s,%3s", xs.name(), ra.name(), rb.name());
  return writeInst(PPC_stxsihx | (xs.encoding() & 31) << 21 |
                   ra.code() << 16 | rb.code() << 11 |
                   ((xs.encoding() >> 5) & 1));
}

// XX3-form, FPR-space only (encoding 0..31 → VSR0..31, all AX/BX/TX = 0).
// Java/JavaScript-style scalar max/min — semantics verified to match
// ECMA-262 Math.max/Math.min including ±0 and NaN propagation. POWER9-only.
BufferOffset Assembler::as_xsmaxjdp(FloatRegister xt, FloatRegister xa,
                                    FloatRegister xb) {
  spew("xsmaxjdp\t%3s,%3s,%3s", xt.name(), xa.name(), xb.name());
  return writeInst(XX3Form(PPC_xsmaxjdp, xt, xa, xb));
}

BufferOffset Assembler::as_xsminjdp(FloatRegister xt, FloatRegister xa,
                                    FloatRegister xb) {
  spew("xsminjdp\t%3s,%3s,%3s", xt.name(), xa.name(), xb.name());
  return writeInst(XX3Form(PPC_xsminjdp, xt, xa, xb));
}

// --- VSX SIMD load/store ---

// For VSX0-31 (FPR), the 6th register bit (TX/SX/BX) is 0.
// X-form: opcode | T << 21 | A << 16 | B << 11 | xo | TX
// lxvx/stxvx are POWER9 (ISA 3.0). lxvd2x/stxvd2x are POWER8 (ISA 2.07).

BufferOffset Assembler::as_lxvx(FloatRegister xt, Register ra, Register rb) {
  spew("lxvx\t%3s,%3s,%3s", xt.name(), ra.name(), rb.name());
  return writeInst(XX1Form(PPC_lxvx, xt.encoding(), ra.code(), rb.code()));
}

BufferOffset Assembler::as_stxvx(FloatRegister xs, Register ra, Register rb) {
  spew("stxvx\t%3s,%3s,%3s", xs.name(), ra.name(), rb.name());
  return writeInst(XX1Form(PPC_stxvx, xs.encoding(), ra.code(), rb.code()));
}

BufferOffset Assembler::as_lxvd2x(FloatRegister xt, Register ra, Register rb) {
  spew("lxvd2x\t%3s,%3s,%3s", xt.name(), ra.name(), rb.name());
  return writeInst(XX1Form(PPC_lxvd2x, xt.encoding(), ra.code(), rb.code()));
}

BufferOffset Assembler::as_stxvd2x(FloatRegister xs, Register ra, Register rb) {
  spew("stxvd2x\t%3s,%3s,%3s", xs.name(), ra.name(), rb.name());
  return writeInst(XX1Form(PPC_stxvd2x, xs.encoding(), ra.code(), rb.code()));
}

// VMX register load/store. See PPC_lvx/PPC_stvx in Assembler-ppc64.h for
// the encoding rationale.
BufferOffset Assembler::as_lvx(uint8_t vrt, Register ra, Register rb) {
  MOZ_ASSERT(vrt < 32);
  spew("lvx\tvr%d,%3s,%3s", vrt, ra.name(), rb.name());
  return writeInst(PPC_lvx | uint32_t(vrt) << 21 | ra.code() << 16 |
                   rb.code() << 11);
}

BufferOffset Assembler::as_stvx(uint8_t vrs, Register ra, Register rb) {
  MOZ_ASSERT(vrs < 32);
  spew("stvx\tvr%d,%3s,%3s", vrs, ra.name(), rb.name());
  return writeInst(PPC_stvx | uint32_t(vrs) << 21 | ra.code() << 16 |
                   rb.code() << 11);
}

// --- VSX SIMD register operations ---

// XX3-form: opcode | T[0:4]<<21 | A[0:4]<<16 | B[0:4]<<11 | xo | AX | BX | TX
// where AX/BX/TX (bits 2/1/0) carry bit 5 of each 6-bit VSR index.
// Encoded by the XX3Form helper above for both VSR0-31 (Single/Double) and
// VSR32-63 (Simd128) operands.
BufferOffset Assembler::as_xxlor(FloatRegister xt, FloatRegister xa,
                                 FloatRegister xb) {
  spew("xxlor\t%3s,%3s,%3s", xt.name(), xa.name(), xb.name());
  return writeInst(XX3Form(PPC_xxlor, xt, xa, xb));
}

BufferOffset Assembler::as_xxland(FloatRegister xt, FloatRegister xa,
                                  FloatRegister xb) {
  spew("xxland\t%3s,%3s,%3s", xt.name(), xa.name(), xb.name());
  return writeInst(XX3Form(PPC_xxland, xt, xa, xb));
}

BufferOffset Assembler::as_xxlxor(FloatRegister xt, FloatRegister xa,
                                  FloatRegister xb) {
  spew("xxlxor\t%3s,%3s,%3s", xt.name(), xa.name(), xb.name());
  return writeInst(XX3Form(PPC_xxlxor, xt, xa, xb));
}

BufferOffset Assembler::as_xxlnor(FloatRegister xt, FloatRegister xa,
                                  FloatRegister xb) {
  spew("xxlnor\t%3s,%3s,%3s", xt.name(), xa.name(), xb.name());
  return writeInst(XX3Form(PPC_xxlnor, xt, xa, xb));
}

BufferOffset Assembler::as_xxlandc(FloatRegister xt, FloatRegister xa,
                                   FloatRegister xb) {
  spew("xxlandc\t%3s,%3s,%3s", xt.name(), xa.name(), xb.name());
  return writeInst(XX3Form(PPC_xxlandc, xt, xa, xb));
}

BufferOffset Assembler::as_xxsel(FloatRegister xt, FloatRegister xa,
                                 FloatRegister xb, FloatRegister xc) {
  spew("xxsel\t%3s,%3s,%3s,%3s", xt.name(), xa.name(), xb.name(), xc.name());
  return writeInst(XX4Form(PPC_xxsel, xt.encoding(), xa.encoding(),
                           xb.encoding(), xc.encoding()));
}

BufferOffset Assembler::as_xxpermdi(FloatRegister xt, FloatRegister xa,
                                    FloatRegister xb, uint8_t dm) {
  MOZ_ASSERT(dm < 4);
  spew("xxpermdi\t%3s,%3s,%3s,%d", xt.name(), xa.name(), xb.name(), dm);
  return writeInst(XX3Form(PPC_xxpermdi | (uint32_t(dm) << 8), xt, xa, xb));
}

// POWER9 (ISA 3.0). XX1-form with two GPR sources.
BufferOffset Assembler::as_mtvsrdd(FloatRegister xt, Register ra, Register rb) {
  spew("mtvsrdd\t%3s,%3s,%3s", xt.name(), ra.name(), rb.name());
  return writeInst(XX1Form(PPC_mtvsrdd, xt.encoding(), ra.code(), rb.code()));
}

// POWER9 (ISA 3.0). XX1-form: move lower doubleword of VSR to GPR.
BufferOffset Assembler::as_mfvsrld(Register rt, FloatRegister xs) {
  spew("mfvsrld\t%3s,%3s", rt.name(), xs.name());
  return writeInst(XX1FormMfvsr(PPC_mfvsrld, rt.code(), xs.encoding()));
}

// --- XX2-form VSX instructions ---

// XX2-form: opcode | T<<21 | UIM<<16_area | B<<11_area | XO<<2 | BX | TX
// For VSR0-31, BX=TX=0.

BufferOffset Assembler::as_xxspltw(FloatRegister xt, FloatRegister xb,
                                   uint8_t uim) {
  MOZ_ASSERT(uim < 4);
  spew("xxspltw\t%3s,%3s,%d", xt.name(), xb.name(), uim);
  return writeInst(XX2Form(PPC_xxspltw, xt.encoding(), xb.encoding(), uim));
}

BufferOffset Assembler::as_xxinsertw(FloatRegister xt, FloatRegister xb,
                                     uint8_t uim) {
  MOZ_ASSERT(uim <= 12 && (uim & 3) == 0);
  spew("xxinsertw\t%3s,%3s,%d", xt.name(), xb.name(), uim);
  return writeInst(XX2Form(PPC_xxinsertw, xt.encoding(), xb.encoding(), uim));
}

BufferOffset Assembler::as_xxextractuw(FloatRegister xt, FloatRegister xb,
                                       uint8_t uim) {
  MOZ_ASSERT(uim <= 12 && (uim & 3) == 0);
  spew("xxextractuw\t%3s,%3s,%d", xt.name(), xb.name(), uim);
  return writeInst(XX2Form(PPC_xxextractuw, xt.encoding(), xb.encoding(), uim));
}

// POWER9 (ISA 3.0). XX1-form-ish: T(5) + UIM8(8) + XO + TX. UIM8 occupies
// bits 18..11 (a non-standard slot that XX1Form doesn't fit), so encode
// inline. TX bit at instruction bit 0 selects the upper half of VSR
// space when xt.encoding() is in 32-63 (Simd128).
BufferOffset Assembler::as_xxspltib(FloatRegister xt, uint8_t imm8) {
  spew("xxspltib\t%3s,%u", xt.name(), imm8);
  uint32_t enc = uint32_t(xt.encoding());
  return writeInst(PPC_xxspltib | (enc & 31) << 21 | (uint32_t)imm8 << 11 |
                   ((enc >> 5) & 1));
}

// --- VMX instructions ---

// VX-form: (4<<26) | VRT<<21 | UIMM<<16 | VRB<<11 | XO
// VRT/VRB are 5-bit raw VR numbers (0-31). Simd128 FloatRegister.encoding()
// returns 32-63; masking with & 31 maps it back to the VR offset 0-31.
BufferOffset Assembler::as_vspltb(FloatRegister vrt, FloatRegister vrb,
                                  uint8_t uim) {
  MOZ_ASSERT(uim < 16);
  spew("vspltb\t%3s,%3s,%d", vrt.name(), vrb.name(), uim);
  return writeInst(PPC_vspltb | (vrt.encoding() & 31) << 21 |
                   (uint32_t)uim << 16 | (vrb.encoding() & 31) << 11);
}

BufferOffset Assembler::as_vsplth(FloatRegister vrt, FloatRegister vrb,
                                  uint8_t uim) {
  MOZ_ASSERT(uim < 8);
  spew("vsplth\t%3s,%3s,%d", vrt.name(), vrb.name(), uim);
  return writeInst(PPC_vsplth | (vrt.encoding() & 31) << 21 |
                   (uint32_t)uim << 16 | (vrb.encoding() & 31) << 11);
}

// VA-form: (4<<26) | VRT<<21 | VRA<<16 | VRB<<11 | SHB<<6 | XO(6-bit)
BufferOffset Assembler::as_vsldoi(FloatRegister vrt, FloatRegister vra,
                                  FloatRegister vrb, uint8_t shb) {
  MOZ_ASSERT(shb < 16);
  spew("vsldoi\t%3s,%3s,%3s,%d", vrt.name(), vra.name(), vrb.name(), shb);
  return writeInst(PPC_vsldoi | (vrt.encoding() & 31) << 21 |
                   (vra.encoding() & 31) << 16 | (vrb.encoding() & 31) << 11 |
                   (uint32_t)shb << 6);
}

// --- VMX integer arithmetic (VR registers only) ---

// VX-form: (4<<26) | VRT<<21 | VRA<<16 | VRB<<11 | XO
// The macro takes raw VR numbers (0-31).
#define DEF_VMX_VVV(op)                                                    \
  BufferOffset Assembler::as_##op(uint8_t vrt, uint8_t vra, uint8_t vrb) { \
    MOZ_ASSERT(vrt < 32 && vra < 32 && vrb < 32);                          \
    spew(#op "\tvr%d,vr%d,vr%d", vrt, vra, vrb);                           \
    return writeInst(PPC_##op | vrt << 21 | vra << 16 | vrb << 11);        \
  }

DEF_VMX_VVV(vaddubm)
DEF_VMX_VVV(vadduhm)
DEF_VMX_VVV(vadduwm)
DEF_VMX_VVV(vaddudm)
DEF_VMX_VVV(vsububm)
DEF_VMX_VVV(vsubuhm)
DEF_VMX_VVV(vsubuwm)
DEF_VMX_VVV(vsubudm)
DEF_VMX_VVV(vaddsbs)
DEF_VMX_VVV(vaddshs)
DEF_VMX_VVV(vaddubs)
DEF_VMX_VVV(vadduhs)
DEF_VMX_VVV(vsubsbs)
DEF_VMX_VVV(vsubshs)
DEF_VMX_VVV(vsububs)
DEF_VMX_VVV(vsubuhs)
DEF_VMX_VVV(vminsb)
DEF_VMX_VVV(vminsh)
DEF_VMX_VVV(vminsw)
DEF_VMX_VVV(vmaxsb)
DEF_VMX_VVV(vmaxsh)
DEF_VMX_VVV(vmaxsw)
DEF_VMX_VVV(vmaxsd)
DEF_VMX_VVV(vminub)
DEF_VMX_VVV(vminuh)
DEF_VMX_VVV(vminuw)
DEF_VMX_VVV(vmaxub)
DEF_VMX_VVV(vmaxuh)
DEF_VMX_VVV(vmaxuw)
DEF_VMX_VVV(vavgub)
DEF_VMX_VVV(vavguh)
DEF_VMX_VVV(vmuluwm)
DEF_VMX_VVV(vmulld)

DEF_VMX_VVV(vslb)
DEF_VMX_VVV(vslh)
DEF_VMX_VVV(vslw)
DEF_VMX_VVV(vsld)
DEF_VMX_VVV(vsrb)
DEF_VMX_VVV(vsrh)
DEF_VMX_VVV(vsrw)
DEF_VMX_VVV(vsrd)
DEF_VMX_VVV(vsrab)
DEF_VMX_VVV(vsrah)
DEF_VMX_VVV(vsraw)
DEF_VMX_VVV(vsrad)
DEF_VMX_VVV(vslo)
DEF_VMX_VVV(vsro)
DEF_VMX_VVV(vcmpequb)
DEF_VMX_VVV(vcmpequh)
DEF_VMX_VVV(vcmpequw)
DEF_VMX_VVV(vcmpequd)
DEF_VMX_VVV(vcmpgtsb)
DEF_VMX_VVV(vcmpgtsh)
DEF_VMX_VVV(vcmpgtsw)
DEF_VMX_VVV(vcmpgtsd)
DEF_VMX_VVV(vcmpgtub)
DEF_VMX_VVV(vcmpgtuh)
DEF_VMX_VVV(vcmpgtuw)
DEF_VMX_VVV(vcmpgtud)
// POWER9 (ISA 3.0). NotEqual compare; saves the xxlnor that vcmpequX needs.
DEF_VMX_VVV(vcmpneb)
DEF_VMX_VVV(vcmpneh)
DEF_VMX_VVV(vcmpnew)

// POWER8+ (ISA 2.07). vbpermq RT,RA,RB: bit-permute quadword.
DEF_VMX_VVV(vbpermq)

#undef DEF_VMX_VVV

// VC-form record forms: same as VX-form above with Rc bit (bit 10 LSB) set.
// vcmpXXX. sets CR6: LT = all-true, EQ = none-true.
#define DEF_VMX_VVV_RC(op)                                                  \
  BufferOffset Assembler::as_##op##_rc(uint8_t vrt, uint8_t vra,            \
                                       uint8_t vrb) {                       \
    MOZ_ASSERT(vrt < 32 && vra < 32 && vrb < 32);                           \
    spew(#op ".\tvr%d,vr%d,vr%d", vrt, vra, vrb);                           \
    return writeInst(PPC_##op | vrt << 21 | vra << 16 | vrb << 11 | 0x400); \
  }

DEF_VMX_VVV_RC(vcmpequb)
DEF_VMX_VVV_RC(vcmpequh)
DEF_VMX_VVV_RC(vcmpequw)
DEF_VMX_VVV_RC(vcmpequd)

#undef DEF_VMX_VVV_RC

// VSX float compare (XX3-form).
#define DEF_VSX_CMP(op)                                               \
  BufferOffset Assembler::as_##op(FloatRegister xt, FloatRegister xa, \
                                  FloatRegister xb) {                 \
    spew(#op "\t%3s,%3s,%3s", xt.name(), xa.name(), xb.name());       \
    return writeInst(XX3Form(PPC_##op, xt, xa, xb));                  \
  }

DEF_VSX_CMP(xvcmpeqsp)
DEF_VSX_CMP(xvcmpgtsp)
DEF_VSX_CMP(xvcmpgesp)
DEF_VSX_CMP(xvcmpeqdp)
DEF_VSX_CMP(xvcmpgtdp)
DEF_VSX_CMP(xvcmpgedp)

#undef DEF_VSX_CMP

// VSX float arithmetic (XX3-form binary).
#define DEF_VSX_BIN(op)                                               \
  BufferOffset Assembler::as_##op(FloatRegister xt, FloatRegister xa, \
                                  FloatRegister xb) {                 \
    spew(#op "\t%3s,%3s,%3s", xt.name(), xa.name(), xb.name());       \
    return writeInst(XX3Form(PPC_##op, xt, xa, xb));                  \
  }
DEF_VSX_BIN(xvaddsp)
DEF_VSX_BIN(xvadddp) DEF_VSX_BIN(xvsubsp) DEF_VSX_BIN(xvsubdp) DEF_VSX_BIN(
    xvmulsp) DEF_VSX_BIN(xvmuldp) DEF_VSX_BIN(xvdivsp) DEF_VSX_BIN(xvdivdp)
    DEF_VSX_BIN(xvminsp) DEF_VSX_BIN(xvmindp) DEF_VSX_BIN(xvmaxsp) DEF_VSX_BIN(
        xvmaxdp) DEF_VSX_BIN(xvmaddasp) DEF_VSX_BIN(xvmaddadp)
        DEF_VSX_BIN(xvnmsubasp) DEF_VSX_BIN(xvnmsubadp)
#undef DEF_VSX_BIN

// VSX unary (XX2-form): op | xt<<21 | xb<<11 | XO<<2
// XX2-form unary VSX op: T + B, no UIM. Uses XX2Form helper for TX/BX bits.
#define DEF_VSX_UN(op)                                                  \
  BufferOffset Assembler::as_##op(FloatRegister xt, FloatRegister xb) { \
    spew(#op "\t%3s,%3s", xt.name(), xb.name());                        \
    return writeInst(XX2Form(PPC_##op, xt.encoding(), xb.encoding()));  \
  }
            DEF_VSX_UN(xvabssp) DEF_VSX_UN(xvabsdp) DEF_VSX_UN(
                xvnegsp) DEF_VSX_UN(xvnegdp) DEF_VSX_UN(xvsqrtsp)
                DEF_VSX_UN(xvsqrtdp) DEF_VSX_UN(xvrspip) DEF_VSX_UN(
                    xvrdpip) DEF_VSX_UN(xvrspim) DEF_VSX_UN(xvrdpim)
                    DEF_VSX_UN(xvrspiz) DEF_VSX_UN(xvrdpiz) DEF_VSX_UN(
                        xvrspic) DEF_VSX_UN(xvrdpic) DEF_VSX_UN(xvcvsxwsp)
                        DEF_VSX_UN(xvcvuxwsp) DEF_VSX_UN(xvcvsxwdp) DEF_VSX_UN(
                            xvcvuxwdp) DEF_VSX_UN(xvcvspsxws)
                            DEF_VSX_UN(xvcvspuxws) DEF_VSX_UN(xvcvdpsxws)
                                DEF_VSX_UN(xvcvdpuxws) DEF_VSX_UN(xvcvdpsp)
                                    DEF_VSX_UN(xvcvspdp)
#undef DEF_VSX_UN

// VMX unary VX-form: (4<<26) | VRT<<21 | 0<<16 | VRB<<11 | XO
#define DEF_VMX_UNARY(op)                                     \
  BufferOffset Assembler::as_##op(uint8_t vrt, uint8_t vrb) { \
    MOZ_ASSERT(vrt < 32 && vrb < 32);                         \
    spew(#op "\tvr%d,vr%d", vrt, vrb);                        \
    return writeInst(PPC_##op | vrt << 21 | vrb << 11);       \
  }
                                        DEF_VMX_UNARY(vupkhsb) DEF_VMX_UNARY(
                                            vupklsb) DEF_VMX_UNARY(vupkhsh)
                                            DEF_VMX_UNARY(vupklsh)
                                                DEF_VMX_UNARY(vupkhsw)
                                                    DEF_VMX_UNARY(vupklsw)
    // POWER9 per-lane integer negate. The VRA field holds the subop code
    // (6 for vnegw, 7 for vnegd) which is already baked into PPC_vneg{w,d}.
    DEF_VMX_UNARY(vnegw) DEF_VMX_UNARY(vnegd) DEF_VMX_UNARY(vpopcntb)
#undef DEF_VMX_UNARY

    // POWER9 addpcis (DX-form). Computes rT = (CIA + 4) + (D << 16).
    // D is a 16-bit signed immediate, split across three instruction fields:
    //   d0 = bits 16..25 (10 bits, D[15:6])
    //   d1 = bits 11..15 (5 bits,  D[5:1])
    //   d2 = bit 31      (1 bit,   D[0])
    // Primary opcode 19, DX subop 2.
    BufferOffset Assembler::as_addpcis(Register rt, int16_t d) {
  spew("addpcis\t%s,%d", rt.name(), (int)d);
  uint32_t D = uint16_t(d);
  uint32_t inst = (19u << 26) | (uint32_t(rt.code()) << 21) |
                  ((D >> 1) & 0x1F) << 16 | ((D >> 6) & 0x3FF) << 6 |
                  (2u << 1) | (D & 1u);
  return writeInst(inst);
}

// -----------------------------------------------------------------------------
// Power ISA v3.1 (POWER10) prefixed instructions.
//
// Layout:
//
//   Prefix word (BE bit numbering from the manual; LE bits in parentheses):
//     [0..5]   primary opcode = 1   (LE 31..26)
//     [6..7]   Type: 00 = 8LS, 10 = MLS   (LE 25..24)
//     [8..10]  reserved = 0   (LE 23..21)
//     [11]     R: 1 = PC-relative (RA must be r0)   (LE 20)
//     [12..13] reserved = 0   (LE 19..18)
//     [14..31] d0: high 18 bits of 34-bit signed immediate   (LE 17..0)
//
//   Suffix (paddi/pld, GPR target):
//     [0..5]   suffix opcode (paddi=14, pld=57)   (LE 31..26)
//     [6..10]  RT   (LE 25..21)
//     [11..15] RA   (LE 20..16)
//     [16..31] d1: low 16 bits of immediate   (LE 15..0)
//
//   Suffix (plxv, VSR target — has the TX bit at suffix bit 5/LE bit 26):
//     [0..4]   plxv 5-bit opcode = 11001 (=25)   (LE 31..27)
//     [5]      TX (high bit of 6-bit XT)   (LE 26)
//     [6..10]  T  (low 5 bits of XT)   (LE 25..21)
//     [11..15] RA   (LE 20..16)
//     [16..31] d1   (LE 15..0)
//
// The prefix and suffix of a prefixed instruction must lie in the same
// 64-byte aligned block at **runtime**. The JitCode allocator only
// guarantees 16-byte alignment, so the buffer-relative offset and the
// runtime address can differ by 0/16/32/48 mod 64. A buffer-only check
// `(currentOffset() & 63) == 60` is correct when the allocator base is
// 64-aligned but misses three of the four 16-aligned base classes — pad
// whenever `(currentOffset() & 15) == 12`, which catches all four. The
// enterNoPool guard prevents the constant-pool flusher from inserting
// bodies between the (optional) nop, prefix, and suffix.

static uint32_t EncodePower10Prefix(uint32_t type, bool R, uint32_t d0) {
  MOZ_ASSERT(type == 0 || type == 2);  // 8LS=0, MLS=2
  MOZ_ASSERT(d0 < (1u << 18));
  return (1u << 26) | (type << 24) | (uint32_t(R ? 1 : 0) << 20) |
         (d0 & 0x3FFFFu);
}

static void SplitImm34(int64_t imm34, uint32_t* d0, uint32_t* d1) {
  MOZ_ASSERT(imm34 >= -(int64_t(1) << 33));
  MOZ_ASSERT(imm34 < (int64_t(1) << 33));
  uint64_t u = uint64_t(imm34) & 0x3FFFFFFFFull;  // low 34 bits
  *d0 = uint32_t(u >> 16) & 0x3FFFFu;             // 18 bits
  *d1 = uint32_t(u) & 0xFFFFu;                    // 16 bits
}

void Assembler::ensurePrefixedAlignment() {
  if ((currentOffset() & 15) == 12) {
    as_nop();
  }
}

// paddi RT, RA, SI, R   (MLS, suffix opcode 14 = addi)
//   R=0: RT = (RA==0 ? 0 : RA) + sign_extend(SI, 34)
//   R=1: RT = CIA(prefix) + sign_extend(SI, 34)   (RA must be r0)
BufferOffset Assembler::as_paddi(Register rt, Register ra, int64_t imm34,
                                  bool R) {
  MOZ_ASSERT_IF(R, ra == r0);
  spew("paddi\t%s,%s,%lld,%d", rt.name(), ra.name(), (long long)imm34,
       R ? 1 : 0);
  uint32_t d0, d1;
  SplitImm34(imm34, &d0, &d1);
  uint32_t prefix = EncodePower10Prefix(/*type=MLS*/ 2, R, d0);
  uint32_t suffix = (14u << 26) | (uint32_t(rt.code()) << 21) |
                    (uint32_t(ra.code()) << 16) | d1;
  // Reservation = nop (worst case) + prefix + suffix.
  m_buffer.enterNoPool(3);
  ensurePrefixedAlignment();
  BufferOffset bo = writeInst(prefix);
  writeInst(suffix);
  m_buffer.leaveNoPool();
  return bo;
}

// pld RT, D(RA), R   (8LS, suffix opcode 57)
BufferOffset Assembler::as_pld(Register rt, Register ra, int64_t imm34,
                                bool R) {
  MOZ_ASSERT_IF(R, ra == r0);
  spew("pld\t%s,%lld(%s),%d", rt.name(), (long long)imm34, ra.name(),
       R ? 1 : 0);
  uint32_t d0, d1;
  SplitImm34(imm34, &d0, &d1);
  uint32_t prefix = EncodePower10Prefix(/*type=8LS*/ 0, R, d0);
  uint32_t suffix = (57u << 26) | (uint32_t(rt.code()) << 21) |
                    (uint32_t(ra.code()) << 16) | d1;
  m_buffer.enterNoPool(3);
  ensurePrefixedAlignment();
  BufferOffset bo = writeInst(prefix);
  writeInst(suffix);
  m_buffer.leaveNoPool();
  return bo;
}

// plxv XT, D(RA), R   (8LS, 5-bit suffix opcode 25, TX in suffix bit 26)
//   XT is 6-bit: TX (high) || T (low 5) — matches lxvx convention.
BufferOffset Assembler::as_plxv(uint8_t xt, Register ra, int64_t imm34,
                                 bool R) {
  MOZ_ASSERT(xt < 64);
  MOZ_ASSERT_IF(R, ra == r0);
  spew("plxv\tvs%u,%lld(%s),%d", xt, (long long)imm34, ra.name(),
       R ? 1 : 0);
  uint32_t d0, d1;
  SplitImm34(imm34, &d0, &d1);
  uint32_t prefix = EncodePower10Prefix(/*type=8LS*/ 0, R, d0);
  uint32_t T = xt & 0x1Fu;
  uint32_t TX = (xt >> 5) & 1u;
  uint32_t suffix = (25u << 27) | (TX << 26) | (T << 21) |
                    (uint32_t(ra.code()) << 16) | d1;
  m_buffer.enterNoPool(3);
  ensurePrefixedAlignment();
  BufferOffset bo = writeInst(prefix);
  writeInst(suffix);
  m_buffer.leaveNoPool();
  return bo;
}

// plfd FRT, D(RA), R   (MLS, suffix opcode 50; D-form-like FPR load)
BufferOffset Assembler::as_plfd(FloatRegister frt, Register ra, int64_t imm34,
                                 bool R) {
  MOZ_ASSERT_IF(R, ra == r0);
  spew("plfd\tf%u,%lld(%s),%d", uint32_t(frt.encoding()),
       (long long)imm34, ra.name(), R ? 1 : 0);
  uint32_t d0, d1;
  SplitImm34(imm34, &d0, &d1);
  uint32_t prefix = EncodePower10Prefix(/*type=MLS*/ 2, R, d0);
  uint32_t suffix = (50u << 26) | (uint32_t(frt.encoding()) << 21) |
                    (uint32_t(ra.code()) << 16) | d1;
  m_buffer.enterNoPool(3);
  ensurePrefixedAlignment();
  BufferOffset bo = writeInst(prefix);
  writeInst(suffix);
  m_buffer.leaveNoPool();
  return bo;
}

// plfs FRT, D(RA), R   (MLS, suffix opcode 48; widens single → double in FPR)
BufferOffset Assembler::as_plfs(FloatRegister frt, Register ra, int64_t imm34,
                                 bool R) {
  MOZ_ASSERT_IF(R, ra == r0);
  spew("plfs\tf%u,%lld(%s),%d", uint32_t(frt.encoding()),
       (long long)imm34, ra.name(), R ? 1 : 0);
  uint32_t d0, d1;
  SplitImm34(imm34, &d0, &d1);
  uint32_t prefix = EncodePower10Prefix(/*type=MLS*/ 2, R, d0);
  uint32_t suffix = (48u << 26) | (uint32_t(frt.encoding()) << 21) |
                    (uint32_t(ra.code()) << 16) | d1;
  m_buffer.enterNoPool(3);
  ensurePrefixedAlignment();
  BufferOffset bo = writeInst(prefix);
  writeInst(suffix);
  m_buffer.leaveNoPool();
  return bo;
}

// pstd RS, D(RA), R   (8LS, suffix opcode 61 = std D-form)
BufferOffset Assembler::as_pstd(Register rs, Register ra, int64_t imm34,
                                 bool R) {
  MOZ_ASSERT_IF(R, ra == r0);
  spew("pstd\t%s,%lld(%s),%d", rs.name(), (long long)imm34, ra.name(),
       R ? 1 : 0);
  uint32_t d0, d1;
  SplitImm34(imm34, &d0, &d1);
  uint32_t prefix = EncodePower10Prefix(/*type=8LS*/ 0, R, d0);
  uint32_t suffix = (61u << 26) | (uint32_t(rs.code()) << 21) |
                    (uint32_t(ra.code()) << 16) | d1;
  m_buffer.enterNoPool(3);
  ensurePrefixedAlignment();
  BufferOffset bo = writeInst(prefix);
  writeInst(suffix);
  m_buffer.leaveNoPool();
  return bo;
}

// pstxv XS, D(RA), R   (8LS, 5-bit suffix opcode 27, SX in suffix bit 26)
//   XS is 6-bit: SX (high) || S (low 5) — matches stxvx convention.
BufferOffset Assembler::as_pstxv(uint8_t xs, Register ra, int64_t imm34,
                                  bool R) {
  MOZ_ASSERT(xs < 64);
  MOZ_ASSERT_IF(R, ra == r0);
  spew("pstxv\tvs%u,%lld(%s),%d", xs, (long long)imm34, ra.name(),
       R ? 1 : 0);
  uint32_t d0, d1;
  SplitImm34(imm34, &d0, &d1);
  uint32_t prefix = EncodePower10Prefix(/*type=8LS*/ 0, R, d0);
  uint32_t sx = (xs >> 5) & 1;
  uint32_t s = xs & 0x1F;
  uint32_t suffix = (27u << 27) | (sx << 26) | (s << 21) |
                    (uint32_t(ra.code()) << 16) | d1;
  m_buffer.enterNoPool(3);
  ensurePrefixedAlignment();
  BufferOffset bo = writeInst(prefix);
  writeInst(suffix);
  m_buffer.leaveNoPool();
  return bo;
}

// pstfd FRS, D(RA), R   (MLS, suffix opcode 54 = stfd)
BufferOffset Assembler::as_pstfd(FloatRegister frs, Register ra, int64_t imm34,
                                  bool R) {
  MOZ_ASSERT_IF(R, ra == r0);
  spew("pstfd\tf%u,%lld(%s),%d", uint32_t(frs.encoding()),
       (long long)imm34, ra.name(), R ? 1 : 0);
  uint32_t d0, d1;
  SplitImm34(imm34, &d0, &d1);
  uint32_t prefix = EncodePower10Prefix(/*type=MLS*/ 2, R, d0);
  uint32_t suffix = (54u << 26) | (uint32_t(frs.encoding()) << 21) |
                    (uint32_t(ra.code()) << 16) | d1;
  m_buffer.enterNoPool(3);
  ensurePrefixedAlignment();
  BufferOffset bo = writeInst(prefix);
  writeInst(suffix);
  m_buffer.leaveNoPool();
  return bo;
}

// pstfs FRS, D(RA), R   (MLS, suffix opcode 52 = stfs)
BufferOffset Assembler::as_pstfs(FloatRegister frs, Register ra, int64_t imm34,
                                  bool R) {
  MOZ_ASSERT_IF(R, ra == r0);
  spew("pstfs\tf%u,%lld(%s),%d", uint32_t(frs.encoding()),
       (long long)imm34, ra.name(), R ? 1 : 0);
  uint32_t d0, d1;
  SplitImm34(imm34, &d0, &d1);
  uint32_t prefix = EncodePower10Prefix(/*type=MLS*/ 2, R, d0);
  uint32_t suffix = (52u << 26) | (uint32_t(frs.encoding()) << 21) |
                    (uint32_t(ra.code()) << 16) | d1;
  m_buffer.enterNoPool(3);
  ensurePrefixedAlignment();
  BufferOffset bo = writeInst(prefix);
  writeInst(suffix);
  m_buffer.leaveNoPool();
  return bo;
}

// POWER10 (ISA 3.1) Vector Extract Mask. RT (GPR) gets the wasm-spec
// bitmask (one bit per lane MSB) directly in low 16/8/4/2 bits. UIM
// is baked into PPC_vextract{b,h,w,d}m (8/9/10/11). Caller must have
// verified HasPOWER10().
#define DEF_VEXTRACT_M(op)                                                 \
  BufferOffset Assembler::as_##op(Register rt, FloatRegister vrb) {        \
    spew(#op "\t%s,vr%u", rt.name(), uint32_t(vrb.encoding() & 31));       \
    return writeInst(PPC_##op | (uint32_t(rt.code()) << 21) |              \
                     ((uint32_t(vrb.encoding()) & 31) << 11));             \
  }
DEF_VEXTRACT_M(vextractbm)
DEF_VEXTRACT_M(vextracthm)
DEF_VEXTRACT_M(vextractwm)
DEF_VEXTRACT_M(vextractdm)
#undef DEF_VEXTRACT_M

// POWER10 (ISA 3.1) Vector Insert Word/Doubleword from GPR. VX-form:
// VRT at bits 21..25, UIM at bits 16..20, RB at bits 11..15.
#define DEF_VINS(op, max_uim)                                              \
  BufferOffset Assembler::as_##op(FloatRegister vrt, Register rb,          \
                                  uint8_t uim) {                           \
    MOZ_ASSERT(uim <= (max_uim));                                          \
    spew(#op "\tvr%u,%s,%u", uint32_t(vrt.encoding() & 31), rb.name(),     \
         uint32_t(uim));                                                   \
    return writeInst(PPC_##op |                                            \
                     ((uint32_t(vrt.encoding()) & 31) << 21) |             \
                     (uint32_t(uim) << 16) |                               \
                     (uint32_t(rb.code()) << 11));                         \
  }
DEF_VINS(vinsw, 12)
DEF_VINS(vinsd, 8)
#undef DEF_VINS

// POWER10 (ISA 3.1) Vector Insert byte/halfword from GPR with
// register-supplied byte position. VX-form: VRT at bits 21..25,
// RA at bits 16..20, RB at bits 11..15. "rx" is right-indexed
// (LE-natural — index 0 = LSB byte).
#define DEF_VINS_RX(op)                                                    \
  BufferOffset Assembler::as_##op(FloatRegister vrt, Register ra,          \
                                  Register rb) {                           \
    spew(#op "\tvr%u,%s,%s", uint32_t(vrt.encoding() & 31), ra.name(),     \
         rb.name());                                                       \
    return writeInst(PPC_##op |                                            \
                     ((uint32_t(vrt.encoding()) & 31) << 21) |             \
                     (uint32_t(ra.code()) << 16) |                         \
                     (uint32_t(rb.code()) << 11));                         \
  }
DEF_VINS_RX(vinsbrx)
DEF_VINS_RX(vinshrx)
#undef DEF_VINS_RX

// POWER9 (ISA 3.0) V-form 3-operand instructions with VRT, UIM, VRB at
// bits 21..25, 16..20, 11..15 respectively (vinsert{b,h}, vextract{ub,uh}).
// Simd128 lives in VSR32-63 (= VR0-31), so we mask VRT and VRB to the
// 5-bit VR field via `encoding() & 31`.
#define DEF_VRT_UIM_VRB(op, max_uim, uim_step)                              \
  BufferOffset Assembler::as_##op(FloatRegister vrt, FloatRegister vrb,    \
                                  uint8_t uim) {                           \
    MOZ_ASSERT(uim <= (max_uim));                                          \
    MOZ_ASSERT((uim) % (uim_step) == 0);                                   \
    spew(#op "\tvr%u,vr%u,%u", uint32_t(vrt.encoding() & 31),              \
         uint32_t(vrb.encoding() & 31), uint32_t(uim));                    \
    return writeInst(PPC_##op |                                            \
                     ((uint32_t(vrt.encoding()) & 31) << 21) |             \
                     (uint32_t(uim) << 16) |                               \
                     ((uint32_t(vrb.encoding()) & 31) << 11));             \
  }
DEF_VRT_UIM_VRB(vinsertb, 15, 1)
DEF_VRT_UIM_VRB(vinserth, 14, 2)
DEF_VRT_UIM_VRB(vextractub, 15, 1)
DEF_VRT_UIM_VRB(vextractuh, 14, 2)
#undef DEF_VRT_UIM_VRB

// VMX binary VX-form pack/merge (re-use DEF_VMX_VVV pattern).
#define DEF_VMX_VVV(op)                                                    \
  BufferOffset Assembler::as_##op(uint8_t vrt, uint8_t vra, uint8_t vrb) { \
    MOZ_ASSERT(vrt < 32 && vra < 32 && vrb < 32);                          \
    spew(#op "\tvr%d,vr%d,vr%d", vrt, vra, vrb);                           \
    return writeInst(PPC_##op | vrt << 21 | vra << 16 | vrb << 11);        \
  }
DEF_VMX_VVV(vpkshss)
DEF_VMX_VVV(vpkswss) DEF_VMX_VVV(vpkshus) DEF_VMX_VVV(vpkswus)
    DEF_VMX_VVV(vmrghb)
        DEF_VMX_VVV(vmrghh) DEF_VMX_VVV(vmrghw) DEF_VMX_VVV(vmrglb)
            DEF_VMX_VVV(vmrglh) DEF_VMX_VVV(vmrglw) DEF_VMX_VVV(vmulesb)
                DEF_VMX_VVV(vmulosb) DEF_VMX_VVV(vmuleub) DEF_VMX_VVV(vmuloub)
                    DEF_VMX_VVV(vmulesh) DEF_VMX_VVV(vmulosh)
                        DEF_VMX_VVV(vmuleuh) DEF_VMX_VVV(vmulouh)
                            DEF_VMX_VVV(vmulesw) DEF_VMX_VVV(vmulosw)
                                DEF_VMX_VVV(vmuleuw) DEF_VMX_VVV(vmulouw)
#undef DEF_VMX_VVV

    // vperm VA-form: (4<<26) | VRT<<21 | VRA<<16 | VRB<<11 | VRC<<6 | XO
    BufferOffset Assembler::as_vperm(uint8_t vrt, uint8_t vra, uint8_t vrb,
                                     uint8_t vrc) {
  MOZ_ASSERT(vrt < 32 && vra < 32 && vrb < 32 && vrc < 32);
  spew("vperm\tvr%d,vr%d,vr%d,vr%d", vrt, vra, vrb, vrc);
  return writeInst(PPC_vperm | vrt << 21 | vra << 16 | vrb << 11 | vrc << 6);
}

// VA-form ternary VMX: (4<<26) | VRT<<21 | VRA<<16 | VRB<<11 | VRC<<6 |
// XO(6-bit)
BufferOffset Assembler::as_vmladduhm(uint8_t vrt, uint8_t vra, uint8_t vrb,
                                     uint8_t vrc) {
  MOZ_ASSERT(vrt < 32 && vra < 32 && vrb < 32 && vrc < 32);
  spew("vmladduhm\tvr%d,vr%d,vr%d,vr%d", vrt, vra, vrb, vrc);
  return writeInst(PPC_vmladduhm | vrt << 21 | vra << 16 | vrb << 11 |
                   vrc << 6);
}

BufferOffset Assembler::as_vmhraddshs(uint8_t vrt, uint8_t vra, uint8_t vrb,
                                      uint8_t vrc) {
  MOZ_ASSERT(vrt < 32 && vra < 32 && vrb < 32 && vrc < 32);
  spew("vmhraddshs\tvr%d,vr%d,vr%d,vr%d", vrt, vra, vrb, vrc);
  return writeInst(PPC_vmhraddshs | vrt << 21 | vra << 16 | vrb << 11 |
                   vrc << 6);
}

BufferOffset Assembler::as_vmsumshm(uint8_t vrt, uint8_t vra, uint8_t vrb,
                                    uint8_t vrc) {
  MOZ_ASSERT(vrt < 32 && vra < 32 && vrb < 32 && vrc < 32);
  spew("vmsumshm\tvr%d,vr%d,vr%d,vr%d", vrt, vra, vrb, vrc);
  return writeInst(PPC_vmsumshm | vrt << 21 | vra << 16 | vrb << 11 |
                   vrc << 6);
}

BufferOffset Assembler::as_vmsumuhm(uint8_t vrt, uint8_t vra, uint8_t vrb,
                                    uint8_t vrc) {
  MOZ_ASSERT(vrt < 32 && vra < 32 && vrb < 32 && vrc < 32);
  spew("vmsumuhm\tvr%d,vr%d,vr%d,vr%d", vrt, vra, vrb, vrc);
  return writeInst(PPC_vmsumuhm | vrt << 21 | vra << 16 | vrb << 11 |
                   vrc << 6);
}

BufferOffset Assembler::as_vspltisb(uint8_t vrt, int8_t simm5) {
  MOZ_ASSERT(vrt < 32);
  MOZ_ASSERT(simm5 >= -16 && simm5 <= 15);
  spew("vspltisb\tvr%d,%d", vrt, simm5);
  return writeInst(PPC_vspltisb | uint32_t(vrt) << 21 |
                   (uint32_t(simm5) & 0x1F) << 16);
}

BufferOffset Assembler::as_vspltish(uint8_t vrt, int8_t simm5) {
  MOZ_ASSERT(vrt < 32);
  MOZ_ASSERT(simm5 >= -16 && simm5 <= 15);
  spew("vspltish\tvr%d,%d", vrt, simm5);
  return writeInst(PPC_vspltish | uint32_t(vrt) << 21 |
                   (uint32_t(simm5) & 0x1F) << 16);
}

BufferOffset Assembler::as_vspltisw(uint8_t vrt, int8_t simm5) {
  MOZ_ASSERT(vrt < 32);
  MOZ_ASSERT(simm5 >= -16 && simm5 <= 15);
  spew("vspltisw\tvr%d,%d", vrt, simm5);
  return writeInst(PPC_vspltisw | uint32_t(vrt) << 21 |
                   (uint32_t(simm5) & 0x1F) << 16);
}

// --- Convenience pseudo-instructions ---

BufferOffset Assembler::xs_trap() {
  spew("trap @ %08x", currentOffset());
  return writeInst(PPC_trap);
}

BufferOffset Assembler::xs_trap_tagged(TrapTag tag) {
  uint32_t tv = PPC_trap | ((uint8_t)tag << 16) | ((uint8_t)tag << 11);
  spew("trap @ %08x ; MARK %d %08x", currentOffset(), (uint8_t)tag, tv);
  return writeInst(tv);
}

BufferOffset Assembler::xs_mr(Register rd, Register ra) {
  return as_or_(rd, ra, ra);
}

BufferOffset Assembler::xs_mtctr(Register ra) {
  return as_mtspr((SPRegisterID)spr_ctr, ra);
}

BufferOffset Assembler::xs_mtlr(Register ra) {
  return as_mtspr((SPRegisterID)spr_lr, ra);
}

BufferOffset Assembler::xs_mflr(Register rd) {
  return as_mfspr(rd, (SPRegisterID)spr_lr);
}

BufferOffset Assembler::xs_mtcr(Register rs) { return as_mtcrf(0xff, rs); }

BufferOffset Assembler::xs_mfxer(Register ra) {
  return as_mfspr(ra, (SPRegisterID)spr_xer);
}

BufferOffset Assembler::xs_mtxer(Register ra) {
  return as_mtspr((SPRegisterID)spr_xer, ra);
}

BufferOffset Assembler::xs_li(Register rd, int16_t im) {
  return as_addi(rd, r0, im, true);
}

BufferOffset Assembler::xs_lis(Register rd, int16_t im) {
  return as_addis(rd, r0, im, true);
}

BufferOffset Assembler::x_subi(Register rd, Register ra, int16_t im) {
  return as_addi(rd, ra, -im);
}

BufferOffset Assembler::x_not(Register rd, Register ra) {
  return as_nor(rd, ra, ra);
}

BufferOffset Assembler::x_slwi(Register rd, Register rs, int n) {
  MOZ_ASSERT(n >= 0 && n < 32);
  return as_rlwinm(rd, rs, n, 0, 31 - n);
}

BufferOffset Assembler::x_sldi(Register rd, Register rs, int n) {
  return as_rldicr(rd, rs, n, 63 - n);
}

BufferOffset Assembler::x_srwi(Register rd, Register rs, int n) {
  MOZ_ASSERT(n >= 0 && n < 32);
  if (n == 0) {
    return as_rlwinm(rd, rs, 0, 0, 31);
  }
  return as_rlwinm(rd, rs, 32 - n, n, 31);
}

BufferOffset Assembler::x_srdi(Register rd, Register rs, int n) {
  MOZ_ASSERT(n >= 0 && n < 64);
  if (n == 0) {
    return as_or_(rd, rs, rs);
  }
  return as_rldicl(rd, rs, 64 - n, n);
}

BufferOffset Assembler::x_bit_value(Register rd, Register rs, unsigned bit) {
  return as_rlwinm(rd, rs, bit + 1, 31, 31);
}

BufferOffset Assembler::x_insertbits0_15(Register rd, Register rs) {
  return as_rlwimi(rd, rs, 0, 16, 31);
}

BufferOffset Assembler::x_sr_mulli(Register rd, Register ra, int16_t im) {
  as_sradi(rd, ra, 63);
  return as_mulli(rd, rd, im);
}

void Assembler::as_break(uint32_t code) {
  spew("break\t%d", code);
  writeInst(PPC_trap);
}

// ========================================================================
// Label binding, retarget, and code label processing.
// ========================================================================

// Forward-declared shape helpers; full definitions and the layout
// commentary live with the WriteLoad64Instructions section below.
static bool IsAddpcisLoad64Stanza(uint32_t enc0);
static uint8_t Load64StanzaDestReg(Instruction* inst0);

InstImm Assembler::invertBranch(InstImm branch, BOffImm16 skipOffset) {
  // Flip the BO condition-true/condition-false bit (bit 24).
  uint32_t data = branch.encode();
  data = (data ^ 0x01000000) & 0xFFFF0003;
  data |= skipOffset.encode();
  branch.setData(data);
  return branch;
}

void Assembler::bind(InstImm* inst, uintptr_t branch, uintptr_t target) {
  intptr_t offset = target - branch;
  Instruction* i0 = (Instruction*)inst;

  if (i0->next()->encode() == PPC_bcl_always_plus4 ||
      IsAddpcisLoad64Stanza(i0->encode())) {
    // Pre-existing long stanza, either P8 (mflr + bcl marker at [1]) or
    // P9+ (addpcis at [0]; major opcode 19). Either way, just register
    // the long jump — the stanza's .quad at [6..7] gets patched later
    // via UpdateLoad64Value.
    addLongJump(BufferOffset(branch), BufferOffset(target));
    return;
  }

  if (i0->isOpcode((uint32_t)PPC_tw)) {
    // Tagged trap stanza. The tag tells us which branch type was reserved.
    TrapTag tag = (TrapTag)inst->traptag();
    Instruction* i1 = i0->next();
    Instruction* i2 = i1->next();
    Instruction* i3 = i2->next();
    Instruction* i4 = i3->next();
    Instruction* i5 = i4->next();
    Instruction* i6 = i5->next();
    Instruction* i7 = i6->next();
    Instruction* i8 = i7->next();
    Instruction* i9 = i8->next();

    switch (tag) {
      case BCTag: {
        // inst[-1] is the original bc instruction.
        Instruction* bc = i0 - 1;
        // Try short bc (offset + 4 because bc is one instruction before tw).
        if (BOffImm16::IsInRange(offset + (intptr_t)sizeof(uint32_t))) {
          bc->setData(((bc->encode() ^ 0x01000000) & 0xFFFF0003) |
                      BOffImm16(offset + sizeof(uint32_t)).encode());
          i0->makeNop();
          i1->makeNop();
          i2->makeNop();
          i3->makeNop();
          i4->makeNop();
          i5->makeNop();
          i6->makeNop();
          i7->makeNop();
          i8->makeNop();
          i9->makeNop();
          return;
        }
        // Try short b (unconditional).
        if (JOffImm26::IsInRange(offset)) {
          i0->setData(PPC_b | JOffImm26(offset).encode());
          i1->makeNop();
          i2->makeNop();
          i3->makeNop();
          i4->makeNop();
          i5->makeNop();
          i6->makeNop();
          i7->makeNop();
          i8->makeNop();
          i9->makeNop();
          return;
        }
        // Long: WriteLoad64 to SecondScratchReg + mtctr + bctr.
        addLongJump(BufferOffset(branch), BufferOffset(target));
        WriteLoad64Instructions(i0, SecondScratchReg,
                                LabelBase::INVALID_OFFSET);
        i8->makeOp_mtctr(SecondScratchReg);
        i9->makeOp_bctr();
        break;
      }
      case CallTag: {
        // For calls, the actual call instruction goes at inst[9] and
        // the return address must be after the stanza.
        intptr_t callOffset = offset - 9 * (intptr_t)sizeof(uint32_t);
        if (JOffImm26::IsInRange(callOffset)) {
          i0->makeNop();
          i1->makeNop();
          i2->makeNop();
          i3->makeNop();
          i4->makeNop();
          i5->makeNop();
          i6->makeNop();
          i7->makeNop();
          i8->makeNop();
          i9->setData(PPC_b | JOffImm26(callOffset).encode() | LinkB);
          return;
        }
        // Long: WriteLoad64 to SecondScratchReg + mtctr + bctrl.
        addLongJump(BufferOffset(branch), BufferOffset(target));
        WriteLoad64Instructions(i0, SecondScratchReg,
                                LabelBase::INVALID_OFFSET);
        i8->makeOp_mtctr(SecondScratchReg);
        i9->makeOp_bctr(LinkB);
        break;
      }
      case BTag: {
        if (JOffImm26::IsInRange(offset)) {
          i0->setData(PPC_b | JOffImm26(offset).encode());
          i1->makeNop();
          i2->makeNop();
          i3->makeNop();
          i4->makeNop();
          i5->makeNop();
          i6->makeNop();
          i7->makeNop();
          i8->makeNop();
          i9->makeNop();
          return;
        }
        // Long: WriteLoad64 to SecondScratchReg + mtctr + bctr.
        addLongJump(BufferOffset(branch), BufferOffset(target));
        WriteLoad64Instructions(i0, SecondScratchReg,
                                LabelBase::INVALID_OFFSET);
        i8->makeOp_mtctr(SecondScratchReg);
        i9->makeOp_bctr();
        break;
      }
      default:
        MOZ_CRASH("Unexpected TrapTag");
    }
    return;
  }

  if (i0->isOpcode(PPC_b)) {
    // Short unconditional branch — set offset, nop next-in-chain slot.
    MOZ_ASSERT(JOffImm26::IsInRange(offset));
    i0->setData((i0->encode() & ~0x03FFFFFC) | JOffImm26(offset).encode());
    i0->next()->makeNop();
    return;
  }

  if (i0->isOpcode(PPC_bc)) {
    // Short conditional branch — preserve upper 16 bits, set offset.
    MOZ_ASSERT(BOffImm16::IsInRange(offset));
    i0->setData((i0->encode() & 0xFFFF0003) | BOffImm16(offset).encode());
    i0->next()->makeNop();
    return;
  }

  MOZ_CRASH("Unexpected instruction in bind");
}

void Assembler::bind(Label* label, BufferOffset boff) {
  if (label->used()) {
    bool more;
    BufferOffset b(label);
    do {
      BufferOffset next;
      InstImm* inst = (InstImm*)editSrc(b);
      Instruction* i1 = ((Instruction*)inst)->next();
      more = (i1->encode() != LabelBase::INVALID_OFFSET);
      if (more) {
        next = BufferOffset(i1->encode());
      }
      bind(inst, b.getOffset(), boff.getOffset());
      b = next;
    } while (more);
  }
  label->bind(boff.getOffset());
}

void Assembler::retarget(Label* label, Label* target) {
  spew("retarget");
  if (label->used() && !oom()) {
    if (target->bound()) {
      bind(label, BufferOffset(target));
    } else if (target->used()) {
      // Prepend label's use chain to target's use chain.
      BufferOffset b(label);
      BufferOffset next;
      do {
        Instruction* inst = (Instruction*)editSrc(b);
        Instruction* i1 = inst->next();
        if (i1->encode() != LabelBase::INVALID_OFFSET) {
          next = BufferOffset(i1->encode());
        } else {
          // End of label's chain — link to target's head.
          i1->setData(target->offset());
          break;
        }
        b = next;
      } while (true);
    }
    // Transfer label's use list to target.
    if (!target->bound()) {
      target->use(label->offset());
    }
  }
  label->reset();
}

void Assembler::processCodeLabels(uint8_t* rawCode) {
  for (const CodeLabel& label : codeLabels_) {
    Bind(rawCode, label);
  }
}

// ========================================================================
// Load64 instruction sequence (8 slots, literal pool format):
//   [0] mflr r0            — save LR
//   [1] bcl 20,0,.+4      — LR = address of [2]
//   [2] mflr rD            — rD = address of [2]
//   [3] mtlr r0            — restore LR
//   [4] ld rD, 16(rD)      — load from [6..7] (offset = 24 - 8 = 16)
//   [5] b .+12             — skip data
//   [6..7] .quad VALUE     — 8-byte data
// ========================================================================

// ========================================================================
// Constant pool callbacks (required by AssemblerBufferWithConstantPools).
// ========================================================================

/* static */
void Assembler::InsertIndexIntoTag(uint8_t* load, uint32_t index) {
  // Stash the pool entry index in the hint word's low 16 bits; the high
  // bits carry the dest reg and load type, consumed by
  // PatchConstantPoolLoad when the pool is resolved.
  uint32_t* inst = (uint32_t*)load;
  *inst = (*inst & 0xFFFF0000) | (index & 0xFFFF);
}

/* static */
bool Assembler::PatchConstantPoolLoad(void* loadAddr, void* constPoolAddr) {
  // Rewrite placeholder instructions with a pool load sequence.
  // Hint word layout (set by loadFromPoolFloat64 / loadFromPoolFloat32 /
  // loadFromPoolSimd128):
  //   bits 0-15:  pool entry index
  //   bits 16-20: destination register (FPR encoding)
  //   bits 21-22: load type (PoolLoadFPR64, PoolLoadSimd128, PoolLoadFPR32)
  //   bits 28-31: sentinel 0xF

  uint32_t* inst = (uint32_t*)loadAddr;

  uint32_t hint = inst[0];
  uint32_t index = hint & 0xFFFF;
  uint32_t destReg = (hint >> 16) & 0x1F;
  uint32_t loadType = (hint >> 21) & 0x3;

  // Displacement: pool entry address relative to inst[1] (mflr target) for the
  // bcl path, or relative to inst[0]+4 (addpcis target = CIA+4, which is the
  // address of inst[1]) for the addpcis path. Both conventions resolve to the
  // same value: (pool entry) − (loadAddr + 4).
  int32_t displacement =
      (int32_t)((uint8_t*)constPoolAddr + index * 4 - ((uint8_t*)loadAddr + 4));

  if (loadType == PoolLoadFPR64 || loadType == PoolLoadFPR32) {
    // Three emission paths:
    //
    // POWER10 (preferred): plfd/plfs FRT, SI(0), R=1 — single PC-relative
    //   prefixed FP load. 8 bytes = 2 slots; slot 2 becomes a nop. If
    //   loadAddr % 64 == 60, plfd would straddle a 64-byte block, so emit
    //   a leading nop at slot 0 and place plfd at slots 1-2 instead.
    //   Reach: ±8 GB (34-bit signed). No LR clobber, no r16 base.
    //
    // POWER9: addpcis + lfd/lfs + nop. 2 real insns, no LR clobber, no
    //   Return Address Stack corruption. Base register is r16.
    //   Displacement splits into (hi << 16) + lo where lo is the 16-bit
    //   signed D-field of lfd/lfs. Reach: ±2 GB.
    //
    // POWER8: bcl + mflr r16 + lfd/lfs. Same clobber + RAS caveat as before.
    //   Kept as a correctness fallback; not exercised today because the
    //   loadConstantDouble/Float32 wrappers skip the pool on POWER8.
    //
    // lfs/plfs (32-bit) auto-expand their result to double-precision in the
    // FPR, replacing the non-pool path's separate xscvspdpn step.
    uint32_t baseReg = SavedScratchRegister.code();
    uint32_t loadOp = (loadType == PoolLoadFPR64) ? PPC_lfd : PPC_lfs;

    if (HasPOWER10()) {
      // MLS prefixed FP load. plfd suffix opcode = 50, plfs = 48. Same
      // alignment-driven slot placement as PoolLoadSimd128 above.
      uint64_t loadAddrBits = reinterpret_cast<uint64_t>(loadAddr);
      // loadAddr is the buffer-time pointer; the final executable base is
      // only 16-byte aligned, so the unsafe straddle is when
      // (loadAddrBits & 15) == 12 (matches ensurePrefixedAlignment above).
      bool needLeadingNop = (loadAddrBits & 15) == 12;
      int prefixSlot = needLeadingNop ? 1 : 0;
      int prefixByteOffset = prefixSlot * 4;
      int64_t SI = int64_t(displacement) + 4 - prefixByteOffset;
      MOZ_ASSERT(SI >= -(int64_t(1) << 33) && SI < (int64_t(1) << 33));
      uint32_t d0 = uint32_t((uint64_t(SI) >> 16) & 0x3FFFFu);
      uint32_t d1 = uint32_t(uint64_t(SI) & 0xFFFFu);
      // Type 2 (MLS), R=1, RA=0.
      uint32_t prefix =
          (1u << 26) | (2u << 24) | (1u << 20) | (d0 & 0x3FFFFu);
      uint32_t suffixOp = (loadType == PoolLoadFPR64) ? 50u : 48u;
      uint32_t suffix = (suffixOp << 26) | (destReg << 21) | d1;

      if (needLeadingNop) {
        inst[0] = NopInst;
        inst[1] = prefix;
        inst[2] = suffix;
      } else {
        inst[0] = prefix;
        inst[1] = suffix;
        inst[2] = NopInst;
      }
    } else if (HasPOWER9()) {
      // Split displacement into addpcis hi field and lfd/lfs lo field so that
      //   target = (CIA + 4) + (hi << 16) + SEXT16(lo).
      // Only 2 slots are reserved on P9 (loadFromPoolFloat{32,64} above);
      // do NOT touch inst[2], it belongs to the next entry.
      int16_t lo = (int16_t)(displacement & 0xFFFF);
      int32_t hiAdj = displacement - lo;
      MOZ_ASSERT((hiAdj & 0xFFFF) == 0);
      int32_t hi = hiAdj >> 16;
      MOZ_ASSERT(hi >= -32768 && hi <= 32767);
      // [0] addpcis r16, hi
      uint32_t Dhi = uint16_t(hi);
      inst[0] = (19u << 26) | (baseReg << 21) | ((Dhi >> 1) & 0x1F) << 16 |
                ((Dhi >> 6) & 0x3FF) << 6 | (2u << 1) | (Dhi & 1u);
      // [1] lfd/lfs fD, lo(r16)
      inst[1] = loadOp | (destReg << 21) | (baseReg << 16) | (uint16_t(lo));
    } else {
      MOZ_ASSERT(displacement >= -32768 && displacement < 32768);
      // [0] bcl 20,0,$+4
      inst[0] = PPC_bcl_always_plus4;
      // [1] mflr r16
      inst[1] = PPC_mfspr | (baseReg << 21) | PPC_SPR(spr_lr);
      // [2] lfd/lfs fD, displacement(r16)
      inst[2] =
          loadOp | (destReg << 21) | (baseReg << 16) | (displacement & 0xFFFF);
    }
  } else if (loadType == PoolLoadSimd128) {
    // Three emission paths (5 slots reserved by loadFromPoolSimd128):
    //
    // POWER10 (preferred): plxv vsD, SI(0), R=1 — single PC-relative
    //   prefixed load, natural-LE byte order (no xxpermdi needed). 8 bytes
    //   = 2 slots; slots 2-4 become nops. If the prefix would straddle a
    //   64-byte block (loadAddr % 64 == 60), emit a leading nop at slot 0
    //   and place plxv at slots 1-2 instead. Reach: ±8 GB (34-bit signed).
    //
    // POWER9: addpcis-equivalent via bcl + mflr + addi + lxvx + nop. 5
    //   real insns, natural LE.
    //
    // POWER8: same prelude + lxvd2x + xxpermdi (BE-DW byte-swap fixup).
    //
    // See PoolLoadFPR64 above for why r16 instead of r12.
    MOZ_ASSERT(displacement >= -32768 && displacement < 32768);
    // Simd128 dest is in VR-namespace (encoding 32-63). Hint stores only
    // the low 5 bits (loadFromPoolSimd128 masks); we set TX unconditionally
    // since PoolLoadSimd128 always targets a Simd128.
    constexpr uint32_t kTX = 1u;
    constexpr uint32_t kAxBxTx_xxpermdi = (1u << 2) | (1u << 1) | 1u;

    if (HasPOWER10()) {
      // Place plxv prefix at the highest 4-byte-aligned offset within
      // the 5 reserved slots that doesn't straddle a 64-byte block.
      uint64_t loadAddrBits = reinterpret_cast<uint64_t>(loadAddr);
      // loadAddr is the buffer-time pointer; the final executable base is
      // only 16-byte aligned, so the unsafe straddle is when
      // (loadAddrBits & 15) == 12 (matches ensurePrefixedAlignment above).
      bool needLeadingNop = (loadAddrBits & 15) == 12;
      int prefixSlot = needLeadingNop ? 1 : 0;
      int prefixByteOffset = prefixSlot * 4;
      // SI = (pool entry addr) - (prefix addr)
      //    = (loadAddr + 4 + displacement) - (loadAddr + prefixByteOffset)
      //    = displacement + 4 - prefixByteOffset
      int64_t SI = int64_t(displacement) + 4 - prefixByteOffset;
      MOZ_ASSERT(SI >= -(int64_t(1) << 33) && SI < (int64_t(1) << 33));
      uint32_t d0 = uint32_t((uint64_t(SI) >> 16) & 0x3FFFFu);
      uint32_t d1 = uint32_t(uint64_t(SI) & 0xFFFFu);
      // Prefix: primary opcode 1, Type 0 (8LS), R=1, d0 at LE bits 17..0.
      uint32_t prefix =
          (1u << 26) | (0u << 24) | (1u << 20) | (d0 & 0x3FFFFu);
      // Suffix: 5-bit opcode 25 at LE 31..27, TX at LE 26, T at LE 25..21,
      //         RA=0 at LE 20..16, d1 at LE 15..0.
      uint32_t suffix = (25u << 27) | (kTX << 26) | (destReg << 21) | d1;

      // P10 reserves 3 slots; only inst[0..2] are written. Slots 3..4
      // belong to the next pool entry on P10.
      if (needLeadingNop) {
        inst[0] = NopInst;
        inst[1] = prefix;
        inst[2] = suffix;
      } else {
        inst[0] = prefix;
        inst[1] = suffix;
        inst[2] = NopInst;
      }
    } else if (HasPOWER9()) {
      // addpcis + addi + lxvx (3 slots) — no LR clobber, no RAS hazard.
      // Same displacement split as the FP scalar P9 path: target =
      // (CIA+4) + (hi << 16) + SEXT16(lo). lxvx is X-form indexed (no
      // immediate offset), so combine the low 16 bits into r16 via addi
      // before the load.
      int16_t lo = (int16_t)(displacement & 0xFFFF);
      int32_t hiAdj = displacement - lo;
      MOZ_ASSERT((hiAdj & 0xFFFF) == 0);
      int32_t hi = hiAdj >> 16;
      MOZ_ASSERT(hi >= -32768 && hi <= 32767);
      uint32_t Dhi = uint16_t(hi);
      uint32_t baseReg = SavedScratchRegister.code();
      // [0] addpcis r16, hi
      inst[0] = (19u << 26) | (baseReg << 21) | ((Dhi >> 1) & 0x1F) << 16 |
                ((Dhi >> 6) & 0x3FF) << 6 | (2u << 1) | (Dhi & 1u);
      // [1] addi r16, r16, lo
      inst[1] = PPC_addi | (baseReg << 21) | (baseReg << 16) | uint16_t(lo);
      // [2] lxvx vsD, 0, r16  (XT[0:4] in bits 21-25, TX at bit 0)
      inst[2] = PPC_lxvx | (destReg << 21) | (baseReg << 11) | kTX;
    } else {
      // P8 fallback: bcl + mflr + addi + lxvd2x + xxpermdi (5 slots).
      // Clobbers LR; correctness-only path.
      uint32_t baseReg = SavedScratchRegister.code();
      inst[0] = PPC_bcl_always_plus4;
      inst[1] = PPC_mfspr | (baseReg << 21) | PPC_SPR(spr_lr);
      inst[2] = PPC_addi | (baseReg << 21) | (baseReg << 16) |
                (displacement & 0xFFFF);
      // lxvd2x XT, RA=0, RB=r16 — loads in BE order on LE.
      inst[3] = PPC_lxvd2x | (destReg << 21) | (baseReg << 11) | kTX;
      // xxpermdi XT, XT, XT, 2 — swap doublewords for LE byte order.
      inst[4] = PPC_xxpermdi | (destReg << 21) | (destReg << 16) |
                (destReg << 11) | (2u << 8) | kAxBxTx_xxpermdi;
    }
  } else {
    MOZ_CRASH("PatchConstantPoolLoad: unsupported load type");
  }

  return false;
}

/* static */
void Assembler::WritePoolGuard(BufferOffset branch, Instruction* inst,
                               BufferOffset dest) {
  // Emit an unconditional branch over the pool data.
  int32_t offset = dest.getOffset() - branch.getOffset();
  MOZ_ASSERT(JOffImm26::IsInRange(offset));
  inst->setData(PPC_b | (offset & 0x03FFFFFC));
}

/* static */
void Assembler::WritePoolHeader(uint8_t* start, Pool* p, bool isNatural) {
  // Write pool identification header.
  // Encode pool size and isNatural flag in a single 32-bit word.
  uint32_t poolSize = p->getPoolSize();
  uint32_t sizeInWords = (poolSize + 4 + 3) >> 2;  // header + data, in words
  MOZ_ASSERT(sizeInWords < (1 << 15));
  uint32_t header = (sizeInWords & 0x7FFF) | (isNatural ? (1 << 15) : 0) |
                    0xFFFF0000;  // sentinel
  *(uint32_t*)start = header;
}

/* static */
void Assembler::PatchShortRangeBranchToVeneer(PPCBuffer*, unsigned rangeIdx,
                                              BufferOffset deadline,
                                              BufferOffset veneer) {
  // PPC64 does not use short-range branch tracking (NumShortBranchRanges = 0).
  MOZ_CRASH("PatchShortRangeBranchToVeneer: should not be called");
}

// Two stanza shapes share the same 8-slot footprint and the same .quad
// location at slots [6..7] (so ExtractLoad64Value / UpdateLoad64Value are
// shape-agnostic):
//
//   POWER8 (no addpcis):
//     [0] mflr r0
//     [1] bcl 20,0,.+4         (LR := pc of [2])
//     [2] mflr rD
//     [3] mtlr r0
//     [4] ld rD, 16(rD)
//     [5] b .+12
//     [6..7] .quad VALUE
//
//   POWER9+ (addpcis):
//     [0] addpcis rD, 0        (rD := NIA = pc of [1])
//     [1] ld rD, 20(rD)        (rD := mem[pc_of_[1] + 20] = mem[slot[6]])
//     [2] b .+24
//     [3..5] NOP, NOP, NOP
//     [6..7] .quad VALUE
//
// The P9+ form drops the bcl/mflr/mtlr LR-bounce (no RAS thrash) and runs
// 2 dynamic insns instead of 6. Distinguished at patch time by inst[0]'s
// major opcode: 31 = mfspr (P8) vs 19 = addpcis (P9+).
static bool IsAddpcisLoad64Stanza(uint32_t enc0) {
  return ((enc0 >> 26) & 0x3f) == 19;
}

// Extract the destination register from a load64 stanza in either shape.
// P8 stores rD in `mflr rD` at slot [2]; P9+ stores rD in `addpcis rD, 0`
// at slot [0]. Both encode RT at LE bits [21..25].
static uint8_t Load64StanzaDestReg(Instruction* inst0) {
  if (IsAddpcisLoad64Stanza(inst0->encode())) {
    return (inst0[0].encode() >> 21) & 0x1f;
  }
  return (inst0[2].encode() >> 21) & 0x1f;
}

/* static */
void Assembler::WriteLoad64Instructions(Instruction* inst0, Register reg,
                                        uint64_t value) {
  Instruction* i1 = inst0->next();
  Instruction* i2 = i1->next();
  Instruction* i3 = i2->next();
  Instruction* i4 = i3->next();
  Instruction* i5 = i4->next();
  Instruction* i6 = i5->next();
  Instruction* i7 = i6->next();

  if (HasPOWER9()) {
    // [0] addpcis rD, 0   (DX-form: opcode=19, XO=2, all D fields = 0)
    inst0->setData(0x4C000004u | (uint32_t(reg.code()) << 21));
    // [1] ld rD, 20(rD)   (rD := *(slot[1] + 20) = *(slot[6]) = .quad)
    i1->setData(PPC_ld | (uint32_t(reg.code()) << 21) |
                (uint32_t(reg.code()) << 16) | 20);
    // [2] b .+24          (skip slots [3..7] to land at slot [8])
    i2->setData(PPC_b | (24 & 0x03FFFFFC));
    // [3..5] NOP filler — unreachable but kept aligned for the patcher.
    i3->setData(NopInst);
    i4->setData(NopInst);
    i5->setData(NopInst);
  } else {
    // [0] mflr r0
    inst0->setData(PPC_mfspr | (r0.code() << 21) | PPC_SPR(spr_lr));
    // [1] bcl 20,0,.+4
    i1->setData(PPC_bcl_always_plus4);
    // [2] mflr rD
    i2->setData(PPC_mfspr | (reg.code() << 21) | PPC_SPR(spr_lr));
    // [3] mtlr r0
    i3->setData(PPC_mtspr | (r0.code() << 21) | PPC_SPR(spr_lr));
    // [4] ld rD, 16(rD)
    i4->setData(PPC_ld | (reg.code() << 21) | (reg.code() << 16) | 16);
    // [5] b .+12
    i5->setData(PPC_b | (12 & 0x03FFFFFC));
  }

  // [6..7] .quad VALUE (low 32 at lower addr, high 32 at higher addr).
  i6->setData((uint32_t)(value & 0xFFFFFFFF));
  i7->setData((uint32_t)(value >> 32));
}

/* static */
uint64_t Assembler::ExtractLoad64Value(Instruction* inst0) {
  // The 8-byte value is at inst0[6..7] in both shapes.
  Instruction* i6 = inst0 + 6;
  Instruction* i7 = inst0 + 7;

  uint64_t lo = (uint64_t)i6->encode();  // low 32 at lower addr
  uint64_t hi = (uint64_t)i7->encode();  // high 32 at higher addr
  return (hi << 32) | lo;
}

/* static */
void Assembler::UpdateLoad64Value(Instruction* inst0, uint64_t value) {
  // Sanity-check that inst0 is the start of a load64 stanza in either shape.
  // P8: inst0[1] == bcl 20,0,.+4. P9+: inst0[0] is addpcis (major opcode 19).
  MOZ_ASSERT(inst0[1].encode() == PPC_bcl_always_plus4 ||
                 IsAddpcisLoad64Stanza(inst0->encode()),
             "UpdateLoad64Value: inst0 is not a load64 stanza");

  // .quad lives at inst0[6..7] in both shapes.
  Instruction* i6 = inst0 + 6;
  Instruction* i7 = inst0 + 7;

  i6->setData((uint32_t)(value & 0xFFFFFFFF));  // low 32 at lower addr
  i7->setData((uint32_t)(value >> 32));         // high 32 at higher addr
}

// ========================================================================
// Patching and toggle operations.
// ========================================================================

/* static */
uint32_t Assembler::PatchWrite_NearCallSize() {
  // 8 instructions for Load64 + mtctr + bctrl = 10 instructions.
  return 10 * sizeof(uint32_t);
}

/* static */
void Assembler::PatchWrite_NearCall(CodeLocationLabel start,
                                    CodeLocationLabel toCall) {
  Instruction* inst = (Instruction*)start.raw();
  uint8_t* dest = toCall.raw();

  Assembler::WriteLoad64Instructions(inst, SavedScratchRegister,
                                     (uint64_t)dest);
  inst[8].makeOp_mtctr(SavedScratchRegister);
  inst[9].makeOp_bctr(LinkB);
  FlushICache(inst, 10 * sizeof(Instruction));
}

/* static */
void Assembler::PatchWrite_Imm32(CodeLocationLabel label, Imm32 imm) {
  uint32_t* l = (uint32_t*)label.raw();
  *(l - 1) = imm.value;
  FlushICache(l - 1, sizeof(uint32_t));
}

void Assembler::PatchDataWithValueCheck(CodeLocationLabel label,
                                        ImmPtr newValue, ImmPtr expectedValue) {
  PatchDataWithValueCheck(label, PatchedImmPtr(newValue.value),
                          PatchedImmPtr(expectedValue.value));
}

void Assembler::PatchDataWithValueCheck(CodeLocationLabel label,
                                        PatchedImmPtr newValue,
                                        PatchedImmPtr expectedValue) {
  Instruction* inst = (Instruction*)label.raw();

  DebugOnly<uint64_t> value = Assembler::ExtractLoad64Value(inst);
  MOZ_ASSERT(value == uint64_t(expectedValue.value));

  Assembler::UpdateLoad64Value(inst, uint64_t(newValue.value));
  FlushICache(inst, 8 * sizeof(Instruction));
}

// ToggleCall toggles the call portion of a toggledCall stanza.
// Layout: 8 load64 instructions + mtctr + bctrl (10 total).
// We toggle the last two instructions (mtctr/bctrl vs nop/nop).
// The destination register is extracted via Load64StanzaDestReg, which
// handles both the P8 (mflr-rD at slot [2]) and P9+ (addpcis-rD at slot
// [0]) shapes.

/* static */
void Assembler::ToggleCall(CodeLocationLabel inst_, bool enabled) {
  Instruction* i0 = (Instruction*)inst_.raw();
  Instruction* i8 = (Instruction*)(inst_.raw() + 8 * sizeof(uint32_t));
  Instruction* i9 = (Instruction*)(inst_.raw() + 9 * sizeof(uint32_t));

  // Accept either P8 stanza (mflr r0 at slot [0]) or P9+ stanza (addpcis at
  // slot [0]; major opcode 19).
  MOZ_ASSERT(i0->encode() == (PPC_mfspr | (r0.code() << 21) | PPC_SPR(spr_lr)) ||
                 IsAddpcisLoad64Stanza(i0->encode()));

  // ToggleCall is idempotent across the same `enabled` value: re-enabling
  // an already-enabled site (or re-disabling a disabled one) is a no-op.
  // Mozilla's debugger machinery may legitimately toggle the same call site
  // multiple times in the same direction (e.g. setting both a breakpoint
  // and a frame.onStep on the same script).
  Register scratch = Register::FromCode(Load64StanzaDestReg(i0));
  uint32_t mtctr = PPC_mtspr | (scratch.code() << 21) | PPC_SPR(spr_ctr);
  uint32_t bctrl = (uint32_t)PPC_bctr | (uint32_t)LinkB;
  if (enabled) {
    MOZ_ASSERT(i8->encode() == NopInst || i8->encode() == mtctr);
    MOZ_ASSERT(i9->encode() == NopInst || i9->encode() == bctrl);
    i8->setData(mtctr);
    i9->setData(bctrl);
  } else {
    MOZ_ASSERT(i8->encode() == NopInst || i8->encode() == mtctr);
    MOZ_ASSERT(i9->encode() == NopInst || i9->encode() == bctrl);
    i8->setData(NopInst);
    i9->setData(NopInst);
  }
  FlushICache(i8, 2 * sizeof(Instruction));
}

// toggledJump emits a trap stanza via jump(label). After binding, the first
// instruction becomes "b offset" (short branch). We toggle between b and ori:
//   b offset:       [010010][LI:24][0][0]
//   ori r0,r0,imm:  [011000][00000][00000][UI:16]
// For short forward jumps (offset < 64KB), bits 25:16 of LI are 0, so
// swapping the opcode preserves the offset in the lower 16 bits.
// ori r0,r0,X is effectively a nop (writes to r0).

/* static */
void Assembler::ToggleToJmp(CodeLocationLabel inst_) {
  Instruction* inst = (Instruction*)inst_.raw();
  MOZ_ASSERT(inst->isOpcode(PPC_ori));
  // Verify RS=0 and RA=0 (r0).
  MOZ_ASSERT((inst->encode() & 0x03E00000) == 0);
  MOZ_ASSERT((inst->encode() & 0x001F0000) == 0);
  // Swap opcode from ori (011000) to b (010010).
  uint32_t encoding = inst->encode();
  encoding = (encoding & 0x03FFFFFF) | (uint32_t)PPC_b;
  inst->setData(encoding);
  FlushICache(inst, sizeof(Instruction));
}

/* static */
void Assembler::ToggleToCmp(CodeLocationLabel inst_) {
  Instruction* inst = (Instruction*)inst_.raw();
  MOZ_ASSERT(inst->isOpcode(PPC_b));
  // Verify short forward branch: upper LI bits (25:16) are 0, AA=0, LK=0.
  MOZ_ASSERT((inst->encode() & 0x03FF0003) == 0);
  // Swap opcode from b (010010) to ori (011000).
  uint32_t encoding = inst->encode();
  encoding = (encoding & 0x03FFFFFF) | (uint32_t)PPC_ori;
  inst->setData(encoding);
  FlushICache(inst, sizeof(Instruction));
}

// ========================================================================
// Bind, tracing, and pointer extraction.
// ========================================================================

void Assembler::Bind(uint8_t* rawCode, const CodeLabel& label) {
  if (label.patchAt().bound()) {
    auto mode = label.linkMode();
    intptr_t offset = label.patchAt().offset();
    intptr_t target = label.target().offset();

    if (mode == CodeLabel::RawPointer) {
      *reinterpret_cast<const void**>(rawCode + offset) = rawCode + target;
    } else {
      MOZ_ASSERT(mode == CodeLabel::MoveImmediate ||
                 mode == CodeLabel::JumpImmediate);
      Instruction* inst = (Instruction*)(rawCode + offset);
      Assembler::UpdateLoad64Value(inst, (uint64_t)(rawCode + target));
    }
  }
}

uintptr_t Assembler::GetPointer(uint8_t* instPtr) {
  Instruction* inst = (Instruction*)instPtr;
  return Assembler::ExtractLoad64Value(inst);
}

static JitCode* CodeFromJump(Instruction* jump) {
  uint8_t* target = (uint8_t*)Assembler::ExtractLoad64Value(jump);
  return JitCode::FromExecutable(target);
}

void Assembler::TraceJumpRelocations(JSTracer* trc, JitCode* code,
                                     CompactBufferReader& reader) {
  while (reader.more()) {
    JitCode* child =
        CodeFromJump((Instruction*)(code->raw() + reader.readUnsigned()));
    TraceManuallyBarrieredEdge(trc, &child, "rel32");
  }
}

static void TraceOneDataRelocation(JSTracer* trc,
                                   mozilla::Maybe<AutoWritableJitCode>& awjc,
                                   JitCode* code, Instruction* inst) {
  void* ptr = (void*)Assembler::ExtractLoad64Value(inst);
  void* prior = ptr;

  uintptr_t word = reinterpret_cast<uintptr_t>(ptr);
  if (word >> JSVAL_TAG_SHIFT) {
    Value v = Value::fromRawBits(word);
    TraceManuallyBarrieredEdge(trc, &v, "jit-masm-value");
    ptr = (void*)v.bitsAsPunboxPointer();
  } else {
    TraceManuallyBarrieredGenericPointerEdge(
        trc, reinterpret_cast<gc::Cell**>(&ptr), "jit-masm-ptr");
  }

  if (ptr != prior) {
    if (awjc.isNothing()) {
      awjc.emplace(code);
    }
    Assembler::UpdateLoad64Value(inst, uint64_t(ptr));
  }
}

/* static */
void Assembler::TraceDataRelocations(JSTracer* trc, JitCode* code,
                                     CompactBufferReader& reader) {
  mozilla::Maybe<AutoWritableJitCode> awjc;
  while (reader.more()) {
    size_t offset = reader.readUnsigned();
    Instruction* inst = (Instruction*)(code->raw() + offset);
    TraceOneDataRelocation(trc, awjc, code, inst);
  }
}

/* static */
uint8_t* Assembler::NextInstruction(uint8_t* instruction, uint32_t* count) {
  if (count != nullptr) {
    *count += sizeof(Instruction);
  }
  return instruction + sizeof(Instruction);
}

// ========================================================================
// UseScratchRegisterScope implementation.
// ========================================================================

UseScratchRegisterScope::UseScratchRegisterScope(Assembler& assembler)
    : available_(assembler.GetScratchRegisterList()),
      old_available_(*available_) {}

UseScratchRegisterScope::UseScratchRegisterScope(Assembler* assembler)
    : available_(assembler->GetScratchRegisterList()),
      old_available_(*available_) {}

UseScratchRegisterScope::~UseScratchRegisterScope() {
  *available_ = old_available_;
}

Register UseScratchRegisterScope::Acquire() {
  MOZ_ASSERT(available_ != nullptr);
  MOZ_ASSERT(!available_->empty());
  Register index = GeneralRegisterSet::FirstRegister(available_->bits());
  available_->takeRegisterIndex(index);
  return index;
}

void UseScratchRegisterScope::Release(const Register& reg) {
  MOZ_ASSERT(available_ != nullptr);
  MOZ_ASSERT(old_available_.hasRegisterIndex(reg));
  MOZ_ASSERT(!available_->hasRegisterIndex(reg));
  Include(GeneralRegisterSet(1 << reg.code()));
}

bool UseScratchRegisterScope::hasAvailable() const {
  return (available_->size()) != 0;
}
