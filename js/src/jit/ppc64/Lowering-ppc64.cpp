/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "jit/ppc64/Lowering-ppc64.h"

#include "mozilla/MathAlgorithms.h"

#include "jit/Lowering.h"
#include "jit/MIR-wasm.h"
#include "jit/MIR.h"
#include "jit/ppc64/Assembler-ppc64.h"
#include "wasm/WasmFeatures.h"  // for wasm::ReportSimdAnalysis

#include "jit/shared/Lowering-shared-inl.h"

using namespace js;
using namespace js::jit;

using mozilla::FloorLog2;

namespace js {
namespace jit {

LTableSwitch* LIRGeneratorPPC64::newLTableSwitch(const LAllocation& in,
                                                 const LDefinition& inputCopy) {
  return new (alloc()) LTableSwitch(in, inputCopy, temp());
}

LTableSwitchV* LIRGeneratorPPC64::newLTableSwitchV(const LBoxAllocation& in) {
  return new (alloc()) LTableSwitchV(in, temp(), tempDouble(), temp());
}

void LIRGeneratorPPC64::lowerForShift(LInstructionHelper<1, 2, 0>* ins,
                                      MDefinition* mir, MDefinition* lhs,
                                      MDefinition* rhs) {
  lowerForALU(ins, mir, lhs, rhs);
}

template <class LInstr>
void LIRGeneratorPPC64::lowerForShiftInt64(LInstr* ins, MDefinition* mir,
                                           MDefinition* lhs, MDefinition* rhs) {
  if constexpr (std::is_same_v<LInstr, LShiftI64>) {
    ins->setLhs(useInt64RegisterAtStart(lhs));
    ins->setRhs(useRegisterOrConstantAtStart(rhs));
  } else {
    ins->setInput(useInt64RegisterAtStart(lhs));
    ins->setCount(useRegisterOrConstantAtStart(rhs));
  }
  defineInt64(ins, mir);
}

template void LIRGeneratorPPC64::lowerForShiftInt64(LShiftI64* ins,
                                                    MDefinition* mir,
                                                    MDefinition* lhs,
                                                    MDefinition* rhs);
template void LIRGeneratorPPC64::lowerForShiftInt64(LRotateI64* ins,
                                                    MDefinition* mir,
                                                    MDefinition* lhs,
                                                    MDefinition* rhs);

void LIRGeneratorPPC64::lowerForALU(LInstructionHelper<1, 1, 0>* ins,
                                    MDefinition* mir, MDefinition* input) {
  ins->setOperand(0, useRegisterAtStart(input));
  define(ins, mir);
}

void LIRGeneratorPPC64::lowerForALU(LInstructionHelper<1, 2, 0>* ins,
                                    MDefinition* mir, MDefinition* lhs,
                                    MDefinition* rhs) {
  ins->setOperand(0, useRegisterAtStart(lhs));
  ins->setOperand(1, useRegisterOrConstantAtStart(rhs));
  define(ins, mir);
}

void LIRGeneratorPPC64::lowerForALUInt64(
    LInstructionHelper<INT64_PIECES, INT64_PIECES, 0>* ins, MDefinition* mir,
    MDefinition* input) {
  ins->setInt64Operand(0, useInt64RegisterAtStart(input));
  defineInt64(ins, mir);
}

void LIRGeneratorPPC64::lowerForALUInt64(
    LInstructionHelper<INT64_PIECES, 2 * INT64_PIECES, 0>* ins,
    MDefinition* mir, MDefinition* lhs, MDefinition* rhs) {
  ins->setInt64Operand(0, useInt64RegisterAtStart(lhs));
  ins->setInt64Operand(INT64_PIECES, useInt64RegisterOrConstantAtStart(rhs));
  defineInt64(ins, mir);
}

void LIRGeneratorPPC64::lowerForMulInt64(LMulI64* ins, MMul* mir,
                                         MDefinition* lhs, MDefinition* rhs) {
  lowerForALUInt64(ins, mir, lhs, rhs);
}

void LIRGeneratorPPC64::lowerForFPU(LInstructionHelper<1, 1, 0>* ins,
                                    MDefinition* mir, MDefinition* input) {
  ins->setOperand(0, useRegisterAtStart(input));
  define(ins, mir);
}

void LIRGeneratorPPC64::lowerForFPU(LInstructionHelper<1, 2, 0>* ins,
                                    MDefinition* mir, MDefinition* lhs,
                                    MDefinition* rhs) {
  ins->setOperand(0, useRegisterAtStart(lhs));
  ins->setOperand(1, useRegisterAtStart(rhs));
  define(ins, mir);
}

LBoxAllocation LIRGeneratorPPC64::useBoxFixed(MDefinition* mir, Register reg1,
                                              Register reg2, bool useAtStart) {
  MOZ_ASSERT(mir->type() == MIRType::Value);

  ensureDefined(mir);
  return LBoxAllocation(LUse(reg1, mir->virtualRegister(), useAtStart));
}

LAllocation LIRGeneratorPPC64::useByteOpRegister(MDefinition* mir) {
  return useRegister(mir);
}

LAllocation LIRGeneratorPPC64::useByteOpRegisterAtStart(MDefinition* mir) {
  return useRegisterAtStart(mir);
}

LAllocation LIRGeneratorPPC64::useByteOpRegisterOrNonDoubleConstant(
    MDefinition* mir) {
  return useRegisterOrNonDoubleConstant(mir);
}

LDefinition LIRGeneratorPPC64::tempByteOpRegister() { return temp(); }

LDefinition LIRGeneratorPPC64::tempToUnbox() { return temp(); }

void LIRGeneratorPPC64::lowerUntypedPhiInput(MPhi* phi, uint32_t inputPosition,
                                             LBlock* block, size_t lirIndex) {
  lowerTypedPhiInput(phi, inputPosition, block, lirIndex);
}

void LIRGeneratorPPC64::lowerInt64PhiInput(MPhi* phi, uint32_t inputPosition,
                                           LBlock* block, size_t lirIndex) {
  lowerTypedPhiInput(phi, inputPosition, block, lirIndex);
}

void LIRGeneratorPPC64::defineInt64Phi(MPhi* phi, size_t lirIndex) {
  defineTypedPhi(phi, lirIndex);
}

void LIRGeneratorPPC64::lowerMulI(MMul* mul, MDefinition* lhs,
                                  MDefinition* rhs) {
  LMulI* lir = new (alloc()) LMulI;
  if (mul->fallible()) {
    assignSnapshot(lir, mul->bailoutKind());
  }
  if (mul->canBeNegativeZero() && !rhs->isConstant()) {
    lir->setOperand(0, useRegister(lhs));
    lir->setOperand(1, useRegister(rhs));
    define(lir, mul);
    return;
  }
  lowerForALU(lir, mul, lhs, rhs);
}

void LIRGeneratorPPC64::lowerDivI(MDiv* div) {
  if (div->rhs()->isConstant()) {
    int32_t rhs = div->rhs()->toConstant()->toInt32();
    int32_t shift = FloorLog2(uint32_t(rhs));
    if (rhs > 0 && 1 << shift == rhs) {
      LDivPowTwoI* lir =
          new (alloc()) LDivPowTwoI(useRegister(div->lhs()), shift);
      if (div->fallible()) {
        assignSnapshot(lir, div->bailoutKind());
      }
      define(lir, div);
      return;
    }
  }
  LDivI* lir = new (alloc())
      LDivI(useRegister(div->lhs()), useRegister(div->rhs()), temp());
  if (div->fallible()) {
    assignSnapshot(lir, div->bailoutKind());
  }
  define(lir, div);
}

void LIRGeneratorPPC64::lowerDivI64(MDiv* div) {
  auto* lir = new (alloc())
      LDivOrModI64(useRegister(div->lhs()), useRegister(div->rhs()));
  defineInt64(lir, div);
}

void LIRGeneratorPPC64::lowerModI(MMod* mod) {
  if (mod->rhs()->isConstant()) {
    int32_t rhs = mod->rhs()->toConstant()->toInt32();
    int32_t shift = FloorLog2(uint32_t(rhs));
    if (rhs > 0 && 1 << shift == rhs) {
      LModPowTwoI* lir =
          new (alloc()) LModPowTwoI(useRegister(mod->lhs()), shift);
      if (mod->fallible()) {
        assignSnapshot(lir, mod->bailoutKind());
      }
      define(lir, mod);
      return;
    } else if (shift < 31 && (1 << (shift + 1)) - 1 == rhs) {
      LModMaskI* lir = new (alloc())
          LModMaskI(useRegister(mod->lhs()), temp(), temp(), shift + 1);
      if (mod->fallible()) {
        assignSnapshot(lir, mod->bailoutKind());
      }
      define(lir, mod);
      return;
    }
  }
  auto* lir =
      new (alloc()) LModI(useRegister(mod->lhs()), useRegister(mod->rhs()));
  if (mod->fallible()) {
    assignSnapshot(lir, mod->bailoutKind());
  }
  define(lir, mod);
}

void LIRGeneratorPPC64::lowerModI64(MMod* mod) {
  auto* lir = new (alloc())
      LDivOrModI64(useRegister(mod->lhs()), useRegister(mod->rhs()));
  defineInt64(lir, mod);
}

void LIRGeneratorPPC64::lowerUDiv(MDiv* div) {
  MDefinition* lhs = div->getOperand(0);
  MDefinition* rhs = div->getOperand(1);
  LUDivOrMod* lir = new (alloc()) LUDivOrMod;
  // useRegisterAtStart: CodeGenerator-ppc64's visitUDivOrMod zero-extends
  // lhs/rhs into their own slots in place before the 32-bit divwu, so the
  // inputs must not be required live after the LIR op begins.
  lir->setOperand(0, useRegisterAtStart(lhs));
  lir->setOperand(1, useRegisterAtStart(rhs));
  if (div->fallible()) {
    assignSnapshot(lir, div->bailoutKind());
  }
  define(lir, div);
}

void LIRGeneratorPPC64::lowerUDivI64(MDiv* div) {
  auto* lir = new (alloc())
      LUDivOrModI64(useRegister(div->lhs()), useRegister(div->rhs()));
  defineInt64(lir, div);
}

void LIRGeneratorPPC64::lowerUMod(MMod* mod) {
  MDefinition* lhs = mod->getOperand(0);
  MDefinition* rhs = mod->getOperand(1);
  LUDivOrMod* lir = new (alloc()) LUDivOrMod;
  // See lowerUDiv above for why useRegisterAtStart is required here.
  lir->setOperand(0, useRegisterAtStart(lhs));
  lir->setOperand(1, useRegisterAtStart(rhs));
  if (mod->fallible()) {
    assignSnapshot(lir, mod->bailoutKind());
  }
  define(lir, mod);
}

void LIRGeneratorPPC64::lowerUModI64(MMod* mod) {
  auto* lir = new (alloc())
      LUDivOrModI64(useRegister(mod->lhs()), useRegister(mod->rhs()));
  defineInt64(lir, mod);
}

void LIRGeneratorPPC64::lowerUrshD(MUrsh* mir) {
  MDefinition* lhs = mir->lhs();
  MDefinition* rhs = mir->rhs();
  MOZ_ASSERT(lhs->type() == MIRType::Int32);
  MOZ_ASSERT(rhs->type() == MIRType::Int32);
  auto* lir = new (alloc()) LUrshD(useRegisterAtStart(lhs),
                                   useRegisterOrConstantAtStart(rhs), temp());
  define(lir, mir);
}

void LIRGeneratorPPC64::lowerPowOfTwoI(MPow* mir) {
  int32_t base = mir->input()->toConstant()->toInt32();
  MDefinition* power = mir->power();
  auto* lir = new (alloc()) LPowOfTwoI(useRegister(power), base);
  assignSnapshot(lir, mir->bailoutKind());
  define(lir, mir);
}

void LIRGeneratorPPC64::lowerBigIntPtrDiv(MBigIntPtrDiv* ins) {
  auto* lir = new (alloc())
      LBigIntPtrDiv(useRegister(ins->lhs()), useRegister(ins->rhs()),
                    LDefinition::BogusTemp(), LDefinition::BogusTemp());
  assignSnapshot(lir, ins->bailoutKind());
  define(lir, ins);
}

void LIRGeneratorPPC64::lowerBigIntPtrMod(MBigIntPtrMod* ins) {
  auto* lir = new (alloc())
      LBigIntPtrMod(useRegister(ins->lhs()), useRegister(ins->rhs()), temp(),
                    LDefinition::BogusTemp());
  if (ins->canBeDivideByZero()) {
    assignSnapshot(lir, ins->bailoutKind());
  }
  define(lir, ins);
}

void LIRGeneratorPPC64::lowerBigIntPtrLsh(MBigIntPtrLsh* ins) {
  auto* lir = new (alloc()) LBigIntPtrLsh(
      useRegister(ins->lhs()), useRegister(ins->rhs()), temp(), temp());
  assignSnapshot(lir, ins->bailoutKind());
  define(lir, ins);
}

void LIRGeneratorPPC64::lowerBigIntPtrRsh(MBigIntPtrRsh* ins) {
  auto* lir = new (alloc()) LBigIntPtrRsh(
      useRegister(ins->lhs()), useRegister(ins->rhs()), temp(), temp());
  assignSnapshot(lir, ins->bailoutKind());
  define(lir, ins);
}

void LIRGeneratorPPC64::lowerTruncateDToInt32(MTruncateToInt32* ins) {
  MDefinition* opd = ins->input();
  MOZ_ASSERT(opd->type() == MIRType::Double);
  define(new (alloc()) LTruncateDToInt32(useRegister(opd), tempDouble()), ins);
}

void LIRGeneratorPPC64::lowerTruncateFToInt32(MTruncateToInt32* ins) {
  MDefinition* opd = ins->input();
  MOZ_ASSERT(opd->type() == MIRType::Float32);
  define(new (alloc()) LTruncateFToInt32(useRegister(opd), tempFloat32()), ins);
}

void LIRGeneratorPPC64::lowerBuiltinInt64ToFloatingPoint(
    MBuiltinInt64ToFloatingPoint* ins) {
  MOZ_CRASH("We don't use it for this architecture");
}

void LIRGeneratorPPC64::lowerWasmSelectI(MWasmSelect* select) {
  auto* lir = new (alloc())
      LWasmSelect(useRegisterAtStart(select->trueExpr()),
                  useAny(select->falseExpr()), useRegister(select->condExpr()));
  defineReuseInput(lir, select, LWasmSelect::TrueExprIndex);
}

void LIRGeneratorPPC64::lowerWasmSelectI64(MWasmSelect* select) {
  auto* lir = new (alloc()) LWasmSelectI64(
      useInt64RegisterAtStart(select->trueExpr()),
      useInt64(select->falseExpr()), useRegister(select->condExpr()));
  defineInt64ReuseInput(lir, select, LWasmSelectI64::TrueExprIndex);
}

void LIRGeneratorPPC64::lowerWasmBuiltinTruncateToInt32(
    MWasmBuiltinTruncateToInt32* ins) {
  MDefinition* opd = ins->input();
  MOZ_ASSERT(opd->type() == MIRType::Double || opd->type() == MIRType::Float32);

  if (opd->type() == MIRType::Double) {
    define(new (alloc()) LWasmBuiltinTruncateDToInt32(
               useRegister(opd), useFixed(ins->instance(), InstanceReg),
               LDefinition::BogusTemp()),
           ins);
    return;
  }

  define(new (alloc()) LWasmBuiltinTruncateFToInt32(
             useRegister(opd), useFixed(ins->instance(), InstanceReg),
             LDefinition::BogusTemp()),
         ins);
}

void LIRGeneratorPPC64::lowerWasmBuiltinTruncateToInt64(
    MWasmBuiltinTruncateToInt64* ins) {
  MOZ_CRASH("We don't use it for this architecture");
}

void LIRGeneratorPPC64::lowerWasmBuiltinDivI64(MWasmBuiltinDivI64* div) {
  MOZ_CRASH("We don't use runtime div for this architecture");
}

void LIRGeneratorPPC64::lowerWasmBuiltinModI64(MWasmBuiltinModI64* mod) {
  MOZ_CRASH("We don't use runtime mod for this architecture");
}

void LIRGeneratorPPC64::lowerAtomicLoad64(MLoadUnboxedScalar* ins) {
  const LUse elements = useRegister(ins->elements());
  const LAllocation index =
      useRegisterOrIndexConstant(ins->index(), ins->storageType());
  auto* lir = new (alloc()) LAtomicLoad64(elements, index);
  defineInt64(lir, ins);
}

void LIRGeneratorPPC64::lowerAtomicStore64(MStoreUnboxedScalar* ins) {
  LUse elements = useRegister(ins->elements());
  LAllocation index =
      useRegisterOrIndexConstant(ins->index(), ins->writeType());
  LInt64Allocation value = useInt64Register(ins->value());
  add(new (alloc()) LAtomicStore64(elements, index, value), ins);
}

// ===============================================================
// LIRGenerator::visit* implementations

void LIRGenerator::visitBox(MBox* box) {
  MDefinition* opd = box->getOperand(0);

  if (opd->isConstant() && box->canEmitAtUses()) {
    emitAtUses(box);
    return;
  }

  if (opd->isConstant()) {
    define(new (alloc()) LValue(opd->toConstant()->toJSValue()), box,
           LDefinition(LDefinition::BOX));
  } else {
    LBox* ins = new (alloc()) LBox(useRegisterAtStart(opd), opd->type());
    define(ins, box, LDefinition(LDefinition::BOX));
  }
}

void LIRGenerator::visitUnbox(MUnbox* unbox) {
  MDefinition* box = unbox->getOperand(0);
  MOZ_ASSERT(box->type() == MIRType::Value);

  LInstructionHelper<1, BOX_PIECES, 0>* lir;
  if (IsFloatingPointType(unbox->type())) {
    MOZ_ASSERT(unbox->type() == MIRType::Double);
    lir = new (alloc()) LUnboxFloatingPoint(useBoxAtStart(box));
  } else if (unbox->fallible()) {
    lir = new (alloc()) LUnbox(useRegisterAtStart(box));
  } else {
    lir = new (alloc()) LUnbox(useAtStart(box));
  }

  if (unbox->fallible()) {
    assignSnapshot(lir, unbox->bailoutKind());
  }

  define(lir, unbox);
}

void LIRGenerator::visitCopySign(MCopySign* ins) {
  MDefinition* lhs = ins->lhs();
  MDefinition* rhs = ins->rhs();

  MOZ_ASSERT(IsFloatingPointType(lhs->type()));
  MOZ_ASSERT(lhs->type() == rhs->type());
  MOZ_ASSERT(lhs->type() == ins->type());

  LInstructionHelper<1, 2, 0>* lir;
  if (lhs->type() == MIRType::Double) {
    lir = new (alloc()) LCopySignD();
  } else {
    lir = new (alloc()) LCopySignF();
  }

  lowerForFPU(lir, ins, lhs, rhs);
}

void LIRGenerator::visitExtendInt32ToInt64(MExtendInt32ToInt64* ins) {
  defineInt64(
      new (alloc()) LExtendInt32ToInt64(useRegisterAtStart(ins->input())), ins);
}

void LIRGenerator::visitSignExtendInt64(MSignExtendInt64* ins) {
  defineInt64(new (alloc())
                  LSignExtendInt64(useInt64RegisterAtStart(ins->input())),
              ins);
}

void LIRGenerator::visitInt64ToFloatingPoint(MInt64ToFloatingPoint* ins) {
  MDefinition* opd = ins->input();
  MOZ_ASSERT(opd->type() == MIRType::Int64);
  MOZ_ASSERT(IsFloatingPointType(ins->type()));
  define(new (alloc()) LInt64ToFloatingPoint(useInt64Register(opd)), ins);
}

void LIRGenerator::visitSubstr(MSubstr* ins) {
  LSubstr* lir = new (alloc())
      LSubstr(useRegister(ins->string()), useRegister(ins->begin()),
              useRegister(ins->length()), temp(), temp(), temp());
  define(lir, ins);
  assignSafepoint(lir, ins);
}

void LIRGenerator::visitReturnImpl(MDefinition* opd, bool isGenerator) {
  MOZ_ASSERT(opd->type() == MIRType::Value);
  LReturn* ins = new (alloc()) LReturn(isGenerator);
  ins->setOperand(0, useFixed(opd, JSReturnReg));
  add(ins);
}
void LIRGenerator::visitCompareExchangeTypedArrayElement(
    MCompareExchangeTypedArrayElement* ins) {
  MOZ_ASSERT(!Scalar::isFloatingType(ins->arrayType()));
  MOZ_ASSERT(ins->elements()->type() == MIRType::Elements);
  MOZ_ASSERT(ins->index()->type() == MIRType::IntPtr);

  const LUse elements = useRegister(ins->elements());
  const LAllocation index =
      useRegisterOrIndexConstant(ins->index(), ins->arrayType());

  if (Scalar::isBigIntType(ins->arrayType())) {
    LInt64Allocation oldval = useInt64Register(ins->oldval());
    LInt64Allocation newval = useInt64Register(ins->newval());

    auto* lir = new (alloc())
        LCompareExchangeTypedArrayElement64(elements, index, oldval, newval);
    defineInt64(lir, ins);
    return;
  }

  const LAllocation oldval = useRegister(ins->oldval());
  const LAllocation newval = useRegister(ins->newval());

  LDefinition outTemp = LDefinition::BogusTemp();
  LDefinition valueTemp = LDefinition::BogusTemp();
  LDefinition offsetTemp = LDefinition::BogusTemp();
  LDefinition maskTemp = LDefinition::BogusTemp();

  if (ins->arrayType() == Scalar::Uint32 && IsFloatingPointType(ins->type())) {
    outTemp = temp();
  }

  if (Scalar::byteSize(ins->arrayType()) < 4) {
    // PPC64 sub-word CAS uses lbarx/lharx + stbcx./sthcx. (POWER7+); only
    // valueTemp is needed, to hold the extsb/extsh-canonicalised oldval
    // for the 32-bit cmpw. offsetTemp/maskTemp are unused (no round-down
    // + bit-isolate dance), and remain BogusTemp.
    valueTemp = temp();
  }

  LCompareExchangeTypedArrayElement* lir = new (alloc())
      LCompareExchangeTypedArrayElement(elements, index, oldval, newval,
                                        outTemp, valueTemp, offsetTemp,
                                        maskTemp);

  define(lir, ins);
}

void LIRGenerator::visitAtomicExchangeTypedArrayElement(
    MAtomicExchangeTypedArrayElement* ins) {
  MOZ_ASSERT(ins->elements()->type() == MIRType::Elements);
  MOZ_ASSERT(ins->index()->type() == MIRType::IntPtr);

  const LUse elements = useRegister(ins->elements());
  const LAllocation index =
      useRegisterOrIndexConstant(ins->index(), ins->arrayType());

  if (Scalar::isBigIntType(ins->arrayType())) {
    LInt64Allocation value = useInt64Register(ins->value());

    auto* lir = new (alloc())
        LAtomicExchangeTypedArrayElement64(elements, index, value);
    defineInt64(lir, ins);
    return;
  }

  MOZ_ASSERT(ins->arrayType() <= Scalar::Uint32);

  const LAllocation value = useRegister(ins->value());

  LDefinition outTemp = LDefinition::BogusTemp();
  LDefinition valueTemp = LDefinition::BogusTemp();
  LDefinition offsetTemp = LDefinition::BogusTemp();
  LDefinition maskTemp = LDefinition::BogusTemp();

  if (ins->arrayType() == Scalar::Uint32) {
    MOZ_ASSERT(ins->type() == MIRType::Double);
    outTemp = temp();
  }

  // PPC64 sub-word atomic exchange uses lbarx/lharx + stbcx./sthcx. directly
  // (POWER7+); valueTemp/offsetTemp/maskTemp are never read by the
  // implementation (see MacroAssembler-ppc64.cpp's AtomicExchange template).
  // Leave them as BogusTemp.

  LAtomicExchangeTypedArrayElement* lir =
      new (alloc()) LAtomicExchangeTypedArrayElement(
          elements, index, value, outTemp, valueTemp, offsetTemp, maskTemp);

  define(lir, ins);
}

void LIRGenerator::visitAtomicTypedArrayElementBinop(
    MAtomicTypedArrayElementBinop* ins) {
  MOZ_ASSERT(ins->arrayType() != Scalar::Uint8Clamped);
  MOZ_ASSERT(!Scalar::isFloatingType(ins->arrayType()));
  MOZ_ASSERT(ins->elements()->type() == MIRType::Elements);
  MOZ_ASSERT(ins->index()->type() == MIRType::IntPtr);

  const LUse elements = useRegister(ins->elements());
  const LAllocation index =
      useRegisterOrIndexConstant(ins->index(), ins->arrayType());

  if (Scalar::isBigIntType(ins->arrayType())) {
    LInt64Allocation value = useInt64Register(ins->value());
    LInt64Definition temp = tempInt64();

    if (ins->isForEffect()) {
      auto* lir = new (alloc()) LAtomicTypedArrayElementBinopForEffect64(
          elements, index, value, temp);
      add(lir, ins);
      return;
    }

    auto* lir = new (alloc())
        LAtomicTypedArrayElementBinop64(elements, index, value, temp);
    defineInt64(lir, ins);
    return;
  }

  LAllocation value = useRegister(ins->value());
  LDefinition valueTemp = LDefinition::BogusTemp();
  LDefinition offsetTemp = LDefinition::BogusTemp();
  LDefinition maskTemp = LDefinition::BogusTemp();

  // PPC64 sub-word atomic-binop uses lbarx/lharx + stbcx./sthcx. (POWER7+).
  // The fetch-op variant needs valueTemp to hold the post-op value being
  // condition-stored (MacroAssembler-ppc64.cpp's AtomicFetchOp); the
  // for-effect variant uses an internal scratch and needs no temps at
  // all. offsetTemp/maskTemp are unused in either path.
  if (Scalar::byteSize(ins->arrayType()) < 4 && !ins->isForEffect()) {
    valueTemp = temp();
  }

  if (ins->isForEffect()) {
    LAtomicTypedArrayElementBinopForEffect* lir =
        new (alloc()) LAtomicTypedArrayElementBinopForEffect(
            elements, index, value, valueTemp, offsetTemp, maskTemp);
    add(lir, ins);
    return;
  }

  LDefinition outTemp = LDefinition::BogusTemp();

  if (ins->arrayType() == Scalar::Uint32 && IsFloatingPointType(ins->type())) {
    outTemp = temp();
  }

  LAtomicTypedArrayElementBinop* lir =
      new (alloc()) LAtomicTypedArrayElementBinop(
          elements, index, value, outTemp, valueTemp, offsetTemp, maskTemp);
  define(lir, ins);
}
void LIRGenerator::visitAsmJSLoadHeap(MAsmJSLoadHeap* ins) {
  MDefinition* base = ins->base();
  MOZ_ASSERT(base->type() == MIRType::Int32);

  MDefinition* boundsCheckLimit = ins->boundsCheckLimit();
  MOZ_ASSERT_IF(ins->needsBoundsCheck(),
                boundsCheckLimit->type() == MIRType::Int32);

  LAllocation baseAlloc = useRegisterAtStart(base);

  LAllocation limitAlloc = ins->needsBoundsCheck()
                               ? useRegisterAtStart(boundsCheckLimit)
                               : LAllocation();

  MOZ_ASSERT(!ins->hasMemoryBase());
  auto* lir =
      new (alloc()) LAsmJSLoadHeap(baseAlloc, limitAlloc, LAllocation());
  define(lir, ins);
}
void LIRGenerator::visitAsmJSStoreHeap(MAsmJSStoreHeap* ins) {
  MDefinition* base = ins->base();
  MOZ_ASSERT(base->type() == MIRType::Int32);

  MDefinition* boundsCheckLimit = ins->boundsCheckLimit();
  MOZ_ASSERT_IF(ins->needsBoundsCheck(),
                boundsCheckLimit->type() == MIRType::Int32);

  LAllocation baseAlloc = useRegisterAtStart(base);

  LAllocation limitAlloc = ins->needsBoundsCheck()
                               ? useRegisterAtStart(boundsCheckLimit)
                               : LAllocation();

  MOZ_ASSERT(!ins->hasMemoryBase());
  add(new (alloc()) LAsmJSStoreHeap(baseAlloc, useRegisterAtStart(ins->value()),
                                    limitAlloc, LAllocation()),
      ins);
}
void LIRGenerator::visitWasmLoad(MWasmLoad* ins) {
  MDefinition* base = ins->base();
  MOZ_ASSERT(base->type() == MIRType::Int32 || base->type() == MIRType::Int64);

  LAllocation memoryBase =
      ins->hasMemoryBase() ? LAllocation(useRegisterAtStart(ins->memoryBase()))
                           : LGeneralReg(HeapReg);

  LAllocation ptr = useRegisterAtStart(base);

  LDefinition ptrCopy = LDefinition::BogusTemp();
  if (ins->access().offset32()) {
    ptrCopy = tempCopy(base, 0);
  }

  if (ins->type() == MIRType::Int64) {
    auto* lir = new (alloc()) LWasmLoadI64(ptr, memoryBase, ptrCopy);
    defineInt64(lir, ins);
    return;
  }

  auto* lir = new (alloc()) LWasmLoad(ptr, memoryBase, ptrCopy);
  define(lir, ins);
}
void LIRGenerator::visitWasmStore(MWasmStore* ins) {
  MDefinition* base = ins->base();
  MOZ_ASSERT(base->type() == MIRType::Int32 || base->type() == MIRType::Int64);

  MDefinition* value = ins->value();
  LAllocation memoryBase =
      ins->hasMemoryBase() ? LAllocation(useRegisterAtStart(ins->memoryBase()))
                           : LGeneralReg(HeapReg);

  LAllocation baseAlloc = useRegisterAtStart(base);

  LDefinition ptrCopy = LDefinition::BogusTemp();
  if (ins->access().offset32()) {
    ptrCopy = tempCopy(base, 0);
  }

  if (ins->access().type() == Scalar::Int64) {
    LInt64Allocation valueAlloc = useInt64RegisterAtStart(value);
    auto* lir =
        new (alloc()) LWasmStoreI64(baseAlloc, valueAlloc, memoryBase, ptrCopy);
    add(lir, ins);
    return;
  }

  LAllocation valueAlloc = useRegisterAtStart(value);
  auto* lir =
      new (alloc()) LWasmStore(baseAlloc, valueAlloc, memoryBase, ptrCopy);
  add(lir, ins);
}
void LIRGenerator::visitWasmTruncateToInt64(MWasmTruncateToInt64* ins) {
  MDefinition* opd = ins->input();
  MOZ_ASSERT(opd->type() == MIRType::Double || opd->type() == MIRType::Float32);

  defineInt64(new (alloc()) LWasmTruncateToInt64(useRegister(opd)), ins);
}
void LIRGenerator::visitWasmUnsignedToDouble(MWasmUnsignedToDouble* ins) {
  MOZ_ASSERT(ins->input()->type() == MIRType::Int32);
  LWasmUint32ToDouble* lir =
      new (alloc()) LWasmUint32ToDouble(useRegisterAtStart(ins->input()));
  define(lir, ins);
}
void LIRGenerator::visitWasmUnsignedToFloat32(MWasmUnsignedToFloat32* ins) {
  MOZ_ASSERT(ins->input()->type() == MIRType::Int32);
  LWasmUint32ToFloat32* lir =
      new (alloc()) LWasmUint32ToFloat32(useRegisterAtStart(ins->input()));
  define(lir, ins);
}
void LIRGenerator::visitWasmCompareExchangeHeap(MWasmCompareExchangeHeap* ins) {
  MDefinition* base = ins->base();
  MOZ_ASSERT(base->type() == MIRType::Int32 || base->type() == MIRType::Int64);
  LAllocation memoryBase = ins->hasMemoryBase()
                               ? LAllocation(useRegister(ins->memoryBase()))
                               : LGeneralReg(HeapReg);

  if (ins->access().type() == Scalar::Int64) {
    auto* lir = new (alloc()) LWasmCompareExchangeI64(
        useRegister(base), useInt64Register(ins->oldValue()),
        useInt64Register(ins->newValue()), memoryBase);
    defineInt64(lir, ins);
    return;
  }

  LDefinition valueTemp = LDefinition::BogusTemp();
  LDefinition offsetTemp = LDefinition::BogusTemp();
  LDefinition maskTemp = LDefinition::BogusTemp();

  // PPC64 sub-word wasm CAS uses lbarx/lharx + stbcx./sthcx. (POWER7+);
  // valueTemp holds the extsb/extsh-canonicalised oldval for cmpw, while
  // offsetTemp/maskTemp are unused (no round-down + bit-isolate dance).
  if (ins->access().byteSize() < 4) {
    valueTemp = temp();
  }

  auto* lir = new (alloc())
      LWasmCompareExchangeHeap(useRegister(base), useRegister(ins->oldValue()),
                               useRegister(ins->newValue()), memoryBase,
                               valueTemp, offsetTemp, maskTemp);

  define(lir, ins);
}
void LIRGenerator::visitWasmAtomicExchangeHeap(MWasmAtomicExchangeHeap* ins) {
  MDefinition* base = ins->base();
  MOZ_ASSERT(base->type() == MIRType::Int32 || base->type() == MIRType::Int64);
  LAllocation memoryBase = ins->hasMemoryBase()
                               ? LAllocation(useRegister(ins->memoryBase()))
                               : LGeneralReg(HeapReg);

  if (ins->access().type() == Scalar::Int64) {
    auto* lir = new (alloc()) LWasmAtomicExchangeI64(
        useRegister(base), useInt64Register(ins->value()), memoryBase);
    defineInt64(lir, ins);
    return;
  }

  // PPC64 sub-word wasm atomic exchange uses lbarx/lharx + stbcx./sthcx.
  // (POWER7+); valueTemp/offsetTemp/maskTemp are never read by the
  // implementation (see MacroAssembler-ppc64.cpp's AtomicExchange template).
  // Pass BogusTemp for all three.
  LDefinition valueTemp = LDefinition::BogusTemp();
  LDefinition offsetTemp = LDefinition::BogusTemp();
  LDefinition maskTemp = LDefinition::BogusTemp();

  auto* lir = new (alloc())
      LWasmAtomicExchangeHeap(useRegister(base), useRegister(ins->value()),
                              memoryBase, valueTemp, offsetTemp, maskTemp);
  define(lir, ins);
}
void LIRGenerator::visitWasmAtomicBinopHeap(MWasmAtomicBinopHeap* ins) {
  MDefinition* base = ins->base();
  MOZ_ASSERT(base->type() == MIRType::Int32 || base->type() == MIRType::Int64);
  LAllocation memoryBase = ins->hasMemoryBase()
                               ? LAllocation(useRegister(ins->memoryBase()))
                               : LGeneralReg(HeapReg);

  if (ins->access().type() == Scalar::Int64) {
    auto* lir = new (alloc())
        LWasmAtomicBinopI64(useRegister(base), useInt64Register(ins->value()),
                            memoryBase, tempInt64());
    defineInt64(lir, ins);
    return;
  }

  LDefinition valueTemp = LDefinition::BogusTemp();
  LDefinition offsetTemp = LDefinition::BogusTemp();
  LDefinition maskTemp = LDefinition::BogusTemp();

  // PPC64 sub-word wasm atomic-binop uses lbarx/lharx + stbcx./sthcx.
  // (POWER7+). The fetch-op variant needs valueTemp for the post-op value
  // being condition-stored; the for-effect variant uses an internal
  // scratch and needs no temps at all. offsetTemp/maskTemp are unused
  // in either path.
  if (ins->access().byteSize() < 4 && ins->hasUses()) {
    valueTemp = temp();
  }

  if (!ins->hasUses()) {
    LWasmAtomicBinopHeapForEffect* lir = new (alloc())
        LWasmAtomicBinopHeapForEffect(useRegister(base),
                                      useRegister(ins->value()), memoryBase,
                                      valueTemp, offsetTemp, maskTemp);
    add(lir, ins);
    return;
  }

  auto* lir = new (alloc())
      LWasmAtomicBinopHeap(useRegister(base), useRegister(ins->value()),
                           memoryBase, valueTemp, offsetTemp, maskTemp);

  define(lir, ins);
}

// SIMD lowering
void LIRGenerator::visitWasmTernarySimd128(MWasmTernarySimd128* ins) {
#ifdef ENABLE_WASM_SIMD
  MOZ_ASSERT(ins->type() == MIRType::Simd128);
  // useRegister for v0/v1 and useRegisterAtStart only for v2 — matches
  // ARM64's V128Bitselect policy. defineReuseInput requires the reused
  // input to be useRegisterAtStart and the others to remain alive
  // (useRegister); reusing all three policies as useRegisterAtStart
  // trips the allocator's "*def->output() != alloc" assertion because
  // v0/v1 may then share the slot with the output.
  LDefinition temp0 = LDefinition::BogusTemp();
  if (ins->simdOp() == wasm::SimdOp::I32x4RelaxedDotI8x16I7x16AddS) {
    temp0 = tempSimd128();
  }
  auto* lir = new (alloc()) LWasmTernarySimd128(
      useRegister(ins->v0()), useRegister(ins->v1()),
      useRegisterAtStart(ins->v2()), temp0,
      ins->simdOp());
  // The PPC64 visitor (CodeGenerator-ppc64.cpp:visitWasmTernarySimd128)
  // emits the FMA / DOT_THEN_ADD chain with v2 as the implicit
  // accumulator. defineReuseInput tells the allocator to put `dest`
  // in v2's slot, eliminating the previous conditional moveSimd128.
  defineReuseInput(lir, ins, LWasmTernarySimd128::V2Index);
#else
  MOZ_CRASH("No SIMD");
#endif
}
void LIRGenerator::visitWasmBinarySimd128(MWasmBinarySimd128* ins) {
#ifdef ENABLE_WASM_SIMD
  MOZ_ASSERT(ins->type() == MIRType::Simd128);
  LDefinition temp0 = LDefinition::BogusTemp();
  LDefinition temp1 = LDefinition::BogusTemp();
  // mulInt64x2 (i64x2.mul) routes through GPRs (mfvsrd/mulld/mtvsrd) and
  // uses an internal ScratchSimd128 + GPR scratches; its FloatRegister
  // temp1/temp2 parameters are inherited from the shared ARM64+PPC64
  // signature but unused on PPC64. Only FP min/max need SIMD temps for
  // the wasm NaN-canonicalisation dance.
  if (ins->simdOp() == wasm::SimdOp::F32x4Min ||
      ins->simdOp() == wasm::SimdOp::F32x4Max ||
      ins->simdOp() == wasm::SimdOp::F64x2Min ||
      ins->simdOp() == wasm::SimdOp::F64x2Max) {
    temp0 = tempSimd128();
    temp1 = tempSimd128();
  }
  auto* lir = new (alloc()) LWasmBinarySimd128(
      useRegisterAtStart(ins->lhs()), useRegisterAtStart(ins->rhs()),
      temp0, temp1, ins->simdOp());
  define(lir, ins);
#else
  MOZ_CRASH("No SIMD");
#endif
}
void LIRGenerator::visitWasmBinarySimd128WithConstant(
    MWasmBinarySimd128WithConstant* ins) {
#ifdef ENABLE_WASM_SIMD
  MOZ_ASSERT(ins->type() == MIRType::Simd128);
  auto* lir = new (alloc()) LWasmBinarySimd128WithConstant(
      useRegisterAtStart(ins->lhs()), LDefinition::BogusTemp(), ins->rhs());
  define(lir, ins);
#else
  MOZ_CRASH("No SIMD");
#endif
}
void LIRGenerator::visitWasmShiftSimd128(MWasmShiftSimd128* ins) {
#ifdef ENABLE_WASM_SIMD
  MOZ_ASSERT(ins->type() == MIRType::Simd128);
  MOZ_ASSERT(ins->rhs()->type() == MIRType::Int32);

  if (ins->rhs()->isConstant()) {
    int32_t shiftCountMask;
    switch (ins->simdOp()) {
      case wasm::SimdOp::I8x16Shl:
      case wasm::SimdOp::I8x16ShrU:
      case wasm::SimdOp::I8x16ShrS:
        shiftCountMask = 7;
        break;
      case wasm::SimdOp::I16x8Shl:
      case wasm::SimdOp::I16x8ShrU:
      case wasm::SimdOp::I16x8ShrS:
        shiftCountMask = 15;
        break;
      case wasm::SimdOp::I32x4Shl:
      case wasm::SimdOp::I32x4ShrU:
      case wasm::SimdOp::I32x4ShrS:
        shiftCountMask = 31;
        break;
      case wasm::SimdOp::I64x2Shl:
      case wasm::SimdOp::I64x2ShrU:
      case wasm::SimdOp::I64x2ShrS:
        shiftCountMask = 63;
        break;
      default:
        MOZ_CRASH("Unexpected shift operation");
    }
    int32_t shiftCount = ins->rhs()->toConstant()->toInt32() & shiftCountMask;
#ifdef DEBUG
    js::wasm::ReportSimdAnalysis("shift -> constant shift");
#endif
    auto* lir = new (alloc())
        LWasmConstantShiftSimd128(useRegisterAtStart(ins->lhs()), shiftCount);
    define(lir, ins);
  } else {
#ifdef DEBUG
    js::wasm::ReportSimdAnalysis("shift -> variable shift");
#endif
    auto* lir = new (alloc()) LWasmVariableShiftSimd128(
        useRegisterAtStart(ins->lhs()), useRegisterAtStart(ins->rhs()));
    define(lir, ins);
  }
#else
  MOZ_CRASH("No SIMD");
#endif
}
#ifdef ENABLE_WASM_SIMD
// Helper: reconstruct raw Wasm byte lane indices from analyzed SimdShuffle.
static SimdConstant ReconstructShuffleBytes(const SimdShuffle& s) {
  int8_t bytes[16];
  if (s.permuteOp) {
    switch (*s.permuteOp) {
      case SimdPermuteOp::MOVE:
        for (int i = 0; i < 16; i++) bytes[i] = i;
        return SimdConstant::CreateX16(bytes);
      case SimdPermuteOp::PERMUTE_32x4: {
        const int32_t* w = reinterpret_cast<const int32_t*>(s.control.bytes());
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++) bytes[i*4+j] = w[i]*4+j;
        return SimdConstant::CreateX16(bytes);
      }
      case SimdPermuteOp::PERMUTE_16x8: {
        const int16_t* h = reinterpret_cast<const int16_t*>(s.control.bytes());
        for (int i = 0; i < 8; i++) {
          int idx = h[i] & 0x7;
          bytes[i*2] = idx*2;
          bytes[i*2+1] = idx*2+1;
        }
        return SimdConstant::CreateX16(bytes);
      }
      case SimdPermuteOp::BROADCAST_8x16: {
        int8_t lane = reinterpret_cast<const int8_t*>(s.control.bytes())[0];
        for (int i = 0; i < 16; i++) bytes[i] = lane;
        return SimdConstant::CreateX16(bytes);
      }
      case SimdPermuteOp::BROADCAST_16x8: {
        int8_t lane = reinterpret_cast<const int8_t*>(s.control.bytes())[0];
        for (int i = 0; i < 8; i++) {
          bytes[i*2] = lane*2; bytes[i*2+1] = lane*2+1;
        }
        return SimdConstant::CreateX16(bytes);
      }
      case SimdPermuteOp::ROTATE_RIGHT_8x16: {
        uint8_t shift = reinterpret_cast<const int8_t*>(s.control.bytes())[0];
        for (int i = 0; i < 16; i++) bytes[i] = (i + shift) % 16;
        return SimdConstant::CreateX16(bytes);
      }
      case SimdPermuteOp::SHIFT_RIGHT_8x16: {
        uint8_t shift = reinterpret_cast<const int8_t*>(s.control.bytes())[0];
        for (int i = 0; i < 16; i++) bytes[i] = (i+shift < 16) ? (i+shift) : 0;
        return SimdConstant::CreateX16(bytes);
      }
      case SimdPermuteOp::SHIFT_LEFT_8x16: {
        uint8_t shift = reinterpret_cast<const int8_t*>(s.control.bytes())[0];
        for (int i = 0; i < 16; i++) bytes[i] = (i >= shift) ? (i-shift) : 0;
        return SimdConstant::CreateX16(bytes);
      }
      case SimdPermuteOp::REVERSE_16x8:
        // Reverse bytes within each 16-bit lane: [1,0,3,2,5,4,...]
        for (int i = 0; i < 8; i++) {
          bytes[i*2] = i*2+1; bytes[i*2+1] = i*2;
        }
        return SimdConstant::CreateX16(bytes);
      case SimdPermuteOp::REVERSE_32x4:
        // Reverse bytes within each 32-bit lane: [3,2,1,0,7,6,5,4,...]
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++) bytes[i*4+j] = i*4+(3-j);
        return SimdConstant::CreateX16(bytes);
      case SimdPermuteOp::REVERSE_64x2:
        // Reverse bytes within each 64-bit lane: [7,6,5,4,3,2,1,0,15,...]
        for (int i = 0; i < 2; i++)
          for (int j = 0; j < 8; j++) bytes[i*8+j] = i*8+(7-j);
        return SimdConstant::CreateX16(bytes);
      default:
        break;
    }
  }
  // Handle SimdShuffleOp (two-operand patterns).
  if (s.shuffleOp) {
    switch (*s.shuffleOp) {
      case SimdShuffleOp::CONCAT_RIGHT_SHIFT_8x16: {
        // control[0] = suffix length. ARM64 uses 16-count as the EXT shift.
        // Reconstruct raw byte indices: EXT(rhs, lhs, 16-count) =
        // take (16-count) bytes from rhs end, then count bytes from lhs start.
        uint8_t count = 16 - reinterpret_cast<const int8_t*>(s.control.bytes())[0];
        for (int i = 0; i < 16; i++) {
          int idx = i + count;
          bytes[i] = (idx < 16) ? (idx + 16) : (idx - 16);
        }
        return SimdConstant::CreateX16(bytes);
      }
      case SimdShuffleOp::BLEND_8x16: {
        // control has 0 (lhs) or -1 (rhs) per byte.
        const int8_t* mask = reinterpret_cast<const int8_t*>(s.control.bytes());
        for (int i = 0; i < 16; i++)
          bytes[i] = mask[i] ? (i + 16) : i;
        return SimdConstant::CreateX16(bytes);
      }
      case SimdShuffleOp::BLEND_16x8: {
        const int16_t* mask = reinterpret_cast<const int16_t*>(s.control.bytes());
        for (int i = 0; i < 8; i++) {
          int base = mask[i] ? (i * 2 + 16) : (i * 2);
          bytes[i * 2] = base;
          bytes[i * 2 + 1] = base + 1;
        }
        return SimdConstant::CreateX16(bytes);
      }
#define INTERLEAVE(name, width, low_start, count) \
      case SimdShuffleOp::name: { \
        for (int i = 0; i < count; i++) { \
          int lhsIdx = low_start + i * width; \
          int rhsIdx = lhsIdx + 16; \
          for (int j = 0; j < width; j++) { \
            bytes[(i * 2) * width + j] = lhsIdx + j; \
            bytes[(i * 2 + 1) * width + j] = rhsIdx + j; \
          } \
        } \
        return SimdConstant::CreateX16(bytes); \
      }
      INTERLEAVE(INTERLEAVE_LOW_8x16, 1, 0, 8)
      INTERLEAVE(INTERLEAVE_HIGH_8x16, 1, 8, 8)
      INTERLEAVE(INTERLEAVE_LOW_16x8, 2, 0, 4)
      INTERLEAVE(INTERLEAVE_HIGH_16x8, 2, 8, 4)
      INTERLEAVE(INTERLEAVE_LOW_32x4, 4, 0, 2)
      INTERLEAVE(INTERLEAVE_HIGH_32x4, 4, 8, 2)
      INTERLEAVE(INTERLEAVE_LOW_64x2, 8, 0, 1)
      INTERLEAVE(INTERLEAVE_HIGH_64x2, 8, 8, 1)
#undef INTERLEAVE
      default:
        break;
    }
  }
  // PERMUTE_8x16, SHUFFLE_BLEND_8x16, etc: control should have raw byte indices.
  // Force to Int8x16 type to avoid assertions from mismatched types.
  if (s.control.type() == SimdConstant::Int8x16) {
    return s.control;
  }
  // Fallback: re-create as Int8x16 from raw bytes.
  memcpy(bytes, s.control.bytes(), 16);
  return SimdConstant::CreateX16(bytes);
}

#endif  // ENABLE_WASM_SIMD

void LIRGenerator::visitWasmShuffleSimd128(MWasmShuffleSimd128* ins) {
#ifdef ENABLE_WASM_SIMD
  MOZ_ASSERT(ins->type() == MIRType::Simd128);
  SimdShuffle s = ins->shuffle();
  switch (s.opd) {
    case SimdShuffle::Operand::LEFT:
    case SimdShuffle::Operand::RIGHT: {
      // Single-operand permute: the analysis has identified that only one
      // input matters (the other is zero or unused).
      LAllocation src;
      if (s.opd == SimdShuffle::Operand::LEFT) {
        src = useRegisterAtStart(ins->lhs());
      } else {
        src = useRegisterAtStart(ins->rhs());
      }
      auto* lir =
          new (alloc()) LWasmPermuteSimd128(src, *s.permuteOp, s.control);
      define(lir, ins);
      break;
    }
    case SimdShuffle::Operand::BOTH:
    case SimdShuffle::Operand::BOTH_SWAPPED: {
      SimdConstant ctrl = ReconstructShuffleBytes(s);
      LAllocation lhs, rhs;
      if (s.opd == SimdShuffle::Operand::BOTH_SWAPPED) {
        lhs = useRegisterAtStart(ins->rhs());
        rhs = useRegisterAtStart(ins->lhs());
      } else {
        lhs = useRegisterAtStart(ins->lhs());
        rhs = useRegisterAtStart(ins->rhs());
      }
      auto* lir = new (alloc()) LWasmShuffleSimd128(
          lhs, rhs, *s.shuffleOp, ctrl);
      define(lir, ins);
      break;
    }
  }
#else
  MOZ_CRASH("No SIMD");
#endif
}
void LIRGenerator::visitWasmReplaceLaneSimd128(MWasmReplaceLaneSimd128* ins) {
#ifdef ENABLE_WASM_SIMD
  MOZ_ASSERT(ins->type() == MIRType::Simd128);
  if (ins->rhs()->type() == MIRType::Int64) {
    auto* lir = new (alloc()) LWasmReplaceInt64LaneSimd128(
        useRegisterAtStart(ins->lhs()), useInt64Register(ins->rhs()));
    defineReuseInput(lir, ins, LWasmReplaceInt64LaneSimd128::LhsIndex);
  } else {
    auto* lir = new (alloc()) LWasmReplaceLaneSimd128(
        useRegisterAtStart(ins->lhs()), useRegister(ins->rhs()));
    defineReuseInput(lir, ins, LWasmReplaceLaneSimd128::LhsIndex);
  }
#else
  MOZ_CRASH("No SIMD");
#endif
}
void LIRGenerator::visitWasmScalarToSimd128(MWasmScalarToSimd128* ins) {
#ifdef ENABLE_WASM_SIMD
  MOZ_ASSERT(ins->type() == MIRType::Simd128);
  if (ins->input()->type() == MIRType::Int64) {
    auto* lir =
        new (alloc()) LWasmInt64ToSimd128(useInt64RegisterAtStart(ins->input()));
    define(lir, ins);
  } else {
    auto* lir =
        new (alloc()) LWasmScalarToSimd128(useRegisterAtStart(ins->input()));
    define(lir, ins);
  }
#else
  MOZ_CRASH("No SIMD");
#endif
}
void LIRGenerator::visitWasmUnarySimd128(MWasmUnarySimd128* ins) {
#ifdef ENABLE_WASM_SIMD
  MOZ_ASSERT(ins->type() == MIRType::Simd128);
  auto* lir = new (alloc())
      LWasmUnarySimd128(useRegisterAtStart(ins->input()),
                        LDefinition::BogusTemp());
  define(lir, ins);
#else
  MOZ_CRASH("No SIMD");
#endif
}
#ifdef ENABLE_WASM_SIMD
bool LIRGeneratorPPC64::canFoldReduceSimd128AndBranch(wasm::SimdOp op) {
  switch (op) {
    case wasm::SimdOp::V128AnyTrue:
    case wasm::SimdOp::I8x16AllTrue:
    case wasm::SimdOp::I16x8AllTrue:
    case wasm::SimdOp::I32x4AllTrue:
    case wasm::SimdOp::I64x2AllTrue:
      return true;
    default:
      return false;
  }
}

bool LIRGeneratorPPC64::canEmitWasmReduceSimd128AtUses(
    MWasmReduceSimd128* ins) {
  if (!ins->canEmitAtUses()) {
    return false;
  }
  if (ins->type() != MIRType::Int32) {
    return false;
  }
  if (!canFoldReduceSimd128AndBranch(ins->simdOp())) {
    return false;
  }
  MUseIterator iter(ins->usesBegin());
  if (iter == ins->usesEnd()) {
    return true;
  }
  MNode* node = iter->consumer();
  if (!node->isDefinition() || !node->toDefinition()->isTest()) {
    return false;
  }
  iter++;
  return iter == ins->usesEnd();
}
#endif  // ENABLE_WASM_SIMD

void LIRGenerator::visitWasmReduceSimd128(MWasmReduceSimd128* ins) {
#ifdef ENABLE_WASM_SIMD
  if (canEmitWasmReduceSimd128AtUses(ins)) {
    emitAtUses(ins);
    return;
  }
  if (ins->type() == MIRType::Int64) {
    auto* lir = new (alloc())
        LWasmReduceSimd128ToInt64(useRegisterAtStart(ins->input()));
    defineInt64(lir, ins);
  } else {
    auto* lir =
        new (alloc()) LWasmReduceSimd128(useRegisterAtStart(ins->input()));
    define(lir, ins);
  }
#else
  MOZ_CRASH("No SIMD");
#endif
}
void LIRGenerator::visitWasmLoadLaneSimd128(MWasmLoadLaneSimd128* ins) {
#ifdef ENABLE_WASM_SIMD
  LUse base = useRegisterAtStart(ins->base());
  LUse inputUse = useRegisterAtStart(ins->value());
  LAllocation memoryBase =
      ins->hasMemoryBase() ? LAllocation(useRegisterAtStart(ins->memoryBase()))
                           : LGeneralReg(HeapReg);
  auto* lir = new (alloc()) LWasmLoadLaneSimd128(base, inputUse, memoryBase);
  define(lir, ins);
#else
  MOZ_CRASH("No SIMD");
#endif
}
void LIRGenerator::visitWasmStoreLaneSimd128(MWasmStoreLaneSimd128* ins) {
#ifdef ENABLE_WASM_SIMD
  LUse base = useRegisterAtStart(ins->base());
  LUse input = useRegisterAtStart(ins->value());
  LAllocation memoryBase =
      ins->hasMemoryBase() ? LAllocation(useRegisterAtStart(ins->memoryBase()))
                           : LGeneralReg(HeapReg);
  auto* lir = new (alloc()) LWasmStoreLaneSimd128(base, input, memoryBase);
  add(lir, ins);
#else
  MOZ_CRASH("No SIMD");
#endif
}

// PPC64 specializes compare+select for {U,}Int32 / {U,}Int64 compare with
// Int32 / Int64 result. The CodeGen visitor
// (CodeGenerator-ppc64.cpp:visitWasmCompareAndSelect) emits
// cmpw/cmplw/cmpd/cmpld + isel = 2 insns, replacing the ~5-7 insns the
// generic path would emit (boolean materialization + test + isel). FP
// specialization is not worthwhile — the generic FP select path already
// runs faster than the specialized integer one and PPC64 lacks a true
// fcsel equivalent (fsel only compares against zero).
bool LIRGeneratorShared::canSpecializeWasmCompareAndSelect(
    MCompare::CompareType compTy, MIRType insTy) {
  const bool insOk = insTy == MIRType::Int32 || insTy == MIRType::Int64;
  const bool cmpOk = compTy == MCompare::Compare_Int32 ||
                     compTy == MCompare::Compare_UInt32 ||
                     compTy == MCompare::Compare_Int64 ||
                     compTy == MCompare::Compare_UInt64;
  return insOk && cmpOk;
}

void LIRGeneratorShared::lowerWasmCompareAndSelect(MWasmSelect* ins,
                                                   MDefinition* lhs,
                                                   MDefinition* rhs,
                                                   MCompare::CompareType compTy,
                                                   JSOp jsop) {
  MOZ_ASSERT(canSpecializeWasmCompareAndSelect(compTy, ins->type()));
  auto* lir = new (alloc()) LWasmCompareAndSelect(
      useRegister(lhs), useRegister(rhs), useRegisterAtStart(ins->trueExpr()),
      useRegister(ins->falseExpr()), compTy, jsop);
  defineReuseInput(lir, ins, LWasmCompareAndSelect::IfTrueExprIndex);
}

// MIR helpers needed by the linker
#ifdef ENABLE_WASM_SIMD
bool MWasmTernarySimd128::specializeBitselectConstantMaskAsShuffle(
    int8_t shuffle[16]) {
  return false;
}
#endif

bool MWasmBinarySimd128::specializeForConstantRhs() { return false; }

#ifdef ENABLE_WASM_SIMD
bool MWasmTernarySimd128::canRelaxBitselect() { return false; }
#endif

#ifdef ENABLE_WASM_SIMD
bool MWasmBinarySimd128::canPmaddubsw() { return false; }
#endif

}  // namespace jit
}  // namespace js
