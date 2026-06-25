/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "jit/ppc64/CodeGenerator-ppc64.h"

#include "mozilla/MathAlgorithms.h"

#include <bit>

#include "builtin/Number.h"
#include "jit/CodeGenerator.h"
#include "jit/InlineScriptTree.h"
#include "jit/JitRuntime.h"
#include "jit/MIR-wasm.h"
#include "jit/MIR.h"
#include "jit/MIRGraph.h"
#include "vm/JSContext.h"
#include "vm/Realm.h"
#include "vm/Shape.h"

#include "jit/shared/CodeGenerator-shared-inl.h"
#include "vm/JSScript-inl.h"

using namespace js;
using namespace js::jit;

using JS::GenericNaN;
using mozilla::NegativeInfinity;

namespace js {
namespace jit {

CodeGeneratorPPC64::CodeGeneratorPPC64(MIRGenerator* gen, LIRGraph* graph,
                                       MacroAssembler* masm,
                                       const wasm::CodeMetadata* codeMeta)
    : CodeGeneratorShared(gen, graph, masm, codeMeta) {}

Operand CodeGeneratorPPC64::ToOperand(const LAllocation& a) {
  if (a.isGeneralReg()) {
    return Operand(a.toGeneralReg()->reg());
  }
  if (a.isFloatReg()) {
    return Operand(a.toFloatReg()->reg());
  }
  return Operand(ToAddress(a));
}

Operand CodeGeneratorPPC64::ToOperand(const LAllocation* a) {
  return ToOperand(*a);
}

MoveOperand CodeGeneratorPPC64::toMoveOperand(LAllocation a) const {
  if (a.isGeneralReg()) {
    return MoveOperand(ToRegister(a));
  }
  if (a.isFloatReg()) {
    return MoveOperand(ToFloatRegister(a));
  }
  MoveOperand::Kind kind = a.isStackArea() ? MoveOperand::Kind::EffectiveAddress
                                           : MoveOperand::Kind::Memory;
  Address address = ToAddress(a);
  MOZ_ASSERT((address.offset & 3) == 0);
  return MoveOperand(address, kind);
}

void CodeGeneratorPPC64::bailoutFrom(Label* label, LSnapshot* snapshot) {
  MOZ_ASSERT_IF(!masm.oom(), label->used());
  MOZ_ASSERT_IF(!masm.oom(), !label->bound());

  encode(snapshot);

  InlineScriptTree* tree = snapshot->mir()->block()->trackedTree();
  auto* ool = new (alloc()) LambdaOutOfLineCode([=, this](OutOfLineCode& ool) {
    // Push snapshotOffset and make sure stack is aligned.
    masm.subPtr(Imm32(sizeof(Value)), StackPointer);
    masm.storePtr(ImmWord(snapshot->snapshotOffset()),
                  Address(StackPointer, 0));
    masm.jump(&deoptLabel_);
  });
  addOutOfLineCode(ool,
                   new (alloc()) BytecodeSite(tree, tree->script()->code()));

  masm.retarget(label, ool->entry());
}

void CodeGeneratorPPC64::bailout(LSnapshot* snapshot) {
  Label label;
  masm.jump(&label);
  bailoutFrom(&label, snapshot);
}

void CodeGeneratorPPC64::bailoutIfFalseBool(Register lhs, LSnapshot* snapshot) {
  Label bail;
  masm.branchTest32(Assembler::Zero, lhs, Imm32(0xFF), &bail);
  bailoutFrom(&bail, snapshot);
}

bool CodeGeneratorPPC64::generateOutOfLineCode() {
  if (!CodeGeneratorShared::generateOutOfLineCode()) {
    return false;
  }

  if (deoptLabel_.used()) {
    masm.bind(&deoptLabel_);

    // Frame size is stored in LR and pushed by GenerateBailoutThunk
    // (via PushBailoutFrame -> pushReturnAddress -> mflr).
    {
      UseScratchRegisterScope temps(masm);
      Register scratch = temps.Acquire();
      masm.movePtr(ImmWord(frameSize()), scratch);
      masm.xs_mtlr(scratch);
    }

    TrampolinePtr handler = gen->jitRuntime()->getGenericBailoutHandler();
    masm.jump(handler);
  }

  return !masm.oom();
}

void CodeGeneratorPPC64::branchToBlock(MBasicBlock* block) {
  Label* label = skipTrivialBlocks(block)->lir()->label();
  masm.jump(label);
}

void CodeGeneratorPPC64::branchToBlock(Assembler::DoubleCondition cond,
                                       FloatRegister lhs, FloatRegister rhs,
                                       MBasicBlock* mir) {
  Label* label = skipTrivialBlocks(mir)->lir()->label();
  masm.branchDouble(cond, lhs, rhs, label);
}

void CodeGeneratorPPC64::branchToBlock(Assembler::FloatFormat fmt,
                                       Assembler::DoubleCondition cond,
                                       FloatRegister lhs, FloatRegister rhs,
                                       MBasicBlock* mir) {
  Label* label = skipTrivialBlocks(mir)->lir()->label();
  if (fmt == Assembler::DoubleFloat) {
    masm.branchDouble(cond, lhs, rhs, label);
  } else {
    masm.branchFloat(cond, lhs, rhs, label);
  }
}

class OutOfLineTableSwitch : public OutOfLineCodeBase<CodeGeneratorPPC64> {
  MTableSwitch* mir_;
  CodeLabel jumpLabel_;

  void accept(CodeGeneratorPPC64* codegen) {
    codegen->visitOutOfLineTableSwitch(this);
  }

 public:
  explicit OutOfLineTableSwitch(MTableSwitch* mir) : mir_(mir) {}

  MTableSwitch* mir() const { return mir_; }
  CodeLabel* jumpLabel() { return &jumpLabel_; }
};

void CodeGeneratorPPC64::emitTableSwitchDispatch(MTableSwitch* mir,
                                                 Register index,
                                                 Register base) {
  Label* defaultcase = skipTrivialBlocks(mir->getDefault())->lir()->label();

  if (mir->low() != 0) {
    masm.subPtr(Imm32(mir->low()), index);
  }

  int32_t cases = mir->numCases();
  masm.branchPtr(Assembler::AboveOrEqual, index, ImmWord(cases), defaultcase);

  OutOfLineTableSwitch* ool = new (alloc()) OutOfLineTableSwitch(mir);
  addOutOfLineCode(ool, mir);

  masm.mov(ool->jumpLabel(), base);

  BaseIndex pointer(base, index, ScalePointer);
  masm.branchToComputedAddress(pointer);
}

void CodeGeneratorPPC64::generateInvalidateEpilogue() {
  // Pad with enough nops so that PatchWrite_NearCall on the last OSI point
  // cannot overlap the invalidation epilogue. The patch area is
  // PatchWrite_NearCallSize (40) bytes; the last OSI point could be right
  // before this epilogue.
  for (size_t i = 0; i < Assembler::PatchWrite_NearCallSize();
       i += Assembler::NopSize()) {
    masm.nop();
  }

  masm.bind(&invalidate_);

  // Push the return address (LR) onto the stack.
  masm.pushReturnAddress();

  invalidateEpilogueData_ = masm.pushWithPatch(ImmWord(uintptr_t(-1)));

  TrampolinePtr thunk = gen->jitRuntime()->getInvalidationThunk();
  masm.jump(thunk);
}

void CodeGeneratorPPC64::visitOutOfLineTableSwitch(OutOfLineTableSwitch* ool) {
  MTableSwitch* mir = ool->mir();

  masm.haltingAlign(sizeof(void*));
  masm.bind(ool->jumpLabel());
  masm.addCodeLabel(*ool->jumpLabel());

  for (size_t i = 0; i < mir->numCases(); i++) {
    LBlock* caseblock = skipTrivialBlocks(mir->getCase(i))->lir();
    Label* caseheader = caseblock->label();
    uint32_t caseoffset = caseheader->offset();

    CodeLabel cl;
    masm.writeCodePointer(&cl);
    cl.target()->bind(caseoffset);
    masm.addCodeLabel(cl);
  }
}

void CodeGeneratorPPC64::visitOutOfLineWasmTruncateCheck(
    OutOfLineWasmTruncateCheck* ool) {
  if (ool->toType() == MIRType::Int32) {
    masm.outOfLineWasmTruncateToInt32Check(ool->input(), ool->output(),
                                           ool->fromType(), ool->flags(),
                                           ool->rejoin(), ool->trapSiteDesc());
  } else {
    MOZ_ASSERT(ool->toType() == MIRType::Int64);
    masm.outOfLineWasmTruncateToInt64Check(ool->input(), ool->output64(),
                                           ool->fromType(), ool->flags(),
                                           ool->rejoin(), ool->trapSiteDesc());
  }
}

void CodeGeneratorPPC64::emitBigIntPtrDiv(LBigIntPtrDiv* ins, Register dividend,
                                          Register divisor, Register output) {
  masm.as_divd(output, dividend, divisor);
}

void CodeGeneratorPPC64::emitBigIntPtrMod(LBigIntPtrMod* ins, Register dividend,
                                          Register divisor, Register output) {
  if (HasPOWER9()) {
    masm.as_modsd(output, dividend, divisor);
  } else {
    masm.as_divd(output, dividend, divisor);
    masm.as_mulld(output, output, divisor);
    masm.as_subf(output, output, dividend);
  }
}

// ===============================================================
// Visitors: Box/Unbox

void CodeGenerator::visitBox(LBox* box) {
  const LAllocation* in = box->getOperand(0);
  ValueOperand result = ToOutValue(box);

  masm.moveValue(TypedOrValueRegister(box->type(), ToAnyRegister(in)), result);
}

void CodeGenerator::visitUnbox(LUnbox* unbox) {
  MUnbox* mir = unbox->mir();

  Register result = ToRegister(unbox->output());

  if (mir->fallible()) {
    ValueOperand value = ToValue(unbox->input());
    Label bail;
    switch (mir->type()) {
      case MIRType::Int32:
        masm.fallibleUnboxInt32(value, result, &bail);
        break;
      case MIRType::Boolean:
        masm.fallibleUnboxBoolean(value, result, &bail);
        break;
      case MIRType::Object:
        masm.fallibleUnboxObject(value, result, &bail);
        break;
      case MIRType::String:
        masm.fallibleUnboxString(value, result, &bail);
        break;
      case MIRType::Symbol:
        masm.fallibleUnboxSymbol(value, result, &bail);
        break;
      case MIRType::BigInt:
        masm.fallibleUnboxBigInt(value, result, &bail);
        break;
      default:
        MOZ_CRASH("Given MIRType cannot be unboxed.");
    }
    bailoutFrom(&bail, unbox->snapshot());
    return;
  }

  LAllocation* input = unbox->getOperand(LUnbox::Input);
  if (input->isGeneralReg()) {
    Register inputReg = ToRegister(input);
    switch (mir->type()) {
      case MIRType::Int32:
        masm.unboxInt32(ValueOperand(inputReg), result);
        break;
      case MIRType::Boolean:
        masm.unboxBoolean(ValueOperand(inputReg), result);
        break;
      case MIRType::Object:
        masm.unboxObject(ValueOperand(inputReg), result);
        break;
      case MIRType::String:
        masm.unboxString(ValueOperand(inputReg), result);
        break;
      case MIRType::Symbol:
        masm.unboxSymbol(ValueOperand(inputReg), result);
        break;
      case MIRType::BigInt:
        masm.unboxBigInt(ValueOperand(inputReg), result);
        break;
      default:
        MOZ_CRASH("Given MIRType cannot be unboxed.");
    }
    return;
  }

  Address inputAddr = ToAddress(input);
  switch (mir->type()) {
    case MIRType::Int32:
      masm.unboxInt32(inputAddr, result);
      break;
    case MIRType::Boolean:
      masm.unboxBoolean(inputAddr, result);
      break;
    case MIRType::Object:
      masm.unboxObject(inputAddr, result);
      break;
    case MIRType::String:
      masm.unboxString(inputAddr, result);
      break;
    case MIRType::Symbol:
      masm.unboxSymbol(inputAddr, result);
      break;
    case MIRType::BigInt:
      masm.unboxBigInt(inputAddr, result);
      break;
    default:
      MOZ_CRASH("Given MIRType cannot be unboxed.");
  }
}

// ===============================================================
// Visitors: Integer Arithmetic

void CodeGenerator::visitAddI(LAddI* ins) {
  LAllocation* lhs = ins->getOperand(0);
  LAllocation* rhs = ins->getOperand(1);
  Register dest = ToRegister(ins->getDef(0));

  if (rhs->isConstant()) {
    Imm32 imm(ToInt32(rhs));
    if (ins->snapshot()) {
      masm.move32(ToRegister(lhs), dest);
      Label overflow;
      masm.branchAdd32(Assembler::Overflow, imm, dest, &overflow);
      bailoutFrom(&overflow, ins->snapshot());
    } else {
      masm.add32(imm, ToRegister(lhs), dest);
    }
  } else {
    Register rhsReg = ToRegister(rhs);
    if (ins->snapshot()) {
      // Use 3-operand add to avoid clobbering rhs when rhs == dest.
      masm.as_add(dest, ToRegister(lhs), rhsReg);
      // Check 32-bit overflow: sign-extend lower 32 and compare.
      masm.as_extsw(SecondScratchReg, dest);
      Label overflow;
      masm.as_cmpd(dest, SecondScratchReg);
      masm.ma_b(Assembler::NotEqual, &overflow);
      masm.as_extsw(dest, dest);
      bailoutFrom(&overflow, ins->snapshot());
    } else {
      masm.as_add(dest, ToRegister(lhs), rhsReg);
      masm.as_extsw(dest, dest);
    }
  }
}

void CodeGenerator::visitAddIntPtr(LAddIntPtr* ins) {
  Register dest = ToRegister(ins->getDef(0));
  Register lhs = ToRegister(ins->getOperand(0));
  const LAllocation* rhs = ins->getOperand(1);

  if (rhs->isConstant()) {
    if (lhs != dest) {
      masm.movePtr(lhs, dest);
    }
    masm.addPtr(ImmWord(ToIntPtr(rhs)), dest);
  } else {
    masm.as_add(dest, lhs, ToRegister(rhs));
  }
}

void CodeGenerator::visitAddI64(LAddI64* lir) {
  Register dest = ToRegister(lir->getDef(0));
  Register lhs = ToRegister(lir->getOperand(0));
  const LAllocation* rhs = lir->getOperand(1);

  if (rhs->isConstant()) {
    if (lhs != dest) {
      masm.movePtr(lhs, dest);
    }
    masm.addPtr(ImmWord(ToInt64(rhs)), dest);
  } else {
    masm.as_add(dest, lhs, ToRegister(rhs));
  }
}

void CodeGenerator::visitSubI(LSubI* ins) {
  LAllocation* lhs = ins->getOperand(0);
  LAllocation* rhs = ins->getOperand(1);
  Register dest = ToRegister(ins->getDef(0));

  if (rhs->isConstant()) {
    Imm32 imm(ToInt32(rhs));
    if (ins->snapshot()) {
      masm.move32(ToRegister(lhs), dest);
      Label overflow;
      masm.branchSub32(Assembler::Overflow, imm, dest, &overflow);
      bailoutFrom(&overflow, ins->snapshot());
    } else {
      masm.move32(ToRegister(lhs), dest);
      masm.sub32(imm, dest);
    }
  } else {
    Register rhsReg = ToRegister(rhs);
    if (ins->snapshot()) {
      // as_subf(d, a, b) computes d = b - a, so subf(dest, rhs, lhs) = lhs -
      // rhs
      masm.as_subf(dest, rhsReg, ToRegister(lhs));
      masm.as_extsw(SecondScratchReg, dest);
      Label overflow;
      masm.as_cmpd(dest, SecondScratchReg);
      masm.ma_b(Assembler::NotEqual, &overflow);
      masm.as_extsw(dest, dest);
      bailoutFrom(&overflow, ins->snapshot());
    } else {
      masm.as_subf(dest, rhsReg, ToRegister(lhs));
      masm.as_extsw(dest, dest);
    }
  }
}

void CodeGenerator::visitSubIntPtr(LSubIntPtr* ins) {
  Register dest = ToRegister(ins->getDef(0));
  Register lhs = ToRegister(ins->getOperand(0));
  const LAllocation* rhs = ins->getOperand(1);

  if (rhs->isConstant()) {
    if (lhs != dest) {
      masm.movePtr(lhs, dest);
    }
    masm.subPtr(Imm32(ToIntPtr(rhs)), dest);
  } else {
    // as_subf(d, a, b) = b - a
    masm.as_subf(dest, ToRegister(rhs), lhs);
  }
}

void CodeGenerator::visitSubI64(LSubI64* lir) {
  Register dest = ToRegister(lir->getDef(0));
  Register lhs = ToRegister(lir->getOperand(0));
  const LAllocation* rhs = lir->getOperand(1);

  if (rhs->isConstant()) {
    if (lhs != dest) {
      masm.movePtr(lhs, dest);
    }
    masm.sub64(Imm64(ToInt64(rhs)), Register64(dest));
  } else {
    // as_subf(d, a, b) = b - a
    masm.as_subf(dest, ToRegister(rhs), lhs);
  }
}

void CodeGenerator::visitMulI(LMulI* ins) {
  Register dest = ToRegister(ins->getDef(0));
  Register lhs = ToRegister(ins->getOperand(0));
  const LAllocation* rhs = ins->getOperand(1);
  MMul* mul = ins->mir();

  if (rhs->isConstant()) {
    int32_t constant = ToInt32(rhs);
    Register src = lhs;

    // Bailout on -0.0 before the special-case handling below, since cases
    // like -1 and 0 return early and would skip the check.
    if (mul->canBeNegativeZero() && constant <= 0) {
      Assembler::Condition cond =
          (constant == 0) ? Assembler::Signed : Assembler::Equal;
      bailoutCmp32(cond, src, Imm32(0), ins->snapshot());
    }

    switch (constant) {
      case -1:
        if (mul->canOverflow()) {
          Label ok;
          masm.branchPtr(Assembler::NotEqual, lhs, ImmWord(INT32_MIN), &ok);
          bailout(ins->snapshot());
          masm.bind(&ok);
        }
        masm.as_neg(dest, src);
        masm.as_extsw(dest, dest);
        return;
      case 0:
        masm.move32(Imm32(0), dest);
        return;
      case 1:
        masm.move32(src, dest);
        return;
      case 2:
        if (mul->canOverflow()) {
          masm.move32(src, dest);
          Label overflow;
          masm.branchAdd32(Assembler::Overflow, dest, dest, &overflow);
          bailoutFrom(&overflow, ins->snapshot());
        } else {
          masm.move32(src, dest);
          masm.add32(dest, dest);
        }
        return;
      default:
        break;
    }

    // Check for power of 2 (positive).
    uint32_t absCst = mozilla::Abs(constant);
    if (absCst > 0 && (absCst & (absCst - 1)) == 0 && !mul->canOverflow()) {
      uint32_t shift = mozilla::FloorLog2(absCst);
      masm.x_slwi(dest, src, shift);
      if (constant < 0) {
        masm.as_neg(dest, dest);
      }
      masm.as_extsw(dest, dest);
      return;
    }

    // General case.
    if (mul->canOverflow()) {
      masm.move32(src, dest);
      Label overflow;
      masm.branchMul32(Assembler::Overflow, Imm32(constant), dest, &overflow);
      bailoutFrom(&overflow, ins->snapshot());
    } else {
      masm.move32(src, dest);
      masm.mul32(Imm32(constant), dest);
    }

    // Check for negative zero (for constants not handled above).
    if (mul->canBeNegativeZero() && constant < 0) {
      Label ok;
      masm.branchPtr(Assembler::NotEqual, dest, ImmWord(0), &ok);
      bailoutCmp32(Assembler::Signed, src, src, ins->snapshot());
      masm.bind(&ok);
    }
    return;
  }

  Register rhsReg = ToRegister(rhs);

  if (mul->canOverflow()) {
    // Use 64-bit multiply so the full result is deterministic, then check
    // whether truncating to 32 bits changes the value. Match the
    // visitAddI/visitSubI ordering: branch first, truncate only on the
    // success path (the bailout discards dest anyway). extsw is
    // non-recording (ISA v3.0B) so it doesn't disturb CR0
    // either way; the choice is for consistency.
    masm.as_mulld(dest, lhs, rhsReg);
    masm.as_extsw(SecondScratchReg, dest);
    Label overflow;
    masm.as_cmpd(dest, SecondScratchReg);
    masm.ma_b(Assembler::NotEqual, &overflow);
    masm.as_extsw(dest, dest);
    bailoutFrom(&overflow, ins->snapshot());
  } else {
    masm.as_mullw(dest, lhs, rhsReg);
    masm.as_extsw(dest, dest);
  }

  if (mul->canBeNegativeZero()) {
    Label done;
    masm.branchPtr(Assembler::NotEqual, dest, ImmWord(0), &done);
    // Result is 0. Check if lhs|rhs was negative.
    {
      UseScratchRegisterScope temps(masm);
      Register scratch = temps.Acquire();
      masm.as_or_(scratch, lhs, rhsReg);
      bailoutCmp32(Assembler::Signed, scratch, scratch, ins->snapshot());
    }
    masm.bind(&done);
  }
}

void CodeGenerator::visitMulIntPtr(LMulIntPtr* ins) {
  Register dest = ToRegister(ins->getDef(0));
  Register lhs = ToRegister(ins->getOperand(0));
  const LAllocation* rhs = ins->getOperand(1);

  if (rhs->isConstant()) {
    if (lhs != dest) {
      masm.movePtr(lhs, dest);
    }
    masm.mulPtr(ImmWord(ToIntPtr(rhs)), dest);
  } else {
    masm.as_mulld(dest, lhs, ToRegister(rhs));
  }
}

void CodeGenerator::visitMulI64(LMulI64* lir) {
  Register dest = ToRegister(lir->getDef(0));
  Register lhs = ToRegister(lir->getOperand(0));
  const LAllocation* rhs = lir->getOperand(1);

  if (rhs->isConstant()) {
    if (lhs != dest) {
      masm.movePtr(lhs, dest);
    }
    masm.mulPtr(ImmWord(ToInt64(rhs)), dest);
  } else {
    masm.as_mulld(dest, lhs, ToRegister(rhs));
  }
}

void CodeGenerator::visitDivI(LDivI* ins) {
  Register lhs = ToRegister(ins->lhs());
  Register rhs = ToRegister(ins->rhs());
  Register dest = ToRegister(ins->output());
  Register temp = ToRegister(ins->temp0());
  MDiv* mir = ins->mir();

  Label done;

  // Handle divide by zero.
  if (mir->canBeDivideByZero()) {
    if (mir->trapOnError()) {
      Label nonZero;
      masm.branchPtr(Assembler::NotEqual, rhs, ImmWord(0), &nonZero);
      masm.wasmTrap(wasm::Trap::IntegerDivideByZero, mir->trapSiteDesc());
      masm.bind(&nonZero);
    } else if (mir->canTruncateInfinities()) {
      Label nonZero;
      masm.branchPtr(Assembler::NotEqual, rhs, ImmWord(0), &nonZero);
      masm.move32(Imm32(0), dest);
      masm.jump(&done);
      masm.bind(&nonZero);
    } else {
      MOZ_ASSERT(mir->fallible());
      bailoutCmp32(Assembler::Equal, rhs, Imm32(0), ins->snapshot());
    }
  }

  // Handle INT32_MIN / -1 overflow.
  if (mir->canBeNegativeOverflow()) {
    Label notMinInt;
    masm.branchPtr(Assembler::NotEqual, lhs, ImmWord(INT32_MIN), &notMinInt);
    masm.branchPtr(Assembler::NotEqual, rhs, ImmWord(-1), &notMinInt);

    if (mir->trapOnError()) {
      masm.wasmTrap(wasm::Trap::IntegerOverflow, mir->trapSiteDesc());
    } else if (mir->canTruncateOverflow()) {
      masm.move32(Imm32(INT32_MIN), dest);
      masm.jump(&done);
    } else {
      MOZ_ASSERT(mir->fallible());
      bailout(ins->snapshot());
    }
    masm.bind(&notMinInt);
  }

  // Handle negative zero.
  if (!mir->canTruncateNegativeZero() && mir->canBeNegativeZero()) {
    Label ok;
    masm.branchPtr(Assembler::NotEqual, lhs, ImmWord(0), &ok);
    bailoutCmp32(Assembler::LessThan, rhs, Imm32(0), ins->snapshot());
    masm.bind(&ok);
  }

  // Perform the division.
  masm.as_divw(dest, lhs, rhs);
  masm.as_extsw(dest, dest);

  // Check remainder if not truncatable.
  if (!mir->canTruncateRemainder()) {
    // Compute remainder: temp = lhs - (dest * rhs)
    masm.as_mullw(temp, dest, rhs);
    masm.as_subf(temp, temp, lhs);  // temp = lhs - temp
    bailoutCmp32(Assembler::NotEqual, temp, Imm32(0), ins->snapshot());
  }

  masm.bind(&done);
}

void CodeGenerator::visitDivPowTwoI(LDivPowTwoI* ins) {
  Register lhs = ToRegister(ins->numerator());
  Register dest = ToRegister(ins->output());
  UseScratchRegisterScope temps(masm);
  Register tmp = temps.Acquire();
  int32_t shift = ins->shift();

  if (shift != 0) {
    MDiv* mir = ins->mir();

    if (!mir->isTruncated()) {
      // If remainder != 0, bailout (check lower 'shift' bits).
      masm.x_slwi(tmp, lhs, 32 - shift);
      bailoutCmp32(Assembler::NotEqual, tmp, Imm32(0), ins->snapshot());
    }

    if (!mir->canBeNegativeDividend()) {
      // Non-negative dividend: simple right shift.
      masm.as_srawi(dest, lhs, shift);
    } else {
      // Need rounding adjustment for negative numbers.
      // Add (1 << shift) - 1 if lhs is negative.
      if (shift > 1) {
        masm.as_srawi(tmp, lhs, 31);
        masm.as_rlwinm(tmp, tmp, 0, 32 - shift, 31);
      } else {
        // shift == 1: extract sign bit into bit 31
        masm.as_rlwinm(tmp, lhs, 1, 31, 31);
      }
      masm.add32(lhs, tmp);
      masm.as_srawi(dest, tmp, shift);
    }
  } else {
    masm.move32(lhs, dest);
  }
}

void CodeGenerator::visitModI(LModI* ins) {
  Register lhs = ToRegister(ins->lhs());
  Register rhs = ToRegister(ins->rhs());
  Register dest = ToRegister(ins->output());
  UseScratchRegisterScope temps(masm);
  Register temp = temps.Acquire();
  MMod* mir = ins->mir();
  Label done;

  // Handle divide by zero.
  if (mir->canBeDivideByZero()) {
    if (mir->isTruncated()) {
      if (mir->trapOnError()) {
        Label nonZero;
        masm.branchPtr(Assembler::NotEqual, rhs, ImmWord(0), &nonZero);
        masm.wasmTrap(wasm::Trap::IntegerDivideByZero, mir->trapSiteDesc());
        masm.bind(&nonZero);
      } else {
        // Truncated division by zero yields integer zero.
        masm.move32(rhs, dest);
        Label nonZero;
        masm.branchPtr(Assembler::NotEqual, rhs, ImmWord(0), &nonZero);
        masm.jump(&done);
        masm.bind(&nonZero);
      }
    } else {
      MOZ_ASSERT(mir->fallible());
      bailoutCmp32(Assembler::Equal, rhs, Imm32(0), ins->snapshot());
    }
  }

  // Handle INT32_MIN % -1.
  // PPC64 divw is undefined for INT32_MIN / -1 (quotient overflows), so we
  // must return 0 explicitly.  The wasm spec also defines rem_s(MIN, -1) = 0.
  if (!mir->isUnsigned()) {
    Label notMinOverflow;
    masm.branchPtr(Assembler::NotEqual, lhs, ImmWord(INT32_MIN),
                   &notMinOverflow);
    masm.branchPtr(Assembler::NotEqual, rhs, ImmWord(-1), &notMinOverflow);
    masm.move32(Imm32(0), dest);
    masm.jump(&done);
    masm.bind(&notMinOverflow);
  }

  if (HasPOWER9()) {
    masm.as_modsw(dest, lhs, rhs);
  } else {
    masm.as_divw(temp, lhs, rhs);
    masm.as_mullw(temp, temp, rhs);
    masm.as_subf(dest, temp, lhs);
  }
  masm.as_extsw(dest, dest);

  // If X%Y == 0 and X < 0, the result is -0, and we need to bail out.
  if (mir->canBeNegativeDividend() && !mir->isTruncated()) {
    MOZ_ASSERT(mir->fallible());
    Label ok;
    masm.branchPtr(Assembler::NotEqual, dest, ImmWord(0), &ok);
    bailoutCmp32(Assembler::Signed, lhs, Imm32(0), ins->snapshot());
    masm.bind(&ok);
  }

  masm.bind(&done);
}

void CodeGenerator::visitModPowTwoI(LModPowTwoI* ins) {
  Register in = ToRegister(ins->getOperand(0));
  Register out = ToRegister(ins->getDef(0));
  MMod* mir = ins->mir();
  int32_t shift = ins->shift();
  uint32_t mask = (uint32_t(1) << shift) - 1;

  if (mir->canBeNegativeDividend() && !mir->isTruncated()) {
    Label nonNeg;
    masm.branchPtr(Assembler::NotEqual, in, ImmWord(0), &nonNeg);
    // in == 0: mod is 0, check for negative zero.
    bailoutCmp32(Assembler::Signed, in, in, ins->snapshot());
    masm.bind(&nonNeg);
  }

  Label negative, done;
  masm.branch32(Assembler::Signed, in, in, &negative);

  // Positive case: just mask.
  masm.and32(Imm32(mask), in, out);
  masm.jump(&done);

  // Negative case: negate, mask, negate back.
  masm.bind(&negative);
  masm.as_neg(out, in);
  masm.and32(Imm32(mask), out);
  masm.as_neg(out, out);
  masm.as_extsw(out, out);

  if (!mir->isTruncated() && mir->canBeNegativeDividend()) {
    Label ok;
    masm.branchPtr(Assembler::NotEqual, out, ImmWord(0), &ok);
    bailout(ins->snapshot());
    masm.bind(&ok);
  }

  masm.bind(&done);
}

void CodeGenerator::visitModMaskI(LModMaskI* ins) {
  Register src = ToRegister(ins->input());
  Register dest = ToRegister(ins->output());
  Register tmp0 = ToRegister(ins->temp0());
  Register tmp1 = ToRegister(ins->temp1());
  MMod* mir = ins->mir();

  if (!mir->isTruncated() && mir->canBeNegativeDividend()) {
    MOZ_ASSERT(mir->fallible());

    Label bail;
    masm.ma_mod_mask(src, dest, tmp0, tmp1, ins->shift(), &bail);
    bailoutFrom(&bail, ins->snapshot());
  } else {
    masm.ma_mod_mask(src, dest, tmp0, tmp1, ins->shift(), nullptr);
  }
}

void CodeGenerator::visitNegI(LNegI* ins) {
  Register input = ToRegister(ins->input());
  Register output = ToRegister(ins->output());
  masm.as_neg(output, input);
  masm.as_extsw(output, output);
}

void CodeGenerator::visitNegI64(LNegI64* ins) {
  Register input = ToRegister64(ins->input()).reg;
  Register output = ToOutRegister64(ins).reg;
  masm.as_neg(output, input);
}

void CodeGenerator::visitUDivOrMod(LUDivOrMod* ins) {
  Register lhs = ToRegister(ins->lhs());
  Register rhs = ToRegister(ins->rhs());
  Register output = ToRegister(ins->output());
  UseScratchRegisterScope temps(masm);
  Register temp = temps.Acquire();
  Label done;

  // Division by zero check.
  if (ins->canBeDivideByZero()) {
    if (ins->mir()->isTruncated()) {
      if (ins->trapOnError()) {
        Label nonZero;
        masm.branch32(Assembler::NotEqual, rhs, Imm32(0), &nonZero);
        masm.wasmTrap(wasm::Trap::IntegerDivideByZero, ins->trapSiteDesc());
        masm.bind(&nonZero);
      } else {
        Label nonZero;
        masm.branch32(Assembler::NotEqual, rhs, Imm32(0), &nonZero);
        masm.move32(Imm32(0), output);
        masm.jump(&done);
        masm.bind(&nonZero);
      }
    } else {
      bailoutCmp32(Assembler::Equal, rhs, Imm32(0), ins->snapshot());
    }
  }

  // Zero-extend both operands to 64 bits for unsigned divide.
  masm.move32To64ZeroExtend(lhs, Register64(lhs));
  masm.move32To64ZeroExtend(rhs, Register64(rhs));

  if (ins->mir()->isDiv()) {
    // Division path: compute quotient. Check remainder if needed.
    if (!ins->mir()->toDiv()->canTruncateRemainder()) {
      if (HasPOWER9()) {
        masm.as_moduw(temp, lhs, rhs);
      } else {
        masm.as_divwu(temp, lhs, rhs);
        masm.as_mullw(temp, temp, rhs);
        masm.as_subf(temp, temp, lhs);
      }
      bailoutCmp32(Assembler::NotEqual, temp, Imm32(0), ins->snapshot());
    }
    masm.as_divwu(output, lhs, rhs);
  } else {
    // Modulo path.
    if (HasPOWER9()) {
      masm.as_moduw(output, lhs, rhs);
    } else {
      masm.as_divwu(temp, lhs, rhs);
      masm.as_mullw(temp, temp, rhs);
      masm.as_subf(output, temp, lhs);
    }
  }

  masm.as_extsw(output, output);

  if (!ins->mir()->isTruncated()) {
    bailoutCmp32(Assembler::LessThan, output, Imm32(0), ins->snapshot());
  }

  masm.bind(&done);
}

void CodeGenerator::visitDivOrModI64(LDivOrModI64* lir) {
  Register lhs = ToRegister(lir->getOperand(0));
  Register rhs = ToRegister(lir->getOperand(1));
  Register output = ToRegister(lir->output());

  Label done;

  // Division by zero trap.
  if (lir->canBeDivideByZero()) {
    Label nonZero;
    masm.branchPtr(Assembler::NotEqual, rhs, ImmWord(0), &nonZero);
    masm.wasmTrap(wasm::Trap::IntegerDivideByZero, lir->trapSiteDesc());
    masm.bind(&nonZero);
  }

  // INT64_MIN / -1 overflow trap (for div only).
  if (lir->canBeNegativeOverflow()) {
    Label notMinInt;
    masm.branchPtr(Assembler::NotEqual, lhs, ImmWord(INT64_MIN), &notMinInt);
    masm.branchPtr(Assembler::NotEqual, rhs, ImmWord(-1), &notMinInt);
    if (lir->mir()->isDiv()) {
      masm.wasmTrap(wasm::Trap::IntegerOverflow, lir->trapSiteDesc());
    } else {
      masm.movePtr(ImmWord(0), output);
      masm.jump(&done);
    }
    masm.bind(&notMinInt);
  }

  if (lir->mir()->isDiv()) {
    masm.as_divd(output, lhs, rhs);
  } else if (HasPOWER9()) {
    masm.as_modsd(output, lhs, rhs);
  } else {
    masm.as_divd(output, lhs, rhs);
    masm.as_mulld(output, output, rhs);
    masm.as_subf(output, output, lhs);
  }

  masm.bind(&done);
}

void CodeGenerator::visitUDivOrModI64(LUDivOrModI64* lir) {
  Register lhs = ToRegister(lir->getOperand(0));
  Register rhs = ToRegister(lir->getOperand(1));
  Register output = ToRegister(lir->output());

  // Division by zero trap.
  if (lir->canBeDivideByZero()) {
    Label nonZero;
    masm.branchPtr(Assembler::NotEqual, rhs, ImmWord(0), &nonZero);
    masm.wasmTrap(wasm::Trap::IntegerDivideByZero, lir->trapSiteDesc());
    masm.bind(&nonZero);
  }

  if (lir->mir()->isDiv()) {
    masm.as_divdu(output, lhs, rhs);
  } else if (HasPOWER9()) {
    masm.as_modud(output, lhs, rhs);
  } else {
    masm.as_divdu(output, lhs, rhs);
    masm.as_mulld(output, output, rhs);
    masm.as_subf(output, output, lhs);
  }
}

// ===============================================================
// Visitors: Bitwise

void CodeGenerator::visitBitNotI(LBitNotI* ins) {
  Register input = ToRegister(ins->input());
  Register dest = ToRegister(ins->output());
  masm.as_nor(dest, input, input);
  masm.as_extsw(dest, dest);
}

void CodeGenerator::visitBitNotI64(LBitNotI64* ins) {
  Register input = ToRegister64(ins->input()).reg;
  Register dest = ToOutRegister64(ins).reg;
  masm.as_nor(dest, input, input);
}

void CodeGenerator::visitBitOpI(LBitOpI* ins) {
  Register dest = ToRegister(ins->getDef(0));
  Register lhs = ToRegister(ins->getOperand(0));
  const LAllocation* rhs = ins->getOperand(1);

  switch (ins->bitop()) {
    case JSOp::BitOr:
      if (rhs->isConstant()) {
        masm.or32(Imm32(ToInt32(rhs)), lhs, dest);
      } else {
        masm.as_or_(dest, lhs, ToRegister(rhs));
        masm.as_extsw(dest, dest);
      }
      break;
    case JSOp::BitXor:
      if (rhs->isConstant()) {
        masm.xor32(Imm32(ToInt32(rhs)), lhs, dest);
      } else {
        masm.as_xor_(dest, lhs, ToRegister(rhs));
        masm.as_extsw(dest, dest);
      }
      break;
    case JSOp::BitAnd:
      if (rhs->isConstant()) {
        masm.and32(Imm32(ToInt32(rhs)), lhs, dest);
      } else {
        masm.as_and_(dest, lhs, ToRegister(rhs));
        masm.as_extsw(dest, dest);
      }
      break;
    default:
      MOZ_CRASH("unexpected binary opcode");
  }
}

void CodeGenerator::visitBitOpI64(LBitOpI64* lir) {
  Register dest = ToRegister(lir->getDef(0));
  Register lhs = ToRegister(lir->getOperand(0));
  const LAllocation* rhs = lir->getOperand(1);

  switch (lir->bitop()) {
    case JSOp::BitOr:
      if (rhs->isConstant()) {
        if (lhs != dest) {
          masm.movePtr(lhs, dest);
        }
        masm.or64(Imm64(ToInt64(rhs)), Register64(dest));
      } else {
        masm.as_or_(dest, lhs, ToRegister(rhs));
      }
      break;
    case JSOp::BitXor:
      if (rhs->isConstant()) {
        if (lhs != dest) {
          masm.movePtr(lhs, dest);
        }
        masm.xor64(Imm64(ToInt64(rhs)), Register64(dest));
      } else {
        masm.as_xor_(dest, lhs, ToRegister(rhs));
      }
      break;
    case JSOp::BitAnd:
      if (rhs->isConstant()) {
        if (lhs != dest) {
          masm.movePtr(lhs, dest);
        }
        masm.and64(Imm64(ToInt64(rhs)), Register64(dest));
      } else {
        masm.as_and_(dest, lhs, ToRegister(rhs));
      }
      break;
    default:
      MOZ_CRASH("unexpected binary opcode");
  }
}

void CodeGenerator::visitShiftI(LShiftI* ins) {
  Register lhs = ToRegister(ins->lhs());
  const LAllocation* rhs = ins->rhs();
  Register dest = ToRegister(ins->output());

  if (rhs->isConstant()) {
    int32_t shift = ToInt32(rhs) & 0x1f;
    switch (ins->bitop()) {
      case JSOp::Lsh:
        if (shift) {
          masm.lshift32(Imm32(shift), lhs, dest);
        } else {
          masm.move32(lhs, dest);
        }
        break;
      case JSOp::Rsh:
        if (shift) {
          masm.rshift32Arithmetic(Imm32(shift), lhs, dest);
        } else {
          masm.move32(lhs, dest);
        }
        break;
      case JSOp::Ursh:
        if (shift) {
          masm.rshift32(Imm32(shift), lhs, dest);
        } else {
          // x >>> 0 can produce values that need to be treated as unsigned.
          masm.move32(lhs, dest);
        }
        if (ins->mir()->toUrsh()->fallible()) {
          // x >>> 0 can produce values that don't fit in signed int32.
          bailoutCmp32(Assembler::LessThan, dest, Imm32(0), ins->snapshot());
        }
        break;
      default:
        MOZ_CRASH("unexpected shift opcode");
    }
  } else {
    Register shiftReg = ToRegister(rhs);
    // PPC slw/srw/sraw use 6 bits of shift amount; JS requires mod 32.
    UseScratchRegisterScope temps(masm);
    Register masked = temps.Acquire();
    masm.as_rlwinm(masked, shiftReg, 0, 27, 31);
    switch (ins->bitop()) {
      case JSOp::Lsh:
        masm.as_slw(dest, lhs, masked);
        masm.as_extsw(dest, dest);
        break;
      case JSOp::Rsh:
        masm.as_sraw(dest, lhs, masked);
        break;
      case JSOp::Ursh:
        masm.as_srw(dest, lhs, masked);
        masm.as_extsw(dest, dest);
        if (ins->mir()->toUrsh()->fallible()) {
          bailoutCmp32(Assembler::LessThan, dest, Imm32(0), ins->snapshot());
        }
        break;
      default:
        MOZ_CRASH("unexpected shift opcode");
    }
  }
}

void CodeGenerator::visitShiftIntPtr(LShiftIntPtr* ins) {
  Register lhs = ToRegister(ins->lhs());
  Register dest = ToRegister(ins->output());

  if (ins->rhs()->isConstant()) {
    // ShiftIntPtr's RHS constant is IntPtr- or Int32-typed, not Int64. Use
    // ToIntPtr() which dispatches on the underlying MIRType (the previous
    // MConstant::toInt64() call asserted when the constant wasn't Int64).
    int32_t shift = int32_t(ToIntPtr(ins->rhs())) & 0x3f;
    switch (ins->bitop()) {
      case JSOp::Lsh:
        if (shift) {
          masm.lshiftPtr(Imm32(shift), lhs, dest);
        } else {
          masm.movePtr(lhs, dest);
        }
        break;
      case JSOp::Rsh:
        if (shift) {
          masm.rshiftPtrArithmetic(Imm32(shift), lhs, dest);
        } else {
          masm.movePtr(lhs, dest);
        }
        break;
      case JSOp::Ursh:
        if (shift) {
          masm.rshiftPtr(Imm32(shift), lhs, dest);
        } else {
          masm.movePtr(lhs, dest);
        }
        break;
      default:
        MOZ_CRASH("unexpected shift opcode");
    }
  } else {
    Register shiftReg = ToRegister(ins->rhs());
    // sld/srd/srad use the low 7 bits of the shift count: counts >= 64
    // produce 0 (sign-fill for srad). Mask to 6 bits for mod-64 semantics.
    UseScratchRegisterScope temps(masm);
    Register masked = temps.Acquire();
    masm.as_rldicl(masked, shiftReg, 0, 58);
    switch (ins->bitop()) {
      case JSOp::Lsh:
        masm.as_sld(dest, lhs, masked);
        break;
      case JSOp::Rsh:
        masm.as_srad(dest, lhs, masked);
        break;
      case JSOp::Ursh:
        masm.as_srd(dest, lhs, masked);
        break;
      default:
        MOZ_CRASH("unexpected shift opcode");
    }
  }
}

void CodeGenerator::visitShiftI64(LShiftI64* lir) {
  Register lhs = ToRegister64(lir->lhs()).reg;
  Register dest = ToOutRegister64(lir).reg;
  const LAllocation* rhs = lir->rhs();

  if (rhs->isConstant()) {
    int32_t shift = int32_t(rhs->toConstant()->toInt64()) & 0x3f;
    switch (lir->bitop()) {
      case JSOp::Lsh:
        if (shift) {
          masm.lshiftPtr(Imm32(shift), lhs, dest);
        } else {
          masm.movePtr(lhs, dest);
        }
        break;
      case JSOp::Rsh:
        if (shift) {
          masm.rshiftPtrArithmetic(Imm32(shift), lhs, dest);
        } else {
          masm.movePtr(lhs, dest);
        }
        break;
      case JSOp::Ursh:
        if (shift) {
          masm.rshiftPtr(Imm32(shift), lhs, dest);
        } else {
          masm.movePtr(lhs, dest);
        }
        break;
      default:
        MOZ_CRASH("unexpected shift opcode");
    }
  } else {
    Register shiftReg = ToRegister(rhs);
    // Wasm i64 shifts require shift count modulo 64. PPC64 sld/srd/srad
    // use a 7-bit shift field, so shifts >= 64 produce 0 (or sign-fill
    // for srad). Mask to 6 bits first.
    UseScratchRegisterScope temps(masm);
    Register masked = temps.Acquire();
    masm.as_rldicl(masked, shiftReg, 0, 58);  // clrldi: keep low 6 bits
    switch (lir->bitop()) {
      case JSOp::Lsh:
        masm.as_sld(dest, lhs, masked);
        break;
      case JSOp::Rsh:
        masm.as_srad(dest, lhs, masked);
        break;
      case JSOp::Ursh:
        masm.as_srd(dest, lhs, masked);
        break;
      default:
        MOZ_CRASH("unexpected shift opcode");
    }
  }
}

void CodeGenerator::visitUrshD(LUrshD* ins) {
  Register lhs = ToRegister(ins->lhs());
  const LAllocation* rhs = ins->rhs();
  FloatRegister dest = ToFloatRegister(ins->output());

  Register temp = ToRegister(ins->temp0());

  if (rhs->isConstant()) {
    int32_t shift = ToInt32(rhs) & 0x1f;
    if (shift) {
      masm.rshift32(Imm32(shift), lhs, temp);
    } else {
      masm.move32(lhs, temp);
    }
  } else {
    masm.move32(lhs, temp);
    masm.rshift32(ToRegister(rhs), temp);
  }

  masm.convertUInt32ToDouble(temp, dest);
}

// ===============================================================
// Visitors: Floating-point arithmetic

void CodeGenerator::visitMathD(LMathD* math) {
  FloatRegister lhs = ToFloatRegister(math->lhs());
  FloatRegister rhs = ToFloatRegister(math->rhs());
  FloatRegister dest = ToFloatRegister(math->output());

  switch (math->jsop()) {
    case JSOp::Add:
      masm.as_fadd(dest, lhs, rhs);
      break;
    case JSOp::Sub:
      masm.as_fsub(dest, lhs, rhs);
      break;
    case JSOp::Mul:
      masm.as_fmul(dest, lhs, rhs);
      break;
    case JSOp::Div:
      masm.as_fdiv(dest, lhs, rhs);
      break;
    default:
      MOZ_CRASH("unexpected double opcode");
  }
}

void CodeGenerator::visitMathF(LMathF* math) {
  FloatRegister lhs = ToFloatRegister(math->lhs());
  FloatRegister rhs = ToFloatRegister(math->rhs());
  FloatRegister dest = ToFloatRegister(math->output());

  switch (math->jsop()) {
    case JSOp::Add:
      masm.as_fadds(dest, lhs, rhs);
      break;
    case JSOp::Sub:
      masm.as_fsubs(dest, lhs, rhs);
      break;
    case JSOp::Mul:
      masm.as_fmuls(dest, lhs, rhs);
      break;
    case JSOp::Div:
      masm.as_fdivs(dest, lhs, rhs);
      break;
    default:
      MOZ_CRASH("unexpected float32 opcode");
  }
}

void CodeGenerator::visitMinMaxD(LMinMaxD* ins) {
  FloatRegister first = ToFloatRegister(ins->first());
  FloatRegister second = ToFloatRegister(ins->second());
  mozilla::DebugOnly<FloatRegister> output = ToFloatRegister(ins->output());

  MOZ_ASSERT(first == output);
  if (ins->mir()->isMax()) {
    masm.maxDouble(second, first, /* handleNaN = */ true);
  } else {
    masm.minDouble(second, first, /* handleNaN = */ true);
  }
}

void CodeGenerator::visitMinMaxF(LMinMaxF* ins) {
  FloatRegister first = ToFloatRegister(ins->first());
  FloatRegister second = ToFloatRegister(ins->second());
  mozilla::DebugOnly<FloatRegister> output = ToFloatRegister(ins->output());

  MOZ_ASSERT(first == output);
  if (ins->mir()->isMax()) {
    masm.maxFloat32(second, first, /* handleNaN = */ true);
  } else {
    masm.minFloat32(second, first, /* handleNaN = */ true);
  }
}

void CodeGenerator::visitNegD(LNegD* ins) {
  FloatRegister input = ToFloatRegister(ins->input());
  FloatRegister output = ToFloatRegister(ins->output());
  masm.as_fneg(output, input);
}

void CodeGenerator::visitNegF(LNegF* ins) {
  FloatRegister input = ToFloatRegister(ins->input());
  FloatRegister output = ToFloatRegister(ins->output());
  masm.as_fneg(output, input);
}

void CodeGenerator::visitPowHalfD(LPowHalfD* ins) {
  FloatRegister input = ToFloatRegister(ins->input());
  FloatRegister output = ToFloatRegister(ins->output());

  Label done, skip;

  // Check for -Infinity.
  masm.loadConstantDouble(NegativeInfinity<double>(), ScratchDoubleReg);
  masm.branchDouble(Assembler::DoubleNotEqualOrUnordered, input,
                    ScratchDoubleReg, &skip);
  masm.loadConstantDouble(std::numeric_limits<double>::infinity(), output);
  masm.jump(&done);

  masm.bind(&skip);
  // Add 0.0 to handle -0.
  masm.loadConstantDouble(0.0, ScratchDoubleReg);
  masm.as_fadd(output, input, ScratchDoubleReg);
  masm.as_fsqrt(output, output);

  masm.bind(&done);
}

void CodeGenerator::visitNotD(LNotD* ins) {
  FloatRegister input = ToFloatRegister(ins->input());
  Register dest = ToRegister(ins->output());

  masm.loadConstantDouble(0.0, ScratchDoubleReg);
  masm.as_fcmpu(input, ScratchDoubleReg);
  masm.ma_cmp_set_dbl(dest, Assembler::DoubleEqualOrUnordered);
}

void CodeGenerator::visitNotF(LNotF* ins) {
  FloatRegister input = ToFloatRegister(ins->input());
  Register dest = ToRegister(ins->output());

  masm.loadConstantFloat32(0.0f, ScratchFloat32Reg);
  masm.as_fcmpu(input, ScratchFloat32Reg);
  masm.ma_cmp_set_dbl(dest, Assembler::DoubleEqualOrUnordered);
}

// ===============================================================
// Visitors: FP comparisons and branches

void CodeGenerator::visitCompareD(LCompareD* comp) {
  FloatRegister lhs = ToFloatRegister(comp->left());
  FloatRegister rhs = ToFloatRegister(comp->right());
  Register dest = ToRegister(comp->output());
  Assembler::DoubleCondition cond =
      comp->mir()->jsop() == JSOp::StrictEq ? Assembler::DoubleEqual
      : comp->mir()->jsop() == JSOp::StrictNe
          ? Assembler::DoubleNotEqualOrUnordered
          : JSOpToDoubleCondition(comp->mir()->jsop());

  masm.as_fcmpu(lhs, rhs);
  masm.ma_cmp_set_dbl(dest, cond);
}

void CodeGenerator::visitCompareF(LCompareF* comp) {
  FloatRegister lhs = ToFloatRegister(comp->left());
  FloatRegister rhs = ToFloatRegister(comp->right());
  Register dest = ToRegister(comp->output());
  Assembler::DoubleCondition cond =
      comp->mir()->jsop() == JSOp::StrictEq ? Assembler::DoubleEqual
      : comp->mir()->jsop() == JSOp::StrictNe
          ? Assembler::DoubleNotEqualOrUnordered
          : JSOpToDoubleCondition(comp->mir()->jsop());

  masm.as_fcmpu(lhs, rhs);
  masm.ma_cmp_set_dbl(dest, cond);
}

void CodeGenerator::visitCompareDAndBranch(LCompareDAndBranch* comp) {
  FloatRegister lhs = ToFloatRegister(comp->left());
  FloatRegister rhs = ToFloatRegister(comp->right());

  Assembler::DoubleCondition cond =
      JSOpToDoubleCondition(comp->cmpMir()->jsop());
  MBasicBlock* ifTrue = comp->ifTrue();
  MBasicBlock* ifFalse = comp->ifFalse();

  if (isNextBlock(ifFalse->lir())) {
    branchToBlock(Assembler::DoubleFloat, cond, lhs, rhs, ifTrue);
  } else {
    branchToBlock(Assembler::DoubleFloat, Assembler::InvertCondition(cond), lhs,
                  rhs, ifFalse);
    jumpToBlock(ifTrue);
  }
}

void CodeGenerator::visitCompareFAndBranch(LCompareFAndBranch* comp) {
  FloatRegister lhs = ToFloatRegister(comp->left());
  FloatRegister rhs = ToFloatRegister(comp->right());

  Assembler::DoubleCondition cond =
      JSOpToDoubleCondition(comp->cmpMir()->jsop());
  MBasicBlock* ifTrue = comp->ifTrue();
  MBasicBlock* ifFalse = comp->ifFalse();

  if (isNextBlock(ifFalse->lir())) {
    branchToBlock(Assembler::SingleFloat, cond, lhs, rhs, ifTrue);
  } else {
    branchToBlock(Assembler::SingleFloat, Assembler::InvertCondition(cond), lhs,
                  rhs, ifFalse);
    jumpToBlock(ifTrue);
  }
}

void CodeGenerator::visitTestDAndBranch(LTestDAndBranch* test) {
  FloatRegister input = ToFloatRegister(test->input());

  MBasicBlock* ifTrue = test->ifTrue();
  MBasicBlock* ifFalse = test->ifFalse();

  masm.loadConstantDouble(0.0, ScratchDoubleReg);

  if (isNextBlock(ifFalse->lir())) {
    branchToBlock(Assembler::DoubleFloat, Assembler::DoubleNotEqual, input,
                  ScratchDoubleReg, ifTrue);
  } else {
    branchToBlock(Assembler::DoubleFloat, Assembler::DoubleEqualOrUnordered,
                  input, ScratchDoubleReg, ifFalse);
    jumpToBlock(ifTrue);
  }
}

void CodeGenerator::visitTestFAndBranch(LTestFAndBranch* test) {
  FloatRegister input = ToFloatRegister(test->input());

  MBasicBlock* ifTrue = test->ifTrue();
  MBasicBlock* ifFalse = test->ifFalse();

  masm.loadConstantFloat32(0.0f, ScratchFloat32Reg);

  if (isNextBlock(ifFalse->lir())) {
    branchToBlock(Assembler::SingleFloat, Assembler::DoubleNotEqual, input,
                  ScratchFloat32Reg, ifTrue);
  } else {
    branchToBlock(Assembler::SingleFloat, Assembler::DoubleEqualOrUnordered,
                  input, ScratchFloat32Reg, ifFalse);
    jumpToBlock(ifTrue);
  }
}

// ===============================================================
// Visitors: Truncation

void CodeGenerator::visitTruncateDToInt32(LTruncateDToInt32* ins) {
  emitTruncateDouble(ToFloatRegister(ins->input()), ToRegister(ins->output()),
                     ins->mir());
}

void CodeGenerator::visitTruncateFToInt32(LTruncateFToInt32* ins) {
  emitTruncateFloat32(ToFloatRegister(ins->input()), ToRegister(ins->output()),
                      ins->mir());
}

// ===============================================================
// Visitors: Int64 / Wasm type conversions

void CodeGenerator::visitExtendInt32ToInt64(LExtendInt32ToInt64* lir) {
  Register input = ToRegister(lir->input());
  Register output = ToRegister(lir->output());

  if (lir->mir()->isUnsigned()) {
    masm.move32To64ZeroExtend(input, Register64(output));
  } else {
    masm.as_extsw(output, input);
  }
}

void CodeGenerator::visitWrapInt64ToInt32(LWrapInt64ToInt32* lir) {
  const LInt64Allocation input = lir->input();
  Register output = ToRegister(lir->output());

  if (lir->mir()->bottomHalf()) {
    if (input.value().isMemory()) {
      masm.load32(ToAddress(input), output);
    } else {
      masm.move64To32(ToRegister64(input), output);
    }
  } else {
    // The only producer of `bottomHalf=false` MWrapInt64ToInt32 in the
    // current MIR pipeline is the GPR-pair argument splitter in
    // WasmIonCompile.cpp, which is gated on JS_CODEGEN_REGISTER_PAIR
    // (32-bit ARM only). PPC64 is 64-bit and never reaches this path.
    // Matches the same defensive crash in x64 / ARM64 backends.
    MOZ_CRASH("Not implemented.");
  }
}

void CodeGenerator::visitSignExtendInt64(LSignExtendInt64* lir) {
  Register64 input = ToRegister64(lir->input());
  Register64 output = ToOutRegister64(lir);

  switch (lir->mir()->mode()) {
    case MSignExtendInt64::Byte:
      masm.as_extsb(output.reg, input.reg);
      break;
    case MSignExtendInt64::Half:
      masm.as_extsh(output.reg, input.reg);
      break;
    case MSignExtendInt64::Word:
      masm.as_extsw(output.reg, input.reg);
      break;
  }
}

void CodeGenerator::visitWasmExtendU32Index(LWasmExtendU32Index* lir) {
  Register input = ToRegister(lir->input());
  Register output = ToRegister(lir->output());
  masm.move32To64ZeroExtend(input, Register64(output));
}

void CodeGenerator::visitWasmWrapU32Index(LWasmWrapU32Index* lir) {
  Register input = ToRegister(lir->input());
  Register output = ToRegister(lir->output());
  masm.move32(input, output);
}

void CodeGenerator::visitWasmTruncateToInt32(LWasmTruncateToInt32* lir) {
  auto input = ToFloatRegister(lir->input());
  auto output = ToRegister(lir->output());

  MWasmTruncateToInt32* mir = lir->mir();
  MIRType fromType = mir->input()->type();

  MOZ_ASSERT(fromType == MIRType::Double || fromType == MIRType::Float32);

  auto* ool = new (alloc()) OutOfLineWasmTruncateCheck(mir, input, output);
  addOutOfLineCode(ool, mir);

  Label* oolEntry = ool->entry();
  if (mir->isUnsigned()) {
    if (fromType == MIRType::Double) {
      masm.wasmTruncateDoubleToUInt32(input, output, mir->isSaturating(),
                                      oolEntry);
    } else if (fromType == MIRType::Float32) {
      masm.wasmTruncateFloat32ToUInt32(input, output, mir->isSaturating(),
                                       oolEntry);
    } else {
      MOZ_CRASH("unexpected type");
    }

    masm.bind(ool->rejoin());
    return;
  }

  if (fromType == MIRType::Double) {
    masm.wasmTruncateDoubleToInt32(input, output, mir->isSaturating(),
                                   oolEntry);
  } else if (fromType == MIRType::Float32) {
    masm.wasmTruncateFloat32ToInt32(input, output, mir->isSaturating(),
                                    oolEntry);
  } else {
    MOZ_CRASH("unexpected type");
  }

  masm.bind(ool->rejoin());
}

void CodeGenerator::visitWasmTruncateToInt64(LWasmTruncateToInt64* lir) {
  FloatRegister input = ToFloatRegister(lir->input());
  Register64 output = ToOutRegister64(lir);

  MWasmTruncateToInt64* mir = lir->mir();
  MIRType fromType = mir->input()->type();

  MOZ_ASSERT(fromType == MIRType::Double || fromType == MIRType::Float32);

  auto* ool = new (alloc()) OutOfLineWasmTruncateCheck(mir, input, output);
  addOutOfLineCode(ool, mir);

  Label* oolEntry = ool->entry();
  Label* oolRejoin = ool->rejoin();
  bool isSaturating = mir->isSaturating();

  if (fromType == MIRType::Double) {
    if (mir->isUnsigned()) {
      masm.wasmTruncateDoubleToUInt64(input, output, isSaturating, oolEntry,
                                      oolRejoin, InvalidFloatReg);
    } else {
      masm.wasmTruncateDoubleToInt64(input, output, isSaturating, oolEntry,
                                     oolRejoin, InvalidFloatReg);
    }
  } else {
    if (mir->isUnsigned()) {
      masm.wasmTruncateFloat32ToUInt64(input, output, isSaturating, oolEntry,
                                       oolRejoin, InvalidFloatReg);
    } else {
      masm.wasmTruncateFloat32ToInt64(input, output, isSaturating, oolEntry,
                                      oolRejoin, InvalidFloatReg);
    }
  }
}

void CodeGenerator::visitInt64ToFloatingPoint(LInt64ToFloatingPoint* lir) {
  Register64 input = ToRegister64(lir->input());
  FloatRegister output = ToFloatRegister(lir->output());
  MIRType outputType = lir->mir()->type();

  if (outputType == MIRType::Double) {
    if (lir->mir()->isUnsigned()) {
      masm.convertUInt64ToDouble(input, output, Register::Invalid());
    } else {
      masm.convertInt64ToDouble(input, output);
    }
  } else {
    if (lir->mir()->isUnsigned()) {
      masm.convertUInt64ToFloat32(input, output, Register::Invalid());
    } else {
      masm.convertInt64ToFloat32(input, output);
    }
  }
}

void CodeGenerator::visitWasmUint32ToDouble(LWasmUint32ToDouble* lir) {
  Register input = ToRegister(lir->input());
  FloatRegister output = ToFloatRegister(lir->output());
  masm.convertUInt32ToDouble(input, output);
}

void CodeGenerator::visitWasmUint32ToFloat32(LWasmUint32ToFloat32* lir) {
  Register input = ToRegister(lir->input());
  FloatRegister output = ToFloatRegister(lir->output());
  masm.convertUInt32ToFloat32(input, output);
}

void CodeGenerator::visitWasmBuiltinTruncateDToInt32(
    LWasmBuiltinTruncateDToInt32* lir) {
  emitTruncateDouble(ToFloatRegister(lir->getOperand(0)),
                     ToRegister(lir->getDef(0)), lir->mir());
}

void CodeGenerator::visitWasmBuiltinTruncateFToInt32(
    LWasmBuiltinTruncateFToInt32* lir) {
  emitTruncateFloat32(ToFloatRegister(lir->getOperand(0)),
                      ToRegister(lir->getDef(0)), lir->mir());
}

// ===============================================================
// Visitors: Wasm load/store

template <typename T>
void CodeGeneratorPPC64::emitWasmLoad(T* lir) {
  const MWasmLoad* mir = lir->mir();
  UseScratchRegisterScope temps(masm);
  Register scratch = temps.Acquire();

  Register memoryBase = ToRegister(lir->memoryBase());
  Register ptr = ToRegister(lir->ptr());
  Register ptrScratch = ToTempRegisterOrInvalid(lir->temp0());

  if (mir->base()->type() == MIRType::Int32) {
    masm.move32To64ZeroExtend(ptr, Register64(scratch));
    ptr = scratch;
    ptrScratch = ptrScratch != InvalidReg ? scratch : InvalidReg;
  }

  masm.wasmLoad(mir->access(), memoryBase, ptr, ptrScratch,
                ToAnyRegister(lir->output()));
}

template <typename T>
void CodeGeneratorPPC64::emitWasmStore(T* lir) {
  const MWasmStore* mir = lir->mir();
  UseScratchRegisterScope temps(masm);
  Register scratch = temps.Acquire();

  Register memoryBase = ToRegister(lir->memoryBase());
  Register ptr = ToRegister(lir->ptr());
  Register ptrScratch = ToTempRegisterOrInvalid(lir->temp0());

  if (mir->base()->type() == MIRType::Int32) {
    masm.move32To64ZeroExtend(ptr, Register64(scratch));
    ptr = scratch;
    ptrScratch = ptrScratch != InvalidReg ? scratch : InvalidReg;
  }

  masm.wasmStore(mir->access(), ToAnyRegister(lir->value()), memoryBase, ptr,
                 ptrScratch);
}

void CodeGenerator::visitWasmLoad(LWasmLoad* lir) { emitWasmLoad(lir); }

void CodeGenerator::visitWasmStore(LWasmStore* lir) { emitWasmStore(lir); }

void CodeGenerator::visitWasmLoadI64(LWasmLoadI64* lir) {
  const MWasmLoad* mir = lir->mir();

  Register memoryBase = ToRegister(lir->memoryBase());
  Register ptrScratch = ToTempRegisterOrInvalid(lir->temp0());

  Register ptrReg = ToRegister(lir->ptr());
  if (mir->base()->type() == MIRType::Int32) {
    masm.move32ZeroExtendToPtr(ptrReg, ptrReg);
  }

  masm.wasmLoadI64(mir->access(), memoryBase, ptrReg, ptrScratch,
                   ToOutRegister64(lir));
}

void CodeGenerator::visitWasmStoreI64(LWasmStoreI64* lir) {
  const MWasmStore* mir = lir->mir();

  Register memoryBase = ToRegister(lir->memoryBase());
  Register ptrScratch = ToTempRegisterOrInvalid(lir->temp0());

  Register ptrReg = ToRegister(lir->ptr());
  if (mir->base()->type() == MIRType::Int32) {
    masm.move32ZeroExtendToPtr(ptrReg, ptrReg);
  }

  masm.wasmStoreI64(mir->access(), ToRegister64(lir->value()), memoryBase,
                    ptrReg, ptrScratch);
}

void CodeGenerator::visitAsmJSLoadHeap(LAsmJSLoadHeap* ins) {
  const MAsmJSLoadHeap* mir = ins->mir();
  MOZ_ASSERT(!mir->hasMemoryBase());

  const LAllocation* ptr = ins->ptr();
  const LDefinition* output = ins->output();
  const LAllocation* boundsCheckLimit = ins->boundsCheckLimit();

  Register ptrReg = ToRegister(ptr);
  Scalar::Type accessType = mir->accessType();
  bool isFloat = accessType == Scalar::Float32 || accessType == Scalar::Float64;
  Label done;

  if (mir->needsBoundsCheck()) {
    Label boundsCheckPassed;
    Register boundsCheckLimitReg = ToRegister(boundsCheckLimit);
    masm.wasmBoundsCheck32(Assembler::Below, ptrReg, boundsCheckLimitReg,
                           &boundsCheckPassed);
    if (isFloat) {
      if (accessType == Scalar::Float32) {
        masm.loadConstantFloat32(GenericNaN(), ToFloatRegister(output));
      } else {
        masm.loadConstantDouble(GenericNaN(), ToFloatRegister(output));
      }
    } else {
      masm.movePtr(ImmWord(0), ToRegister(output));
    }
    masm.jump(&done);
    masm.bind(&boundsCheckPassed);
  }

  UseScratchRegisterScope temps(masm);
  Register scratch = temps.Acquire();
  masm.move32To64ZeroExtend(ptrReg, Register64(scratch));

  switch (accessType) {
    case Scalar::Int8:
      masm.as_lbzx(ToRegister(output), HeapReg, scratch);
      masm.as_extsb(ToRegister(output), ToRegister(output));
      break;
    case Scalar::Uint8:
      masm.as_lbzx(ToRegister(output), HeapReg, scratch);
      break;
    case Scalar::Int16:
      masm.as_lhax(ToRegister(output), HeapReg, scratch);
      break;
    case Scalar::Uint16:
      masm.as_lhzx(ToRegister(output), HeapReg, scratch);
      break;
    case Scalar::Int32:
      masm.as_lwzx(ToRegister(output), HeapReg, scratch);
      masm.as_extsw(ToRegister(output), ToRegister(output));
      break;
    case Scalar::Uint32:
      masm.as_lwzx(ToRegister(output), HeapReg, scratch);
      break;
    case Scalar::Float64:
      masm.as_lfdx(ToFloatRegister(output), HeapReg, scratch);
      break;
    case Scalar::Float32:
      masm.as_lfsx(ToFloatRegister(output), HeapReg, scratch);
      break;
    default:
      MOZ_CRASH("unexpected array type");
  }

  if (done.used()) {
    masm.bind(&done);
  }
}

void CodeGenerator::visitAsmJSStoreHeap(LAsmJSStoreHeap* ins) {
  const MAsmJSStoreHeap* mir = ins->mir();
  MOZ_ASSERT(!mir->hasMemoryBase());

  const LAllocation* value = ins->value();
  const LAllocation* ptr = ins->ptr();
  const LAllocation* boundsCheckLimit = ins->boundsCheckLimit();

  Register ptrReg = ToRegister(ptr);

  Label done;
  if (mir->needsBoundsCheck()) {
    Register boundsCheckLimitReg = ToRegister(boundsCheckLimit);
    masm.wasmBoundsCheck32(Assembler::AboveOrEqual, ptrReg, boundsCheckLimitReg,
                           &done);
  }

  UseScratchRegisterScope temps(masm);
  Register scratch = temps.Acquire();
  masm.move32To64ZeroExtend(ptrReg, Register64(scratch));

  switch (mir->accessType()) {
    case Scalar::Int8:
    case Scalar::Uint8:
      masm.as_stbx(ToRegister(value), HeapReg, scratch);
      break;
    case Scalar::Int16:
    case Scalar::Uint16:
      masm.as_sthx(ToRegister(value), HeapReg, scratch);
      break;
    case Scalar::Int32:
    case Scalar::Uint32:
      masm.as_stwx(ToRegister(value), HeapReg, scratch);
      break;
    case Scalar::Float64:
      masm.as_stfdx(ToFloatRegister(value), HeapReg, scratch);
      break;
    case Scalar::Float32:
      masm.as_stfsx(ToFloatRegister(value), HeapReg, scratch);
      break;
    default:
      MOZ_CRASH("unexpected array type");
  }

  if (done.used()) {
    masm.bind(&done);
  }
}

void CodeGenerator::visitWasmStackArg(LWasmStackArg* ins) {
  const MWasmStackArg* mir = ins->mir();
  if (ins->arg()->isConstant()) {
    // An i32 stack arg must be stored as 32 bits: a 64-bit store places the
    // value in the low half of the 8-byte slot, which on big-endian is the
    // high address word, while the callee reads a 32-bit value at the slot
    // offset (the low address word) and would see 0.
    if (mir->input()->type() == MIRType::Int32) {
      masm.store32(Imm32(ToInt32(ins->arg())),
                   Address(StackPointer, mir->spOffset()));
    } else {
      masm.storePtr(ImmWord(ToInt32(ins->arg())),
                    Address(StackPointer, mir->spOffset()));
    }
  } else {
    if (ins->arg()->isGeneralReg()) {
      if (mir->input()->type() == MIRType::Int32) {
        masm.store32(ToRegister(ins->arg()),
                     Address(StackPointer, mir->spOffset()));
      } else {
        masm.storePtr(ToRegister(ins->arg()),
                      Address(StackPointer, mir->spOffset()));
      }
    } else if (mir->input()->type() == MIRType::Double) {
      masm.storeDouble(ToFloatRegister(ins->arg()),
                       Address(StackPointer, mir->spOffset()));
#ifdef ENABLE_WASM_SIMD
    } else if (mir->input()->type() == MIRType::Simd128) {
      masm.storeUnalignedSimd128(ToFloatRegister(ins->arg()),
                                 Address(StackPointer, mir->spOffset()));
#endif
    } else {
      masm.storeFloat32(ToFloatRegister(ins->arg()),
                        Address(StackPointer, mir->spOffset()));
    }
  }
}

void CodeGenerator::visitWasmStackArgI64(LWasmStackArgI64* ins) {
  const MWasmStackArg* mir = ins->mir();
  Address dst(StackPointer, mir->spOffset());
  if (IsConstant(ins->arg())) {
    masm.store64(Imm64(ToInt64(ins->arg())), dst);
  } else {
    masm.store64(ToRegister64(ins->arg()), dst);
  }
}

void CodeGenerator::visitWasmSelect(LWasmSelect* ins) {
  MIRType mirType = ins->mir()->type();

  Register cond = ToRegister(ins->condExpr());
  const LAllocation* falseExpr = ins->falseExpr();

  if (mirType == MIRType::Int32 || mirType == MIRType::WasmAnyRef) {
    Register out = ToRegister(ins->output());
    MOZ_ASSERT(ToRegister(ins->trueExpr()) == out,
               "true expr input is reused for output");
    if (falseExpr->isGeneralReg()) {
      masm.moveIfZero(out, ToRegister(falseExpr), cond);
    } else {
      masm.cmp32Load32(Assembler::Zero, cond, cond, ToAddress(falseExpr), out);
    }
    return;
  }

  FloatRegister out = ToFloatRegister(ins->output());
  MOZ_ASSERT(ToFloatRegister(ins->trueExpr()) == out,
             "true expr input is reused for output");

  if (falseExpr->isFloatReg()) {
    Label done;
    // The select condition is a 32-bit value; test 32 bits so high-bit garbage
    // does not make a zero condition read as non-zero.
    masm.branchTest32(Assembler::NonZero, cond, cond, &done);
    if (mirType == MIRType::Float32) {
      masm.moveFloat32(ToFloatRegister(falseExpr), out);
    } else if (mirType == MIRType::Double) {
      masm.moveDouble(ToFloatRegister(falseExpr), out);
    } else if (mirType == MIRType::Simd128) {
      masm.moveSimd128(ToFloatRegister(falseExpr), out);
    } else {
      MOZ_CRASH("unhandled type in visitWasmSelect!");
    }
    masm.bind(&done);
  } else {
    Label done;
    // The select condition is a 32-bit value; test 32 bits so high-bit garbage
    // does not make a zero condition read as non-zero.
    masm.branchTest32(Assembler::NonZero, cond, cond, &done);

    if (mirType == MIRType::Float32) {
      masm.loadFloat32(ToAddress(falseExpr), out);
    } else if (mirType == MIRType::Double) {
      masm.loadDouble(ToAddress(falseExpr), out);
    } else if (mirType == MIRType::Simd128) {
      masm.loadUnalignedSimd128(ToAddress(falseExpr), out);
    } else {
      MOZ_CRASH("unhandled type in visitWasmSelect!");
    }

    masm.bind(&done);
  }
}

void CodeGenerator::visitWasmSelectI64(LWasmSelectI64* lir) {
  MOZ_ASSERT(lir->mir()->type() == MIRType::Int64);

  Register cond = ToRegister(lir->condExpr());
  LInt64Allocation falseExpr = lir->falseExpr();

  Register64 out = ToOutRegister64(lir);
  MOZ_ASSERT(ToRegister64(lir->trueExpr()) == out,
             "true expr is reused for input");

  if (falseExpr.value().isGeneralReg()) {
    masm.moveIfZero(out.reg, ToRegister(falseExpr.value()), cond);
  } else {
    Label done;
    // The select condition is a 32-bit value; test 32 bits so high-bit garbage
    // does not make a zero condition read as non-zero.
    masm.branchTest32(Assembler::NonZero, cond, cond, &done);
    masm.loadPtr(ToAddress(falseExpr.value()), out.reg);
    masm.bind(&done);
  }
}

void CodeGenerator::visitWasmCompareAndSelect(LWasmCompareAndSelect* ins) {
  MCompare::CompareType compTy = ins->compareType();
  MIRType insTy = ins->mir()->type();
  const bool cmpIs32 = compTy == MCompare::Compare_Int32 ||
                       compTy == MCompare::Compare_UInt32;
  const bool cmpIs64 = compTy == MCompare::Compare_Int64 ||
                       compTy == MCompare::Compare_UInt64;
  const bool selIsInt = insTy == MIRType::Int32 || insTy == MIRType::Int64;

  MOZ_RELEASE_ASSERT(
      (cmpIs32 || cmpIs64) && selIsInt,
      "CodeGenerator::visitWasmCompareAndSelect: unexpected types");

  Register trueExprAndDest = ToRegister(ins->output());
  MOZ_ASSERT(ToRegister(ins->ifTrueExpr()) == trueExprAndDest,
             "true expr input is reused for output");

  Assembler::Condition cond =
      Assembler::InvertCondition(JSOpToCondition(compTy, ins->jsop()));
  Register lhs = ToRegister(ins->leftExpr());
  Register rhs = ToRegister(ins->rightExpr());
  Register falseExpr = ToRegister(ins->ifFalseExpr());

  // isel operates on the whole 64-bit GPR regardless of compare width; only
  // the compare instruction differs (cmpw/cmplw vs cmpd/cmpld).
  if (cmpIs32) {
    masm.cmp32Move32(cond, lhs, rhs, falseExpr, trueExprAndDest);
  } else {
    masm.cmpPtrMovePtr(cond, lhs, rhs, falseExpr, trueExprAndDest);
  }
}

void CodeGenerator::visitWasmAddOffset(LWasmAddOffset* lir) {
  MWasmAddOffset* mir = lir->mir();
  Register base = ToRegister(lir->base());
  Register out = ToRegister(lir->output());

  Label ok;
  masm.ma_add32TestCarry(Assembler::CarryClear, out, base, Imm32(mir->offset()),
                         &ok);
  masm.wasmTrap(wasm::Trap::OutOfBounds, mir->trapSiteDesc());
  masm.bind(&ok);
}

void CodeGenerator::visitWasmAddOffset64(LWasmAddOffset64* lir) {
  MWasmAddOffset* mir = lir->mir();
  Register64 base = ToRegister64(lir->base());
  Register64 out = ToOutRegister64(lir);

  Label ok;
  masm.ma_addPtrTestCarry(Assembler::CarryClear, out.reg, base.reg,
                          ImmWord(mir->offset()), &ok);
  masm.wasmTrap(wasm::Trap::OutOfBounds, mir->trapSiteDesc());
  masm.bind(&ok);
}

// ===============================================================
// Visitors: Effective Address

void CodeGenerator::visitEffectiveAddress2(LEffectiveAddress2* ins) {
  const MEffectiveAddress2* mir = ins->mir();
  Register output = ToRegister(ins->output());

  // EA = index * scale + displacement (no base register)
  masm.movePtr(ImmWord(0), output);
  BaseIndex addr(output, ToRegister(ins->index()), mir->scale(),
                 mir->displacement());
  masm.computeEffectiveAddress(addr, output);
  // Sign-extend to 32-bit
  masm.as_extsw(output, output);
}

void CodeGenerator::visitEffectiveAddress3(LEffectiveAddress3* ins) {
  const MEffectiveAddress3* mir = ins->mir();
  Register output = ToRegister(ins->output());

  BaseIndex addr(ToRegister(ins->base()), ToRegister(ins->index()),
                 mir->scale(), mir->displacement());
  masm.computeEffectiveAddress(addr, output);
  // Sign-extend to 32-bit
  masm.as_extsw(output, output);
}

void CodeGenerator::visitWasmMulI64WideHI64(LWasmMulI64WideHI64* ins) {
  Register lhs = ToRegister(ins->lhs());
  Register rhs = ToRegister(ins->rhs());
  Register output = ToRegister(ins->output());

  if (ins->isSigned()) {
    masm.as_mulhd(output, lhs, rhs);
  } else {
    masm.as_mulhdu(output, lhs, rhs);
  }
}

// ===============================================================
// Visitors: Typed Array Atomics

void CodeGenerator::visitCompareExchangeTypedArrayElement(
    LCompareExchangeTypedArrayElement* lir) {
  Register elements = ToRegister(lir->elements());
  AnyRegister output = ToAnyRegister(lir->output());
  Register outTemp = ToTempRegisterOrInvalid(lir->temp0());

  Register oldval = ToRegister(lir->oldval());
  Register newval = ToRegister(lir->newval());
  Register valueTemp = ToTempRegisterOrInvalid(lir->temp1());
  Register offsetTemp = ToTempRegisterOrInvalid(lir->temp2());
  Register maskTemp = ToTempRegisterOrInvalid(lir->temp3());
  Scalar::Type arrayType = lir->mir()->arrayType();

  auto dest = ToAddressOrBaseIndex(elements, lir->index(), arrayType);

  dest.match([&](const auto& dest) {
    masm.compareExchangeJS(arrayType, Synchronization::Full(), dest, oldval,
                           newval, valueTemp, offsetTemp, maskTemp, outTemp,
                           output);
  });
}

void CodeGenerator::visitAtomicExchangeTypedArrayElement(
    LAtomicExchangeTypedArrayElement* lir) {
  Register elements = ToRegister(lir->elements());
  AnyRegister output = ToAnyRegister(lir->output());
  Register outTemp = ToTempRegisterOrInvalid(lir->temp0());

  Register value = ToRegister(lir->value());
  Register valueTemp = ToTempRegisterOrInvalid(lir->temp1());
  Register offsetTemp = ToTempRegisterOrInvalid(lir->temp2());
  Register maskTemp = ToTempRegisterOrInvalid(lir->temp3());
  Scalar::Type arrayType = lir->mir()->arrayType();

  auto dest = ToAddressOrBaseIndex(elements, lir->index(), arrayType);

  dest.match([&](const auto& dest) {
    masm.atomicExchangeJS(arrayType, Synchronization::Full(), dest, value,
                          valueTemp, offsetTemp, maskTemp, outTemp, output);
  });
}

void CodeGenerator::visitAtomicTypedArrayElementBinop(
    LAtomicTypedArrayElementBinop* lir) {
  MOZ_ASSERT(!lir->mir()->isForEffect());

  AnyRegister output = ToAnyRegister(lir->output());
  Register elements = ToRegister(lir->elements());
  Register outTemp = ToTempRegisterOrInvalid(lir->temp0());
  Register valueTemp = ToTempRegisterOrInvalid(lir->temp1());
  Register offsetTemp = ToTempRegisterOrInvalid(lir->temp2());
  Register maskTemp = ToTempRegisterOrInvalid(lir->temp3());
  Register value = ToRegister(lir->value());
  Scalar::Type arrayType = lir->mir()->arrayType();

  auto mem = ToAddressOrBaseIndex(elements, lir->index(), arrayType);

  mem.match([&](const auto& mem) {
    masm.atomicFetchOpJS(arrayType, Synchronization::Full(),
                         lir->mir()->operation(), value, mem, valueTemp,
                         offsetTemp, maskTemp, outTemp, output);
  });
}

void CodeGenerator::visitAtomicTypedArrayElementBinopForEffect(
    LAtomicTypedArrayElementBinopForEffect* lir) {
  MOZ_ASSERT(lir->mir()->isForEffect());

  Register elements = ToRegister(lir->elements());
  Register valueTemp = ToTempRegisterOrInvalid(lir->temp0());
  Register offsetTemp = ToTempRegisterOrInvalid(lir->temp1());
  Register maskTemp = ToTempRegisterOrInvalid(lir->temp2());
  Register value = ToRegister(lir->value());
  Scalar::Type arrayType = lir->mir()->arrayType();

  auto mem = ToAddressOrBaseIndex(elements, lir->index(), arrayType);

  mem.match([&](const auto& mem) {
    masm.atomicEffectOpJS(arrayType, Synchronization::Full(),
                          lir->mir()->operation(), value, mem, valueTemp,
                          offsetTemp, maskTemp);
  });
}

void CodeGenerator::visitCompareExchangeTypedArrayElement64(
    LCompareExchangeTypedArrayElement64* lir) {
  Register elements = ToRegister(lir->elements());
  Register64 oldval = ToRegister64(lir->oldval());
  Register64 newval = ToRegister64(lir->newval());
  Register64 out = ToOutRegister64(lir);
  Scalar::Type arrayType = lir->mir()->arrayType();

  auto dest = ToAddressOrBaseIndex(elements, lir->index(), arrayType);

  dest.match([&](const auto& dest) {
    masm.compareExchange64(Synchronization::Full(), dest, oldval, newval, out);
  });
}

void CodeGenerator::visitAtomicExchangeTypedArrayElement64(
    LAtomicExchangeTypedArrayElement64* lir) {
  Register elements = ToRegister(lir->elements());
  Register64 value = ToRegister64(lir->value());
  Register64 out = ToOutRegister64(lir);
  Scalar::Type arrayType = lir->mir()->arrayType();

  auto dest = ToAddressOrBaseIndex(elements, lir->index(), arrayType);

  dest.match([&](const auto& dest) {
    masm.atomicExchange64(Synchronization::Full(), dest, value, out);
  });
}

void CodeGenerator::visitAtomicTypedArrayElementBinop64(
    LAtomicTypedArrayElementBinop64* lir) {
  MOZ_ASSERT(lir->mir()->hasUses());

  Register elements = ToRegister(lir->elements());
  Register64 value = ToRegister64(lir->value());
  Register64 temp = ToRegister64(lir->temp0());
  Register64 out = ToOutRegister64(lir);

  Scalar::Type arrayType = lir->mir()->arrayType();
  AtomicOp atomicOp = lir->mir()->operation();

  auto dest = ToAddressOrBaseIndex(elements, lir->index(), arrayType);

  dest.match([&](const auto& dest) {
    masm.atomicFetchOp64(Synchronization::Full(), atomicOp, value, dest, temp,
                         out);
  });
}

void CodeGenerator::visitAtomicTypedArrayElementBinopForEffect64(
    LAtomicTypedArrayElementBinopForEffect64* lir) {
  MOZ_ASSERT(!lir->mir()->hasUses());

  Register elements = ToRegister(lir->elements());
  Register64 value = ToRegister64(lir->value());
  Register64 temp = ToRegister64(lir->temp0());

  Scalar::Type arrayType = lir->mir()->arrayType();
  AtomicOp atomicOp = lir->mir()->operation();

  auto dest = ToAddressOrBaseIndex(elements, lir->index(), arrayType);

  dest.match([&](const auto& dest) {
    masm.atomicEffectOp64(Synchronization::Full(), atomicOp, value, dest, temp);
  });
}

void CodeGenerator::visitAtomicLoad64(LAtomicLoad64* lir) {
  Register elements = ToRegister(lir->elements());
  Register64 out = ToOutRegister64(lir);
  Scalar::Type storageType = lir->mir()->storageType();

  auto source = ToAddressOrBaseIndex(elements, lir->index(), storageType);

  auto sync = Synchronization::Load();
  masm.memoryBarrierBefore(sync);
  source.match([&](const auto& source) { masm.load64(source, out); });
  masm.memoryBarrierAfter(sync);
}

void CodeGenerator::visitAtomicStore64(LAtomicStore64* lir) {
  Register elements = ToRegister(lir->elements());
  Register64 value = ToRegister64(lir->value());
  Scalar::Type writeType = lir->mir()->writeType();

  auto dest = ToAddressOrBaseIndex(elements, lir->index(), writeType);

  auto sync = Synchronization::Store();
  masm.memoryBarrierBefore(sync);
  dest.match([&](const auto& dest) { masm.store64(value, dest); });
  masm.memoryBarrierAfter(sync);
}

// Wasm Atomics
void CodeGenerator::visitWasmCompareExchangeHeap(
    LWasmCompareExchangeHeap* ins) {
  MWasmCompareExchangeHeap* mir = ins->mir();
  Register memoryBase = ToRegister(ins->memoryBase());
  Register ptrReg = ToRegister(ins->ptr());
  BaseIndex srcAddr(memoryBase, ptrReg, TimesOne, mir->access().offset32());

  Register oldval = ToRegister(ins->oldValue());
  Register newval = ToRegister(ins->newValue());
  Register valueTemp = ToTempRegisterOrInvalid(ins->temp0());
  Register offsetTemp = ToTempRegisterOrInvalid(ins->temp1());
  Register maskTemp = ToTempRegisterOrInvalid(ins->temp2());

  masm.wasmCompareExchange(mir->access(), srcAddr, oldval, newval, valueTemp,
                           offsetTemp, maskTemp, ToRegister(ins->output()));
}

void CodeGenerator::visitWasmAtomicExchangeHeap(LWasmAtomicExchangeHeap* ins) {
  MWasmAtomicExchangeHeap* mir = ins->mir();
  Register memoryBase = ToRegister(ins->memoryBase());
  Register ptrReg = ToRegister(ins->ptr());
  Register value = ToRegister(ins->value());
  BaseIndex srcAddr(memoryBase, ptrReg, TimesOne, mir->access().offset32());

  Register valueTemp = ToTempRegisterOrInvalid(ins->temp0());
  Register offsetTemp = ToTempRegisterOrInvalid(ins->temp1());
  Register maskTemp = ToTempRegisterOrInvalid(ins->temp2());

  masm.wasmAtomicExchange(mir->access(), srcAddr, value, valueTemp, offsetTemp,
                          maskTemp, ToRegister(ins->output()));
}

void CodeGenerator::visitWasmAtomicBinopHeap(LWasmAtomicBinopHeap* ins) {
  MOZ_ASSERT(ins->mir()->hasUses());

  MWasmAtomicBinopHeap* mir = ins->mir();
  Register memoryBase = ToRegister(ins->memoryBase());
  Register ptrReg = ToRegister(ins->ptr());
  Register valueTemp = ToTempRegisterOrInvalid(ins->temp0());
  Register offsetTemp = ToTempRegisterOrInvalid(ins->temp1());
  Register maskTemp = ToTempRegisterOrInvalid(ins->temp2());

  BaseIndex srcAddr(memoryBase, ptrReg, TimesOne, mir->access().offset32());

  masm.wasmAtomicFetchOp(mir->access(), mir->operation(),
                         ToRegister(ins->value()), srcAddr, valueTemp,
                         offsetTemp, maskTemp, ToRegister(ins->output()));
}

void CodeGenerator::visitWasmAtomicBinopHeapForEffect(
    LWasmAtomicBinopHeapForEffect* ins) {
  MOZ_ASSERT(!ins->mir()->hasUses());

  MWasmAtomicBinopHeap* mir = ins->mir();
  Register memoryBase = ToRegister(ins->memoryBase());
  Register ptrReg = ToRegister(ins->ptr());
  Register valueTemp = ToTempRegisterOrInvalid(ins->temp0());
  Register offsetTemp = ToTempRegisterOrInvalid(ins->temp1());
  Register maskTemp = ToTempRegisterOrInvalid(ins->temp2());

  BaseIndex srcAddr(memoryBase, ptrReg, TimesOne, mir->access().offset32());
  masm.wasmAtomicEffectOp(mir->access(), mir->operation(),
                          ToRegister(ins->value()), srcAddr, valueTemp,
                          offsetTemp, maskTemp);
}

void CodeGenerator::visitWasmCompareExchangeI64(LWasmCompareExchangeI64* lir) {
  Register memoryBase = ToRegister(lir->memoryBase());
  Register ptr = ToRegister(lir->ptr());
  Register64 oldValue = ToRegister64(lir->oldValue());
  Register64 newValue = ToRegister64(lir->newValue());
  Register64 output = ToOutRegister64(lir);
  uint32_t offset = lir->mir()->access().offset32();

  BaseIndex addr(memoryBase, ptr, TimesOne, offset);
  masm.wasmCompareExchange64(lir->mir()->access(), addr, oldValue, newValue,
                             output);
}

void CodeGenerator::visitWasmAtomicExchangeI64(LWasmAtomicExchangeI64* lir) {
  Register memoryBase = ToRegister(lir->memoryBase());
  Register ptr = ToRegister(lir->ptr());
  Register64 value = ToRegister64(lir->value());
  Register64 output = ToOutRegister64(lir);
  uint32_t offset = lir->mir()->access().offset32();

  BaseIndex addr(memoryBase, ptr, TimesOne, offset);
  masm.wasmAtomicExchange64(lir->mir()->access(), addr, value, output);
}

void CodeGenerator::visitWasmAtomicBinopI64(LWasmAtomicBinopI64* lir) {
  Register memoryBase = ToRegister(lir->memoryBase());
  Register ptr = ToRegister(lir->ptr());
  Register64 value = ToRegister64(lir->value());
  Register64 output = ToOutRegister64(lir);
  Register64 temp = ToRegister64(lir->temp0());
  uint32_t offset = lir->mir()->access().offset32();

  BaseIndex addr(memoryBase, ptr, TimesOne, offset);

  masm.wasmAtomicFetchOp64(lir->mir()->access(), lir->mir()->operation(), value,
                           addr, temp, output);
}

// SIMD code generators.
void CodeGenerator::visitSimd128(LSimd128* ins) {
  FloatRegister dest = ToFloatRegister(ins->output());
  masm.loadConstantSimd128(ins->simd128(), dest);
}
void CodeGenerator::visitWasmTernarySimd128(LWasmTernarySimd128* ins) {
  FloatRegister v0 = ToFloatRegister(ins->v0());
  FloatRegister v1 = ToFloatRegister(ins->v1());
  FloatRegister v2 = ToFloatRegister(ins->v2());
  FloatRegister dest = ToFloatRegister(ins->output());
  switch (ins->simdOp()) {
    case wasm::SimdOp::V128Bitselect:
      // bitselect(v0, v1, v2): result = (v0 & v2) | (v1 & ~v2)
      // xxsel: XC=0→XA, XC=1→XB → (XA & ~XC) | (XB & XC)
      // Need XA=v1, XB=v0, XC=v2.
      masm.as_xxsel(dest, v1, v0, v2);
      break;
    case wasm::SimdOp::I8x16RelaxedLaneSelect:
    case wasm::SimdOp::I16x8RelaxedLaneSelect:
    case wasm::SimdOp::I32x4RelaxedLaneSelect:
    case wasm::SimdOp::I64x2RelaxedLaneSelect:
      // relaxed laneSelect(v0, v1, mask=v2): same as bitselect
      masm.as_xxsel(dest, v1, v0, v2);
      break;
    // Lowering uses defineReuseInput on V2Index for ternary ops — the
    // allocator is required to place `dest` in v2's slot. Assert that
    // here; the FMA/dot helpers write their result through v2 in-place,
    // so dest == v2 makes the trailing moveSimd128 unnecessary.
    case wasm::SimdOp::I32x4RelaxedDotI8x16I7x16AddS:
      MOZ_ASSERT(dest == v2);
      masm.dotInt8x16Int7x16ThenAdd(v0, v1, v2,
                                    ToFloatRegister(ins->temp0()));
      break;
    case wasm::SimdOp::F32x4RelaxedMadd:
      MOZ_ASSERT(dest == v2);
      masm.fmaFloat32x4(v0, v1, v2);
      break;
    case wasm::SimdOp::F64x2RelaxedMadd:
      MOZ_ASSERT(dest == v2);
      masm.fmaFloat64x2(v0, v1, v2);
      break;
    case wasm::SimdOp::F32x4RelaxedNmadd:
      MOZ_ASSERT(dest == v2);
      masm.fnmaFloat32x4(v0, v1, v2);
      break;
    case wasm::SimdOp::F64x2RelaxedNmadd:
      MOZ_ASSERT(dest == v2);
      masm.fnmaFloat64x2(v0, v1, v2);
      break;
    default:
      MOZ_CRASH("PPC64: NYI SIMD ternary op");
  }
}
void CodeGenerator::visitWasmBinarySimd128(LWasmBinarySimd128* ins) {
  FloatRegister lhs = ToFloatRegister(ins->lhs());
  FloatRegister rhs = ToFloatRegister(ins->rhs());
  FloatRegister dest = ToFloatRegister(ins->output());
  switch (ins->simdOp()) {
    // Bitwise
    case wasm::SimdOp::V128And:
      masm.bitwiseAndSimd128(lhs, rhs, dest);
      break;
    case wasm::SimdOp::V128Or:
      masm.bitwiseOrSimd128(lhs, rhs, dest);
      break;
    case wasm::SimdOp::V128Xor:
      masm.bitwiseXorSimd128(lhs, rhs, dest);
      break;
    case wasm::SimdOp::V128AndNot:
      masm.bitwiseAndNotSimd128(lhs, rhs, dest);
      break;
    // Integer add
    case wasm::SimdOp::I8x16Add:
      masm.addInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8Add:
      masm.addInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4Add:
      masm.addInt32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2Add:
      masm.addInt64x2(lhs, rhs, dest);
      break;
    // Integer sub
    case wasm::SimdOp::I8x16Sub:
      masm.subInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8Sub:
      masm.subInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4Sub:
      masm.subInt32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2Sub:
      masm.subInt64x2(lhs, rhs, dest);
      break;
    // Saturating add
    case wasm::SimdOp::I8x16AddSatS:
      masm.addSatInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16AddSatU:
      masm.unsignedAddSatInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8AddSatS:
      masm.addSatInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8AddSatU:
      masm.unsignedAddSatInt16x8(lhs, rhs, dest);
      break;
    // Saturating sub
    case wasm::SimdOp::I8x16SubSatS:
      masm.subSatInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16SubSatU:
      masm.unsignedSubSatInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8SubSatS:
      masm.subSatInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8SubSatU:
      masm.unsignedSubSatInt16x8(lhs, rhs, dest);
      break;
    // Integer multiply
    case wasm::SimdOp::I16x8Mul:
      masm.mulInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4Mul:
      masm.mulInt32x4(lhs, rhs, dest);
      break;
    // Integer min/max signed
    case wasm::SimdOp::I8x16MinS:
      masm.minInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16MaxS:
      masm.maxInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8MinS:
      masm.minInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8MaxS:
      masm.maxInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4MinS:
      masm.minInt32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4MaxS:
      masm.maxInt32x4(lhs, rhs, dest);
      break;
    // Integer min/max unsigned
    case wasm::SimdOp::I8x16MinU:
      masm.unsignedMinInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16MaxU:
      masm.unsignedMaxInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8MinU:
      masm.unsignedMinInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8MaxU:
      masm.unsignedMaxInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4MinU:
      masm.unsignedMinInt32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4MaxU:
      masm.unsignedMaxInt32x4(lhs, rhs, dest);
      break;
    // Average unsigned
    case wasm::SimdOp::I8x16AvgrU:
      masm.unsignedAverageInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8AvgrU:
      masm.unsignedAverageInt16x8(lhs, rhs, dest);
      break;
    // Q15 multiply
    case wasm::SimdOp::I16x8Q15MulrSatS:
      masm.q15MulrSatInt16x8(lhs, rhs, dest);
      break;
    // Integer compare
    case wasm::SimdOp::I8x16Eq:
      masm.compareInt8x16(Assembler::Equal, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16Ne:
      masm.compareInt8x16(Assembler::NotEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16LtS:
      masm.compareInt8x16(Assembler::LessThan, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16GtS:
      masm.compareInt8x16(Assembler::GreaterThan, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16LeS:
      masm.compareInt8x16(Assembler::LessThanOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16GeS:
      masm.compareInt8x16(Assembler::GreaterThanOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16LtU:
      masm.compareInt8x16(Assembler::Below, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16GtU:
      masm.compareInt8x16(Assembler::Above, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16LeU:
      masm.compareInt8x16(Assembler::BelowOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16GeU:
      masm.compareInt8x16(Assembler::AboveOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8Eq:
      masm.compareInt16x8(Assembler::Equal, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8Ne:
      masm.compareInt16x8(Assembler::NotEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8LtS:
      masm.compareInt16x8(Assembler::LessThan, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8GtS:
      masm.compareInt16x8(Assembler::GreaterThan, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8LeS:
      masm.compareInt16x8(Assembler::LessThanOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8GeS:
      masm.compareInt16x8(Assembler::GreaterThanOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8LtU:
      masm.compareInt16x8(Assembler::Below, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8GtU:
      masm.compareInt16x8(Assembler::Above, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8LeU:
      masm.compareInt16x8(Assembler::BelowOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8GeU:
      masm.compareInt16x8(Assembler::AboveOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4Eq:
      masm.compareInt32x4(Assembler::Equal, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4Ne:
      masm.compareInt32x4(Assembler::NotEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4LtS:
      masm.compareInt32x4(Assembler::LessThan, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4GtS:
      masm.compareInt32x4(Assembler::GreaterThan, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4LeS:
      masm.compareInt32x4(Assembler::LessThanOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4GeS:
      masm.compareInt32x4(Assembler::GreaterThanOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4LtU:
      masm.compareInt32x4(Assembler::Below, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4GtU:
      masm.compareInt32x4(Assembler::Above, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4LeU:
      masm.compareInt32x4(Assembler::BelowOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4GeU:
      masm.compareInt32x4(Assembler::AboveOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2Eq:
      masm.compareInt64x2(Assembler::Equal, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2Ne:
      masm.compareInt64x2(Assembler::NotEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2LtS:
      masm.compareInt64x2(Assembler::LessThan, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2GtS:
      masm.compareInt64x2(Assembler::GreaterThan, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2LeS:
      masm.compareInt64x2(Assembler::LessThanOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2GeS:
      masm.compareInt64x2(Assembler::GreaterThanOrEqual, lhs, rhs, dest);
      break;
    // Float compare
    case wasm::SimdOp::F32x4Eq:
      masm.compareFloat32x4(Assembler::Equal, lhs, rhs, dest);
      break;
    case wasm::SimdOp::F32x4Ne:
      masm.compareFloat32x4(Assembler::NotEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::F32x4Lt:
      masm.compareFloat32x4(Assembler::LessThan, lhs, rhs, dest);
      break;
    case wasm::SimdOp::F32x4Gt:
      masm.compareFloat32x4(Assembler::GreaterThan, lhs, rhs, dest);
      break;
    case wasm::SimdOp::F32x4Le:
      masm.compareFloat32x4(Assembler::LessThanOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::F32x4Ge:
      masm.compareFloat32x4(Assembler::GreaterThanOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::F64x2Eq:
      masm.compareFloat64x2(Assembler::Equal, lhs, rhs, dest);
      break;
    case wasm::SimdOp::F64x2Ne:
      masm.compareFloat64x2(Assembler::NotEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::F64x2Lt:
      masm.compareFloat64x2(Assembler::LessThan, lhs, rhs, dest);
      break;
    case wasm::SimdOp::F64x2Gt:
      masm.compareFloat64x2(Assembler::GreaterThan, lhs, rhs, dest);
      break;
    case wasm::SimdOp::F64x2Le:
      masm.compareFloat64x2(Assembler::LessThanOrEqual, lhs, rhs, dest);
      break;
    case wasm::SimdOp::F64x2Ge:
      masm.compareFloat64x2(Assembler::GreaterThanOrEqual, lhs, rhs, dest);
      break;
    // Float arithmetic
    case wasm::SimdOp::F32x4Add:
      masm.addFloat32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::F32x4Sub:
      masm.subFloat32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::F32x4Mul:
      masm.mulFloat32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::F32x4Div:
      masm.divFloat32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::F32x4Min:
      masm.minFloat32x4(lhs, rhs, dest, ToFloatRegister(ins->getTemp(0)),
                         ToFloatRegister(ins->getTemp(1)));
      break;
    case wasm::SimdOp::F32x4Max:
      masm.maxFloat32x4(lhs, rhs, dest, ToFloatRegister(ins->getTemp(0)),
                         ToFloatRegister(ins->getTemp(1)));
      break;
    case wasm::SimdOp::F32x4PMin:
      masm.pseudoMinFloat32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::F32x4PMax:
      masm.pseudoMaxFloat32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::F64x2Add:
      masm.addFloat64x2(lhs, rhs, dest);
      break;
    case wasm::SimdOp::F64x2Sub:
      masm.subFloat64x2(lhs, rhs, dest);
      break;
    case wasm::SimdOp::F64x2Mul:
      masm.mulFloat64x2(lhs, rhs, dest);
      break;
    case wasm::SimdOp::F64x2Div:
      masm.divFloat64x2(lhs, rhs, dest);
      break;
    case wasm::SimdOp::F64x2Min:
      masm.minFloat64x2(lhs, rhs, dest, ToFloatRegister(ins->getTemp(0)),
                         ToFloatRegister(ins->getTemp(1)));
      break;
    case wasm::SimdOp::F64x2Max:
      masm.maxFloat64x2(lhs, rhs, dest, ToFloatRegister(ins->getTemp(0)),
                         ToFloatRegister(ins->getTemp(1)));
      break;
    case wasm::SimdOp::F64x2PMin:
      masm.pseudoMinFloat64x2(lhs, rhs, dest);
      break;
    case wasm::SimdOp::F64x2PMax:
      masm.pseudoMaxFloat64x2(lhs, rhs, dest);
      break;
    // Narrow
    case wasm::SimdOp::I8x16NarrowI16x8S:
      masm.narrowInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16NarrowI16x8U:
      masm.unsignedNarrowInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8NarrowI32x4S:
      masm.narrowInt32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8NarrowI32x4U:
      masm.unsignedNarrowInt32x4(lhs, rhs, dest);
      break;
    // i64 multiply
    case wasm::SimdOp::I64x2Mul: {
      FloatRegister temp0 = ToTempFloatRegisterOrInvalid(ins->temp0());
      FloatRegister temp1f = ToTempFloatRegisterOrInvalid(ins->temp1());
      masm.mulInt64x2(lhs, rhs, dest, temp0, temp1f);
      break;
    }
    // Extended multiply
    case wasm::SimdOp::I16x8ExtmulLowI8x16S:
      masm.extMulLowInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8ExtmulHighI8x16S:
      masm.extMulHighInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8ExtmulLowI8x16U:
      masm.unsignedExtMulLowInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8ExtmulHighI8x16U:
      masm.unsignedExtMulHighInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4ExtmulLowI16x8S:
      masm.extMulLowInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4ExtmulHighI16x8S:
      masm.extMulHighInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4ExtmulLowI16x8U:
      masm.unsignedExtMulLowInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4ExtmulHighI16x8U:
      masm.unsignedExtMulHighInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2ExtmulLowI32x4S:
      masm.extMulLowInt32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2ExtmulHighI32x4S:
      masm.extMulHighInt32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2ExtmulLowI32x4U:
      masm.unsignedExtMulLowInt32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2ExtmulHighI32x4U:
      masm.unsignedExtMulHighInt32x4(lhs, rhs, dest);
      break;
    // Dot product
    case wasm::SimdOp::I32x4DotI16x8S:
      masm.widenDotInt16x8(lhs, rhs, dest);
      break;
    // Relaxed binary ops
    case wasm::SimdOp::F32x4RelaxedMin:
      masm.minFloat32x4Relaxed(rhs, lhs);
      if (dest != lhs) masm.moveSimd128(lhs, dest);
      break;
    case wasm::SimdOp::F32x4RelaxedMax:
      masm.maxFloat32x4Relaxed(rhs, lhs);
      if (dest != lhs) masm.moveSimd128(lhs, dest);
      break;
    case wasm::SimdOp::F64x2RelaxedMin:
      masm.minFloat64x2Relaxed(rhs, lhs);
      if (dest != lhs) masm.moveSimd128(lhs, dest);
      break;
    case wasm::SimdOp::F64x2RelaxedMax:
      masm.maxFloat64x2Relaxed(rhs, lhs);
      if (dest != lhs) masm.moveSimd128(lhs, dest);
      break;
    case wasm::SimdOp::I8x16RelaxedSwizzle:
      masm.swizzleInt8x16Relaxed(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8RelaxedQ15MulrS:
      masm.q15MulrInt16x8Relaxed(lhs, rhs, dest);
      break;
    // Swizzle
    case wasm::SimdOp::I8x16Swizzle:
      masm.swizzleInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8RelaxedDotI8x16I7x16S:
      masm.dotInt8x16Int7x16(lhs, rhs, dest);
      break;
    default:
      MOZ_CRASH("PPC64: NYI SIMD binary op");
  }
}
void CodeGenerator::visitWasmBinarySimd128WithConstant(
    LWasmBinarySimd128WithConstant* ins) {
  FloatRegister lhs = ToFloatRegister(ins->lhs());
  FloatRegister dest = ToFloatRegister(ins->output());
  SimdConstant rhs = ins->rhs();
  // Load the constant into scratch, then use the binary op.
  ScratchSimd128Scope scratch(masm);
  masm.loadConstantSimd128(rhs, scratch);
  switch (ins->mir()->simdOp()) {
    // Bitwise
    case wasm::SimdOp::V128And:
      masm.bitwiseAndSimd128(lhs, scratch, dest);
      break;
    case wasm::SimdOp::V128Or:
      masm.bitwiseOrSimd128(lhs, scratch, dest);
      break;
    case wasm::SimdOp::V128Xor:
      masm.bitwiseXorSimd128(lhs, scratch, dest);
      break;
    case wasm::SimdOp::V128AndNot:
      masm.bitwiseAndNotSimd128(lhs, scratch, dest);
      break;
    // Integer add
    case wasm::SimdOp::I8x16Add:
      masm.addInt8x16(lhs, scratch, dest);
      break;
    case wasm::SimdOp::I16x8Add:
      masm.addInt16x8(lhs, scratch, dest);
      break;
    case wasm::SimdOp::I32x4Add:
      masm.addInt32x4(lhs, scratch, dest);
      break;
    case wasm::SimdOp::I64x2Add:
      masm.addInt64x2(lhs, scratch, dest);
      break;
    // Integer sub
    case wasm::SimdOp::I8x16Sub:
      masm.subInt8x16(lhs, scratch, dest);
      break;
    case wasm::SimdOp::I16x8Sub:
      masm.subInt16x8(lhs, scratch, dest);
      break;
    case wasm::SimdOp::I32x4Sub:
      masm.subInt32x4(lhs, scratch, dest);
      break;
    case wasm::SimdOp::I64x2Sub:
      masm.subInt64x2(lhs, scratch, dest);
      break;
    // Integer multiply (16-/32-bit lanes; I64x2 unreachable, see below)
    case wasm::SimdOp::I16x8Mul:
      masm.mulInt16x8(lhs, scratch, dest);
      break;
    case wasm::SimdOp::I32x4Mul:
      masm.mulInt32x4(lhs, scratch, dest);
      break;
    case wasm::SimdOp::I64x2Mul:
      // Unreachable on PPC64: MWasmBinarySimd128::specializeForConstantRhs
      // returns false in Lowering-ppc64.cpp, so MIR with a constant rhs
      // to I64x2Mul is never created on this backend.
      //
      // The previous in-place implementation was broken in three ways:
      // hard-coded VR0/VR1 staging assumed an ordering that didn't match
      // the surrounding code; a dead `mfvsrd(a, f0)` clobbered `a`
      // immediately before the next mfvsrd; and the trailing
      // `xxpermdi(dest, scratch, dest, 0)` with DM=0 placed lane-0 in the
      // wrong half. Rather than ship dead-but-broken code, crash loudly
      // if reachability ever changes — the future enabler must write a
      // correct lowering (e.g. via masm.mulInt64x2 with explicit temps).
      MOZ_CRASH("PPC64: I64x2Mul with constant rhs unimplemented "
                "(specializeForConstantRhs returns false)");
    // Compare
    case wasm::SimdOp::I8x16Eq:
      masm.compareInt8x16(Assembler::Equal, lhs, scratch, dest);
      break;
    case wasm::SimdOp::I8x16Ne:
      masm.compareInt8x16(Assembler::NotEqual, lhs, scratch, dest);
      break;
    default:
      MOZ_CRASH("PPC64: NYI SIMD binary-with-constant op");
  }
}
void CodeGenerator::visitWasmVariableShiftSimd128(
    LWasmVariableShiftSimd128* ins) {
  FloatRegister lhs = ToFloatRegister(ins->lhs());
  Register rhs = ToRegister(ins->rhs());
  FloatRegister dest = ToFloatRegister(ins->output());
  switch (ins->mir()->simdOp()) {
    case wasm::SimdOp::I8x16Shl:
      masm.leftShiftInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16ShrS:
      masm.rightShiftInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I8x16ShrU:
      masm.unsignedRightShiftInt8x16(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8Shl:
      masm.leftShiftInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8ShrS:
      masm.rightShiftInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I16x8ShrU:
      masm.unsignedRightShiftInt16x8(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4Shl:
      masm.leftShiftInt32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4ShrS:
      masm.rightShiftInt32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I32x4ShrU:
      masm.unsignedRightShiftInt32x4(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2Shl:
      masm.leftShiftInt64x2(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2ShrS:
      masm.rightShiftInt64x2(lhs, rhs, dest);
      break;
    case wasm::SimdOp::I64x2ShrU:
      masm.unsignedRightShiftInt64x2(lhs, rhs, dest);
      break;
    default:
      MOZ_CRASH("PPC64: NYI SIMD variable shift op");
  }
}
void CodeGenerator::visitWasmConstantShiftSimd128(
    LWasmConstantShiftSimd128* ins) {
  FloatRegister src = ToFloatRegister(ins->src());
  FloatRegister dest = ToFloatRegister(ins->output());
  int32_t shift = ins->shift();
  switch (ins->mir()->simdOp()) {
    case wasm::SimdOp::I8x16Shl:
      masm.leftShiftInt8x16(Imm32(shift), src, dest);
      break;
    case wasm::SimdOp::I8x16ShrS:
      masm.rightShiftInt8x16(Imm32(shift), src, dest);
      break;
    case wasm::SimdOp::I8x16ShrU:
      masm.unsignedRightShiftInt8x16(Imm32(shift), src, dest);
      break;
    case wasm::SimdOp::I16x8Shl:
      masm.leftShiftInt16x8(Imm32(shift), src, dest);
      break;
    case wasm::SimdOp::I16x8ShrS:
      masm.rightShiftInt16x8(Imm32(shift), src, dest);
      break;
    case wasm::SimdOp::I16x8ShrU:
      masm.unsignedRightShiftInt16x8(Imm32(shift), src, dest);
      break;
    case wasm::SimdOp::I32x4Shl:
      masm.leftShiftInt32x4(Imm32(shift), src, dest);
      break;
    case wasm::SimdOp::I32x4ShrS:
      masm.rightShiftInt32x4(Imm32(shift), src, dest);
      break;
    case wasm::SimdOp::I32x4ShrU:
      masm.unsignedRightShiftInt32x4(Imm32(shift), src, dest);
      break;
    case wasm::SimdOp::I64x2Shl:
      masm.leftShiftInt64x2(Imm32(shift), src, dest);
      break;
    case wasm::SimdOp::I64x2ShrS:
      masm.rightShiftInt64x2(Imm32(shift), src, dest);
      break;
    case wasm::SimdOp::I64x2ShrU:
      masm.unsignedRightShiftInt64x2(Imm32(shift), src, dest);
      break;
    default:
      MOZ_CRASH("PPC64: NYI SIMD constant shift op");
  }
}
void CodeGenerator::visitWasmSignReplicationSimd128(
    LWasmSignReplicationSimd128* ins) {
  FloatRegister src = ToFloatRegister(ins->src());
  FloatRegister dest = ToFloatRegister(ins->output());
  // Sign replication = arithmetic right shift by max amount (all sign bits).
  switch (ins->mir()->simdOp()) {
    case wasm::SimdOp::I8x16ShrS:
      masm.rightShiftInt8x16(Imm32(7), src, dest);
      break;
    case wasm::SimdOp::I16x8ShrS:
      masm.rightShiftInt16x8(Imm32(15), src, dest);
      break;
    case wasm::SimdOp::I32x4ShrS:
      masm.rightShiftInt32x4(Imm32(31), src, dest);
      break;
    case wasm::SimdOp::I64x2ShrS:
      masm.rightShiftInt64x2(Imm32(63), src, dest);
      break;
    default:
      MOZ_CRASH("Unexpected sign replication op");
  }
}
void CodeGenerator::visitWasmShuffleSimd128(LWasmShuffleSimd128* ins) {
  FloatRegister lhs = ToFloatRegister(ins->lhs());
  FloatRegister rhs = ToFloatRegister(ins->rhs());
  FloatRegister dest = ToFloatRegister(ins->output());
  SimdConstant ctrl = ins->control();
  const uint8_t* lanes = reinterpret_cast<const uint8_t*>(ctrl.bytes());
  masm.shuffleInt8x16(lanes, lhs, rhs, dest);
}
void CodeGenerator::visitWasmPermuteSimd128(LWasmPermuteSimd128* ins) {
  FloatRegister src = ToFloatRegister(ins->src());
  FloatRegister dest = ToFloatRegister(ins->output());
  // PPC64: the shuffle analysis transforms control bytes into specialized
  // formats. Reconstruct raw Wasm byte indices for our vperm implementation.
  SimdConstant ctrl = ins->control();
  uint8_t rawLanes[16];
  switch (ins->op()) {
    case SimdPermuteOp::MOVE:
      masm.moveSimd128(src, dest);
      return;
    case SimdPermuteOp::PERMUTE_32x4: {
      const int32_t* words = reinterpret_cast<const int32_t*>(ctrl.bytes());
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          rawLanes[i * 4 + j] = words[i] * 4 + j;
      break;
    }
    case SimdPermuteOp::PERMUTE_16x8: {
      // control has int16 halfword indices. High byte of halfs[0] may have
      // platform-specific flags (Perm16x8Action). Mask to get the index only.
      const int16_t* halfs = reinterpret_cast<const int16_t*>(ctrl.bytes());
      for (int i = 0; i < 8; i++) {
        int hwIdx = halfs[i] & 0x7;
        rawLanes[i * 2] = hwIdx * 2;
        rawLanes[i * 2 + 1] = hwIdx * 2 + 1;
      }
      break;
    }
    case SimdPermuteOp::BROADCAST_8x16: {
      uint8_t lane = reinterpret_cast<const int8_t*>(ctrl.bytes())[0];
      for (int i = 0; i < 16; i++) rawLanes[i] = lane;
      break;
    }
    case SimdPermuteOp::BROADCAST_16x8: {
      uint8_t lane = reinterpret_cast<const int8_t*>(ctrl.bytes())[0];
      for (int i = 0; i < 8; i++) {
        rawLanes[i * 2] = lane * 2;
        rawLanes[i * 2 + 1] = lane * 2 + 1;
      }
      break;
    }
    case SimdPermuteOp::ROTATE_RIGHT_8x16: {
      uint8_t shift = reinterpret_cast<const int8_t*>(ctrl.bytes())[0];
      for (int i = 0; i < 16; i++) rawLanes[i] = (i + shift) % 16;
      break;
    }
    case SimdPermuteOp::SHIFT_LEFT_8x16: {
      // Shifted-out positions must be zero. Use index 16+ to pick from zero.
      uint8_t shift = reinterpret_cast<const int8_t*>(ctrl.bytes())[0];
      for (int i = 0; i < 16; i++)
        rawLanes[i] = (i >= shift) ? (i - shift) : (16 + i);
      goto needsZeroRhs;
    }
    case SimdPermuteOp::SHIFT_RIGHT_8x16: {
      uint8_t shift = reinterpret_cast<const int8_t*>(ctrl.bytes())[0];
      for (int i = 0; i < 16; i++)
        rawLanes[i] = (i + shift < 16) ? (i + shift) : (16 + i);
      goto needsZeroRhs;
    }
    case SimdPermuteOp::REVERSE_16x8: {
      // Reverse bytes within each 16-bit lane: [1,0,3,2,5,4,...]
      for (int i = 0; i < 8; i++) {
        rawLanes[i * 2] = i * 2 + 1;
        rawLanes[i * 2 + 1] = i * 2;
      }
      break;
    }
    case SimdPermuteOp::REVERSE_32x4: {
      // Reverse bytes within each 32-bit lane: [3,2,1,0,7,6,5,4,...]
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          rawLanes[i * 4 + j] = i * 4 + (3 - j);
      break;
    }
    case SimdPermuteOp::REVERSE_64x2: {
      // Reverse bytes within each 64-bit lane: [7,6,5,4,3,2,1,0,15,...]
      for (int i = 0; i < 2; i++)
        for (int j = 0; j < 8; j++)
          rawLanes[i * 8 + j] = i * 8 + (7 - j);
      break;
    }
    case SimdPermuteOp::ZERO_EXTEND_8x16_TO_16x8:
    case SimdPermuteOp::ZERO_EXTEND_8x16_TO_32x4:
    case SimdPermuteOp::ZERO_EXTEND_8x16_TO_64x2:
    case SimdPermuteOp::ZERO_EXTEND_16x8_TO_32x4:
    case SimdPermuteOp::ZERO_EXTEND_16x8_TO_64x2:
    case SimdPermuteOp::ZERO_EXTEND_32x4_TO_64x2: {
      const int8_t* bytes = reinterpret_cast<const int8_t*>(ctrl.bytes());
      for (int i = 0; i < 16; i++) rawLanes[i] = bytes[i];
      goto needsZeroRhs;
    }
    default: {
      // PERMUTE_8x16 and others: control has raw byte indices.
      const int8_t* bytes = reinterpret_cast<const int8_t*>(ctrl.bytes());
      for (int i = 0; i < 16; i++) rawLanes[i] = bytes[i];
      break;
    }
  }
  masm.shuffleInt8x16(rawLanes, src, src, dest);
  return;

  needsZeroRhs: {
    // Wasm convention: rawLanes[i] in 0..15 selects src.LE_byte[idx], and
    // rawLanes[i] >= 16 means "zero". Without spilling, we can't satisfy
    // vperm's three-input constraint AND keep src alive when dest == src.
    // Strategy: vperm src with itself (any valid byte for the "zero"
    // positions, bytes get masked out below), then AND with a mask that
    // zeros those positions.
    int8_t ctrl[16], mask[16];
    for (unsigned i = 0; i < 16; i++) {
      uint8_t idx = rawLanes[i];
      if (idx < 16) {
        ctrl[i] = 15 - idx;
        mask[i] = -1;
      } else {
        ctrl[i] = 0;
        mask[i] = 0;
      }
    }
    ScratchSimd128Scope scratch(masm);
    masm.loadConstantSimd128(SimdConstant::CreateX16(ctrl), scratch);
    masm.as_vperm(dest.encoding() & 31,
                  src.encoding() & 31,
                  src.encoding() & 31,
                  scratch.encoding() & 31);
    masm.loadConstantSimd128(SimdConstant::CreateX16(mask), scratch);
    masm.as_xxland(dest, dest, scratch);
    return;
  }
}
void CodeGenerator::visitWasmReplaceLaneSimd128(LWasmReplaceLaneSimd128* ins) {
  FloatRegister lhsDest = ToFloatRegister(ins->output());
  MOZ_ASSERT(ToFloatRegister(ins->lhs()) == lhsDest);
  uint32_t lane = ins->mir()->laneIndex();
  switch (ins->mir()->simdOp()) {
    case wasm::SimdOp::I8x16ReplaceLane:
      masm.replaceLaneInt8x16(lane, ToRegister(ins->rhs()), lhsDest);
      break;
    case wasm::SimdOp::I16x8ReplaceLane:
      masm.replaceLaneInt16x8(lane, ToRegister(ins->rhs()), lhsDest);
      break;
    case wasm::SimdOp::I32x4ReplaceLane:
      masm.replaceLaneInt32x4(lane, ToRegister(ins->rhs()), lhsDest);
      break;
    case wasm::SimdOp::F32x4ReplaceLane:
      masm.replaceLaneFloat32x4(lane, ToFloatRegister(ins->rhs()), lhsDest);
      break;
    case wasm::SimdOp::F64x2ReplaceLane:
      masm.replaceLaneFloat64x2(lane, ToFloatRegister(ins->rhs()), lhsDest);
      break;
    default:
      MOZ_CRASH("PPC64: NYI SIMD replace lane op");
  }
}
void CodeGenerator::visitWasmReplaceInt64LaneSimd128(
    LWasmReplaceInt64LaneSimd128* ins) {
  MOZ_ASSERT(ins->mir()->simdOp() == wasm::SimdOp::I64x2ReplaceLane);
  FloatRegister lhsDest = ToFloatRegister(ins->output());
  MOZ_ASSERT(ToFloatRegister(ins->lhs()) == lhsDest);
  masm.replaceLaneInt64x2(ins->mir()->laneIndex(),
                          ToRegister64(ins->rhs()), lhsDest);
}
void CodeGenerator::visitWasmScalarToSimd128(LWasmScalarToSimd128* ins) {
  FloatRegister dest = ToFloatRegister(ins->output());
  switch (ins->mir()->simdOp()) {
    case wasm::SimdOp::I8x16Splat:
      masm.splatX16(ToRegister(ins->src()), dest);
      break;
    case wasm::SimdOp::I16x8Splat:
      masm.splatX8(ToRegister(ins->src()), dest);
      break;
    case wasm::SimdOp::I32x4Splat:
      masm.splatX4(ToRegister(ins->src()), dest);
      break;
    case wasm::SimdOp::F32x4Splat:
      masm.splatX4(ToFloatRegister(ins->src()), dest);
      break;
    case wasm::SimdOp::F64x2Splat:
      masm.splatX2(ToFloatRegister(ins->src()), dest);
      break;
    default:
      MOZ_CRASH("PPC64: NYI SIMD scalar-to-simd op");
  }
}
void CodeGenerator::visitWasmInt64ToSimd128(LWasmInt64ToSimd128* ins) {
  FloatRegister dest = ToFloatRegister(ins->output());
  switch (ins->mir()->simdOp()) {
    case wasm::SimdOp::I64x2Splat:
      masm.splatX2(ToRegister64(ins->src()), dest);
      break;
    default:
      MOZ_CRASH("PPC64: NYI SIMD int64-to-simd op");
  }
}
void CodeGenerator::visitWasmUnarySimd128(LWasmUnarySimd128* ins) {
  FloatRegister src = ToFloatRegister(ins->src());
  FloatRegister dest = ToFloatRegister(ins->output());
  switch (ins->mir()->simdOp()) {
    case wasm::SimdOp::I8x16Neg:
      masm.negInt8x16(src, dest);
      break;
    case wasm::SimdOp::I16x8Neg:
      masm.negInt16x8(src, dest);
      break;
    case wasm::SimdOp::I32x4Neg:
      masm.negInt32x4(src, dest);
      break;
    case wasm::SimdOp::I64x2Neg:
      masm.negInt64x2(src, dest);
      break;
    case wasm::SimdOp::I8x16Abs:
      masm.absInt8x16(src, dest);
      break;
    case wasm::SimdOp::I16x8Abs:
      masm.absInt16x8(src, dest);
      break;
    case wasm::SimdOp::I32x4Abs:
      masm.absInt32x4(src, dest);
      break;
    case wasm::SimdOp::I64x2Abs:
      masm.absInt64x2(src, dest);
      break;
    case wasm::SimdOp::V128Not:
      masm.bitwiseNotSimd128(src, dest);
      break;
    case wasm::SimdOp::F32x4Neg:
      masm.negFloat32x4(src, dest);
      break;
    case wasm::SimdOp::F64x2Neg:
      masm.negFloat64x2(src, dest);
      break;
    case wasm::SimdOp::F32x4Abs:
      masm.absFloat32x4(src, dest);
      break;
    case wasm::SimdOp::F64x2Abs:
      masm.absFloat64x2(src, dest);
      break;
    case wasm::SimdOp::F32x4Sqrt:
      masm.sqrtFloat32x4(src, dest);
      break;
    case wasm::SimdOp::F64x2Sqrt:
      masm.sqrtFloat64x2(src, dest);
      break;
    case wasm::SimdOp::F32x4Ceil:
      masm.ceilFloat32x4(src, dest);
      break;
    case wasm::SimdOp::F64x2Ceil:
      masm.ceilFloat64x2(src, dest);
      break;
    case wasm::SimdOp::F32x4Floor:
      masm.floorFloat32x4(src, dest);
      break;
    case wasm::SimdOp::F64x2Floor:
      masm.floorFloat64x2(src, dest);
      break;
    case wasm::SimdOp::F32x4Trunc:
      masm.truncFloat32x4(src, dest);
      break;
    case wasm::SimdOp::F64x2Trunc:
      masm.truncFloat64x2(src, dest);
      break;
    case wasm::SimdOp::F32x4Nearest:
      masm.nearestFloat32x4(src, dest);
      break;
    case wasm::SimdOp::F64x2Nearest:
      masm.nearestFloat64x2(src, dest);
      break;
    // Conversions
    case wasm::SimdOp::F32x4ConvertI32x4S:
      masm.convertInt32x4ToFloat32x4(src, dest);
      break;
    case wasm::SimdOp::F32x4ConvertI32x4U:
      masm.unsignedConvertInt32x4ToFloat32x4(src, dest);
      break;
    case wasm::SimdOp::I32x4TruncSatF32x4S:
      masm.truncSatFloat32x4ToInt32x4(src, dest);
      break;
    case wasm::SimdOp::I32x4TruncSatF32x4U:
      masm.unsignedTruncSatFloat32x4ToInt32x4(src, dest);
      break;
    case wasm::SimdOp::F64x2ConvertLowI32x4S:
      masm.convertInt32x4ToFloat64x2(src, dest);
      break;
    case wasm::SimdOp::F64x2ConvertLowI32x4U:
      masm.unsignedConvertInt32x4ToFloat64x2(src, dest);
      break;
    case wasm::SimdOp::F32x4DemoteF64x2Zero:
      masm.convertFloat64x2ToFloat32x4(src, dest);
      break;
    case wasm::SimdOp::F64x2PromoteLowF32x4:
      masm.convertFloat32x4ToFloat64x2(src, dest);
      break;
    case wasm::SimdOp::I32x4TruncSatF64x2SZero:
      masm.truncSatFloat64x2ToInt32x4(src, dest, ScratchSimd128Reg);
      break;
    case wasm::SimdOp::I32x4TruncSatF64x2UZero:
      masm.unsignedTruncSatFloat64x2ToInt32x4(src, dest, ScratchSimd128Reg);
      break;
    // Widen
    case wasm::SimdOp::I16x8ExtendLowI8x16S:
      masm.widenLowInt8x16(src, dest);
      break;
    case wasm::SimdOp::I16x8ExtendHighI8x16S:
      masm.widenHighInt8x16(src, dest);
      break;
    case wasm::SimdOp::I16x8ExtendLowI8x16U:
      masm.unsignedWidenLowInt8x16(src, dest);
      break;
    case wasm::SimdOp::I16x8ExtendHighI8x16U:
      masm.unsignedWidenHighInt8x16(src, dest);
      break;
    case wasm::SimdOp::I32x4ExtendLowI16x8S:
      masm.widenLowInt16x8(src, dest);
      break;
    case wasm::SimdOp::I32x4ExtendHighI16x8S:
      masm.widenHighInt16x8(src, dest);
      break;
    case wasm::SimdOp::I32x4ExtendLowI16x8U:
      masm.unsignedWidenLowInt16x8(src, dest);
      break;
    case wasm::SimdOp::I32x4ExtendHighI16x8U:
      masm.unsignedWidenHighInt16x8(src, dest);
      break;
    case wasm::SimdOp::I64x2ExtendLowI32x4S:
      masm.widenLowInt32x4(src, dest);
      break;
    case wasm::SimdOp::I64x2ExtendHighI32x4S:
      masm.widenHighInt32x4(src, dest);
      break;
    case wasm::SimdOp::I64x2ExtendLowI32x4U:
      masm.unsignedWidenLowInt32x4(src, dest);
      break;
    case wasm::SimdOp::I64x2ExtendHighI32x4U:
      masm.unsignedWidenHighInt32x4(src, dest);
      break;
    // Extended add pairwise
    case wasm::SimdOp::I16x8ExtaddPairwiseI8x16S:
      masm.extAddPairwiseInt8x16(src, dest);
      break;
    case wasm::SimdOp::I16x8ExtaddPairwiseI8x16U:
      masm.unsignedExtAddPairwiseInt8x16(src, dest);
      break;
    case wasm::SimdOp::I32x4ExtaddPairwiseI16x8S:
      masm.extAddPairwiseInt16x8(src, dest);
      break;
    case wasm::SimdOp::I32x4ExtaddPairwiseI16x8U:
      masm.unsignedExtAddPairwiseInt16x8(src, dest);
      break;
    // Relaxed truncation
    case wasm::SimdOp::I32x4RelaxedTruncF32x4S:
      masm.truncFloat32x4ToInt32x4Relaxed(src, dest);
      break;
    case wasm::SimdOp::I32x4RelaxedTruncF32x4U:
      masm.unsignedTruncFloat32x4ToInt32x4Relaxed(src, dest);
      break;
    case wasm::SimdOp::I32x4RelaxedTruncF64x2SZero:
      masm.truncFloat64x2ToInt32x4Relaxed(src, dest);
      break;
    case wasm::SimdOp::I32x4RelaxedTruncF64x2UZero:
      masm.unsignedTruncFloat64x2ToInt32x4Relaxed(src, dest);
      break;
    // Popcnt
    case wasm::SimdOp::I8x16Popcnt:
      masm.popcntInt8x16(src, dest);
      break;
    default:
      MOZ_CRASH("PPC64: NYI SIMD unary op");
  }
}
void CodeGenerator::visitWasmReduceSimd128(LWasmReduceSimd128* ins) {
  FloatRegister src = ToFloatRegister(ins->src());
  uint32_t imm = ins->mir()->imm();
  switch (ins->mir()->simdOp()) {
    case wasm::SimdOp::I8x16ExtractLaneS:
      masm.extractLaneInt8x16(imm, src, ToRegister(ins->output()));
      break;
    case wasm::SimdOp::I8x16ExtractLaneU:
      masm.unsignedExtractLaneInt8x16(imm, src, ToRegister(ins->output()));
      break;
    case wasm::SimdOp::I16x8ExtractLaneS:
      masm.extractLaneInt16x8(imm, src, ToRegister(ins->output()));
      break;
    case wasm::SimdOp::I16x8ExtractLaneU:
      masm.unsignedExtractLaneInt16x8(imm, src, ToRegister(ins->output()));
      break;
    case wasm::SimdOp::I32x4ExtractLane:
      masm.extractLaneInt32x4(imm, src, ToRegister(ins->output()));
      break;
    case wasm::SimdOp::F32x4ExtractLane:
      masm.extractLaneFloat32x4(imm, src, ToFloatRegister(ins->output()));
      break;
    case wasm::SimdOp::F64x2ExtractLane:
      masm.extractLaneFloat64x2(imm, src, ToFloatRegister(ins->output()));
      break;
    case wasm::SimdOp::V128AnyTrue:
      masm.anyTrueSimd128(src, ToRegister(ins->output()));
      break;
    case wasm::SimdOp::I8x16AllTrue:
      masm.allTrueInt8x16(src, ToRegister(ins->output()));
      break;
    case wasm::SimdOp::I16x8AllTrue:
      masm.allTrueInt16x8(src, ToRegister(ins->output()));
      break;
    case wasm::SimdOp::I32x4AllTrue:
      masm.allTrueInt32x4(src, ToRegister(ins->output()));
      break;
    case wasm::SimdOp::I64x2AllTrue:
      masm.allTrueInt64x2(src, ToRegister(ins->output()));
      break;
    case wasm::SimdOp::I8x16Bitmask:
      masm.bitmaskInt8x16(src, ToRegister(ins->output()), ScratchSimd128Reg);
      break;
    case wasm::SimdOp::I16x8Bitmask:
      masm.bitmaskInt16x8(src, ToRegister(ins->output()), ScratchSimd128Reg);
      break;
    case wasm::SimdOp::I32x4Bitmask:
      masm.bitmaskInt32x4(src, ToRegister(ins->output()), ScratchSimd128Reg);
      break;
    case wasm::SimdOp::I64x2Bitmask:
      masm.bitmaskInt64x2(src, ToRegister(ins->output()), ScratchSimd128Reg);
      break;
    default:
      MOZ_CRASH("PPC64: NYI SIMD reduce op");
  }
}
void CodeGenerator::visitWasmReduceAndBranchSimd128(
    LWasmReduceAndBranchSimd128* ins) {
  FloatRegister src = ToFloatRegister(ins->src());
  UseScratchRegisterScope temps(masm);
  Register tmp = temps.Acquire();
  switch (ins->simdOp()) {
    case wasm::SimdOp::V128AnyTrue:
      masm.anyTrueSimd128(src, tmp);
      break;
    case wasm::SimdOp::I8x16AllTrue:
      masm.allTrueInt8x16(src, tmp);
      break;
    case wasm::SimdOp::I16x8AllTrue:
      masm.allTrueInt16x8(src, tmp);
      break;
    case wasm::SimdOp::I32x4AllTrue:
      masm.allTrueInt32x4(src, tmp);
      break;
    case wasm::SimdOp::I64x2AllTrue:
      masm.allTrueInt64x2(src, tmp);
      break;
    case wasm::SimdOp::I8x16Bitmask:
      masm.bitmaskInt8x16(src, tmp, ScratchSimd128Reg);
      break;
    case wasm::SimdOp::I16x8Bitmask:
      masm.bitmaskInt16x8(src, tmp, ScratchSimd128Reg);
      break;
    case wasm::SimdOp::I32x4Bitmask:
      masm.bitmaskInt32x4(src, tmp, ScratchSimd128Reg);
      break;
    case wasm::SimdOp::I64x2Bitmask:
      masm.bitmaskInt64x2(src, tmp, ScratchSimd128Reg);
      break;
    default:
      MOZ_CRASH("PPC64: NYI SIMD reduce-and-branch op");
  }
  masm.as_cmpdi(tmp, 0);
  // Branch to ifTrue if nonzero, fall through to ifFalse.
  Label* ifTrue = skipTrivialBlocks(ins->ifTrue())->lir()->label();
  Label* ifFalse = skipTrivialBlocks(ins->ifFalse())->lir()->label();
  masm.ma_b(Assembler::NotEqual, ifTrue);
  masm.jump(ifFalse);
}
void CodeGenerator::visitWasmReduceSimd128ToInt64(
    LWasmReduceSimd128ToInt64* ins) {
  FloatRegister src = ToFloatRegister(ins->src());
  Register64 dest = ToOutRegister64(ins);
  switch (ins->mir()->simdOp()) {
    case wasm::SimdOp::I64x2ExtractLane:
      masm.extractLaneInt64x2(ins->mir()->imm(), src, dest);
      break;
    default:
      MOZ_CRASH("PPC64: NYI SIMD reduce-to-int64 op");
  }
}
static inline wasm::MemoryAccessDesc DeriveMemoryAccessDesc(
    const wasm::MemoryAccessDesc& access, Scalar::Type type) {
  return wasm::MemoryAccessDesc(access.memoryIndex(), type, access.align(),
                                access.offset32(), access.trapDesc(),
                                access.isHugeMemory());
}

void CodeGenerator::visitWasmLoadLaneSimd128(LWasmLoadLaneSimd128* ins) {
  const MWasmLoadLaneSimd128* mir = ins->mir();
  Register memoryBase = ToRegister(ins->memoryBase());
  Register ptr = ToRegister(ins->ptr());
  FloatRegister src = ToFloatRegister(ins->src());
  FloatRegister dest = ToFloatRegister(ins->output());
  UseScratchRegisterScope temps(masm);
  Register tmp = temps.Acquire();
  masm.moveSimd128(src, dest);
  switch (mir->laneSize()) {
    case 1:
      masm.wasmLoad(DeriveMemoryAccessDesc(mir->access(), Scalar::Int8),
                    memoryBase, ptr, ptr, AnyRegister(tmp));
      masm.replaceLaneInt8x16(mir->laneIndex(), tmp, dest);
      break;
    case 2:
      masm.wasmLoad(DeriveMemoryAccessDesc(mir->access(), Scalar::Int16),
                    memoryBase, ptr, ptr, AnyRegister(tmp));
      masm.replaceLaneInt16x8(mir->laneIndex(), tmp, dest);
      break;
    case 4:
      masm.wasmLoad(DeriveMemoryAccessDesc(mir->access(), Scalar::Int32),
                    memoryBase, ptr, ptr, AnyRegister(tmp));
      masm.replaceLaneInt32x4(mir->laneIndex(), tmp, dest);
      break;
    case 8: {
      masm.wasmLoadI64(DeriveMemoryAccessDesc(mir->access(), Scalar::Int64),
                       memoryBase, ptr, ptr,
                       Register64(tmp));
      masm.replaceLaneInt64x2(mir->laneIndex(), Register64(tmp), dest);
      break;
    }
    default:
      MOZ_CRASH("Unexpected lane size");
  }
}
void CodeGenerator::visitWasmStoreLaneSimd128(LWasmStoreLaneSimd128* ins) {
  const MWasmStoreLaneSimd128* mir = ins->mir();
  Register memoryBase = ToRegister(ins->memoryBase());
  Register ptr = ToRegister(ins->ptr());
  FloatRegister src = ToFloatRegister(ins->src());
  UseScratchRegisterScope temps(masm);
  Register tmp = temps.Acquire();
  switch (mir->laneSize()) {
    case 1:
      masm.unsignedExtractLaneInt8x16(mir->laneIndex(), src, tmp);
      masm.wasmStore(DeriveMemoryAccessDesc(mir->access(), Scalar::Int8),
                     AnyRegister(tmp), memoryBase, ptr, ptr);
      break;
    case 2:
      masm.unsignedExtractLaneInt16x8(mir->laneIndex(), src, tmp);
      masm.wasmStore(DeriveMemoryAccessDesc(mir->access(), Scalar::Int16),
                     AnyRegister(tmp), memoryBase, ptr, ptr);
      break;
    case 4:
      masm.extractLaneInt32x4(mir->laneIndex(), src, tmp);
      masm.wasmStore(DeriveMemoryAccessDesc(mir->access(), Scalar::Int32),
                     AnyRegister(tmp), memoryBase, ptr, ptr);
      break;
    case 8:
      masm.extractLaneInt64x2(mir->laneIndex(), src, Register64(tmp));
      masm.wasmStoreI64(DeriveMemoryAccessDesc(mir->access(), Scalar::Int64),
                        Register64(tmp), memoryBase, ptr, ptr);
      break;
    default:
      MOZ_CRASH("Unexpected lane size");
  }
}

}  // namespace jit
}  // namespace js
