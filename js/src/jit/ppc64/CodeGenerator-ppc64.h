/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef jit_ppc64_CodeGenerator_ppc64_h
#define jit_ppc64_CodeGenerator_ppc64_h

#include "jit/ppc64/Assembler-ppc64.h"
#include "jit/shared/CodeGenerator-shared.h"

namespace js {
namespace jit {

class CodeGeneratorPPC64;
class OutOfLineTableSwitch;

using OutOfLineWasmTruncateCheck =
    OutOfLineWasmTruncateCheckBase<CodeGeneratorPPC64>;

class CodeGeneratorPPC64 : public CodeGeneratorShared {
  friend class MoveResolverPPC64;

 protected:
  CodeGeneratorPPC64(MIRGenerator* gen, LIRGraph* graph, MacroAssembler* masm,
                     const wasm::CodeMetadata* codeMeta);

  MoveOperand toMoveOperand(LAllocation a) const;

  void bailoutIfFalseBool(Register lhs, LSnapshot* snapshot);

 protected:
  bool generateOutOfLineCode();
  void branchToBlock(MBasicBlock* block);

  template <typename T>
  void branchToBlock(Assembler::Condition cond, Register lhs, T rhs,
                     MBasicBlock* mir) {
    Label* label = skipTrivialBlocks(mir)->lir()->label();
    masm.branch32(cond, lhs, rhs, label);
  }
  void branchToBlock(Assembler::DoubleCondition cond, FloatRegister lhs,
                     FloatRegister rhs, MBasicBlock* mir);
  void branchToBlock(Assembler::FloatFormat fmt,
                     Assembler::DoubleCondition cond, FloatRegister lhs,
                     FloatRegister rhs, MBasicBlock* mir);

  void emitTableSwitchDispatch(MTableSwitch* mir, Register index,
                               Register base);

  void emitBigIntPtrDiv(LBigIntPtrDiv* ins, Register dividend, Register divisor,
                        Register output);
  void emitBigIntPtrMod(LBigIntPtrMod* ins, Register dividend, Register divisor,
                        Register output);

  void generateInvalidateEpilogue();

  template <typename T>
  void emitWasmLoad(T* lir);
  template <typename T>
  void emitWasmStore(T* lir);

 public:
  void visitOutOfLineTableSwitch(OutOfLineTableSwitch* ool);
  void visitOutOfLineWasmTruncateCheck(OutOfLineWasmTruncateCheck* ool);
};

typedef CodeGeneratorPPC64 CodeGeneratorSpecific;

}  // namespace jit
}  // namespace js

#endif /* jit_ppc64_CodeGenerator_ppc64_h */
