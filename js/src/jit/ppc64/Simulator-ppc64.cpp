/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "jit/ppc64/Simulator-ppc64.h"

#include <cinttypes>
#include <cmath>
#include <cstring>
#include <float.h>
#include <limits>

#include "jit/AtomicOperations.h"
#include "jit/ppc64/Assembler-ppc64.h"
#include "js/Conversions.h"
#include "threading/LockGuard.h"
#include "vm/Float16.h"
#include "vm/JSContext.h"
#include "vm/Runtime.h"
#include "wasm/WasmInstance.h"
#include "wasm/WasmSignalHandlers.h"

#define I8(v) static_cast<int8_t>(v)
#define I16(v) static_cast<int16_t>(v)
#define U16(v) static_cast<uint16_t>(v)
#define I32(v) static_cast<int32_t>(v)
#define U32(v) static_cast<uint32_t>(v)
#define I64(v) static_cast<int64_t>(v)
#define U64(v) static_cast<uint64_t>(v)
#define I128(v) static_cast<__int128_t>(v)
#define U128(v) static_cast<__uint128_t>(v)

namespace js {
namespace jit {

static int64_t MultiplyHighSigned(int64_t u, int64_t v) {
  uint64_t u0, v0, w0;
  int64_t u1, v1, w1, w2, t;

  u0 = u & 0xFFFFFFFFL;
  u1 = u >> 32;
  v0 = v & 0xFFFFFFFFL;
  v1 = v >> 32;

  w0 = u0 * v0;
  t = u1 * v0 + (w0 >> 32);
  w1 = t & 0xFFFFFFFFL;
  w2 = t >> 32;
  w1 = u0 * v1 + w1;

  return u1 * v1 + w2 + (w1 >> 32);
}

static uint64_t MultiplyHighUnsigned(uint64_t u, uint64_t v) {
  uint64_t u0, v0, w0;
  uint64_t u1, v1, w1, w2, t;

  u0 = u & 0xFFFFFFFFL;
  u1 = u >> 32;
  v0 = v & 0xFFFFFFFFL;
  v1 = v >> 32;

  w0 = u0 * v0;
  t = u1 * v0 + (w0 >> 32);
  w1 = t & 0xFFFFFFFFL;
  w2 = t >> 32;
  w1 = u0 * v1 + w1;

  return u1 * v1 + w2 + (w1 >> 32);
}

inline constexpr uint32_t RotateLeft32(uint32_t value, uint32_t shift) {
  return (value << shift) | (value >> ((32 - shift) & 31));
}

inline constexpr uint64_t RotateLeft64(uint64_t value, uint64_t shift) {
  return (value << shift) | (value >> ((64 - shift) & 63));
}

// Generate a 64-bit mask with bits mb..me set (PPC numbering: 0 = MSB = bit
// 63 in C).  When mb <= me, a contiguous range is set; when mb > me, the
// mask wraps around (bits 0..me and mb..63 are set).
static inline uint64_t MASK64(unsigned mb, unsigned me) {
  MOZ_ASSERT(mb < 64 && me < 64);
  uint64_t mask_begin = ~0ULL >> mb;
  uint64_t mask_end = ~0ULL << (63 - me);
  if (mb <= me) {
    return mask_begin & mask_end;
  }
  return mask_begin | mask_end;
}

static inline uint32_t MASK32(unsigned mb, unsigned me) {
  MOZ_ASSERT(mb < 32 && me < 32);
  uint32_t mask_begin = ~0U >> mb;
  uint32_t mask_end = ~0U << (31 - me);
  if (mb <= me) {
    return mask_begin & mask_end;
  }
  return mask_begin | mask_end;
}

// Count leading zeros.
static inline int CountLeadingZeros64(uint64_t value) {
  if (value == 0) return 64;
  return __builtin_clzll(value);
}

static inline int CountLeadingZeros32(uint32_t value) {
  if (value == 0) return 32;
  return __builtin_clz(value);
}

static inline int CountTrailingZeros64(uint64_t value) {
  if (value == 0) return 64;
  return __builtin_ctzll(value);
}

static inline int CountTrailingZeros32(uint32_t value) {
  if (value == 0) return 32;
  return __builtin_ctz(value);
}

static inline int PopCount64(uint64_t value) {
  return __builtin_popcountll(value);
}

static inline int PopCount32(uint32_t value) {
  return __builtin_popcount(value);
}

static inline uint64_t PopCountPerByte(uint64_t value) {
  uint64_t result = 0;
  for (int i = 0; i < 8; i++) {
    uint8_t byte = (value >> (i * 8)) & 0xFF;
    result |= (uint64_t)__builtin_popcount(byte) << (i * 8);
  }
  return result;
}

// PPC64 C argument slots: PPC64 ELFv2 ABI does not require C argument
// slots on the stack for register-passed arguments, but we reserve the
// link area (32 bytes).
const int kCArgSlotCount = 0;
const int kCArgsSlotsSize = kCArgSlotCount * sizeof(uintptr_t);

// -----------------------------------------------------------------------------
// PPC64 SimInstruction.

class SimInstruction {
 public:
  enum {
    kInstrSize = 4,
    kPCReadOffset = 0
  };

  inline Instr instructionBits() const {
    return *reinterpret_cast<const Instr*>(this);
  }

  inline void setInstructionBits(Instr value) {
    *reinterpret_cast<Instr*>(this) = value;
  }

  inline int bit(int nr) const { return (instructionBits() >> nr) & 1; }

  inline uint32_t bits(int hi, int lo) const {
    return (instructionBits() >> lo) & ((2U << (hi - lo)) - 1);
  }

  inline uint32_t opcode() const { return bits(31, 26); }

  inline uint32_t rtValue() const { return bits(25, 21); }
  inline uint32_t rsValue() const { return bits(25, 21); }
  inline uint32_t raValue() const { return bits(20, 16); }
  inline uint32_t rbValue() const { return bits(15, 11); }
  inline uint32_t rcValue() const { return bits(10, 6); }

  inline uint32_t boValue() const { return bits(25, 21); }
  inline uint32_t biValue() const { return bits(20, 16); }

  // D-form 16-bit immediate (sign-extend to get signed value).
  inline int16_t imm16Value() const { return I16(bits(15, 0)); }
  inline uint16_t uimm16Value() const { return U16(bits(15, 0)); }

  // DS-form 14-bit displacement (bits 2..15, 4-byte aligned).
  inline int16_t ds14Value() const {
    return I16(bits(15, 2) << 2);
  }

  // B-form 14-bit branch displacement (bits 2..15, 4-byte aligned).
  inline int32_t bd16Value() const {
    int16_t raw = I16(bits(15, 2) << 2);
    return (int32_t)raw;
  }

  // I-form 24-bit branch offset (bits 2..25, sign-extended, 4-byte aligned).
  inline int32_t li26Value() const {
    int32_t raw = I32(bits(25, 2) << 2);
    // Sign-extend from 26 bits.
    return (raw << 6) >> 6;
  }

  // Extended opcode for X-form / XO-form (bits 1..10).
  inline uint32_t xoValue() const { return bits(10, 1); }

  // Extended opcode for XL-form (bits 1..10).
  inline uint32_t xlValue() const { return bits(10, 1); }

  // MD-form SH field: sh[0:4] in instruction bits 15:11, sh[5] in bit 1.
  // Assembler encodes: ((sh & 0x1f) << 11) | ((sh & 0x20) >> 4).
  inline uint32_t mdSHValue() const {
    return bits(15, 11) | (bit(1) << 5);
  }
  // mb/me for MD-form (rldicl/rldicr/rldic/rldimi): 6-bit field split as
  // mb[0:4] in instruction bits 10:6 and mb[5] in bit 5.
  inline uint32_t mdMBValue() const {
    return bits(10, 6) | (bit(5) << 5);
  }
  inline uint32_t mdMEValue() const { return mdMBValue(); }

  // MD-form XO (bits 2..4).
  inline uint32_t mdXOValue() const { return bits(4, 2); }

  // MDS-form (rldcl, rldcr): mb[0:4] in bits 10:6, mb[5] in bit 5.
  inline uint32_t mdsMBValue() const {
    return bits(10, 6) | (bit(5) << 5);
  }

  // M-form fields (32-bit rotate/mask).
  inline uint32_t mSHValue() const { return bits(15, 11); }
  inline uint32_t mMBValue() const { return bits(10, 6); }
  inline uint32_t mMEValue() const { return bits(5, 1); }

  // Rc bit.
  inline bool rcBit() const { return bit(0); }

  // AA bit for branch instructions.
  inline bool aaBit() const { return bit(1); }

  // LK bit for branch instructions.
  inline bool lkBit() const { return bit(0); }

  // OE bit for XO-form arithmetic.
  inline bool oeBit() const { return bit(10); }

  // L bit for compare instructions (bit 21).
  inline bool lBit() const { return bit(21); }

  // BF field (bits 23..25) for compares.
  inline uint32_t bfValue() const { return bits(25, 23); }

  bool isTrap() const {
    uint32_t instr = instructionBits();
    // PPC_trap = 0x7FE00008 (tw 31,0,0).
    // Don't treat the call-redirection instruction or wasm trap as a
    // debugger trap.
    if (instr == kCallRedirInstr) return false;
    if (instr == 0x7FE00008) return false;
    // Any other tw instruction with TO=31 is a trap.
    if (opcode() == 31 && (xoValue() == 4)) return true;
    return false;
  }

 private:
  SimInstruction() = delete;
  SimInstruction(const SimInstruction& other) = delete;
  void operator=(const SimInstruction& other) = delete;
};

// -----------------------------------------------------------------------------
// ICache.

class CachePage {
 public:
  static const int LINE_VALID = 0;
  static const int LINE_INVALID = 1;

  static const int kPageShift = 12;
  static const int kPageSize = 1 << kPageShift;
  static const int kPageMask = kPageSize - 1;
  static const int kLineShift = 2;
  static const int kLineLength = 1 << kLineShift;
  static const int kLineMask = kLineLength - 1;

  CachePage() { memset(&validity_map_, LINE_INVALID, sizeof(validity_map_)); }

  char* validityByte(int offset) {
    return &validity_map_[offset >> kLineShift];
  }

  char* cachedData(int offset) { return &data_[offset]; }

 private:
  char data_[kPageSize];
  static const int kValidityMapSize = kPageSize >> kLineShift;
  char validity_map_[kValidityMapSize];
};

class AutoLockSimulatorCache : public LockGuard<Mutex> {
  using Base = LockGuard<Mutex>;

 public:
  explicit AutoLockSimulatorCache()
      : Base(SimulatorProcess::singleton_->cacheLock_) {}
};

mozilla::Atomic<size_t, mozilla::ReleaseAcquire>
    SimulatorProcess::ICacheCheckingDisableCount(1);
SimulatorProcess* SimulatorProcess::singleton_ = nullptr;

int64_t Simulator::StopSimAt = -1;

// -----------------------------------------------------------------------------
// Simulator Create / Destroy.

Simulator* Simulator::Create() {
  auto sim = MakeUnique<Simulator>();
  if (!sim) {
    return nullptr;
  }

  if (!sim->init()) {
    return nullptr;
  }

  int64_t stopAt;
  char* stopAtStr = getenv("PPC64_SIM_STOP_AT");
  if (stopAtStr && sscanf(stopAtStr, "%" PRIi64, &stopAt) == 1) {
    fprintf(stderr, "\nStopping simulation at icount %" PRIi64 "\n", stopAt);
    Simulator::StopSimAt = stopAt;
  }

  return sim.release();
}

void Simulator::Destroy(Simulator* sim) { js_delete(sim); }

// -----------------------------------------------------------------------------
// Debugger.

class ppc64Debugger {
 public:
  explicit ppc64Debugger(Simulator* sim) : sim_(sim) {}

  void stop(SimInstruction* instr);
  void debug();
  void printAllRegs();
  void printAllRegsIncludingFPU();

 private:
  static const Instr kBreakpointInstr = 0x7FE00008;  // PPC_trap
  static const Instr kNopInstr = 0x60000000;          // PPC_nop

  Simulator* sim_;

  int64_t getRegisterValue(int regnum);
  int64_t getFPURegisterValueLong(int regnum);
  float getFPURegisterValueFloat(int regnum);
  double getFPURegisterValueDouble(int regnum);
  bool getValue(const char* desc, int64_t* value);

  bool setBreakpoint(SimInstruction* breakpc);
  bool deleteBreakpoint(SimInstruction* breakpc);

  void undoBreakpoints();
  void redoBreakpoints();
};

[[maybe_unused]] static void UNIMPLEMENTED() {
  printf("UNIMPLEMENTED instruction.\n");
  MOZ_CRASH();
}
[[maybe_unused]] static void UNREACHABLE() {
  printf("UNREACHABLE instruction.\n");
  MOZ_CRASH();
}
[[maybe_unused]] static void UNSUPPORTED() {
  printf("Unsupported instruction.\n");
  MOZ_CRASH();
}

void ppc64Debugger::stop(SimInstruction* instr) {
  uint32_t code = 0;
  char* msg = *reinterpret_cast<char**>(sim_->get_pc() +
                                        SimInstruction::kInstrSize);
  if (!sim_->watchedStops_[code].desc_) {
    sim_->watchedStops_[code].desc_ = msg;
  }
  if (code != kMaxStopCode) {
    printf("Simulator hit stop %u: %s\n", code, msg);
  } else {
    printf("Simulator hit %s\n", msg);
  }
  sim_->set_pc(sim_->get_pc() + 2 * SimInstruction::kInstrSize);
  debug();
}

int64_t ppc64Debugger::getRegisterValue(int regnum) {
  if (regnum == kPCRegister) {
    return sim_->get_pc();
  }
  return sim_->getRegister(regnum);
}

int64_t ppc64Debugger::getFPURegisterValueLong(int regnum) {
  return sim_->getFpuRegister(regnum);
}

float ppc64Debugger::getFPURegisterValueFloat(int regnum) {
  return sim_->getFpuRegisterFloat(regnum);
}

double ppc64Debugger::getFPURegisterValueDouble(int regnum) {
  return sim_->getFpuRegisterDouble(regnum);
}

bool ppc64Debugger::getValue(const char* desc, int64_t* value) {
  Register reg = Register::FromName(desc);
  if (reg != InvalidReg) {
    *value = getRegisterValue(reg.code());
    return true;
  }

  if (strncmp(desc, "0x", 2) == 0) {
    return sscanf(desc + 2, "%" PRIx64, reinterpret_cast<uint64_t*>(value)) ==
           1;
  }
  return sscanf(desc, "%" PRIu64, reinterpret_cast<uint64_t*>(value)) == 1;
}

bool ppc64Debugger::setBreakpoint(SimInstruction* breakpc) {
  if (sim_->break_pc_ != nullptr) {
    return false;
  }

  sim_->break_pc_ = breakpc;
  sim_->break_instr_ = breakpc->instructionBits();
  return true;
}

bool ppc64Debugger::deleteBreakpoint(SimInstruction* breakpc) {
  if (sim_->break_pc_ != nullptr) {
    sim_->break_pc_->setInstructionBits(sim_->break_instr_);
  }

  sim_->break_pc_ = nullptr;
  sim_->break_instr_ = 0;
  return true;
}

void ppc64Debugger::undoBreakpoints() {
  if (sim_->break_pc_) {
    sim_->break_pc_->setInstructionBits(sim_->break_instr_);
  }
}

void ppc64Debugger::redoBreakpoints() {
  if (sim_->break_pc_) {
    sim_->break_pc_->setInstructionBits(kBreakpointInstr);
  }
}

void ppc64Debugger::printAllRegs() {
  int64_t value;
  for (uint32_t i = 0; i < Registers::Total; i++) {
    value = getRegisterValue(i);
    printf("%3s: 0x%016" PRIx64 " %20" PRIi64 "   ", Registers::GetName(i),
           value, value);

    if (i % 2) {
      printf("\n");
    }
  }
  printf("\n");

  value = getRegisterValue(Simulator::pc);
  printf("  pc: 0x%016" PRIx64 "\n", value);
  printf("  lr: 0x%016" PRIx64 "\n", sim_->getLR());
  printf(" ctr: 0x%016" PRIx64 "\n", sim_->getCTR());
  printf("  cr: 0x%08x\n", sim_->getCR());
  printf(" xer: 0x%016" PRIx64 "\n", sim_->getXER());
}

void ppc64Debugger::printAllRegsIncludingFPU() {
  printAllRegs();

  printf("\n\n");
  for (uint32_t i = 0; i < FloatRegisters::TotalPhys; i++) {
    printf("%3s: 0x%016" PRIx64 "\tflt: %-8.4g\tdbl: %-16.4g\n",
           FloatRegisters::GetName(i), getFPURegisterValueLong(i),
           getFPURegisterValueFloat(i), getFPURegisterValueDouble(i));
  }
}

static char* ReadLine(const char* prompt) {
  UniqueChars result;
  char lineBuf[256];
  int offset = 0;
  bool keepGoing = true;
  fprintf(stdout, "%s", prompt);
  fflush(stdout);
  while (keepGoing) {
    if (fgets(lineBuf, sizeof(lineBuf), stdin) == nullptr) {
      return nullptr;
    }
    int len = strlen(lineBuf);
    if (len > 0 && lineBuf[len - 1] == '\n') {
      keepGoing = false;
    }
    if (!result) {
      result.reset(js_pod_malloc<char>(len + 1));
      if (!result) {
        return nullptr;
      }
    } else {
      int new_len = offset + len + 1;
      char* new_result = js_pod_malloc<char>(new_len);
      if (!new_result) {
        return nullptr;
      }
      memcpy(new_result, result.get(), offset * sizeof(char));
      result.reset(new_result);
    }
    memcpy(result.get() + offset, lineBuf, len * sizeof(char));
    offset += len;
  }

  MOZ_ASSERT(result);
  result[offset] = '\0';
  return result.release();
}

static void DisassembleInstruction(uint64_t pc) {
  printf("  0x%016" PRIx64 ":  %08x\n", pc,
         *reinterpret_cast<uint32_t*>(pc));
}

void ppc64Debugger::debug() {
  intptr_t lastPC = -1;
  bool done = false;

#define COMMAND_SIZE 63
#define ARG_SIZE 255

#define STR(a) #a
#define XSTR(a) STR(a)

  char cmd[COMMAND_SIZE + 1];
  char arg1[ARG_SIZE + 1];
  char arg2[ARG_SIZE + 1];
  char* argv[3] = {cmd, arg1, arg2};

  cmd[COMMAND_SIZE] = 0;
  arg1[ARG_SIZE] = 0;
  arg2[ARG_SIZE] = 0;

  undoBreakpoints();

  while (!done && (sim_->get_pc() != Simulator::end_sim_pc)) {
    if (lastPC != sim_->get_pc()) {
      DisassembleInstruction(sim_->get_pc());
      lastPC = sim_->get_pc();
    }
    char* line = ReadLine("sim> ");
    if (line == nullptr) {
      break;
    } else {
      char* last_input = sim_->lastDebuggerInput();
      if (strcmp(line, "\n") == 0 && last_input != nullptr) {
        line = last_input;
      } else {
        sim_->setLastDebuggerInput(line);
      }
      int argc = sscanf(line,
                              "%" XSTR(COMMAND_SIZE) "s "
                              "%" XSTR(ARG_SIZE) "s "
                              "%" XSTR(ARG_SIZE) "s",
                              cmd, arg1, arg2);
      if ((strcmp(cmd, "si") == 0) || (strcmp(cmd, "stepi") == 0)) {
        SimInstruction* instr =
            reinterpret_cast<SimInstruction*>(sim_->get_pc());
        if (!instr->isTrap()) {
          sim_->instructionDecode(instr);
        } else {
          printf("/!\\ Jumping over generated breakpoint.\n");
          sim_->set_pc(sim_->get_pc() + SimInstruction::kInstrSize);
        }
        sim_->icount_++;
      } else if ((strcmp(cmd, "c") == 0) || (strcmp(cmd, "cont") == 0)) {
        sim_->instructionDecode(
            reinterpret_cast<SimInstruction*>(sim_->get_pc()));
        sim_->icount_++;
        done = true;
      } else if ((strcmp(cmd, "p") == 0) || (strcmp(cmd, "print") == 0)) {
        if (argc == 2) {
          int64_t value;
          if (strcmp(arg1, "all") == 0) {
            printAllRegs();
          } else if (strcmp(arg1, "allf") == 0) {
            printAllRegsIncludingFPU();
          } else {
            Register reg = Register::FromName(arg1);
            FloatRegisters::Code fReg = FloatRegisters::FromName(arg1);
            if (reg != InvalidReg) {
              value = getRegisterValue(reg.code());
              printf("%s: 0x%016" PRIx64 " %20" PRIi64 " \n", arg1, value,
                     value);
            } else if (fReg != FloatRegisters::Invalid) {
              printf("%3s: 0x%016" PRIx64 "\tflt: %-8.4g\tdbl: %-16.4g\n",
                     FloatRegisters::GetName(fReg),
                     getFPURegisterValueLong(fReg),
                     getFPURegisterValueFloat(fReg),
                     getFPURegisterValueDouble(fReg));
            } else {
              printf("%s unrecognized\n", arg1);
            }
          }
        } else {
          printf("print <register> or print <fpu register> single\n");
        }
      } else if (strcmp(cmd, "stack") == 0 || strcmp(cmd, "mem") == 0) {
        int64_t* cur = nullptr;
        int64_t* end = nullptr;
        int next_arg = 1;

        if (strcmp(cmd, "stack") == 0) {
          cur = reinterpret_cast<int64_t*>(sim_->getRegister(Simulator::sp));
        } else {
          int64_t value;
          if (!getValue(arg1, &value)) {
            printf("%s unrecognized\n", arg1);
            continue;
          }
          cur = reinterpret_cast<int64_t*>(value);
          next_arg++;
        }

        int64_t words;
        if (argc == next_arg) {
          words = 10;
        } else {
          if (!getValue(argv[next_arg], &words)) {
            words = 10;
          }
        }
        end = cur + words;

        while (cur < end) {
          printf("  %p:  0x%016" PRIx64 " %20" PRIi64, cur, *cur, *cur);
          printf("\n");
          cur++;
        }

      } else if ((strcmp(cmd, "disasm") == 0) || (strcmp(cmd, "dpc") == 0) ||
                 (strcmp(cmd, "di") == 0)) {
        uint8_t* cur = nullptr;
        uint8_t* end = nullptr;

        if (argc == 1) {
          cur = reinterpret_cast<uint8_t*>(sim_->get_pc());
          end = cur + (10 * SimInstruction::kInstrSize);
        } else if (argc == 2) {
          Register reg = Register::FromName(arg1);
          if (reg != InvalidReg || strncmp(arg1, "0x", 2) == 0) {
            int64_t value;
            if (getValue(arg1, &value)) {
              cur = reinterpret_cast<uint8_t*>(value);
              end = cur + (10 * SimInstruction::kInstrSize);
            }
          } else {
            int64_t value;
            if (getValue(arg1, &value)) {
              cur = reinterpret_cast<uint8_t*>(sim_->get_pc());
              end = cur + (value * SimInstruction::kInstrSize);
            }
          }
        } else {
          int64_t value1;
          int64_t value2;
          if (getValue(arg1, &value1) && getValue(arg2, &value2)) {
            cur = reinterpret_cast<uint8_t*>(value1);
            end = cur + (value2 * SimInstruction::kInstrSize);
          }
        }

        while (cur < end) {
          DisassembleInstruction(uint64_t(cur));
          cur += SimInstruction::kInstrSize;
        }
      } else if (strcmp(cmd, "gdb") == 0) {
        printf("relinquishing control to gdb\n");
#if defined(__x86_64__)
        asm("int $3");
#elif defined(__aarch64__)
        asm("brk #0xf000");
#endif
        printf("regaining control from gdb\n");
      } else if (strcmp(cmd, "break") == 0) {
        if (argc == 2) {
          int64_t value;
          if (getValue(arg1, &value)) {
            if (!setBreakpoint(reinterpret_cast<SimInstruction*>(value))) {
              printf("setting breakpoint failed\n");
            }
          } else {
            printf("%s unrecognized\n", arg1);
          }
        } else {
          printf("break <address>\n");
        }
      } else if (strcmp(cmd, "del") == 0) {
        if (!deleteBreakpoint(nullptr)) {
          printf("deleting breakpoint failed\n");
        }
      } else if (strcmp(cmd, "flags") == 0) {
        printf("CR: 0x%08x   XER: 0x%016" PRIx64 "\n", sim_->getCR(),
               sim_->getXER());
      } else if (strcmp(cmd, "stop") == 0) {
        int64_t value;
        intptr_t stop_pc = sim_->get_pc() - 2 * SimInstruction::kInstrSize;
        SimInstruction* stop_instr =
            reinterpret_cast<SimInstruction*>(stop_pc);
        SimInstruction* msg_address = reinterpret_cast<SimInstruction*>(
            stop_pc + SimInstruction::kInstrSize);
        if ((argc == 2) && (strcmp(arg1, "unstop") == 0)) {
          if (sim_->isStopInstruction(stop_instr)) {
            stop_instr->setInstructionBits(kNopInstr);
            msg_address->setInstructionBits(kNopInstr);
          } else {
            printf("Not at debugger stop.\n");
          }
        } else if (argc == 3) {
          if (strcmp(arg1, "info") == 0) {
            if (strcmp(arg2, "all") == 0) {
              printf("Stop information:\n");
              for (uint32_t i = kMaxWatchpointCode + 1; i <= kMaxStopCode;
                   i++) {
                sim_->printStopInfo(i);
              }
            } else if (getValue(arg2, &value)) {
              sim_->printStopInfo(value);
            } else {
              printf("Unrecognized argument.\n");
            }
          } else if (strcmp(arg1, "enable") == 0) {
            if (strcmp(arg2, "all") == 0) {
              for (uint32_t i = kMaxWatchpointCode + 1; i <= kMaxStopCode;
                   i++) {
                sim_->enableStop(i);
              }
            } else if (getValue(arg2, &value)) {
              sim_->enableStop(value);
            } else {
              printf("Unrecognized argument.\n");
            }
          } else if (strcmp(arg1, "disable") == 0) {
            if (strcmp(arg2, "all") == 0) {
              for (uint32_t i = kMaxWatchpointCode + 1; i <= kMaxStopCode;
                   i++) {
                sim_->disableStop(i);
              }
            } else if (getValue(arg2, &value)) {
              sim_->disableStop(value);
            } else {
              printf("Unrecognized argument.\n");
            }
          }
        } else {
          printf("Wrong usage. Use help command for more information.\n");
        }
      } else if ((strcmp(cmd, "h") == 0) || (strcmp(cmd, "help") == 0)) {
        printf("cont\n");
        printf("  continue execution (alias 'c')\n");
        printf("stepi\n");
        printf("  step one instruction (alias 'si')\n");
        printf("print <register>\n");
        printf("  print register content (alias 'p')\n");
        printf("  use register name 'all' to print all registers\n");
        printf("stack [<words>]\n");
        printf("  dump stack content, default dump 10 words)\n");
        printf("mem <address> [<words>]\n");
        printf("  dump memory content, default dump 10 words)\n");
        printf("flags\n");
        printf("  print CR and XER\n");
        printf("disasm [<instructions>]\n");
        printf("disasm [<address/register>]\n");
        printf("disasm [[<address/register>] <instructions>]\n");
        printf("  disassemble code, default is 10 instructions\n");
        printf("  from pc (alias 'di')\n");
        printf("gdb\n");
        printf("  enter gdb\n");
        printf("break <address>\n");
        printf("  set a break point on the address\n");
        printf("del\n");
        printf("  delete the breakpoint\n");
      } else {
        printf("Unknown command: %s\n", cmd);
      }
    }
  }

  redoBreakpoints();

#undef COMMAND_SIZE
#undef ARG_SIZE

#undef STR
#undef XSTR
}

// -----------------------------------------------------------------------------
// ICache helpers.

static bool AllOnOnePage(uintptr_t start, int size) {
  intptr_t start_page = (start & ~CachePage::kPageMask);
  intptr_t end_page = ((start + size) & ~CachePage::kPageMask);
  return start_page == end_page;
}

void Simulator::setLastDebuggerInput(char* input) {
  js_free(lastDebuggerInput_);
  lastDebuggerInput_ = input;
}

static CachePage* GetCachePageLocked(SimulatorProcess::ICacheMap& i_cache,
                                     void* page) {
  SimulatorProcess::ICacheMap::AddPtr p = i_cache.lookupForAdd(page);
  if (p) {
    return p->value();
  }
  AutoEnterOOMUnsafeRegion oomUnsafe;
  CachePage* new_page = js_new<CachePage>();
  if (!new_page || !i_cache.add(p, page, new_page)) {
    oomUnsafe.crash("Simulator CachePage");
  }
  return new_page;
}

static void FlushOnePageLocked(SimulatorProcess::ICacheMap& i_cache,
                               intptr_t start, int size) {
  MOZ_ASSERT(size <= CachePage::kPageSize);
  MOZ_ASSERT(AllOnOnePage(start, size - 1));
  MOZ_ASSERT((start & CachePage::kLineMask) == 0);
  MOZ_ASSERT((size & CachePage::kLineMask) == 0);
  void* page = reinterpret_cast<void*>(start & (~CachePage::kPageMask));
  int offset = (start & CachePage::kPageMask);
  CachePage* cache_page = GetCachePageLocked(i_cache, page);
  char* valid_bytemap = cache_page->validityByte(offset);
  memset(valid_bytemap, CachePage::LINE_INVALID, size >> CachePage::kLineShift);
}

static void FlushICacheLocked(SimulatorProcess::ICacheMap& i_cache,
                              void* start_addr, size_t size) {
  intptr_t start = reinterpret_cast<intptr_t>(start_addr);
  int intra_line = (start & CachePage::kLineMask);
  start -= intra_line;
  size += intra_line;
  size = ((size - 1) | CachePage::kLineMask) + 1;
  int offset = (start & CachePage::kPageMask);
  while (!AllOnOnePage(start, size - 1)) {
    int bytes_to_flush = CachePage::kPageSize - offset;
    FlushOnePageLocked(i_cache, start, bytes_to_flush);
    start += bytes_to_flush;
    size -= bytes_to_flush;
    MOZ_ASSERT((start & CachePage::kPageMask) == 0);
    offset = 0;
  }
  if (size != 0) {
    FlushOnePageLocked(i_cache, start, size);
  }
}

/* static */
void SimulatorProcess::checkICacheLocked(SimInstruction* instr) {
  intptr_t address = reinterpret_cast<intptr_t>(instr);
  void* page = reinterpret_cast<void*>(address & (~CachePage::kPageMask));
  void* line = reinterpret_cast<void*>(address & (~CachePage::kLineMask));
  int offset = (address & CachePage::kPageMask);
  CachePage* cache_page = GetCachePageLocked(icache(), page);
  char* cache_valid_byte = cache_page->validityByte(offset);
  bool cache_hit = (*cache_valid_byte == CachePage::LINE_VALID);
  char* cached_line = cache_page->cachedData(offset & ~CachePage::kLineMask);

  if (cache_hit) {
    mozilla::DebugOnly<int> cmpret =
        memcmp(reinterpret_cast<void*>(instr), cache_page->cachedData(offset),
               SimInstruction::kInstrSize);
    MOZ_ASSERT(cmpret == 0);
  } else {
    memcpy(cached_line, line, CachePage::kLineLength);
    *cache_valid_byte = CachePage::LINE_VALID;
  }
}

HashNumber SimulatorProcess::ICacheHasher::hash(const Lookup& l) {
  return U32(reinterpret_cast<uintptr_t>(l)) >> 2;
}

bool SimulatorProcess::ICacheHasher::match(const Key& k, const Lookup& l) {
  MOZ_ASSERT((reinterpret_cast<intptr_t>(k) & CachePage::kPageMask) == 0);
  MOZ_ASSERT((reinterpret_cast<intptr_t>(l) & CachePage::kPageMask) == 0);
  return k == l;
}

/* static */
void SimulatorProcess::FlushICache(void* start_addr, size_t size) {
  if (!ICacheCheckingDisableCount) {
    AutoLockSimulatorCache als;
    js::jit::FlushICacheLocked(icache(), start_addr, size);
  }
}

// -----------------------------------------------------------------------------
// Redirection.

class Redirection {
  friend class SimulatorProcess;

  Redirection(void* nativeFunction, ABIFunctionType type)
      : nativeFunction_(nativeFunction),
        swiInstruction_(kCallRedirInstr),
        type_(type),
        next_(nullptr) {
    next_ = SimulatorProcess::redirection();
    if (!SimulatorProcess::ICacheCheckingDisableCount) {
      FlushICacheLocked(SimulatorProcess::icache(), addressOfSwiInstruction(),
                        SimInstruction::kInstrSize);
    }
    SimulatorProcess::setRedirection(this);
  }

 public:
  void* addressOfSwiInstruction() { return &swiInstruction_; }
  void* nativeFunction() const { return nativeFunction_; }
  ABIFunctionType type() const { return type_; }

  static Redirection* Get(void* nativeFunction, ABIFunctionType type) {
    AutoLockSimulatorCache als;

    Redirection* current = SimulatorProcess::redirection();
    for (; current != nullptr; current = current->next_) {
      if (current->nativeFunction_ == nativeFunction) {
        MOZ_ASSERT(current->type() == type);
        return current;
      }
    }

    AutoEnterOOMUnsafeRegion oomUnsafe;
    Redirection* redir = js_pod_malloc<Redirection>(1);
    if (!redir) {
      oomUnsafe.crash("Simulator redirection");
    }
    new (redir) Redirection(nativeFunction, type);
    return redir;
  }

  static Redirection* FromSwiInstruction(SimInstruction* swiInstruction) {
    uint8_t* addrOfSwi = reinterpret_cast<uint8_t*>(swiInstruction);
    uint8_t* addrOfRedirection =
        addrOfSwi - offsetof(Redirection, swiInstruction_);
    return reinterpret_cast<Redirection*>(addrOfRedirection);
  }

 private:
  void* nativeFunction_;
  uint32_t swiInstruction_;
  ABIFunctionType type_;
  Redirection* next_;
};

// -----------------------------------------------------------------------------
// Simulator constructor / destructor / init.

Simulator::Simulator() {
  stack_ = nullptr;
  stackLimit_ = 0;
  pc_modified_ = false;
  icount_ = 0;
  break_count_ = 0;
  break_pc_ = nullptr;
  break_instr_ = 0;
  single_stepping_ = false;
  single_step_callback_ = nullptr;
  single_step_callback_arg_ = nullptr;

  for (int i = 0; i < Register::kNumSimuRegisters; i++) {
    registers_[i] = 0;
  }
  for (int i = 0; i < Simulator::FPURegister::kNumFPURegisters; i++) {
    FPUregisters_[i] = 0;
  }

  LR_ = 0;
  CTR_ = 0;
  CR_ = 0;
  XER_ = 0;
  FPSCR_ = 0;
  LLBit_ = false;
  LLAddr_ = 0;
  lastLLValue_ = 0;

  // Initialize PC and LR to a known bad value that will cause an
  // access violation if the simulator ever tries to execute it.
  registers_[pc] = bad_ra;
  LR_ = bad_ra;

  lastDebuggerInput_ = nullptr;
}

bool Simulator::init() {
  static const size_t stackSize = 2 * 1024 * 1024;
  stack_ = js_pod_malloc<char>(stackSize);
  if (!stack_) {
    return false;
  }

  // Leave a safety margin of 1MB to prevent overrunning the stack.
  stackLimit_ = reinterpret_cast<uintptr_t>(stack_) + 1024 * 1024;

  // The sp is initialized to point to the bottom (high address) of the
  // allocated stack area.
  registers_[sp] = reinterpret_cast<int64_t>(stack_) + stackSize - 64;

  // Zero-initialize VR namespace. Simulated PPC64 does not guarantee any
  // value in VRs at entry, but zeroing avoids uninitialized-read false
  // positives in tools and makes regression traces deterministic.
  memset(VRregisters_, 0, sizeof(VRregisters_));

  return true;
}

Simulator::~Simulator() { js_free(stack_); }

SimulatorProcess::SimulatorProcess()
    : cacheLock_(mutexid::SimulatorCacheLock), redirection_(nullptr) {
  if (getenv("PPC64_SIM_ICACHE_CHECKS")) {
    ICacheCheckingDisableCount = 0;
  }
}

SimulatorProcess::~SimulatorProcess() {
  Redirection* r = redirection_;
  while (r) {
    Redirection* next = r->next_;
    js_delete(r);
    r = next;
  }
}

/* static */
void* Simulator::RedirectNativeFunction(void* nativeFunction,
                                        ABIFunctionType type) {
  Redirection* redirection = Redirection::Get(nativeFunction, type);
  return redirection->addressOfSwiInstruction();
}

Simulator* Simulator::Current() {
  JSContext* cx = TlsContext.get();
  MOZ_ASSERT(CurrentThreadCanAccessRuntime(cx->runtime()));
  return cx->simulator();
}

// -----------------------------------------------------------------------------
// Register accessors.

void Simulator::setRegister(int reg, int64_t value) {
  MOZ_ASSERT((reg >= 0) && (reg < Register::kNumSimuRegisters));
  if (reg == pc) {
    pc_modified_ = true;
  }
  registers_[reg] = value;
}

int64_t Simulator::getRegister(int reg) const {
  MOZ_ASSERT((reg >= 0) && (reg < Register::kNumSimuRegisters));
  return registers_[reg] + ((reg == pc) ? SimInstruction::kPCReadOffset : 0);
}

void Simulator::setFpuRegister(int fpureg, int64_t value) {
  MOZ_ASSERT((fpureg >= 0) &&
             (fpureg < Simulator::FPURegister::kNumFPURegisters));
  FPUregisters_[fpureg] = value;
}

void Simulator::setFpuRegisterWord(int fpureg, int32_t value) {
  MOZ_ASSERT((fpureg >= 0) &&
             (fpureg < Simulator::FPURegister::kNumFPURegisters));
  int32_t* pword;
  pword = reinterpret_cast<int32_t*>(&FPUregisters_[fpureg]);
  *pword = value;
}

// Promote f32 → f64 preserving NaN payload, like PPC64's `lfs` and
// `xscvspdpn`. The plain C cast `(double)f32_nan` is permitted by the
// standard to quiet a signaling NaN, which on x86/ARM hosts visibly
// transforms 0x7FA00000 (sNaN) into a qNaN such as 0x7FE00000 — breaking
// every wasm test that loads a constant sNaN bit pattern. Manually
// reconstruct the f64 NaN with the same sign + payload (payload shifted
// left by 29 to fill the wider mantissa).
static double promoteFloatPreservingNaN(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
    uint64_t sign = uint64_t(bits >> 31) & 1u;
    uint64_t payload = uint64_t(bits & 0x007FFFFFu);
    uint64_t dbits = (sign << 63) | (uint64_t(0x7FFu) << 52) | (payload << 29);
    double d;
    memcpy(&d, &dbits, sizeof(d));
    return d;
  }
  return (double)f;
}

// Demote f64 → f32 preserving NaN payload (non-signaling: matches PPC64
// `stfs` / `xscvdpspn`, and wasm `lfs`-equivalent stores). Truncates the
// lower 29 bits of the f64 payload (those bits cannot be represented in
// the narrower f32 mantissa); if the truncation would yield a payload of
// zero (which would degrade the NaN to an Infinity), force the LSB so
// the result is still a NaN. This intentionally does NOT set the quiet
// bit — that's the job of the explicit-quieting op `xscvdpsp` and
// f32.demote_f64's wasm-level lowering.
static float demoteDoublePreservingNaN(double d) {
  uint64_t bits;
  memcpy(&bits, &d, sizeof(bits));
  if ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL &&
      (bits & 0x000FFFFFFFFFFFFFULL) != 0) {
    uint32_t sign = uint32_t(bits >> 63) & 1u;
    uint32_t payload = uint32_t((bits >> 29) & 0x007FFFFFu);
    if (payload == 0) payload = 1;
    uint32_t fbits = (sign << 31) | 0x7F800000u | payload;
    float f;
    memcpy(&f, &fbits, sizeof(f));
    return f;
  }
  return (float)d;
}

void Simulator::setFpuRegisterFloat(int fpureg, float value) {
  MOZ_ASSERT((fpureg >= 0) &&
             (fpureg < Simulator::FPURegister::kNumFPURegisters));
  // ELFv2 ABI: single-precision values in FPRs are stored as their
  // double-precision representation. Promote and store the full 8 bytes,
  // not just the low 4. (Otherwise the upper 4 bytes are stale, matching
  // the layout that fctid/fcfid/lfd would read but NOT what the JIT and
  // the C ABI expect for a 'float' parameter.) Use the NaN-preserving
  // helper so a signaling-NaN return value isn't quieted into a qNaN.
  double promoted = promoteFloatPreservingNaN(value);
  memcpy(&FPUregisters_[fpureg], &promoted, sizeof(promoted));
}

void Simulator::setFpuRegisterDouble(int fpureg, double value) {
  MOZ_ASSERT((fpureg >= 0) &&
             (fpureg < Simulator::FPURegister::kNumFPURegisters));
  *mozilla::BitwiseCast<double*>(&FPUregisters_[fpureg]) = value;
}

int64_t Simulator::getFpuRegister(int fpureg) const {
  MOZ_ASSERT((fpureg >= 0) &&
             (fpureg < Simulator::FPURegister::kNumFPURegisters));
  return FPUregisters_[fpureg];
}

int32_t Simulator::getFpuRegisterWord(int fpureg) const {
  MOZ_ASSERT((fpureg >= 0) &&
             (fpureg < Simulator::FPURegister::kNumFPURegisters));
  return *mozilla::BitwiseCast<int32_t*>(&FPUregisters_[fpureg]);
}

int32_t Simulator::getFpuRegisterSignedWord(int fpureg) const {
  MOZ_ASSERT((fpureg >= 0) &&
             (fpureg < Simulator::FPURegister::kNumFPURegisters));
  return *mozilla::BitwiseCast<int32_t*>(&FPUregisters_[fpureg]);
}

float Simulator::getFpuRegisterFloat(int fpureg) const {
  MOZ_ASSERT((fpureg >= 0) &&
             (fpureg < Simulator::FPURegister::kNumFPURegisters));
  // ELFv2 ABI: single-precision values are passed/returned in FPRs as their
  // double-precision representation. Read the full 8 bytes as double, then
  // narrow to float — matching the `frsp` the C callee would do, and matching
  // what real PPC64 hardware sees when the FPR was loaded via `lfs`. Use the
  // NaN-preserving helper so a signaling-NaN parameter isn't quieted.
  double promoted;
  memcpy(&promoted, &FPUregisters_[fpureg], sizeof(promoted));
  return demoteDoublePreservingNaN(promoted);
}

double Simulator::getFpuRegisterDouble(int fpureg) const {
  MOZ_ASSERT((fpureg >= 0) &&
             (fpureg < Simulator::FPURegister::kNumFPURegisters));
  return *mozilla::BitwiseCast<double*>(&FPUregisters_[fpureg]);
}

void Simulator::setVRBytes(int vreg, const uint8_t bytes[16]) {
  MOZ_ASSERT((vreg >= 0) && (vreg < kNumVRRegisters));
  memcpy(VRregisters_[vreg], bytes, 16);
}

void Simulator::getVRBytes(int vreg, uint8_t bytes[16]) const {
  MOZ_ASSERT((vreg >= 0) && (vreg < kNumVRRegisters));
  memcpy(bytes, VRregisters_[vreg], 16);
}

void Simulator::getVSR128(int vsr, uint8_t bytes[16]) const {
  MOZ_ASSERT((vsr >= 0) && (vsr < kNumFPURegisters + kNumVRRegisters));
  if (vsr < kNumFPURegisters) {
    // VSR 0-31: FPR view. The FPR scalar lives in BE DW0 of the VSR,
    // which on PPC64LE register storage maps to LE bytes 8-15.
    // DW1 is undefined per ISA; we model it as zero.
    // `lfd f0,(mem); xxlor <vr>,f0,f0; stxvx <vr>,...` writes the
    // double's 8 bytes to the HIGH half of the 16-byte store (LE
    // bytes 8-15).
    int64_t val = FPUregisters_[vsr];
    memset(bytes, 0, 8);
    memcpy(bytes + 8, &val, 8);
  } else {
    memcpy(bytes, VRregisters_[vsr - kNumFPURegisters], 16);
  }
}

void Simulator::setVSR128(int vsr, const uint8_t bytes[16]) {
  MOZ_ASSERT((vsr >= 0) && (vsr < kNumFPURegisters + kNumVRRegisters));
  if (vsr < kNumFPURegisters) {
    // FPR scalar at BE DW0 = LE bytes 8-15. DW1 is architecturally
    // discarded on VSR-to-FPR writes.
    int64_t val;
    memcpy(&val, bytes + 8, 8);
    FPUregisters_[vsr] = val;
  } else {
    memcpy(VRregisters_[vsr - kNumFPURegisters], bytes, 16);
  }
}

void Simulator::setCallResultDouble(double result) {
  setFpuRegisterDouble(Simulator::f1, result);
}

void Simulator::setCallResultFloat(float result) {
  setFpuRegisterFloat(Simulator::f1, result);
}

void Simulator::setCallResult(int64_t res) { setRegister(r3, res); }

#ifdef XP_DARWIN
void Simulator::setCallResult(intptr_t res) {
  setRegister(r3, I64(res));
}
#endif

void Simulator::setCallResult(__int128 res) {
  setRegister(r3, I64(res));
  setRegister(r4, I64(res >> 64));
}

void Simulator::set_pc(int64_t value) {
  pc_modified_ = true;
  registers_[pc] = value;
}

bool Simulator::has_bad_pc() const {
  return ((registers_[pc] == bad_ra) || (registers_[pc] == end_sim_pc));
}

int64_t Simulator::get_pc() const { return registers_[pc]; }

JS::ProfilingFrameIterator::RegisterState Simulator::registerState() {
  wasm::RegisterState state;
  state.pc = (void*)get_pc();
  state.fp = (void*)getRegister(fp);
  state.sp = (void*)getRegister(sp);
  state.lr = (void*)getLR();
  return state;
}

// -----------------------------------------------------------------------------
// Memory access helpers.

uint8_t Simulator::readBU(uint64_t addr) {
  if (handleWasmSegFault(addr, 1)) {
    return 0xff;
  }
  uint8_t* ptr = reinterpret_cast<uint8_t*>(addr);
  return *ptr;
}

int8_t Simulator::readB(uint64_t addr) {
  if (handleWasmSegFault(addr, 1)) {
    return -1;
  }
  int8_t* ptr = reinterpret_cast<int8_t*>(addr);
  return *ptr;
}

void Simulator::writeB(uint64_t addr, uint8_t value) {
  if (handleWasmSegFault(addr, 1)) {
    return;
  }
  uint8_t* ptr = reinterpret_cast<uint8_t*>(addr);
  *ptr = value;
}

void Simulator::writeB(uint64_t addr, int8_t value) {
  if (handleWasmSegFault(addr, 1)) {
    return;
  }
  int8_t* ptr = reinterpret_cast<int8_t*>(addr);
  *ptr = value;
}

uint16_t Simulator::readHU(uint64_t addr, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 2)) {
    return 0xffff;
  }
  uint16_t* ptr = reinterpret_cast<uint16_t*>(addr);
  return *ptr;
}

int16_t Simulator::readH(uint64_t addr, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 2)) {
    return -1;
  }
  int16_t* ptr = reinterpret_cast<int16_t*>(addr);
  return *ptr;
}

void Simulator::writeH(uint64_t addr, uint16_t value, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 2)) {
    return;
  }
  uint16_t* ptr = reinterpret_cast<uint16_t*>(addr);
  LLBit_ = false;
  *ptr = value;
}

void Simulator::writeH(uint64_t addr, int16_t value, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 2)) {
    return;
  }
  int16_t* ptr = reinterpret_cast<int16_t*>(addr);
  LLBit_ = false;
  *ptr = value;
}

uint32_t Simulator::readWU(uint64_t addr, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 4)) {
    return -1;
  }
  uint32_t* ptr = reinterpret_cast<uint32_t*>(addr);
  return *ptr;
}

int32_t Simulator::readW(uint64_t addr, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 4)) {
    return -1;
  }
  int32_t* ptr = reinterpret_cast<int32_t*>(addr);
  return *ptr;
}

void Simulator::writeW(uint64_t addr, uint32_t value, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 4)) {
    return;
  }
  uint32_t* ptr = reinterpret_cast<uint32_t*>(addr);
  LLBit_ = false;
  *ptr = value;
}

void Simulator::writeW(uint64_t addr, int32_t value, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 4)) {
    return;
  }
  int32_t* ptr = reinterpret_cast<int32_t*>(addr);
  LLBit_ = false;
  *ptr = value;
}

int64_t Simulator::readDW(uint64_t addr, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 8)) {
    return -1;
  }
  int64_t* ptr = reinterpret_cast<int64_t*>(addr);
  return *ptr;
}

void Simulator::writeDW(uint64_t addr, int64_t value, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 8)) {
    return;
  }
  int64_t* ptr = reinterpret_cast<int64_t*>(addr);
  LLBit_ = false;
  *ptr = value;
}

double Simulator::readD(uint64_t addr, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 8)) {
    return NAN;
  }
  double* ptr = reinterpret_cast<double*>(addr);
  return *ptr;
}

void Simulator::writeD(uint64_t addr, double value, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 8)) {
    return;
  }
  double* ptr = reinterpret_cast<double*>(addr);
  LLBit_ = false;
  *ptr = value;
}

// Byte-wide load-reserve / store-conditional (lbarx / stbcx.).
// Byte accesses have no alignment requirement.
uint8_t Simulator::loadLinkedB(uint64_t addr, SimInstruction* instr) {
  if (handleWasmSegFault(addr, 1)) {
    return 0;
  }
  volatile uint8_t* ptr = reinterpret_cast<volatile uint8_t*>(addr);
  uint8_t value = *ptr;
  lastLLValue_ = value;
  LLAddr_ = addr;
  LLBit_ = true;
  return value;
}

int Simulator::storeConditionalB(uint64_t addr, uint8_t value,
                                 SimInstruction* instr) {
  if (addr != LLAddr_) {
    printf("stbcx. to bad address: 0x%016" PRIx64 ", pc=0x%016" PRIxPTR
           ", expected: 0x%016" PRIxPTR "\n",
           addr, reinterpret_cast<intptr_t>(instr), LLAddr_);
    MOZ_CRASH();
  }
  SharedMem<uint8_t*> ptr =
      SharedMem<uint8_t*>::shared(reinterpret_cast<uint8_t*>(addr));
  if (!LLBit_) {
    return 0;
  }
  LLBit_ = false;
  LLAddr_ = 0;
  uint8_t expected = uint8_t(lastLLValue_);
  uint8_t old =
      AtomicOperations::compareExchangeSeqCst(ptr, expected, value);
  return (old == expected) ? 1 : 0;
}

// Halfword-wide load-reserve / store-conditional (lharx / sthcx.).
// 2-byte aligned per ISA.
uint16_t Simulator::loadLinkedH(uint64_t addr, SimInstruction* instr) {
  if ((addr & 1) == 0) {
    if (handleWasmSegFault(addr, 2)) {
      return 0;
    }
    volatile uint16_t* ptr = reinterpret_cast<volatile uint16_t*>(addr);
    uint16_t value = *ptr;
    lastLLValue_ = value;
    LLAddr_ = addr;
    LLBit_ = true;
    return value;
  }
  printf("Unaligned lharx at 0x%016" PRIx64 ", pc=0x%016" PRIxPTR "\n", addr,
         reinterpret_cast<intptr_t>(instr));
  MOZ_CRASH();
  return 0;
}

int Simulator::storeConditionalH(uint64_t addr, uint16_t value,
                                 SimInstruction* instr) {
  if (addr != LLAddr_) {
    printf("sthcx. to bad address: 0x%016" PRIx64 ", pc=0x%016" PRIxPTR
           ", expected: 0x%016" PRIxPTR "\n",
           addr, reinterpret_cast<intptr_t>(instr), LLAddr_);
    MOZ_CRASH();
  }
  if ((addr & 1) == 0) {
    SharedMem<uint16_t*> ptr =
        SharedMem<uint16_t*>::shared(reinterpret_cast<uint16_t*>(addr));
    if (!LLBit_) {
      return 0;
    }
    LLBit_ = false;
    LLAddr_ = 0;
    uint16_t expected = uint16_t(lastLLValue_);
    uint16_t old =
        AtomicOperations::compareExchangeSeqCst(ptr, expected, value);
    return (old == expected) ? 1 : 0;
  }
  printf("Unaligned sthcx. at 0x%016" PRIx64 ", pc=0x%016" PRIxPTR "\n", addr,
         reinterpret_cast<intptr_t>(instr));
  MOZ_CRASH();
  return 0;
}

int32_t Simulator::loadLinkedW(uint64_t addr, SimInstruction* instr) {
  if ((addr & 3) == 0) {
    if (handleWasmSegFault(addr, 4)) {
      return -1;
    }

    volatile int32_t* ptr = reinterpret_cast<volatile int32_t*>(addr);
    int32_t value = *ptr;
    lastLLValue_ = value;
    LLAddr_ = addr;
    LLBit_ = true;
    return value;
  }
  printf("Unaligned lwarx at 0x%016" PRIx64 ", pc=0x%016" PRIxPTR "\n", addr,
         reinterpret_cast<intptr_t>(instr));
  MOZ_CRASH();
  return 0;
}

int Simulator::storeConditionalW(uint64_t addr, int32_t value,
                                 SimInstruction* instr) {
  if (addr != LLAddr_) {
    printf("stwcx. to bad address: 0x%016" PRIx64 ", pc=0x%016" PRIxPTR
           ", expected: 0x%016" PRIxPTR "\n",
           addr, reinterpret_cast<intptr_t>(instr), LLAddr_);
    MOZ_CRASH();
  }

  if ((addr & 3) == 0) {
    SharedMem<int32_t*> ptr =
        SharedMem<int32_t*>::shared(reinterpret_cast<int32_t*>(addr));

    if (!LLBit_) {
      return 0;
    }

    LLBit_ = false;
    LLAddr_ = 0;
    int32_t expected = int32_t(lastLLValue_);
    int32_t old =
        AtomicOperations::compareExchangeSeqCst(ptr, expected, value);
    return (old == expected) ? 1 : 0;
  }
  printf("Unaligned stwcx. at 0x%016" PRIx64 ", pc=0x%016" PRIxPTR "\n", addr,
         reinterpret_cast<intptr_t>(instr));
  MOZ_CRASH();
  return 0;
}

int64_t Simulator::loadLinkedD(uint64_t addr, SimInstruction* instr) {
  if ((addr & kPointerAlignmentMask) == 0) {
    if (handleWasmSegFault(addr, 8)) {
      return -1;
    }

    volatile int64_t* ptr = reinterpret_cast<volatile int64_t*>(addr);
    int64_t value = *ptr;
    lastLLValue_ = value;
    LLAddr_ = addr;
    LLBit_ = true;
    return value;
  }
  printf("Unaligned ldarx at 0x%016" PRIx64 ", pc=0x%016" PRIxPTR "\n", addr,
         reinterpret_cast<intptr_t>(instr));
  MOZ_CRASH();
  return 0;
}

int Simulator::storeConditionalD(uint64_t addr, int64_t value,
                                 SimInstruction* instr) {
  if (addr != LLAddr_) {
    printf("stdcx. to bad address: 0x%016" PRIx64 ", pc=0x%016" PRIxPTR
           ", expected: 0x%016" PRIxPTR "\n",
           addr, reinterpret_cast<intptr_t>(instr), LLAddr_);
    MOZ_CRASH();
  }

  if ((addr & kPointerAlignmentMask) == 0) {
    SharedMem<int64_t*> ptr =
        SharedMem<int64_t*>::shared(reinterpret_cast<int64_t*>(addr));

    if (!LLBit_) {
      return 0;
    }

    LLBit_ = false;
    LLAddr_ = 0;
    int64_t expected = lastLLValue_;
    int64_t old =
        AtomicOperations::compareExchangeSeqCst(ptr, expected, value);
    return (old == expected) ? 1 : 0;
  }
  printf("Unaligned stdcx. at 0x%016" PRIx64 ", pc=0x%016" PRIxPTR "\n", addr,
         reinterpret_cast<intptr_t>(instr));
  MOZ_CRASH();
  return 0;
}

// -----------------------------------------------------------------------------
// Stack limit / recursion helpers.

uintptr_t Simulator::stackLimit() const { return stackLimit_; }

uintptr_t* Simulator::addressOfStackLimit() { return &stackLimit_; }

bool Simulator::overRecursed(uintptr_t newsp) const {
  if (newsp == 0) {
    newsp = getRegister(sp);
  }
  return newsp <= stackLimit();
}

bool Simulator::overRecursedWithExtra(uint32_t extra) const {
  uintptr_t newsp = getRegister(sp) - extra;
  return newsp <= stackLimit();
}

void Simulator::format(SimInstruction* instr, const char* format) {
  printf("Simulator found unsupported instruction:\n 0x%016" PRIxPTR
         ": %08x %s\n",
         reinterpret_cast<intptr_t>(instr), instr->instructionBits(), format);
  MOZ_CRASH();
}

// -----------------------------------------------------------------------------
// softwareInterrupt - handle kCallRedirInstr (PPC_stop) and PPC_trap.

ABI_FUNCTION_TYPE_SIM_PROTOTYPES

void Simulator::softwareInterrupt(SimInstruction* instr) {
  uint32_t instrBits = instr->instructionBits();

  if (instrBits == kCallRedirInstr) {
    Redirection* redirection = Redirection::FromSwiInstruction(instr);
    uintptr_t nativeFn =
        reinterpret_cast<uintptr_t>(redirection->nativeFunction());

    // Get the SP for reading stack arguments.
    int64_t* sp_ = reinterpret_cast<int64_t*>(getRegister(sp));
    // Skip past the PPC64 ELFv2 link area (4 doublewords = 32 bytes).
    sp_ = reinterpret_cast<int64_t*>(reinterpret_cast<uintptr_t>(sp_) + 32);

    // PPC64 ELFv2: integer args in r3-r10, FP args in f1-f13.
    int64_t a0_ = getRegister(r3);
    int64_t a1_ = getRegister(r4);
    int64_t a2_ = getRegister(r5);
    int64_t a3_ = getRegister(r6);
    int64_t a4_ = getRegister(r7);
    int64_t a5_ = getRegister(r8);
    int64_t a6_ = getRegister(r9);
    int64_t a7_ = getRegister(r10);
    // PPC64 ELFv2: FP args in f1-f13, mapped to f0_s..f12_s and f0_d..f12_d.
    float f0_s = getFpuRegisterFloat(Simulator::f1);
    float f1_s = getFpuRegisterFloat(Simulator::f2);
    float f2_s = getFpuRegisterFloat(Simulator::f3);
    float f3_s = getFpuRegisterFloat(Simulator::f4);
    float f4_s = getFpuRegisterFloat(Simulator::f5);
    float f5_s = getFpuRegisterFloat(Simulator::f6);
    float f6_s = getFpuRegisterFloat(Simulator::f7);
    float f7_s = getFpuRegisterFloat(Simulator::f8);
    float f8_s = getFpuRegisterFloat(Simulator::f9);
    float f9_s = getFpuRegisterFloat(Simulator::f10);
    float f10_s = getFpuRegisterFloat(Simulator::f11);
    float f11_s = getFpuRegisterFloat(Simulator::f12);
    float f12_s = getFpuRegisterFloat(Simulator::f13);
    double f0_d = getFpuRegisterDouble(Simulator::f1);
    double f1_d = getFpuRegisterDouble(Simulator::f2);
    double f2_d = getFpuRegisterDouble(Simulator::f3);
    double f3_d = getFpuRegisterDouble(Simulator::f4);
    double f4_d = getFpuRegisterDouble(Simulator::f5);
    double f5_d = getFpuRegisterDouble(Simulator::f6);
    double f6_d = getFpuRegisterDouble(Simulator::f7);
    double f7_d = getFpuRegisterDouble(Simulator::f8);
    double f8_d = getFpuRegisterDouble(Simulator::f9);
    double f9_d = getFpuRegisterDouble(Simulator::f10);
    double f10_d = getFpuRegisterDouble(Simulator::f11);
    double f11_d = getFpuRegisterDouble(Simulator::f12);
    double f12_d = getFpuRegisterDouble(Simulator::f13);

    // Suppress unused-variable warnings for higher FP arg registers.
    // They exist for ABI completeness but few function types use >5 FP args.
    (void)f4_s; (void)f5_s; (void)f6_s; (void)f7_s; (void)f8_s; (void)f9_s;
    (void)f10_s; (void)f11_s; (void)f12_s;
    (void)f4_d; (void)f5_d; (void)f6_d; (void)f7_d; (void)f8_d; (void)f9_d;
    (void)f10_d; (void)f11_d; (void)f12_d;

    int64_t saved_lr = getLR();

    bool stack_aligned = (getRegister(sp) & (ABIStackAlignment - 1)) == 0;
    if (!stack_aligned) {
      fprintf(stderr, "Runtime call with unaligned stack!\n");
      MOZ_CRASH();
    }

    if (single_stepping_) {
      single_step_callback_(single_step_callback_arg_, this, nullptr);
    }

    switch (redirection->type()) {
      ABI_FUNCTION_TYPE_PPC64_SIM_DISPATCH

      default:
        MOZ_CRASH("Unknown function type.");
    }

    if (single_stepping_) {
      single_step_callback_(single_step_callback_arg_, this, nullptr);
    }

    setLR(saved_lr);
    set_pc(getLR());
  } else if (instrBits == 0x7FE00008) {
    // PPC_trap: used for wasm traps.
    uint8_t* newPC;
    if (wasm::HandleIllegalInstruction(registerState(), &newPC)) {
      set_pc(int64_t(newPC));
      return;
    }
    MOZ_CRASH("Unexpected trap instruction");
  } else {
    // Other trap-like instructions: enter debugger.
    ppc64Debugger dbg(this);
    dbg.debug();
  }
}

// -----------------------------------------------------------------------------
// Stop/breakpoint helpers.

bool Simulator::isWatchpoint(uint32_t code) {
  return (code <= kMaxWatchpointCode);
}

void Simulator::printWatchpoint(uint32_t code) {
  ppc64Debugger dbg(this);
  ++break_count_;
  printf("\n---- break %d marker: %20" PRIi64 "  (instr count: %20" PRIi64
         ") ----\n",
         code, break_count_, icount_);
  dbg.printAllRegs();
}

void Simulator::handleStop(uint32_t code, SimInstruction* instr) {
  if (isEnabledStop(code)) {
    ppc64Debugger dbg(this);
    dbg.stop(instr);
  } else {
    set_pc(get_pc() + SimInstruction::kInstrSize);
  }
}

bool Simulator::isStopInstruction(SimInstruction* instr) {
  return instr->instructionBits() == kCallRedirInstr;
}

bool Simulator::isEnabledStop(uint32_t code) {
  MOZ_ASSERT(code <= kMaxStopCode);
  MOZ_ASSERT(code > kMaxWatchpointCode);
  return !(watchedStops_[code].count_ & kStopDisabledBit);
}

void Simulator::enableStop(uint32_t code) {
  if (!isEnabledStop(code)) {
    watchedStops_[code].count_ &= ~kStopDisabledBit;
  }
}

void Simulator::disableStop(uint32_t code) {
  if (isEnabledStop(code)) {
    watchedStops_[code].count_ |= kStopDisabledBit;
  }
}

void Simulator::increaseStopCounter(uint32_t code) {
  MOZ_ASSERT(code <= kMaxStopCode);
  if ((watchedStops_[code].count_ & ~(1 << 31)) == 0x7fffffff) {
    printf(
        "Stop counter for code %i has overflowed.\n"
        "Enabling this code and reseting the counter to 0.\n",
        code);
    watchedStops_[code].count_ = 0;
    enableStop(code);
  } else {
    watchedStops_[code].count_++;
  }
}

void Simulator::printStopInfo(uint32_t code) {
  if (code <= kMaxWatchpointCode) {
    printf("That is a watchpoint, not a stop.\n");
    return;
  } else if (code > kMaxStopCode) {
    printf("Code too large, only %u stops can be used\n", kMaxStopCode + 1);
    return;
  }
  const char* state = isEnabledStop(code) ? "Enabled" : "Disabled";
  int32_t count = watchedStops_[code].count_ & ~kStopDisabledBit;
  if (count != 0) {
    if (watchedStops_[code].desc_) {
      printf("stop %i - 0x%x: \t%s, \tcounter = %i, \t%s\n", code, code,
             state, count, watchedStops_[code].desc_);
    } else {
      printf("stop %i - 0x%x: \t%s, \tcounter = %i\n", code, code, state,
             count);
    }
  }
}

// =============================================================================
// Instruction decoders.
// =============================================================================

// Compute effective address for D-form instructions.
// If RA==0, the base is 0 (not GPR[0]).
static inline int64_t DFormEA(Simulator* sim, SimInstruction* instr,
                              int16_t offset) {
  uint32_t ra = instr->raValue();
  int64_t base = (ra == 0) ? 0 : sim->getRegister(ra);
  return base + offset;
}

// Compute effective address for DS-form instructions.
static inline int64_t DSFormEA(Simulator* sim, SimInstruction* instr,
                               int16_t offset) {
  uint32_t ra = instr->raValue();
  int64_t base = (ra == 0) ? 0 : sim->getRegister(ra);
  return base + offset;
}

// Compute effective address for X-form indexed instructions.
// If RA==0, base is 0 (not GPR[0]).
static inline int64_t XFormEA(Simulator* sim, SimInstruction* instr) {
  uint32_t ra = instr->raValue();
  uint32_t rb = instr->rbValue();
  int64_t base = (ra == 0) ? 0 : sim->getRegister(ra);
  return base + sim->getRegister(rb);
}

// Compute effective address for X-form indexed updates (RA != 0 required).
static inline int64_t XFormEAUpdate(Simulator* sim, SimInstruction* instr) {
  uint32_t ra = instr->raValue();
  uint32_t rb = instr->rbValue();
  return sim->getRegister(ra) + sim->getRegister(rb);
}

// -----------------------------------------------------------------------------
// decodeDFormALU: addi, addis, ori, oris, xori, xoris, andi., andis.,
//                 cmpi, cmpli, subfic, addic, addic., mulli, twi

void Simulator::decodeDFormALU(SimInstruction* instr) {
  uint32_t opcode = instr->opcode();
  uint32_t rt = instr->rtValue();
  uint32_t ra = instr->raValue();
  int16_t si = instr->imm16Value();
  uint16_t ui = instr->uimm16Value();

  switch (opcode) {
    case 14: {
      // addi: RT = (RA|0) + SI
      int64_t base = (ra == 0) ? 0 : getRegister(ra);
      setRegister(rt, base + (int64_t)si);
      break;
    }
    case 15: {
      // addis: RT = (RA|0) + (SI << 16)
      int64_t base = (ra == 0) ? 0 : getRegister(ra);
      setRegister(rt, base + ((int64_t)si << 16));
      break;
    }
    case 24: {
      // ori: RA = RS | UI
      setRegister(ra, getRegister(rt) | (uint64_t)ui);
      break;
    }
    case 25: {
      // oris: RA = RS | (UI << 16)
      setRegister(ra, getRegister(rt) | ((uint64_t)ui << 16));
      break;
    }
    case 26: {
      // xori: RA = RS ^ UI
      setRegister(ra, getRegister(rt) ^ (uint64_t)ui);
      break;
    }
    case 27: {
      // xoris: RA = RS ^ (UI << 16)
      setRegister(ra, getRegister(rt) ^ ((uint64_t)ui << 16));
      break;
    }
    case 28: {
      // andi.: RA = RS & UI, update CR0
      int64_t result = getRegister(rt) & (uint64_t)ui;
      setRegister(ra, result);
      updateCR0(result);
      break;
    }
    case 29: {
      // andis.: RA = RS & (UI << 16), update CR0
      int64_t result = getRegister(rt) & ((uint64_t)ui << 16);
      setRegister(ra, result);
      updateCR0(result);
      break;
    }
    case 11: {
      // cmpi: compare RA with SI, signed
      uint32_t bf = instr->bfValue();
      bool l = instr->lBit();
      if (l) {
        // 64-bit compare
        setCRFieldCmp(bf, getRegister(ra), (int64_t)si);
      } else {
        // 32-bit compare
        int32_t ra32 = I32(getRegister(ra));
        setCRFieldCmp(bf, (int64_t)ra32, (int64_t)(int32_t)si);
      }
      break;
    }
    case 10: {
      // cmpli: compare RA with UI, unsigned
      uint32_t bf = instr->bfValue();
      bool l = instr->lBit();
      if (l) {
        // 64-bit unsigned compare
        setCRFieldCmpU(bf, U64(getRegister(ra)), (uint64_t)ui);
      } else {
        // 32-bit unsigned compare
        uint32_t ra32 = U32(getRegister(ra));
        setCRFieldCmpU(bf, (uint64_t)ra32, (uint64_t)ui);
      }
      break;
    }
    case 8: {
      // subfic: RT = SI - RA, set CA
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t imm = U64((int64_t)si);
      uint64_t result = imm + ~ra_val + 1;
      setRegister(rt, I64(result));
      // CA is set if there is a carry out of the addition (~RA + IMM + 1).
      // Equivalently, CA = (IMM >= RA) for unsigned interpretation of the
      // full 64-bit subtraction.
      bool carry = (imm >= ra_val) || (imm == 0 && ra_val == 0);
      // More precise: carry = (~ra_val + imm) would overflow, or adding 1
      // overflows.
      uint64_t tmp = ~ra_val + imm;
      carry = (tmp < ~ra_val) || (tmp < imm) || (result < tmp);
      // Simplify: CA if no borrow.
      carry = (U64((int64_t)si) >= ra_val);
      if (ra_val == 0) carry = true;
      // Actually, subfic CA: carry out of ~RA + IMM + 1.
      // CA = (IMM > RA - 1) when RA != 0, CA = 1 when RA == 0.
      // Or just: the unsigned result of (SI - RA) is valid (no borrow).
      // Let's compute it correctly:
      {
        __uint128_t wide = (__uint128_t)(~ra_val) + (__uint128_t)imm + 1;
        carry = (wide >> 64) != 0;
      }
      setXERCA(carry);
      break;
    }
    case 12: {
      // addic: RT = RA + SI, set CA
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t imm = U64((int64_t)si);
      uint64_t result = ra_val + imm;
      setRegister(rt, I64(result));
      setXERCA(result < ra_val);
      break;
    }
    case 13: {
      // addic.: RT = RA + SI, set CA, update CR0
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t imm = U64((int64_t)si);
      uint64_t result = ra_val + imm;
      setRegister(rt, I64(result));
      setXERCA(result < ra_val);
      updateCR0(I64(result));
      break;
    }
    case 7: {
      // mulli: RT = RA * SI (low 64 bits)
      int64_t result = getRegister(ra) * (int64_t)si;
      setRegister(rt, result);
      break;
    }
    case 3: {
      // twi: Trap Word Immediate. We don't implement trapping in the
      // simulator; just continue.
      break;
    }
    default:
      MOZ_CRASH_UNSAFE_PRINTF("decodeDFormALU: unhandled opcode %u", opcode);
  }
}

// -----------------------------------------------------------------------------
// decodeDFormLoad: lwz(32), lbz(34), lhz(40), lha(42), lfs(48), lfd(50)
//   and update variants

void Simulator::decodeDFormLoad(SimInstruction* instr) {
  uint32_t opcode = instr->opcode();
  uint32_t rt = instr->rtValue();
  int16_t si = instr->imm16Value();
  uint64_t ea = DFormEA(this, instr, si);

  switch (opcode) {
    case 32:
      // lwz
      setRegister(rt, U64(readWU(ea, instr)));
      break;
    case 33: {
      // lwzu: RA != 0, load and update RA
      setRegister(rt, U64(readWU(ea, instr)));
      setRegister(instr->raValue(), ea);
      break;
    }
    case 34:
      // lbz
      setRegister(rt, U64(readBU(ea)));
      break;
    case 35: {
      // lbzu
      setRegister(rt, U64(readBU(ea)));
      setRegister(instr->raValue(), ea);
      break;
    }
    case 40:
      // lhz
      setRegister(rt, U64(readHU(ea, instr)));
      break;
    case 41: {
      // lhzu
      setRegister(rt, U64(readHU(ea, instr)));
      setRegister(instr->raValue(), ea);
      break;
    }
    case 42:
      // lha (half-word, sign-extended)
      setRegister(rt, (int64_t)readH(ea, instr));
      break;
    case 43: {
      // lhau
      setRegister(rt, (int64_t)readH(ea, instr));
      setRegister(instr->raValue(), ea);
      break;
    }
    case 48: {
      // lfs: load float single, widen to double in FPR (NaN-preserving;
      // matches Power ISA `lfs` which uses xscvspdpn semantics)
      if (handleWasmSegFault(ea, 4)) break;
      float val = *reinterpret_cast<float*>(ea);
      setFpuRegisterDouble(rt, promoteFloatPreservingNaN(val));
      break;
    }
    case 49: {
      // lfsu
      if (handleWasmSegFault(ea, 4)) break;
      float val = *reinterpret_cast<float*>(ea);
      setFpuRegisterDouble(rt, promoteFloatPreservingNaN(val));
      setRegister(instr->raValue(), ea);
      break;
    }
    case 50: {
      // lfd: load float double
      double val = readD(ea, instr);
      setFpuRegisterDouble(rt, val);
      break;
    }
    case 51: {
      // lfdu
      double val = readD(ea, instr);
      setFpuRegisterDouble(rt, val);
      setRegister(instr->raValue(), ea);
      break;
    }
    default:
      MOZ_CRASH_UNSAFE_PRINTF("decodeDFormLoad: unhandled opcode %u", opcode);
  }
}

// -----------------------------------------------------------------------------
// decodeDFormStore: stw(36), stwu(37), stb(38), sth(44), stfs(52), stfd(54)
//   and update variants

void Simulator::decodeDFormStore(SimInstruction* instr) {
  uint32_t opcode = instr->opcode();
  uint32_t rs = instr->rsValue();
  int16_t si = instr->imm16Value();

  // For stores, the effective address calculation differs for update forms:
  // - Non-update: EA = (RA|0) + D
  // - Update: EA = RA + D (RA must not be 0)
  bool isUpdate = false;
  switch (opcode) {
    case 37: case 39: case 45: case 53: case 55:
      isUpdate = true;
      break;
  }

  uint64_t ea;
  if (isUpdate) {
    ea = getRegister(instr->raValue()) + (int64_t)si;
  } else {
    ea = DFormEA(this, instr, si);
  }

  switch (opcode) {
    case 36:
      // stw
      writeW(ea, I32(getRegister(rs)), instr);
      break;
    case 38:
      // stb
      writeB(ea, (uint8_t)(getRegister(rs) & 0xFF));
      break;
    case 39:
      // stbu
      writeB(ea, (uint8_t)(getRegister(rs) & 0xFF));
      setRegister(instr->raValue(), ea);
      break;
    case 44:
      // sth
      writeH(ea, U16(getRegister(rs)), instr);
      break;
    case 45:
      // sthu
      writeH(ea, U16(getRegister(rs)), instr);
      setRegister(instr->raValue(), ea);
      break;
    case 52: {
      // stfs: convert double in FPR to single and store (NaN-preserving;
      // matches Power ISA `stfs` which uses xscvdpspn semantics)
      double dval = getFpuRegisterDouble(rs);
      float fval = demoteDoublePreservingNaN(dval);
      if (handleWasmSegFault(ea, 4)) break;
      *reinterpret_cast<float*>(ea) = fval;
      LLBit_ = false;
      break;
    }
    case 53: {
      // stfsu
      double dval = getFpuRegisterDouble(rs);
      float fval = demoteDoublePreservingNaN(dval);
      if (handleWasmSegFault(ea, 4)) break;
      *reinterpret_cast<float*>(ea) = fval;
      LLBit_ = false;
      setRegister(instr->raValue(), ea);
      break;
    }
    case 54:
      // stfd
      writeD(ea, getFpuRegisterDouble(rs), instr);
      break;
    case 55:
      // stfdu
      writeD(ea, getFpuRegisterDouble(rs), instr);
      setRegister(instr->raValue(), ea);
      break;
    default:
      MOZ_CRASH_UNSAFE_PRINTF("decodeDFormStore: unhandled opcode %u", opcode);
  }
}

// -----------------------------------------------------------------------------
// decodeDSForm: ld(58/0), lwa(58/2), std(62/0), stdu(62/1)

void Simulator::decodeDSForm(SimInstruction* instr) {
  uint32_t opcode = instr->opcode();
  uint32_t rt = instr->rtValue();
  int16_t ds = instr->ds14Value();
  uint32_t xo = instr->bits(1, 0);

  if (opcode == 58) {
    uint64_t ea = DSFormEA(this, instr, ds);
    switch (xo) {
      case 0:
        // ld
        setRegister(rt, readDW(ea, instr));
        break;
      case 1: {
        // ldu
        setRegister(rt, readDW(ea, instr));
        setRegister(instr->raValue(), ea);
        break;
      }
      case 2:
        // lwa (load word algebraic, sign-extended to 64)
        setRegister(rt, (int64_t)readW(ea, instr));
        break;
      default:
        MOZ_CRASH_UNSAFE_PRINTF("decodeDSForm: opcode 58, xo=%u", xo);
    }
  } else if (opcode == 62) {
    // For std/stdu, EA uses RA directly (no RA|0 rule).
    uint64_t ea;
    if (xo == 1) {
      // stdu: update form
      ea = getRegister(instr->raValue()) + (int64_t)ds;
    } else {
      ea = DSFormEA(this, instr, ds);
    }
    switch (xo) {
      case 0:
        // std
        writeDW(ea, getRegister(rt), instr);
        break;
      case 1:
        // stdu
        writeDW(ea, getRegister(rt), instr);
        setRegister(instr->raValue(), ea);
        break;
      default:
        MOZ_CRASH_UNSAFE_PRINTF("decodeDSForm: opcode 62, xo=%u", xo);
    }
  } else {
    MOZ_CRASH_UNSAFE_PRINTF("decodeDSForm: unhandled opcode %u", opcode);
  }
}

// -----------------------------------------------------------------------------
// decodeXForm: Major opcode 31 (X-form, XO-form, etc.)
// This is the largest decoder covering most ALU, indexed load/store, SPR,
// and atomic instructions.

void Simulator::decodeXForm(SimInstruction* instr) {
  uint32_t xo = instr->xoValue();
  uint32_t rt = instr->rtValue();
  uint32_t ra = instr->raValue();
  uint32_t rb = instr->rbValue();
  bool rc = instr->rcBit();

  // Many instructions share major opcode 31. Switch on extended opcode.
  // For XO-form with OE=1, the xoValue() includes bit 10, so
  // addo (266 | 512 = 778) etc. are separate cases.

  // First check for isel which uses bits 1-5 = 15 (XO = 15 in bits 1..5).
  if ((xo & 0x1F) == 15) {
    // isel: if CR[BC] then RT=RA else RT=RB
    // BC is in bits 6..10 (the rc field position).
    uint32_t bc = instr->rcValue();
    uint32_t crField = bc / 4;
    uint32_t crBit = bc % 4;
    uint8_t crFieldVal = getCRField(crField);
    // PPC CR field bits: bit3=LT(8), bit2=GT(4), bit1=EQ(2), bit0=SO(1)
    // Bit numbering within field: 0=LT, 1=GT, 2=EQ, 3=SO
    bool bitSet;
    switch (crBit) {
      case 0: bitSet = (crFieldVal & kCRFieldLT) != 0; break;
      case 1: bitSet = (crFieldVal & kCRFieldGT) != 0; break;
      case 2: bitSet = (crFieldVal & kCRFieldEQ) != 0; break;
      case 3: bitSet = (crFieldVal & kCRFieldSO) != 0; break;
      default: bitSet = false; break;
    }
    int64_t raVal = (ra == 0) ? 0 : getRegister(ra);
    int64_t rbVal = getRegister(rb);
    setRegister(rt, bitSet ? raVal : rbVal);
    return;
  }

  switch (xo) {
    // --- Arithmetic ---
    case 266: {
      // add
      int64_t result = getRegister(ra) + getRegister(rb);
      setRegister(rt, result);
      if (rc) updateCR0(result);
      break;
    }
    case 778: {
      // addo
      int64_t ra_val = getRegister(ra);
      int64_t rb_val = getRegister(rb);
      int64_t result = ra_val + rb_val;
      setRegister(rt, result);
      // Overflow if signs of inputs are same but result sign differs.
      bool ov = ((ra_val ^ result) & (rb_val ^ result)) < 0;
      setXEROV(ov);
      if (rc) updateCR0(result);
      break;
    }
    case 10: {
      // addc
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t rb_val = U64(getRegister(rb));
      uint64_t result = ra_val + rb_val;
      setRegister(rt, I64(result));
      setXERCA(result < ra_val);
      if (rc) updateCR0(I64(result));
      break;
    }
    case 138: {
      // adde: RT = RA + RB + CA
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t rb_val = U64(getRegister(rb));
      uint64_t ca = getXERCA() ? 1ULL : 0ULL;
      uint64_t result = ra_val + rb_val + ca;
      setRegister(rt, I64(result));
      // Carry-out: when ca==0, only the ra+rb wrap matters; when ca==1,
      // an additional wrap occurs iff result <= ra_val.
      bool newCA = ca ? (result <= ra_val) : (result < ra_val);
      setXERCA(newCA);
      if (rc) updateCR0(I64(result));
      break;
    }
    case 234: {
      // addme: RT = RA + CA - 1
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t ca = getXERCA() ? 1ULL : 0ULL;
      uint64_t result = ra_val + ca + ~0ULL;  // + CA + (-1)
      setRegister(rt, I64(result));
      // CA if carry out of (RA + CA + 0xFFFFFFFFFFFFFFFF)
      bool newCA = (ra_val != 0) || (ca != 0);
      setXERCA(newCA);
      if (rc) updateCR0(I64(result));
      break;
    }
    case 202: {
      // addze: RT = RA + CA
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t ca = getXERCA() ? 1ULL : 0ULL;
      uint64_t result = ra_val + ca;
      setRegister(rt, I64(result));
      setXERCA(result < ra_val);
      if (rc) updateCR0(I64(result));
      break;
    }
    case 40: {
      // subf: RT = RB - RA
      int64_t result = getRegister(rb) - getRegister(ra);
      setRegister(rt, result);
      if (rc) updateCR0(result);
      break;
    }
    case 552: {
      // subfo: RT = RB - RA, set OV
      int64_t ra_val = getRegister(ra);
      int64_t rb_val = getRegister(rb);
      int64_t result = rb_val - ra_val;
      setRegister(rt, result);
      bool ov = ((rb_val ^ ra_val) & (rb_val ^ result)) < 0;
      setXEROV(ov);
      if (rc) updateCR0(result);
      break;
    }
    case 8: {
      // subfc: RT = ~RA + RB + 1
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t rb_val = U64(getRegister(rb));
      uint64_t result = ~ra_val + rb_val + 1;
      setRegister(rt, I64(result));
      // CA = no borrow = (RB >= RA unsigned)
      setXERCA(rb_val >= ra_val);
      if (rc) updateCR0(I64(result));
      break;
    }
    case 136: {
      // subfe: RT = ~RA + RB + CA
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t rb_val = U64(getRegister(rb));
      uint64_t ca = getXERCA() ? 1ULL : 0ULL;
      uint64_t result = ~ra_val + rb_val + ca;
      setRegister(rt, I64(result));
      __uint128_t wide = (__uint128_t)(~ra_val) + (__uint128_t)rb_val + ca;
      setXERCA((wide >> 64) != 0);
      if (rc) updateCR0(I64(result));
      break;
    }
    case 232: {
      // subfze: RT = ~RA + CA
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t ca = getXERCA() ? 1ULL : 0ULL;
      uint64_t result = ~ra_val + ca;
      setRegister(rt, I64(result));
      setXERCA(ca > ra_val);  // CA if ~RA + CA overflows
      if (rc) updateCR0(I64(result));
      break;
    }
    case 104: {
      // neg: RT = -RA
      int64_t result = -getRegister(ra);
      setRegister(rt, result);
      if (rc) updateCR0(result);
      break;
    }

    // --- Multiply ---
    case 233: {
      // mulld: RT = RA * RB (low 64 bits)
      int64_t result = getRegister(ra) * getRegister(rb);
      setRegister(rt, result);
      if (rc) updateCR0(result);
      break;
    }
    case 745: {
      // mulldo: RT = RA * RB, set OV
      int64_t ra_val = getRegister(ra);
      int64_t rb_val = getRegister(rb);
      int64_t result = ra_val * rb_val;
      setRegister(rt, result);
      // OV if high part of full 128-bit product is not all-sign.
      int64_t hi = MultiplyHighSigned(ra_val, rb_val);
      bool ov = (hi != (result >> 63));
      setXEROV(ov);
      if (rc) updateCR0(result);
      break;
    }
    case 235: {
      // mullw: RT = sign_ext(RA[32:63] * RB[32:63])
      int64_t result = (int64_t)I32(getRegister(ra)) *
                       (int64_t)I32(getRegister(rb));
      setRegister(rt, result);
      if (rc) updateCR0(result);
      break;
    }
    case 747: {
      // mullwo
      int64_t ra_val = I32(getRegister(ra));
      int64_t rb_val = I32(getRegister(rb));
      int64_t result = ra_val * rb_val;
      setRegister(rt, result);
      bool ov = (result != (int64_t)I32(result));
      setXEROV(ov);
      if (rc) updateCR0(result);
      break;
    }
    case 73: {
      // mulhd: RT = high 64 bits of RA * RB (signed)
      setRegister(rt, MultiplyHighSigned(getRegister(ra), getRegister(rb)));
      if (rc) updateCR0(getRegister(rt));
      break;
    }
    case 9: {
      // mulhdu: RT = high 64 bits of RA * RB (unsigned)
      setRegister(rt, I64(MultiplyHighUnsigned(U64(getRegister(ra)),
                                               U64(getRegister(rb)))));
      if (rc) updateCR0(getRegister(rt));
      break;
    }
    case 75: {
      // mulhw: RT = high 32 bits of (RA[32:63] * RB[32:63]), signed
      int64_t result =
          (int64_t)I32(getRegister(ra)) * (int64_t)I32(getRegister(rb));
      setRegister(rt, result >> 32);
      if (rc) updateCR0(getRegister(rt));
      break;
    }
    case 11: {
      // mulhwu: RT = high 32 bits, unsigned
      uint64_t result =
          (uint64_t)U32(getRegister(ra)) * (uint64_t)U32(getRegister(rb));
      setRegister(rt, I64(result >> 32));
      if (rc) updateCR0(getRegister(rt));
      break;
    }

    // --- Divide ---
    case 489: {
      // divd: RT = RA / RB (signed, 64-bit)
      int64_t ra_val = getRegister(ra);
      int64_t rb_val = getRegister(rb);
      if (rb_val == 0 || (ra_val == INT64_MIN && rb_val == -1)) {
        setRegister(rt, 0);
      } else {
        setRegister(rt, ra_val / rb_val);
      }
      if (rc) updateCR0(getRegister(rt));
      break;
    }
    case 1001: {
      // divdo
      int64_t ra_val = getRegister(ra);
      int64_t rb_val = getRegister(rb);
      bool ov = (rb_val == 0) || (ra_val == INT64_MIN && rb_val == -1);
      if (ov) {
        setRegister(rt, 0);
      } else {
        setRegister(rt, ra_val / rb_val);
      }
      setXEROV(ov);
      if (rc) updateCR0(getRegister(rt));
      break;
    }
    case 457: {
      // divdu: unsigned 64-bit divide
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t rb_val = U64(getRegister(rb));
      if (rb_val == 0) {
        setRegister(rt, 0);
      } else {
        setRegister(rt, I64(ra_val / rb_val));
      }
      if (rc) updateCR0(getRegister(rt));
      break;
    }
    case 969: {
      // divduo
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t rb_val = U64(getRegister(rb));
      bool ov = (rb_val == 0);
      if (ov) {
        setRegister(rt, 0);
      } else {
        setRegister(rt, I64(ra_val / rb_val));
      }
      setXEROV(ov);
      if (rc) updateCR0(getRegister(rt));
      break;
    }
    case 491: {
      // divw: signed 32-bit divide
      int32_t ra_val = I32(getRegister(ra));
      int32_t rb_val = I32(getRegister(rb));
      if (rb_val == 0 || (ra_val == INT32_MIN && rb_val == -1)) {
        setRegister(rt, 0);
      } else {
        setRegister(rt, (int64_t)(ra_val / rb_val));
      }
      if (rc) updateCR0(getRegister(rt));
      break;
    }
    case 1003: {
      // divwo
      int32_t ra_val = I32(getRegister(ra));
      int32_t rb_val = I32(getRegister(rb));
      bool ov = (rb_val == 0) || (ra_val == INT32_MIN && rb_val == -1);
      if (ov) {
        setRegister(rt, 0);
      } else {
        setRegister(rt, (int64_t)(ra_val / rb_val));
      }
      setXEROV(ov);
      if (rc) updateCR0(getRegister(rt));
      break;
    }
    case 459: {
      // divwu: unsigned 32-bit divide
      uint32_t ra_val = U32(getRegister(ra));
      uint32_t rb_val = U32(getRegister(rb));
      if (rb_val == 0) {
        setRegister(rt, 0);
      } else {
        setRegister(rt, (int64_t)(ra_val / rb_val));
      }
      if (rc) updateCR0(getRegister(rt));
      break;
    }
    case 971: {
      // divwuo
      uint32_t ra_val = U32(getRegister(ra));
      uint32_t rb_val = U32(getRegister(rb));
      bool ov = (rb_val == 0);
      if (ov) {
        setRegister(rt, 0);
      } else {
        setRegister(rt, (int64_t)(ra_val / rb_val));
      }
      setXEROV(ov);
      if (rc) updateCR0(getRegister(rt));
      break;
    }

    // --- POWER9 modulo (ISA 3.0) ---
    // Result of "undefined" division (rb_val == 0, or signed INT_MIN / -1)
    // is implementation-defined per Power ISA; matching the divX behaviour
    // above, we yield 0 in those cases. Rc has no encoding for these ops.
    case 779: {
      // modsw: RT = RA % RB (signed, 32-bit)
      int32_t ra_val = I32(getRegister(ra));
      int32_t rb_val = I32(getRegister(rb));
      if (rb_val == 0 || (ra_val == INT32_MIN && rb_val == -1)) {
        setRegister(rt, 0);
      } else {
        setRegister(rt, (int64_t)(ra_val % rb_val));
      }
      break;
    }
    case 267: {
      // moduw: RT = RA % RB (unsigned, 32-bit)
      uint32_t ra_val = U32(getRegister(ra));
      uint32_t rb_val = U32(getRegister(rb));
      if (rb_val == 0) {
        setRegister(rt, 0);
      } else {
        setRegister(rt, (int64_t)(ra_val % rb_val));
      }
      break;
    }
    case 777: {
      // modsd: RT = RA % RB (signed, 64-bit)
      int64_t ra_val = getRegister(ra);
      int64_t rb_val = getRegister(rb);
      if (rb_val == 0 || (ra_val == INT64_MIN && rb_val == -1)) {
        setRegister(rt, 0);
      } else {
        setRegister(rt, ra_val % rb_val);
      }
      break;
    }
    case 265: {
      // modud: RT = RA % RB (unsigned, 64-bit)
      uint64_t ra_val = U64(getRegister(ra));
      uint64_t rb_val = U64(getRegister(rb));
      if (rb_val == 0) {
        setRegister(rt, 0);
      } else {
        setRegister(rt, I64(ra_val % rb_val));
      }
      break;
    }

    // --- Logical ---
    case 28: {
      // and: RA = RS & RB
      int64_t result = getRegister(rt) & getRegister(rb);
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }
    case 60: {
      // andc: RA = RS & ~RB
      int64_t result = getRegister(rt) & ~getRegister(rb);
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }
    case 444: {
      // or: RA = RS | RB
      int64_t result = getRegister(rt) | getRegister(rb);
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }
    case 412: {
      // orc: RA = RS | ~RB
      int64_t result = getRegister(rt) | ~getRegister(rb);
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }
    case 316: {
      // xor: RA = RS ^ RB
      int64_t result = getRegister(rt) ^ getRegister(rb);
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }
    case 476: {
      // nand: RA = ~(RS & RB)
      int64_t result = ~(getRegister(rt) & getRegister(rb));
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }
    case 124: {
      // nor: RA = ~(RS | RB)
      int64_t result = ~(getRegister(rt) | getRegister(rb));
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }
    case 284: {
      // eqv: RA = ~(RS ^ RB)
      int64_t result = ~(getRegister(rt) ^ getRegister(rb));
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }

    // --- Shifts ---
    case 27: {
      // sld: RA = RS << RB[58:63] if RB[57]==0, else RA=0
      uint64_t shift = U64(getRegister(rb));
      uint64_t rs_val = U64(getRegister(rt));
      int64_t result;
      if (shift & 0x40) {
        result = 0;
      } else {
        result = I64(rs_val << (shift & 0x3F));
      }
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }
    case 24: {
      // slw: RA = RS[32:63] << RB[59:63] if RB[58]==0, else RA=0 (32-bit)
      uint32_t shift = U32(getRegister(rb));
      uint32_t rs_val = U32(getRegister(rt));
      uint32_t result;
      if (shift & 0x20) {
        result = 0;
      } else {
        result = rs_val << (shift & 0x1F);
      }
      setRegister(ra, (int64_t)(int32_t)result);
      if (rc) updateCR0(getRegister(ra));
      break;
    }
    case 539: {
      // srd: RA = RS >> RB[58:63] if RB[57]==0, else RA=0 (logical)
      uint64_t shift = U64(getRegister(rb));
      uint64_t rs_val = U64(getRegister(rt));
      int64_t result;
      if (shift & 0x40) {
        result = 0;
      } else {
        result = I64(rs_val >> (shift & 0x3F));
      }
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }
    case 536: {
      // srw: RA = RS[32:63] >> RB[59:63] logical (32-bit)
      uint32_t shift = U32(getRegister(rb));
      uint32_t rs_val = U32(getRegister(rt));
      uint32_t result;
      if (shift & 0x20) {
        result = 0;
      } else {
        result = rs_val >> (shift & 0x1F);
      }
      setRegister(ra, (int64_t)(int32_t)result);
      if (rc) updateCR0(getRegister(ra));
      break;
    }
    case 794: {
      // srad: RA = RS >> RB[58:63] arithmetic (64-bit), set CA
      uint64_t shift = U64(getRegister(rb));
      int64_t rs_val = getRegister(rt);
      int64_t result;
      bool carry;
      if (shift & 0x40) {
        result = rs_val >> 63;  // all sign bits
        carry = (rs_val < 0);
      } else {
        uint32_t sh = shift & 0x3F;
        result = rs_val >> sh;
        // CA = 1 if RS is negative and any 1-bits were shifted out.
        carry = (rs_val < 0) && ((rs_val & ((1ULL << sh) - 1)) != 0);
      }
      setRegister(ra, result);
      setXERCA(carry);
      if (rc) updateCR0(result);
      break;
    }
    case 792: {
      // sraw: RA = RS[32:63] >> RB[59:63] arithmetic (32-bit), set CA
      uint32_t shift = U32(getRegister(rb));
      int32_t rs_val = I32(getRegister(rt));
      int32_t result;
      bool carry;
      if (shift & 0x20) {
        result = rs_val >> 31;
        carry = (rs_val < 0);
      } else {
        uint32_t sh = shift & 0x1F;
        result = rs_val >> sh;
        carry = (rs_val < 0) && ((rs_val & ((1U << sh) - 1)) != 0);
      }
      setRegister(ra, (int64_t)result);
      setXERCA(carry);
      if (rc) updateCR0(getRegister(ra));
      break;
    }
    case 826:
    case 827: {
      // sradi RA, RS, SH: RA = EXTS(RS) >> sh arithmetic (64-bit), set CA.
      // XS-form, XO=413 (9-bit, bits 21-29), sh[5] at bit 30, Rc at bit 31.
      // Our xoValue() extracts bits 10:1 (10 bits)
      // which yields 413*2 + sh[5] = 826 (sh[5]=0) or 827 (sh[5]=1).
      // sh[0:4] at instruction bits 15:11 (= raValue field position, but
      // for this XS-form they're the SH[0:4] subfield).
      uint32_t sh = instr->bits(15, 11) | (instr->bit(1) << 5);
      int64_t rs_val = getRegister(rt);
      int64_t result = (sh == 0) ? rs_val : (rs_val >> sh);
      // CA := rs_val < 0 && any bits shifted out are 1.
      bool carry = (rs_val < 0) && sh > 0 &&
                   ((U64(rs_val) & ((1ULL << sh) - 1)) != 0);
      setRegister(ra, result);
      setXERCA(carry);
      if (rc) updateCR0(result);
      break;
    }
    case 824: {
      // srawi: RA = RS[32:63] >> SH arithmetic (32-bit), set CA
      uint32_t sh = instr->bits(15, 11);
      int32_t rs_val = I32(getRegister(rt));
      int32_t result = rs_val >> sh;
      bool carry = (rs_val < 0) && sh > 0 &&
                   ((U32(rs_val) & ((1U << sh) - 1)) != 0);
      setRegister(ra, (int64_t)result);
      setXERCA(carry);
      if (rc) updateCR0(getRegister(ra));
      break;
    }

    // --- Extend / count ---
    case 954: {
      // extsb: RA = sign_ext(RS[56:63])
      int64_t result = (int64_t)(int8_t)(getRegister(rt) & 0xFF);
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }
    case 922: {
      // extsh: RA = sign_ext(RS[48:63])
      int64_t result = (int64_t)(int16_t)(getRegister(rt) & 0xFFFF);
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }
    case 986: {
      // extsw: RA = sign_ext(RS[32:63])
      int64_t result = (int64_t)(int32_t)(getRegister(rt) & 0xFFFFFFFF);
      setRegister(ra, result);
      if (rc) updateCR0(result);
      break;
    }
    case 58: {
      // cntlzd: RA = count leading zeros of RS (64-bit)
      setRegister(ra, CountLeadingZeros64(U64(getRegister(rt))));
      if (rc) updateCR0(getRegister(ra));
      break;
    }
    case 26: {
      // cntlzw: RA = count leading zeros of RS[32:63] (32-bit)
      setRegister(ra, CountLeadingZeros32(U32(getRegister(rt))));
      if (rc) updateCR0(getRegister(ra));
      break;
    }
    case 570: {
      // cnttzd
      setRegister(ra, CountTrailingZeros64(U64(getRegister(rt))));
      if (rc) updateCR0(getRegister(ra));
      break;
    }
    case 538: {
      // cnttzw
      setRegister(ra, CountTrailingZeros32(U32(getRegister(rt))));
      if (rc) updateCR0(getRegister(ra));
      break;
    }
    case 506: {
      // popcntd
      setRegister(ra, PopCount64(U64(getRegister(rt))));
      break;
    }
    case 378: {
      // popcntw: popcount each 32-bit half independently, sum in each half
      uint64_t val = U64(getRegister(rt));
      uint32_t lo = PopCount32(U32(val));
      uint32_t hi = PopCount32(U32(val >> 32));
      setRegister(ra, I64(((uint64_t)hi << 32) | lo));
      break;
    }
    case 122: {
      // popcntb: popcount each byte independently
      setRegister(ra, I64(PopCountPerByte(U64(getRegister(rt)))));
      break;
    }
    case 187: {
      // brd (POWER10): RA = byte-reverse(RS) full 64-bit doubleword.
      setRegister(ra, I64(__builtin_bswap64(U64(getRegister(rt)))));
      break;
    }
    case 219: {
      // brh (POWER10): byte-reverse each of the 4 halfwords in RS.
      uint64_t v = U64(getRegister(rt));
      uint64_t out = ((v & 0xFF00FF00FF00FF00ULL) >> 8) |
                     ((v & 0x00FF00FF00FF00FFULL) << 8);
      setRegister(ra, I64(out));
      break;
    }
    case 155: {
      // brw (POWER10): byte-reverse each of the 2 words in RS.
      uint64_t v = U64(getRegister(rt));
      uint64_t out = ((uint64_t)__builtin_bswap32((uint32_t)(v >> 32)) << 32) |
                     (uint64_t)__builtin_bswap32((uint32_t)v);
      setRegister(ra, I64(out));
      break;
    }

    // --- Compare (X-form) ---
    case 0: {
      // cmp (cmpw/cmpd): signed compare
      uint32_t bf = instr->bfValue();
      bool l = instr->lBit();
      if (l) {
        setCRFieldCmp(bf, getRegister(ra), getRegister(rb));
      } else {
        setCRFieldCmp(bf, (int64_t)I32(getRegister(ra)),
                      (int64_t)I32(getRegister(rb)));
      }
      break;
    }
    case 32: {
      // cmpl (cmplw/cmpld): unsigned compare
      uint32_t bf = instr->bfValue();
      bool l = instr->lBit();
      if (l) {
        setCRFieldCmpU(bf, U64(getRegister(ra)), U64(getRegister(rb)));
      } else {
        setCRFieldCmpU(bf, (uint64_t)U32(getRegister(ra)),
                       (uint64_t)U32(getRegister(rb)));
      }
      break;
    }

    // --- Trap ---
    case 4: {
      // tw: Trap Word. The JIT uses this for debugging / tagging.
      // In the simulator we just treat it as a NOP (the JIT uses tagged
      // trap words that are never actually reached during normal execution,
      // they serve as metadata for the patcher).
      break;
    }

    // --- SPR ---
    case 339: {
      // mfspr: RT = SPR
      // SPR encoding: spr[4:0] at bits 16..20, spr[9:5] at bits 11..15
      uint32_t spr_lo = instr->raValue();  // bits 16..20
      uint32_t spr_hi = instr->rbValue();  // bits 11..15
      uint32_t spr = (spr_lo) | (spr_hi << 5);
      switch (spr) {
        case 8:  // LR
          setRegister(rt, getLR());
          break;
        case 9:  // CTR
          setRegister(rt, getCTR());
          break;
        case 1:  // XER
          setRegister(rt, I64(getXER()));
          break;
        default:
          MOZ_CRASH_UNSAFE_PRINTF("mfspr: unhandled SPR %u", spr);
      }
      break;
    }
    case 467: {
      // mtspr: SPR = RS
      uint32_t spr_lo = instr->raValue();
      uint32_t spr_hi = instr->rbValue();
      uint32_t spr = (spr_lo) | (spr_hi << 5);
      int64_t val = getRegister(rt);
      switch (spr) {
        case 8:  // LR
          setLR(val);
          break;
        case 9:  // CTR
          setCTR(val);
          break;
        case 1:  // XER
          setXER(U64(val));
          break;
        default:
          MOZ_CRASH_UNSAFE_PRINTF("mtspr: unhandled SPR %u", spr);
      }
      break;
    }
    case 19: {
      // mfocrf: read one CR field selected by the FXM bitmask into RT.
      // (Plain mfcr shares this XO with FXM=0; we model both by reading
      // the full CR — the JIT only emits mfocrf and the bits outside the
      // selected field are spec'd "undefined", so reading the full CR is
      // a valid implementation.)
      setRegister(rt, (int64_t)getCR());
      break;
    }
    case 144: {
      // mtcrf: move to CR fields
      // FXM field is in bits 12..19.
      uint32_t fxm = instr->bits(19, 12);
      uint32_t rs_val = U32(getRegister(rt));
      uint32_t cr = getCR();
      for (int i = 0; i < 8; i++) {
        if (fxm & (0x80 >> i)) {
          uint32_t shift = 4 * (7 - i);
          cr = (cr & ~(0xFu << shift)) | (rs_val & (0xFu << shift));
        }
      }
      setCR(cr);
      break;
    }
    case 576: {
      // mcrxrx: move XER[OV,OV32,CA,CA32] to CR field BF
      uint32_t bf = instr->bfValue();
      uint8_t field = 0;
      if (getXEROV()) field |= 0x8;
      // OV32 at bit 19 of XER
      if ((getXER() >> kXEROV32Bit) & 1) field |= 0x4;
      if (getXERCA()) field |= 0x2;
      if ((getXER() >> kXERCA32Bit) & 1) field |= 0x1;
      setCRField(bf, field);
      break;
    }
    case 384:
    case 416: {
      // POWER10 setbc/setbcr: RT = (CR[BI]==N) ? 1 : 0
      // BI at bits 11..15; xo=384 (setbc, N=1), xo=416 (setbcr, N=0).
      uint32_t bi = instr->raValue();
      uint32_t crField = bi / 4;
      uint32_t crBit = bi % 4;
      uint8_t crFieldVal = getCRField(crField);
      bool bitSet;
      switch (crBit) {
        case 0: bitSet = (crFieldVal & kCRFieldLT) != 0; break;
        case 1: bitSet = (crFieldVal & kCRFieldGT) != 0; break;
        case 2: bitSet = (crFieldVal & kCRFieldEQ) != 0; break;
        case 3: bitSet = (crFieldVal & kCRFieldSO) != 0; break;
        default: bitSet = false; break;
      }
      bool want = (xo == 384) ? bitSet : !bitSet;
      setRegister(rt, want ? 1 : 0);
      break;
    }

    // --- Indexed loads ---
    case 21: {
      // ldx: RT = [RA|0 + RB], 8 bytes
      uint64_t ea = XFormEA(this, instr);
      setRegister(rt, readDW(ea, instr));
      break;
    }
    case 53: {
      // ldux: RT = [RA + RB], update RA
      uint64_t ea = XFormEAUpdate(this, instr);
      setRegister(rt, readDW(ea, instr));
      setRegister(ra, ea);
      break;
    }
    case 23: {
      // lwzx: RT = zero_ext([RA|0 + RB], 4 bytes)
      uint64_t ea = XFormEA(this, instr);
      setRegister(rt, U64(readWU(ea, instr)));
      break;
    }
    case 341: {
      // lwax: RT = sign_ext([RA|0 + RB], 4 bytes)
      uint64_t ea = XFormEA(this, instr);
      setRegister(rt, (int64_t)readW(ea, instr));
      break;
    }
    case 87: {
      // lbzx
      uint64_t ea = XFormEA(this, instr);
      setRegister(rt, U64(readBU(ea)));
      break;
    }
    case 279: {
      // lhzx
      uint64_t ea = XFormEA(this, instr);
      setRegister(rt, U64(readHU(ea, instr)));
      break;
    }
    case 343: {
      // lhax
      uint64_t ea = XFormEA(this, instr);
      setRegister(rt, (int64_t)readH(ea, instr));
      break;
    }
    case 535: {
      // lfsx: load float single indexed, widen to double (NaN-preserving)
      uint64_t ea = XFormEA(this, instr);
      if (!handleWasmSegFault(ea, 4)) {
        float val = *reinterpret_cast<float*>(ea);
        setFpuRegisterDouble(rt, promoteFloatPreservingNaN(val));
      }
      break;
    }
    case 599: {
      // lfdx: load float double indexed
      uint64_t ea = XFormEA(this, instr);
      setFpuRegisterDouble(rt, readD(ea, instr));
      break;
    }
    case 855: {
      // lfiwax: load float as integer word algebraic
      uint64_t ea = XFormEA(this, instr);
      int32_t val = readW(ea, instr);
      setFpuRegister(rt, (int64_t)val);
      break;
    }
    case 887: {
      // lfiwzx: load float as integer word zero
      uint64_t ea = XFormEA(this, instr);
      uint32_t val = readWU(ea, instr);
      setFpuRegister(rt, (int64_t)(uint64_t)val);
      break;
    }

    // --- Indexed stores ---
    case 149: {
      // stdx
      uint64_t ea = XFormEA(this, instr);
      writeDW(ea, getRegister(rt), instr);
      break;
    }
    case 151: {
      // stwx
      uint64_t ea = XFormEA(this, instr);
      writeW(ea, I32(getRegister(rt)), instr);
      break;
    }
    case 215: {
      // stbx
      uint64_t ea = XFormEA(this, instr);
      writeB(ea, (uint8_t)(getRegister(rt) & 0xFF));
      break;
    }
    case 407: {
      // sthx
      uint64_t ea = XFormEA(this, instr);
      writeH(ea, U16(getRegister(rt)), instr);
      break;
    }
    case 663: {
      // stfsx: store float single indexed (NaN-preserving)
      uint64_t ea = XFormEA(this, instr);
      if (!handleWasmSegFault(ea, 4)) {
        float fval = demoteDoublePreservingNaN(getFpuRegisterDouble(rt));
        *reinterpret_cast<float*>(ea) = fval;
        LLBit_ = false;
      }
      break;
    }
    case 727: {
      // stfdx: store float double indexed
      uint64_t ea = XFormEA(this, instr);
      writeD(ea, getFpuRegisterDouble(rt), instr);
      break;
    }

    // --- Byte-reversed stores ---
    case 662: {
      // stwbrx
      uint64_t ea = XFormEA(this, instr);
      uint32_t val = U32(getRegister(rt));
      writeW(ea, (int32_t)__builtin_bswap32(val), instr);
      break;
    }

    // --- Atomic load/store ---
    //
    // Load-reserve and store-conditional. Sub-word variants
    // (lbarx/lharx/stbcx./sthcx.) were added in ISA v2.06 (POWER7+).
    // Word/doubleword variants (lwarx/stwcx./ldarx/stdcx.) go back
    // to the base ISA.
    case 52: {
      // lbarx RT, RA, RB, EH
      uint64_t ea = XFormEA(this, instr);
      uint8_t val = loadLinkedB(ea, instr);
      setRegister(rt, (int64_t)val);
      break;
    }
    case 116: {
      // lharx RT, RA, RB, EH
      uint64_t ea = XFormEA(this, instr);
      uint16_t val = loadLinkedH(ea, instr);
      setRegister(rt, (int64_t)val);
      break;
    }
    case 694: {
      // stbcx. RS, RA, RB: always Rc=1.
      uint64_t ea = XFormEA(this, instr);
      uint8_t val = uint8_t(getRegister(rt));
      int result = storeConditionalB(ea, val, instr);
      if (result) {
        setCRField(0, kCRFieldEQ | (kCRFieldSO * getXERSO()));
      } else {
        setCRField(0, kCRFieldSO * getXERSO());
      }
      break;
    }
    case 726: {
      // sthcx. RS, RA, RB: always Rc=1.
      uint64_t ea = XFormEA(this, instr);
      uint16_t val = uint16_t(getRegister(rt));
      int result = storeConditionalH(ea, val, instr);
      if (result) {
        setCRField(0, kCRFieldEQ | (kCRFieldSO * getXERSO()));
      } else {
        setCRField(0, kCRFieldSO * getXERSO());
      }
      break;
    }
    case 20: {
      // lwarx
      uint64_t ea = XFormEA(this, instr);
      int32_t val = loadLinkedW(ea, instr);
      setRegister(rt, (int64_t)val);
      break;
    }
    case 150: {
      // stwcx.
      uint64_t ea = XFormEA(this, instr);
      int32_t val = I32(getRegister(rt));
      int result = storeConditionalW(ea, val, instr);
      // stwcx. always updates CR0: EQ if store succeeded, else clear.
      if (result) {
        setCRField(0, kCRFieldEQ | (kCRFieldSO * getXERSO()));
      } else {
        setCRField(0, kCRFieldSO * getXERSO());
      }
      break;
    }
    case 84: {
      // ldarx
      uint64_t ea = XFormEA(this, instr);
      int64_t val = loadLinkedD(ea, instr);
      setRegister(rt, val);
      break;
    }
    case 214: {
      // stdcx.
      uint64_t ea = XFormEA(this, instr);
      int64_t val = getRegister(rt);
      int result = storeConditionalD(ea, val, instr);
      if (result) {
        setCRField(0, kCRFieldEQ | (kCRFieldSO * getXERSO()));
      } else {
        setCRField(0, kCRFieldSO * getXERSO());
      }
      break;
    }

    // --- Synchronization ---
    case 598:
      // sync / lwsync / ptesync: no-op in simulator
      break;
    case 854:
      // eieio: no-op in simulator
      break;

    // --- GPR <-> VSR move (major opcode 31, XX1-form) ---
    //
    // Two sub-encodings:
    //   mtvsr* XT,RA{,RB}: XX1Form — XT at bits 25:21 (5) + TX at bit 0 (1);
    //                      RA at bits 20:16; RB (if any) at bits 15:11.
    //   mfvsr* RA,XS:      XX1FormMfvsr — XS at bits 25:21 (5) + SX at bit 0 (1);
    //                      RA (GPR dest) at bits 20:16.
    //
    // The original decoder treated "rsValue()" (bits 25:21 = VSR field) as a
    // GPR index — doubly wrong: the GPR side lives at bits 20:16 (= raValue())
    // and the VSR side is 6 bits (5-bit field + extension bit at bit 0). Fixed
    // here and extended for the full VSR namespace (0-63).
    // The ISA names each field in BE. "XT.DW0" is the BE doubleword which on
    // PPC64LE register storage lives at LE bytes 8-15 (our bytes[] is LE-natural:
    // bytes[0] = lowest address). With `mtvsrd / mfvsrd / mtvsrdd / mfvsrld
    // / stxvx`: mtvsrd of 0x1122334455667788 produces `00 00 00 00 00 00 00 00
    // 88 77 66 55 44 33 22 11` in memory (LE bytes 8-15 hold the GPR bits with
    // LSB at byte 8). Matching semantics here means the sim respects
    // the full Power ISA, not a self-consistent LE-reversed
    // convention.
    case 51: {
      // mfvsrd RA, XS: GPR[RA] = XS.DW0 = LE bytes 8..15.
      int xs = int(instr->rtValue() | (instr->bit(0) << 5));  // T + SX(TX)
      uint8_t bytes[16];
      getVSR128(xs, bytes);
      int64_t val;
      memcpy(&val, bytes + 8, 8);
      setRegister(instr->raValue(), val);
      break;
    }
    case 211: {
      // mtvsrwa XT, RA: XT.DW0 = sign_ext_64(RA[32:63]); XT.DW1 = 0.
      // POWER8+ (ISA 2.07). Combines extsw + mtvsrd. LE layout: bytes
      // 8-15 ← sign-extended low 32 of RA; bytes 0-7 ← 0.
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t bytes[16];
      int64_t val = (int64_t)(int32_t)getRegister(instr->raValue());
      memset(bytes, 0, 8);
      memcpy(bytes + 8, &val, 8);
      setVSR128(xt, bytes);
      break;
    }
    case 179: {
      // mtvsrd XT, RA: XT.DW0 = RA; XT.DW1 = 0.
      // LE layout: bytes 8-15 ← RA, bytes 0-7 ← 0.
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t bytes[16];
      int64_t val = getRegister(instr->raValue());
      memset(bytes, 0, 8);
      memcpy(bytes + 8, &val, 8);
      setVSR128(xt, bytes);
      break;
    }
    case 243: {
      // mtvsrwz XT, RA: XT.DW0 = zero_ext(RA[32:63]); XT.DW1 = 0.
      // The 32-bit value lives in the low 32 bits of DW0 = BE word 1,
      // which on LE storage is LE bytes 8..11 (LE word 2); LE bytes
      // 12..15 = 0 (upper half of DW0 = BE word 0 = zero-extended).
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t bytes[16];
      uint32_t lo = U32(getRegister(instr->raValue()));
      memset(bytes, 0, 16);
      bytes[8]  = (uint8_t)(lo);
      bytes[9]  = (uint8_t)(lo >> 8);
      bytes[10] = (uint8_t)(lo >> 16);
      bytes[11] = (uint8_t)(lo >> 24);
      setVSR128(xt, bytes);
      break;
    }
    case 307: {
      // mfvsrld RA, XS: GPR[RA] = XS.DW1 = LE bytes 0..7.
      // POWER9.
      int xs = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t bytes[16];
      getVSR128(xs, bytes);
      int64_t val;
      memcpy(&val, bytes, 8);
      setRegister(instr->raValue(), val);
      break;
    }
    case 403: {
      // mtvsrws XT, RA (POWER9): splat low 32 bits of RA into all four
      // word elements of XT. The same 32-bit value appears in lanes 0..3,
      // so the byte layout is identical in LE and BE —
      // bytes 0..15 = lo | lo | lo | lo.
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t bytes[16];
      uint32_t lo = U32(getRegister(instr->raValue()));
      uint64_t val = ((uint64_t)lo << 32) | lo;
      memcpy(bytes, &val, 8);
      memcpy(bytes + 8, &val, 8);
      setVSR128(xt, bytes);
      break;
    }
    case 435: {
      // mtvsrdd XT, RA, RB: XT.DW0 = RA; XT.DW1 = RB. POWER9.
      // LE: bytes 8-15 ← RA, bytes 0-7 ← RB.
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t bytes[16];
      int64_t dw0 = getRegister(instr->raValue());
      int64_t dw1 = getRegister(instr->rbValue());
      memcpy(bytes,     &dw1, 8);
      memcpy(bytes + 8, &dw0, 8);
      setVSR128(xt, bytes);
      break;
    }

    // --- VMX vector memory (major opcode 31) ---
    //
    // lvx / stvx / lvxl / stvxl.
    //   EA = (RA|0) + RB; EA = EA & ~0xF (alignment)
    //   lvx:  VRT[0:127] <- MEM(EA, 16)       bytes[0] = *(EA+0)
    //   stvx: MEM(EA, 16) <- VRS[0:127]       *(EA+0) = bytes[0]
    // lvxl / stvxl are identical in effect to lvx / stvx (the "l" form
    // hints "least recently used"; semantically indistinguishable).
    case 103: {
      // lvx: VRT = MEM(EA & ~0xF, 16 bytes)
      uint64_t ea = XFormEA(this, instr) & ~uint64_t(0xF);
      if (handleWasmSegFault(ea, 16)) break;
      memcpy(VRregisters_[rt], reinterpret_cast<const void*>(ea), 16);
      break;
    }
    case 231: {
      // stvx: MEM(EA & ~0xF, 16 bytes) = VRS
      uint64_t ea = XFormEA(this, instr) & ~uint64_t(0xF);
      if (handleWasmSegFault(ea, 16)) break;
      memcpy(reinterpret_cast<void*>(ea), VRregisters_[rt], 16);
      break;
    }
    case 359: {
      // lvxl: semantically identical to lvx
      uint64_t ea = XFormEA(this, instr) & ~uint64_t(0xF);
      if (handleWasmSegFault(ea, 16)) break;
      memcpy(VRregisters_[rt], reinterpret_cast<const void*>(ea), 16);
      break;
    }
    case 487: {
      // stvxl: semantically identical to stvx
      uint64_t ea = XFormEA(this, instr) & ~uint64_t(0xF);
      if (handleWasmSegFault(ea, 16)) break;
      memcpy(reinterpret_cast<void*>(ea), VRregisters_[rt], 16);
      break;
    }

    // --- VSX vector memory indexed (major opcode 31) ---
    //
    // These ops take a 6-bit VSR register,
    // encoded as 5-bit T/S + 1-bit TX/SX extension at instruction LSB
    // bit 0 (= our instr->bit(0)). EA = (RA|0) + RB. 16-byte access,
    // not forced-aligned (hardware may handle misaligned via sub-access
    // or alignment interrupt per impl).
    //
    // Byte-order note: lxvx/stxvx perform a natural 16-byte LE
    // memcpy. lxvd2x/stxvd2x on real PPC64 LE hardware load/store
    // doublewords in BE-pair order — i.e. lxvd2x places memory bytes
    // 0-7 in the register's BE-DW0 (= LE bytes 8-15) and bytes 8-15
    // in BE-DW1 (= LE bytes 0-7). The JIT brackets every wasm SIMD
    // load/store with a compensating xxpermdi DM=2 so the net effect
    // is a natural LE byte order. The constant pool emits the same
    // lxvd2x + xxpermdi sequence (per PatchConstantPoolLoad) but
    // assumes the hardware semantics, not a plain memcpy. So the sim
    // must match real-hardware lxvd2x/stxvd2x semantics including the
    // BE-DW byte order — otherwise the post-load xxpermdi unswaps
    // bytes that were never swapped, and constant-pool Simd128 loads
    // (e.g. shuffle masks) come out with halves transposed.
    case 268: {
      // lxvx: XT = MEM((RA|0)+RB, 16)
      uint64_t ea = XFormEA(this, instr);
      if (handleWasmSegFault(ea, 16)) break;
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t buf[16];
      memcpy(buf, reinterpret_cast<const void*>(ea), 16);
      setVSR128(xt, buf);
      break;
    }
    case 396: {
      // stxvx: MEM((RA|0)+RB, 16) = XS
      uint64_t ea = XFormEA(this, instr);
      if (handleWasmSegFault(ea, 16)) break;
      int xs = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t buf[16];
      getVSR128(xs, buf);
      memcpy(reinterpret_cast<void*>(ea), buf, 16);
      break;
    }
    case 813: {
      // lxsihzx XT, RA, RB: P9 (ISA 3.0). Load halfword to VSR & zero,
      // indexed. MEM(EA, 2) (LE-natural halfword) is placed in dw[0]
      // low 16 bits; the rest of the VSR is zeroed. In sim LE-byte
      // storage, that is bytes[8..9] (low byte at bytes[8]).
      uint64_t ea = XFormEA(this, instr);
      if (handleWasmSegFault(ea, 2)) break;
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      uint16_t halfword = readH(ea, instr);
      uint8_t buf[16];
      memset(buf, 0, 16);
      buf[8] = (uint8_t)(halfword & 0xFF);
      buf[9] = (uint8_t)((halfword >> 8) & 0xFF);
      setVSR128(xt, buf);
      break;
    }
    case 941: {
      // stxsihx XS, RA, RB: P9 (ISA 3.0). Store halfword from VSR,
      // indexed. dw[0] low 16 bits (sim bytes[8..9] in host-LE order)
      // are written as a halfword at MEM(EA, 2).
      uint64_t ea = XFormEA(this, instr);
      if (handleWasmSegFault(ea, 2)) break;
      int xs = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t buf[16];
      getVSR128(xs, buf);
      uint16_t halfword =
          (uint16_t)buf[8] | ((uint16_t)buf[9] << 8);
      writeH(ea, halfword, instr);
      break;
    }
    case 844: {
      // lxvd2x: XT = MEM((RA|0)+RB, 16) with BE-DW byte ordering.
      // Memory bytes 0-7 land in BE-DW0 (= LE bytes 8-15); memory
      // bytes 8-15 land in BE-DW1 (= LE bytes 0-7).
      uint64_t ea = XFormEA(this, instr);
      if (handleWasmSegFault(ea, 16)) break;
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t mem[16], buf[16];
      memcpy(mem, reinterpret_cast<const void*>(ea), 16);
      memcpy(buf, mem + 8, 8);
      memcpy(buf + 8, mem, 8);
      setVSR128(xt, buf);
      break;
    }
    case 972: {
      // stxvd2x: MEM((RA|0)+RB, 16) = XS with BE-DW byte ordering.
      // Inverse of lxvd2x: register LE bytes 0-7 → memory bytes 8-15;
      // LE bytes 8-15 → memory bytes 0-7.
      uint64_t ea = XFormEA(this, instr);
      if (handleWasmSegFault(ea, 16)) break;
      int xs = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t buf[16], mem[16];
      getVSR128(xs, buf);
      memcpy(mem, buf + 8, 8);
      memcpy(mem + 8, buf, 8);
      memcpy(reinterpret_cast<void*>(ea), mem, 16);
      break;
    }

    default:
      MOZ_CRASH_UNSAFE_PRINTF(
          "decodeXForm: unimplemented XO=%u (instruction 0x%08x)", xo,
          instr->instructionBits());
  }
}

// -----------------------------------------------------------------------------
// decodeRotateMask: rlwinm(21), rlwnm(23), rlwimi(20),
//   rldicl(30), rldicr(30), rldic(30), rldimi(30), rldcl(30), rldcr(30)

void Simulator::decodeRotateMask(SimInstruction* instr) {
  uint32_t opcode = instr->opcode();

  if (opcode == 21) {
    // rlwinm: RA = ROTL32(RS,SH) & MASK(MB,ME), Rc
    uint32_t rs_val = U32(getRegister(instr->rsValue()));
    uint32_t sh = instr->mSHValue();
    uint32_t mb = instr->mMBValue();
    uint32_t me = instr->mMEValue();
    uint32_t rotated = RotateLeft32(rs_val, sh);
    uint32_t mask = MASK32(mb, me);
    int64_t result = (int64_t)(uint64_t)(rotated & mask);
    setRegister(instr->raValue(), result);
    if (instr->rcBit()) updateCR0(result);
  } else if (opcode == 23) {
    // rlwnm: RA = ROTL32(RS,RB[27:31]) & MASK(MB,ME), Rc
    uint32_t rs_val = U32(getRegister(instr->rsValue()));
    uint32_t sh = U32(getRegister(instr->rbValue())) & 0x1F;
    uint32_t mb = instr->mMBValue();
    uint32_t me = instr->mMEValue();
    uint32_t rotated = RotateLeft32(rs_val, sh);
    uint32_t mask = MASK32(mb, me);
    int64_t result = (int64_t)(uint64_t)(rotated & mask);
    setRegister(instr->raValue(), result);
    if (instr->rcBit()) updateCR0(result);
  } else if (opcode == 20) {
    // rlwimi: RA = (ROTL32(RS,SH) & MASK) | (RA & ~MASK), Rc
    uint32_t rs_val = U32(getRegister(instr->rsValue()));
    uint32_t sh = instr->mSHValue();
    uint32_t mb = instr->mMBValue();
    uint32_t me = instr->mMEValue();
    uint32_t rotated = RotateLeft32(rs_val, sh);
    uint32_t mask = MASK32(mb, me);
    uint32_t ra_val = U32(getRegister(instr->raValue()));
    int64_t result = (int64_t)(uint64_t)((rotated & mask) | (ra_val & ~mask));
    setRegister(instr->raValue(), result);
    if (instr->rcBit()) updateCR0(result);
  } else if (opcode == 30) {
    // MD-form / MDS-form: 64-bit rotate/mask
    uint32_t rs = instr->rsValue();
    uint64_t rs_val = U64(getRegister(rs));
    uint32_t ra_reg = instr->raValue();

    // Determine which sub-opcode: bits 2..4 for MD-form, bit 4 for MDS.
    // MD: bits 2..4
    // MDS: bit 4 (rldcl has bit4=0, bit3..2=00 with bit1=1; rldcr has
    //      bit4=0, bit3..2=01 with bit1=1). Actually:
    //   rldicl:  30 | MD-XO=0 (bits 2..4 = 000), bit1=0
    //   rldicr:  30 | MD-XO=1 (bits 2..4 = 001), bit1=0
    //   rldic:   30 | MD-XO=2 (bits 2..4 = 010), bit1=0
    //   rldimi:  30 | MD-XO=3 (bits 2..4 = 011), bit1=0
    //   rldcl:   30 | MDS, bit4=0, bit3..1=000, bit0=Rc => bits 1..4=1000
    //            Actually rldcl: bits 1..4 = 1000, i.e. bit(4)=1,bit(3)=0,
    //            bit(2)=0,bit(1)=0
    //   rldcr:   30 | MDS, bits 1..4 = 1001
    //
    // Let's check bit 4 first: if bit(4)==1, it's MDS-form (rldcl/rldcr).
    if (instr->bit(4)) {
      // MDS-form: shift amount from RB register
      uint32_t sh = U32(getRegister(instr->rbValue())) & 0x3F;
      uint64_t rotated = RotateLeft64(rs_val, sh);
      uint32_t mb = instr->mdsMBValue();

      if (!instr->bit(1)) {
        // rldcl: RA = ROTL64(RS, RB[58:63]) & MASK(mb, 63)
        uint64_t mask = MASK64(mb, 63);
        int64_t result = I64(rotated & mask);
        setRegister(ra_reg, result);
        if (instr->rcBit()) updateCR0(result);
      } else {
        // rldcr: RA = ROTL64(RS, RB[58:63]) & MASK(0, me)
        uint32_t me = instr->mdsMBValue();
        uint64_t mask = MASK64(0, me);
        int64_t result = I64(rotated & mask);
        setRegister(ra_reg, result);
        if (instr->rcBit()) updateCR0(result);
      }
    } else {
      // MD-form
      uint32_t sh = instr->mdSHValue();
      uint64_t rotated = RotateLeft64(rs_val, sh);
      uint32_t xo_md = instr->bits(3, 2);

      switch (xo_md) {
        case 0: {
          // rldicl: RA = ROTL64(RS, SH) & MASK(mb, 63)
          uint32_t mb = instr->mdMBValue();
          uint64_t mask = MASK64(mb, 63);
          int64_t result = I64(rotated & mask);
          setRegister(ra_reg, result);
          if (instr->rcBit()) updateCR0(result);
          break;
        }
        case 1: {
          // rldicr: RA = ROTL64(RS, SH) & MASK(0, me)
          uint32_t me = instr->mdMEValue();
          uint64_t mask = MASK64(0, me);
          int64_t result = I64(rotated & mask);
          setRegister(ra_reg, result);
          if (instr->rcBit()) updateCR0(result);
          break;
        }
        case 2: {
          // rldic: RA = ROTL64(RS, SH) & MASK(mb, ~SH)
          // Actually: MASK(mb, 63-SH)
          uint32_t mb = instr->mdMBValue();
          uint64_t mask = MASK64(mb, 63 - sh);
          int64_t result = I64(rotated & mask);
          setRegister(ra_reg, result);
          if (instr->rcBit()) updateCR0(result);
          break;
        }
        case 3: {
          // rldimi: RA = (ROTL64(RS,SH) & MASK) | (RA & ~MASK)
          uint32_t mb = instr->mdMBValue();
          uint64_t mask = MASK64(mb, 63 - sh);
          uint64_t ra_val = U64(getRegister(ra_reg));
          int64_t result = I64((rotated & mask) | (ra_val & ~mask));
          setRegister(ra_reg, result);
          if (instr->rcBit()) updateCR0(result);
          break;
        }
        default:
          MOZ_CRASH_UNSAFE_PRINTF("decodeRotateMask: MD xo=%u", xo_md);
      }
    }
  } else {
    MOZ_CRASH_UNSAFE_PRINTF("decodeRotateMask: opcode=%u", opcode);
  }
}

// -----------------------------------------------------------------------------
// CR-bit accessors used by the XL-form CR-logic ops (crand, crandc, cror,
// crorc, crxor, creqv). Bit index is in BIF*4+x form: field=b/4, bit=b%4
// where 0=LT, 1=GT, 2=EQ, 3=SO.
static inline uint8_t CRBitMask(uint32_t bitInField) {
  switch (bitInField) {
    case 0: return kCRFieldLT;
    case 1: return kCRFieldGT;
    case 2: return kCRFieldEQ;
    case 3: return kCRFieldSO;
  }
  return 0;
}

static inline bool GetCRBit(Simulator& s, uint32_t b) {
  return (s.getCRField(b / 4) & CRBitMask(b % 4)) != 0;
}

static inline void SetCRBit(Simulator& s, uint32_t b, bool val) {
  uint8_t fv = s.getCRField(b / 4);
  uint8_t mask = CRBitMask(b % 4);
  s.setCRField(b / 4, val ? (fv | mask) : (fv & ~mask));
}

// -----------------------------------------------------------------------------
// decodeBranch: b(18), bc(16), XL-form(19)

void Simulator::decodeBranch(SimInstruction* instr) {
  uint32_t opcode = instr->opcode();

  if (opcode == 18) {
    // b / bl: I-form unconditional branch
    int32_t offset = instr->li26Value();
    bool lk = instr->lkBit();
    bool aa = instr->aaBit();

    int64_t target;
    if (aa) {
      target = (int64_t)offset;
    } else {
      target = get_pc() + (int64_t)offset;
    }

    if (lk) {
      setLR(get_pc() + SimInstruction::kInstrSize);
    }

    set_pc(target);
    return;
  }

  if (opcode == 16) {
    // bc / bcl: B-form conditional branch
    uint32_t bo = instr->boValue();
    uint32_t bi = instr->biValue();
    int32_t bd = instr->bd16Value();
    bool lk = instr->lkBit();
    bool aa = instr->aaBit();

    // Decrement CTR if BO[2] (bit 2 of BO, which is bo & 0x04) is clear.
    if (!(bo & 0x04)) {
      setCTR(getCTR() - 1);
    }

    // Evaluate CTR condition.
    bool ctr_ok = (bo & 0x04) ||
                  ((getCTR() != 0) ^ ((bo & 0x02) != 0));

    // Evaluate CR condition.
    uint32_t crField = bi / 4;
    uint32_t crBit = bi % 4;
    uint8_t crFieldVal = getCRField(crField);
    bool crBitSet;
    switch (crBit) {
      case 0: crBitSet = (crFieldVal & kCRFieldLT) != 0; break;
      case 1: crBitSet = (crFieldVal & kCRFieldGT) != 0; break;
      case 2: crBitSet = (crFieldVal & kCRFieldEQ) != 0; break;
      case 3: crBitSet = (crFieldVal & kCRFieldSO) != 0; break;
      default: crBitSet = false; break;
    }
    bool cond_ok = (bo & 0x10) || (crBitSet == ((bo & 0x08) != 0));

    if (ctr_ok && cond_ok) {
      int64_t target;
      if (aa) {
        target = (int64_t)bd;
      } else {
        target = get_pc() + (int64_t)bd;
      }
      if (lk) {
        setLR(get_pc() + SimInstruction::kInstrSize);
      }
      set_pc(target);
    } else {
      // Branch not taken.
      set_pc(get_pc() + SimInstruction::kInstrSize);
    }
    return;
  }

  if (opcode == 19) {
    // XL-form: bclr, bcctr, crand, crandc, cror, crorc, crxor, creqv,
    //          mcrf, isync
    uint32_t xl = instr->xlValue();

    switch (xl) {
      case 16: {
        // bclr: conditional branch to LR
        uint32_t bo = instr->boValue();
        uint32_t bi = instr->biValue();
        bool lk = instr->lkBit();

        if (!(bo & 0x04)) {
          setCTR(getCTR() - 1);
        }

        bool ctr_ok = (bo & 0x04) ||
                      ((getCTR() != 0) ^ ((bo & 0x02) != 0));

        uint32_t crField = bi / 4;
        uint32_t crBit = bi % 4;
        uint8_t crFieldVal = getCRField(crField);
        bool crBitSet;
        switch (crBit) {
          case 0: crBitSet = (crFieldVal & kCRFieldLT) != 0; break;
          case 1: crBitSet = (crFieldVal & kCRFieldGT) != 0; break;
          case 2: crBitSet = (crFieldVal & kCRFieldEQ) != 0; break;
          case 3: crBitSet = (crFieldVal & kCRFieldSO) != 0; break;
          default: crBitSet = false; break;
        }
        bool cond_ok = (bo & 0x10) || (crBitSet == ((bo & 0x08) != 0));

        if (ctr_ok && cond_ok) {
          int64_t target = getLR() & ~3LL;
          if (lk) {
            setLR(get_pc() + SimInstruction::kInstrSize);
          }
          set_pc(target);
        } else {
          set_pc(get_pc() + SimInstruction::kInstrSize);
        }
        break;
      }
      case 528: {
        // bcctr: conditional branch to CTR
        uint32_t bo = instr->boValue();
        uint32_t bi = instr->biValue();
        bool lk = instr->lkBit();

        // CTR is not decremented for bcctr.
        uint32_t crField = bi / 4;
        uint32_t crBit = bi % 4;
        uint8_t crFieldVal = getCRField(crField);
        bool crBitSet;
        switch (crBit) {
          case 0: crBitSet = (crFieldVal & kCRFieldLT) != 0; break;
          case 1: crBitSet = (crFieldVal & kCRFieldGT) != 0; break;
          case 2: crBitSet = (crFieldVal & kCRFieldEQ) != 0; break;
          case 3: crBitSet = (crFieldVal & kCRFieldSO) != 0; break;
          default: crBitSet = false; break;
        }
        bool cond_ok = (bo & 0x10) || (crBitSet == ((bo & 0x08) != 0));

        if (cond_ok) {
          int64_t target = getCTR() & ~3LL;
          if (lk) {
            setLR(get_pc() + SimInstruction::kInstrSize);
          }
          set_pc(target);
        } else {
          set_pc(get_pc() + SimInstruction::kInstrSize);
        }
        break;
      }
      case 257: {
        // crand: CR[BT] = CR[BA] & CR[BB]
        uint32_t bt = instr->rtValue();
        uint32_t ba = instr->raValue();
        uint32_t bb = instr->rbValue();
        SetCRBit(*this, bt, GetCRBit(*this, ba) && GetCRBit(*this, bb));
        break;
      }
      case 129: {
        // crandc: CR[BT] = CR[BA] & ~CR[BB]
        uint32_t bt = instr->rtValue();
        uint32_t ba = instr->raValue();
        uint32_t bb = instr->rbValue();
        SetCRBit(*this, bt, GetCRBit(*this, ba) && !GetCRBit(*this, bb));
        break;
      }
      case 449: {
        // cror: CR[BT] = CR[BA] | CR[BB]
        uint32_t bt = instr->rtValue();
        uint32_t ba = instr->raValue();
        uint32_t bb = instr->rbValue();
        SetCRBit(*this, bt, GetCRBit(*this, ba) || GetCRBit(*this, bb));
        break;
      }
      case 417: {
        // crorc: CR[BT] = CR[BA] | ~CR[BB]
        uint32_t bt = instr->rtValue();
        uint32_t ba = instr->raValue();
        uint32_t bb = instr->rbValue();
        SetCRBit(*this, bt, GetCRBit(*this, ba) || !GetCRBit(*this, bb));
        break;
      }
      case 193: {
        // crxor: CR[BT] = CR[BA] ^ CR[BB]
        uint32_t bt = instr->rtValue();
        uint32_t ba = instr->raValue();
        uint32_t bb = instr->rbValue();
        SetCRBit(*this, bt, GetCRBit(*this, ba) ^ GetCRBit(*this, bb));
        break;
      }
      case 289: {
        // creqv: CR[BT] = ~(CR[BA] ^ CR[BB])
        uint32_t bt = instr->rtValue();
        uint32_t ba = instr->raValue();
        uint32_t bb = instr->rbValue();
        SetCRBit(*this, bt, !(GetCRBit(*this, ba) ^ GetCRBit(*this, bb)));
        break;
      }
      case 150: {
        // isync: no-op in simulator
        break;
      }
      case 370: {
        // PPC_stop (0x4C0002E4) decoded as XL-form opcode 19, XL=370.
        // This is our kCallRedirInstr. Handle via softwareInterrupt.
        softwareInterrupt(instr);
        break;
      }
      case 2: {
        // POWER9 addpcis rT, D (DX-form). Computes rT = (CIA + 4) +
        // (sext16(D) << 16). The 16-bit signed displacement D is split
        // across three sub-fields:
        //   d0 = bits LE 6..15 (10 bits) — D[15:6]
        //   d1 = bits LE 16..20 (5 bits)  — D[5:1]
        //   d2 = bit  LE 0      (1 bit)   — D[0]
        // (Mirrors the encoder in Assembler-ppc64.cpp:as_addpcis.)
        uint32_t rt = instr->rtValue();
        uint32_t d0 = instr->bits(15, 6);
        uint32_t d1 = instr->bits(20, 16);
        uint32_t d2 = instr->bit(0);
        int16_t D = (int16_t)((d0 << 6) | (d1 << 1) | d2);
        int64_t cia = reinterpret_cast<int64_t>(instr);
        setRegister(rt, cia + SimInstruction::kInstrSize +
                            (static_cast<int64_t>(D) << 16));
        break;
      }
      default:
        MOZ_CRASH_UNSAFE_PRINTF("decodeBranch: XL opcode 19, xl=%u", xl);
    }
    return;
  }

  MOZ_CRASH_UNSAFE_PRINTF("decodeBranch: opcode=%u", opcode);
}

// -----------------------------------------------------------------------------
// decodeFP: Major opcodes 59 (A-form single) and 63 (X-form / A-form double)

void Simulator::decodeFP(SimInstruction* instr) {
  uint32_t opcode = instr->opcode();
  uint32_t rt = instr->rtValue();  // FRT
  uint32_t ra = instr->raValue();  // FRA
  uint32_t rb = instr->rbValue();  // FRB
  uint32_t rc_reg = instr->rcValue();  // FRC (A-form)

  if (opcode == 63) {
    // X-form and A-form double-precision instructions.
    // For A-form, the sub-opcode is in bits 1..5.
    // For X-form, the sub-opcode is in bits 1..10.
    uint32_t xo_a = instr->bits(5, 1);  // A-form sub-opcode
    uint32_t xo_x = instr->bits(10, 1); // X-form sub-opcode

    // Try A-form first (5-bit sub-opcode in bits 1..5).
    switch (xo_a) {
      case 21: {
        // fadd
        double result = getFpuRegisterDouble(ra) + getFpuRegisterDouble(rb);
        setFpuRegisterDouble(rt, result);
        return;
      }
      case 20: {
        // fsub
        double result = getFpuRegisterDouble(ra) - getFpuRegisterDouble(rb);
        setFpuRegisterDouble(rt, result);
        return;
      }
      case 25: {
        // fmul: FRT = FRA * FRC (note: FRC, not FRB!)
        double result = getFpuRegisterDouble(ra) * getFpuRegisterDouble(rc_reg);
        setFpuRegisterDouble(rt, result);
        return;
      }
      case 18: {
        // fdiv
        double result = getFpuRegisterDouble(ra) / getFpuRegisterDouble(rb);
        setFpuRegisterDouble(rt, result);
        return;
      }
      case 22: {
        // fsqrt
        double result = sqrt(getFpuRegisterDouble(rb));
        setFpuRegisterDouble(rt, result);
        return;
      }
      case 29: {
        // fmadd: FRT = FRA * FRC + FRB
        double result = std::fma(getFpuRegisterDouble(ra),
                                 getFpuRegisterDouble(rc_reg),
                                 getFpuRegisterDouble(rb));
        setFpuRegisterDouble(rt, result);
        return;
      }
      case 30: {
        // fnmsub: FRT = -(FRA * FRC - FRB)
        double result = -(std::fma(getFpuRegisterDouble(ra),
                                   getFpuRegisterDouble(rc_reg),
                                   -getFpuRegisterDouble(rb)));
        setFpuRegisterDouble(rt, result);
        return;
      }
      case 28: {
        // fmsub: FRT = FRA * FRC - FRB
        double result = std::fma(getFpuRegisterDouble(ra),
                                 getFpuRegisterDouble(rc_reg),
                                 -getFpuRegisterDouble(rb));
        setFpuRegisterDouble(rt, result);
        return;
      }
      case 31: {
        // fnmadd: FRT = -(FRA * FRC + FRB)
        double result = -(std::fma(getFpuRegisterDouble(ra),
                                   getFpuRegisterDouble(rc_reg),
                                   getFpuRegisterDouble(rb)));
        setFpuRegisterDouble(rt, result);
        return;
      }
      case 23: {
        // fsel: FRT = (FRA >= 0) ? FRC : FRB
        double fra = getFpuRegisterDouble(ra);
        setFpuRegisterDouble(rt, (fra >= 0.0) ? getFpuRegisterDouble(rc_reg)
                                               : getFpuRegisterDouble(rb));
        return;
      }
      case 26: {
        // frsqrte: FRT = 1.0 / sqrt(FRB) (estimate)
        double result = 1.0 / sqrt(getFpuRegisterDouble(rb));
        setFpuRegisterDouble(rt, result);
        return;
      }
    }

    // X-form (10-bit sub-opcode).
    switch (xo_x) {
      case 72: {
        // fmr: FRT = FRB
        setFpuRegisterDouble(rt, getFpuRegisterDouble(rb));
        break;
      }
      case 40: {
        // fneg: FRT = -FRB
        setFpuRegisterDouble(rt, -getFpuRegisterDouble(rb));
        break;
      }
      case 264: {
        // fabs: FRT = |FRB|
        setFpuRegisterDouble(rt, fabs(getFpuRegisterDouble(rb)));
        break;
      }
      case 136: {
        // fnabs: FRT = -|FRB|
        setFpuRegisterDouble(rt, -fabs(getFpuRegisterDouble(rb)));
        break;
      }
      case 8: {
        // fcpsgn: FRT = sign(FRA) || magnitude(FRB)
        double fra = getFpuRegisterDouble(ra);
        double frb = getFpuRegisterDouble(rb);
        setFpuRegisterDouble(rt, std::copysign(frb, fra));
        break;
      }
      case 0: {
        // fcmpu: compare FRA, FRB unordered
        uint32_t bf = instr->bfValue();
        double fra = getFpuRegisterDouble(ra);
        double frb = getFpuRegisterDouble(rb);
        uint8_t field = 0;
        if (std::isnan(fra) || std::isnan(frb)) {
          field = kCRFieldSO;
        } else if (fra < frb) {
          field = kCRFieldLT;
        } else if (fra > frb) {
          field = kCRFieldGT;
        } else {
          field = kCRFieldEQ;
        }
        setCRField(bf, field);
        break;
      }
      case 32: {
        // fcmpo: compare FRA, FRB ordered
        uint32_t bf = instr->bfValue();
        double fra = getFpuRegisterDouble(ra);
        double frb = getFpuRegisterDouble(rb);
        uint8_t field = 0;
        if (std::isnan(fra) || std::isnan(frb)) {
          field = kCRFieldSO;
        } else if (fra < frb) {
          field = kCRFieldLT;
        } else if (fra > frb) {
          field = kCRFieldGT;
        } else {
          field = kCRFieldEQ;
        }
        setCRField(bf, field);
        break;
      }
      // For fctid* and fctiw* the ISA specifies that bit 23 of FPSCR (VXCVI,
      // "invalid op for integer convert") is set when the source is NaN, +Inf,
      // -Inf, or out of the destination's range. Wasm's out-of-range trap
      // sequence is `mtfsb0 23; fctidz; mfvsrd; mcrfs cr0,5; bt SOBit,trap`,
      // so the simulator MUST update VXCVI here for the trap to fire. With
      // FPSCR_ in the low-half PPC layout (PPC bit N → int64 bit (31-N)),
      // VXCVI lives at int64 bit (31-23) = 8.
      case 814: {
        // fctid: convert double to int64 (current rounding)
        double frb = getFpuRegisterDouble(rb);
        int64_t result;
        bool invalid = false;
        if (std::isnan(frb)) {
          result = INT64_MIN;
          invalid = true;
        } else if (frb >= -(double)INT64_MIN || frb < (double)INT64_MIN) {
          result = (frb < 0) ? INT64_MIN : INT64_MAX;
          invalid = true;
        } else {
          switch (FPSCR_ & kFPSCRRNMask) {
            case RN: result = (int64_t)llrint(frb); break;
            case RZ: result = (int64_t)frb; break;
            case RP: result = (int64_t)ceil(frb); break;
            case RM: result = (int64_t)floor(frb); break;
            default: result = (int64_t)frb; break;
          }
        }
        if (invalid) FPSCR_ |= (1ULL << 8);  /* VXCVI: PPC bit 23 in low-half layout */
        setFpuRegister(rt, result);
        break;
      }
      case 815: {
        // fctidz: convert double to int64 (round toward zero)
        double frb = getFpuRegisterDouble(rb);
        int64_t result;
        bool invalid = false;
        if (std::isnan(frb)) {
          result = INT64_MIN;
          invalid = true;
        } else if (frb >= -(double)INT64_MIN) {
          result = INT64_MAX;
          invalid = true;
        } else if (frb < (double)INT64_MIN) {
          result = INT64_MIN;
          invalid = true;
        } else {
          result = (int64_t)frb;
        }
        if (invalid) FPSCR_ |= (1ULL << 8);  /* VXCVI: PPC bit 23 in low-half layout */
        setFpuRegister(rt, result);
        break;
      }
      case 942: {
        // fctidu: convert double to uint64 (current rounding).
        // VXCVI is signaled when source is NaN, ±Inf, or the rounded value
        // is outside [0, 2^64-1]. Notably,
        // a negative source whose rounded value is 0 (e.g. -0.4 in RN, or
        // any value in (-1, 0) in RZ) is NOT invalid.
        double frb = getFpuRegisterDouble(rb);
        uint64_t result;
        bool invalid = false;
        if (std::isnan(frb)) {
          result = 0;
          invalid = true;
        } else if (frb >= -2.0 * (double)INT64_MIN /* 2^64 */) {
          result = UINT64_MAX;
          invalid = true;
        } else {
          double rounded;
          switch (FPSCR_ & kFPSCRRNMask) {
            case RN: rounded = nearbyint(frb); break;
            case RZ: rounded = trunc(frb); break;
            case RP: rounded = ceil(frb); break;
            case RM: rounded = floor(frb); break;
            default: rounded = trunc(frb); break;
          }
          if (rounded < 0.0) {
            result = 0;
            invalid = true;
          } else {
            result = (uint64_t)rounded;
          }
        }
        if (invalid) FPSCR_ |= (1ULL << 8);  /* VXCVI: PPC bit 23 in low-half layout */
        setFpuRegister(rt, I64(result));
        break;
      }
      case 943: {
        // fctiduz: convert double to uint64 (round toward zero).
        // Same VXCVI rule as fctidu but rounding is fixed to truncate
        // toward zero. Source in (-1, 0) truncates to 0 — VALID.
        double frb = getFpuRegisterDouble(rb);
        uint64_t result;
        bool invalid = false;
        if (std::isnan(frb)) {
          result = 0;
          invalid = true;
        } else if (frb >= -2.0 * (double)INT64_MIN /* 2^64 */) {
          result = UINT64_MAX;
          invalid = true;
        } else if (frb <= -1.0) {
          // Truncated value is negative — invalid for unsigned.
          result = 0;
          invalid = true;
        } else {
          // Source is in (-1, 2^64); truncation toward zero yields a value
          // in [0, 2^64).
          result = (uint64_t)trunc(frb);
        }
        if (invalid) FPSCR_ |= (1ULL << 8);  /* VXCVI: PPC bit 23 in low-half layout */
        setFpuRegister(rt, I64(result));
        break;
      }
      case 14: {
        // fctiw: convert double to int32 (current rounding).
        // Invalid range: rounded value < INT32_MIN or > INT32_MAX. The
        // double-precision boundary on the negative side is INT32_MIN-1 =
        // -2^31-1 = -2147483649.0 (exactly representable; doubles in
        // (-2^31-1, -2^31) all round-to-nearest to -2^31 which is valid).
        double frb = getFpuRegisterDouble(rb);
        int32_t result;
        bool invalid = false;
        if (std::isnan(frb)) {
          result = INT32_MIN;
          invalid = true;
        } else {
          double rounded;
          switch (FPSCR_ & kFPSCRRNMask) {
            case RN: rounded = nearbyint(frb); break;
            case RZ: rounded = trunc(frb); break;
            case RP: rounded = ceil(frb); break;
            case RM: rounded = floor(frb); break;
            default: rounded = trunc(frb); break;
          }
          if (rounded > (double)INT32_MAX) {
            result = INT32_MAX;
            invalid = true;
          } else if (rounded < (double)INT32_MIN) {
            result = INT32_MIN;
            invalid = true;
          } else {
            result = (int32_t)rounded;
          }
        }
        if (invalid) FPSCR_ |= (1ULL << 8);  /* VXCVI: PPC bit 23 in low-half layout */
        setFpuRegister(rt, (int64_t)result);
        break;
      }
      case 15: {
        // fctiwz: convert double to int32 (round toward zero).
        // Truncation of a value in (-2^31-1, INT32_MIN) toward zero gives
        // INT32_MIN — valid. Only `frb <= -2^31-1` (i.e. `frb < INT32_MIN-1+1`
        // = `frb < -2147483648` ... wait, simplest: check truncated value in
        // range AFTER truncation.)
        double frb = getFpuRegisterDouble(rb);
        int32_t result;
        bool invalid = false;
        if (std::isnan(frb)) {
          result = INT32_MIN;
          invalid = true;
        } else {
          double truncated = trunc(frb);
          if (truncated > (double)INT32_MAX) {
            result = INT32_MAX;
            invalid = true;
          } else if (truncated < (double)INT32_MIN) {
            result = INT32_MIN;
            invalid = true;
          } else {
            result = (int32_t)truncated;
          }
        }
        if (invalid) FPSCR_ |= (1ULL << 8);  /* VXCVI: PPC bit 23 in low-half layout */
        setFpuRegister(rt, (int64_t)result);
        break;
      }
      case 142: {
        // fctiwu: convert double to uint32 (current rounding). The check is
        // on the ROUNDED value: VXCVI iff rounded < 0 or rounded > UINT32_MAX.
        double frb = getFpuRegisterDouble(rb);
        uint32_t result;
        bool invalid = false;
        if (std::isnan(frb)) {
          result = 0;
          invalid = true;
        } else {
          double rounded;
          switch (FPSCR_ & kFPSCRRNMask) {
            case RN: rounded = nearbyint(frb); break;
            case RZ: rounded = trunc(frb); break;
            case RP: rounded = ceil(frb); break;
            case RM: rounded = floor(frb); break;
            default: rounded = trunc(frb); break;
          }
          if (rounded < 0.0) {
            result = 0;
            invalid = true;
          } else if (rounded > (double)UINT32_MAX) {
            result = UINT32_MAX;
            invalid = true;
          } else {
            result = (uint32_t)rounded;
          }
        }
        if (invalid) FPSCR_ |= (1ULL << 8);  /* VXCVI: PPC bit 23 in low-half layout */
        setFpuRegister(rt, (int64_t)(uint64_t)result);
        break;
      }
      case 143: {
        // fctiwuz: convert double to uint32 (round toward zero).
        // Source in (-1, 0) truncates to 0 — VALID.
        double frb = getFpuRegisterDouble(rb);
        uint32_t result;
        bool invalid = false;
        if (std::isnan(frb)) {
          result = 0;
          invalid = true;
        } else {
          double truncated = trunc(frb);
          if (truncated > (double)UINT32_MAX) {
            result = UINT32_MAX;
            invalid = true;
          } else if (truncated < 0.0) {
            result = 0;
            invalid = true;
          } else {
            result = (uint32_t)truncated;
          }
        }
        if (invalid) FPSCR_ |= (1ULL << 8);  /* VXCVI: PPC bit 23 in low-half layout */
        setFpuRegister(rt, (int64_t)(uint64_t)result);
        break;
      }
      case 846: {
        // fcfid: convert int64 in FPR to double
        int64_t val = getFpuRegister(rb);
        setFpuRegisterDouble(rt, (double)val);
        break;
      }
      case 974: {
        // fcfidu: convert uint64 in FPR to double
        uint64_t val = U64(getFpuRegister(rb));
        setFpuRegisterDouble(rt, (double)val);
        break;
      }
      case 12: {
        // frsp: round double to single precision (then re-extend in FPR).
        // sNaN inputs are quieted (the result payload MSB is set).
        // wasm f32.demote_f64 lowers to this op when
        // not using xscvdpsp directly.
        double frb = getFpuRegisterDouble(rb);
        float result = demoteDoublePreservingNaN(frb);
        uint32_t fbits;
        memcpy(&fbits, &result, sizeof(fbits));
        if ((fbits & 0x7F800000u) == 0x7F800000u &&
            (fbits & 0x007FFFFFu) != 0) {
          fbits |= 0x00400000u;
          memcpy(&result, &fbits, sizeof(result));
        }
        setFpuRegisterDouble(rt, promoteFloatPreservingNaN(result));
        break;
      }
      case 392: {
        // frin: round to nearest integer (ties away from zero)
        double frb = getFpuRegisterDouble(rb);
        setFpuRegisterDouble(rt, round(frb));
        break;
      }
      case 424: {
        // friz: round toward zero
        double frb = getFpuRegisterDouble(rb);
        setFpuRegisterDouble(rt, trunc(frb));
        break;
      }
      case 456: {
        // frip: round toward +infinity (ceil). XO=456.
        double frb = getFpuRegisterDouble(rb);
        setFpuRegisterDouble(rt, ceil(frb));
        break;
      }
      case 488: {
        // frim: round toward -infinity (floor). XO=488.
        double frb = getFpuRegisterDouble(rb);
        setFpuRegisterDouble(rt, floor(frb));
        break;
      }
      case 583: {
        // mffs: FRT = FPSCR (as double bit pattern)
        setFpuRegister(rt, I64(FPSCR_));
        break;
      }
      // FPSCR is treated as a 32-bit register stored in the low 32 bits of
      // FPSCR_ (uint64_t), with PPC bit numbering: PPC bit N (where bit 0 is
      // the MSB) lives at int64 bit (31-N). Field F (4 bits) covers PPC bits
      // 4F..4F+3 → int64 bit-LSB (28-4F) to bit-MSB (31-4F). This matches
      // mcrfs, mtfsfi, kFPSCRRNMask (which checks bits 30-31 PPC = int64 bits
      // 0-1), and mffs (which copies FPSCR into FPR bits 32..63 PPC = int64
      // bits 0..31). Earlier mtfsb0/mtfsb1 used (63-bt) which placed bits in
      // the high half of FPSCR_ where mcrfs etc. would never see them — so
      // the wasm trap sequence `mtfsb0 23; fctidz; mcrfs cr0,5; bt SO,oolEntry`
      // could not detect VXCVI.
      case 70: {
        // mtfsb0: clear FPSCR bit. XO=70.
        // (Cases 38 and 70 had the labels swapped, so wasm's
        // `mtfsb0 23; fctidz; mcrfs cr0,5; bt SO,trap` sequence accidentally
        // SET VXCVI before the convert ran, causing every fctid* to trap.)
        uint32_t bt = instr->rtValue();
        FPSCR_ &= ~(1ULL << (31 - bt));
        break;
      }
      case 64: {
        // mcrfs: copy FPSCR field to CR field
        uint32_t bf = instr->bfValue();
        uint32_t bfa = instr->bits(20, 18);
        uint32_t shift = 4 * (7 - bfa);
        uint8_t val = (FPSCR_ >> shift) & 0xF;
        setCRField(bf, val);
        break;
      }
      default:
        MOZ_CRASH_UNSAFE_PRINTF(
            "decodeFP: opcode 63, xo_x=%u (instruction 0x%08x)", xo_x,
            instr->instructionBits());
    }
  } else if (opcode == 59) {
    // A-form single-precision instructions.
    uint32_t xo_a = instr->bits(5, 1);

    switch (xo_a) {
      case 21: {
        // fadds
        double result = (double)((float)(getFpuRegisterDouble(ra) +
                                         getFpuRegisterDouble(rb)));
        setFpuRegisterDouble(rt, result);
        break;
      }
      case 20: {
        // fsubs
        double result = (double)((float)(getFpuRegisterDouble(ra) -
                                         getFpuRegisterDouble(rb)));
        setFpuRegisterDouble(rt, result);
        break;
      }
      case 25: {
        // fmuls: FRT = (float)(FRA * FRC)
        double result = (double)((float)(getFpuRegisterDouble(ra) *
                                         getFpuRegisterDouble(rc_reg)));
        setFpuRegisterDouble(rt, result);
        break;
      }
      case 18: {
        // fdivs
        double result = (double)((float)(getFpuRegisterDouble(ra) /
                                         getFpuRegisterDouble(rb)));
        setFpuRegisterDouble(rt, result);
        break;
      }
      case 22: {
        // fsqrts
        double result = (double)sqrtf((float)getFpuRegisterDouble(rb));
        setFpuRegisterDouble(rt, result);
        break;
      }
      case 29: {
        // fmadds
        double result = (double)((float)std::fma(getFpuRegisterDouble(ra),
                                                 getFpuRegisterDouble(rc_reg),
                                                 getFpuRegisterDouble(rb)));
        setFpuRegisterDouble(rt, result);
        break;
      }
      case 30: {
        // fnmsubs
        double result = (double)(-(float)std::fma(getFpuRegisterDouble(ra),
                                                  getFpuRegisterDouble(rc_reg),
                                                  -getFpuRegisterDouble(rb)));
        setFpuRegisterDouble(rt, result);
        break;
      }
      case 28: {
        // fmsubs
        double result = (double)((float)std::fma(getFpuRegisterDouble(ra),
                                                 getFpuRegisterDouble(rc_reg),
                                                 -getFpuRegisterDouble(rb)));
        setFpuRegisterDouble(rt, result);
        break;
      }
      case 31: {
        // fnmadds
        double result = (double)(-(float)std::fma(getFpuRegisterDouble(ra),
                                                  getFpuRegisterDouble(rc_reg),
                                                  getFpuRegisterDouble(rb)));
        setFpuRegisterDouble(rt, result);
        break;
      }
      default: {
        // Try X-form sub-opcodes for opcode 59 (e.g., fcfids, fcfidus).
        uint32_t xo_x = instr->bits(10, 1);
        switch (xo_x) {
          case 846: {
            // fcfids: convert int64 to float single (result stored as double)
            int64_t val = getFpuRegister(rb);
            setFpuRegisterDouble(rt, (double)(float)val);
            break;
          }
          case 974: {
            // fcfidus: convert uint64 to float single
            uint64_t val = U64(getFpuRegister(rb));
            setFpuRegisterDouble(rt, (double)(float)val);
            break;
          }
          default:
            MOZ_CRASH_UNSAFE_PRINTF(
                "decodeFP: opcode 59, xo_a=%u xo_x=%u", xo_a, xo_x);
        }
        break;
      }
    }
  } else {
    MOZ_CRASH_UNSAFE_PRINTF("decodeFP: opcode=%u", opcode);
  }
}

// -----------------------------------------------------------------------------
// decodeVMX: Major opcode 4 (AltiVec/VMX vector ops on VR0-VR31).
//
// VR-form (VX-form): bits 0-5 = primary opcode (4), bits 6-10 = VRT,
// bits 11-15 = VRA, bits 16-20 = VRB, bits 21-31 = XO (11 bits).
// XO extracted via `instructionBits() & 0x7FF`.
//
// Helpers below pack/unpack each VR via the VRregisters_ byte storage
// (16 bytes, big-endian PPC numbering: bytes[0] is the most-significant
// byte of the architectural register, but on PPC64 LE wasm the lane
// ordering is what the JIT expects). All ops here use byte-level
// accessors for consistency with the existing VMX memory ops.

void Simulator::decodeVMX(SimInstruction* instr) {
  uint32_t xo = instr->instructionBits() & 0x7FFu;
  uint32_t vrt = instr->rtValue();   // bits 6..10
  uint32_t vra = instr->raValue();   // bits 11..15
  uint32_t vrb = instr->rbValue();   // bits 16..20
  uint32_t uimm = instr->raValue();  // VA-form: 5-bit immediate at bits 11..15

  uint8_t a[16], b[16], r[16];
  getVRBytes(vra, a);
  getVRBytes(vrb, b);

  // Helpers for treating the byte storage as typed lane arrays.
  // The PPC64LE wasm SIMD lowering stores each lane's bytes in
  // little-endian order, so lane i of an N-byte element occupies bytes
  // (i*N) .. (i*N + N - 1) with the LSB at byte (i*N). For example,
  // a v128.const i32x4 0x12345678 has bytes [78 56 34 12 …].
  #define LANE_U8(buf, i)  ((uint8_t)(buf)[(i)])
  #define LANE_S8(buf, i)  ((int8_t)(buf)[(i)])
  #define LANE_U16(buf, i)                                     \
    ((uint16_t)((uint16_t)(buf)[(i) * 2] |                    \
                ((uint16_t)(buf)[(i) * 2 + 1] << 8)))
  #define LANE_S16(buf, i) ((int16_t)LANE_U16(buf, i))
  #define LANE_U32(buf, i)                                     \
    ((uint32_t)((uint32_t)(buf)[(i) * 4] |                    \
                ((uint32_t)(buf)[(i) * 4 + 1] << 8) |         \
                ((uint32_t)(buf)[(i) * 4 + 2] << 16) |        \
                ((uint32_t)(buf)[(i) * 4 + 3] << 24)))
  #define LANE_S32(buf, i) ((int32_t)LANE_U32(buf, i))
  #define LANE_U64(buf, i)                                     \
    ((uint64_t)((uint64_t)(buf)[(i) * 8] |                    \
                ((uint64_t)(buf)[(i) * 8 + 1] << 8) |         \
                ((uint64_t)(buf)[(i) * 8 + 2] << 16) |        \
                ((uint64_t)(buf)[(i) * 8 + 3] << 24) |        \
                ((uint64_t)(buf)[(i) * 8 + 4] << 32) |        \
                ((uint64_t)(buf)[(i) * 8 + 5] << 40) |        \
                ((uint64_t)(buf)[(i) * 8 + 6] << 48) |        \
                ((uint64_t)(buf)[(i) * 8 + 7] << 56)))
  #define LANE_S64(buf, i) ((int64_t)LANE_U64(buf, i))
  #define SET_LANE_U8(buf, i, v)  do { (buf)[(i)] = (uint8_t)(v); } while (0)
  #define SET_LANE_U16(buf, i, v) do {                                       \
      (buf)[(i) * 2]     = (uint8_t)((uint16_t)(v) & 0xFF);                 \
      (buf)[(i) * 2 + 1] = (uint8_t)(((uint16_t)(v) >> 8) & 0xFF);          \
    } while (0)
  #define SET_LANE_U32(buf, i, v) do {                                       \
      (buf)[(i) * 4]     = (uint8_t)((uint32_t)(v) & 0xFF);                 \
      (buf)[(i) * 4 + 1] = (uint8_t)(((uint32_t)(v) >> 8) & 0xFF);          \
      (buf)[(i) * 4 + 2] = (uint8_t)(((uint32_t)(v) >> 16) & 0xFF);         \
      (buf)[(i) * 4 + 3] = (uint8_t)(((uint32_t)(v) >> 24) & 0xFF);         \
    } while (0)
  #define SET_LANE_U64(buf, i, v) do {                                       \
      (buf)[(i) * 8]     = (uint8_t)((uint64_t)(v) & 0xFF);                 \
      (buf)[(i) * 8 + 1] = (uint8_t)(((uint64_t)(v) >> 8) & 0xFF);          \
      (buf)[(i) * 8 + 2] = (uint8_t)(((uint64_t)(v) >> 16) & 0xFF);         \
      (buf)[(i) * 8 + 3] = (uint8_t)(((uint64_t)(v) >> 24) & 0xFF);         \
      (buf)[(i) * 8 + 4] = (uint8_t)(((uint64_t)(v) >> 32) & 0xFF);         \
      (buf)[(i) * 8 + 5] = (uint8_t)(((uint64_t)(v) >> 40) & 0xFF);         \
      (buf)[(i) * 8 + 6] = (uint8_t)(((uint64_t)(v) >> 48) & 0xFF);         \
      (buf)[(i) * 8 + 7] = (uint8_t)(((uint64_t)(v) >> 56) & 0xFF);         \
    } while (0)

  // --- VA-form pre-dispatch ---
  //
  // VA-form has a 6-bit XO at bits 26-31 and a 5-bit VRC at bits 21-25.
  // decodeVMX's 11-bit XO mask conflates VRC with
  // XO, so a plain `switch (xo)` over 11-bit values only matches when
  // VRC == 0. Peel off the three VA-form ops actually used by the JIT
  // (vmladduhm, vsel, vperm) before the main switch so any VRC value
  // works. vsldoi (XO=44) is VX-form with SH at bits 22-25, not VA —
  // handled in the switch below.
  {
    uint32_t va_xo = xo & 0x3Fu;
    if (va_xo == 32 || va_xo == 33 || va_xo == 34 || va_xo == 38 ||
        va_xo == 40 || va_xo == 42 || va_xo == 43) {
      uint32_t vrc = (instr->instructionBits() >> 6) & 0x1F;
      uint8_t cv[16];
      getVRBytes(vrc, cv);
      if (va_xo == 32) {
        // vmhaddshs VT,VA,VB,VC : VT[i] = sat_s16(
        //   (s32)VA.h[i] * (s32)VB.h[i] >> 15 + (s32)VC.h[i])
        // (no rounding term — use vmhraddshs for the rounded form).
        for (int i = 0; i < 8; i++) {
          int32_t prod = (int32_t)LANE_S16(a, i) * (int32_t)LANE_S16(b, i);
          int32_t sum = (prod >> 15) + (int32_t)LANE_S16(cv, i);
          if (sum > INT16_MAX) sum = INT16_MAX;
          if (sum < INT16_MIN) sum = INT16_MIN;
          SET_LANE_U16(r, i, (uint16_t)(int16_t)sum);
        }
      } else if (va_xo == 33) {
        // vmhraddshs VT,VA,VB,VC : rounded Q15 multiply-add-saturate.
        //   VT[i] = sat_s16(((s32)VA.h[i] * (s32)VB.h[i] + 0x4000)
        //                   >> 15 + (s32)VC.h[i])
        // Used by wasm i16x8.q15mulr_sat_s (VC is zero).
        for (int i = 0; i < 8; i++) {
          int32_t prod = (int32_t)LANE_S16(a, i) * (int32_t)LANE_S16(b, i);
          int32_t sum = ((prod + 0x4000) >> 15) + (int32_t)LANE_S16(cv, i);
          if (sum > INT16_MAX) sum = INT16_MAX;
          if (sum < INT16_MIN) sum = INT16_MIN;
          SET_LANE_U16(r, i, (uint16_t)(int16_t)sum);
        }
      } else if (va_xo == 34) {
        // vmladduhm VT,VA,VB,VC : VT = low16(VA*VB + VC)
        for (int i = 0; i < 8; i++) {
          uint16_t prod = LANE_U16(a, i) * LANE_U16(b, i);
          SET_LANE_U16(r, i, prod + LANE_U16(cv, i));
        }
      } else if (va_xo == 40) {
        // vmsumshm VT,VA,VB,VC : pairwise multiply-sum of signed halfwords
        // into i32 lanes, modulo i32 wrap.
        //   VT.i32[k] = VC.i32[k] + VA.i16[2k]*VB.i16[2k]
        //                         + VA.i16[2k+1]*VB.i16[2k+1]
        // Used by wasm i32x4.dot_i16x8_s with VC = 0, and by
        // i32x4.extadd_pairwise_i16x8_s with VB = splat(1) and VC = 0.
        for (int k = 0; k < 4; k++) {
          int32_t a0 = (int32_t)LANE_S16(a, 2 * k);
          int32_t a1 = (int32_t)LANE_S16(a, 2 * k + 1);
          int32_t b0 = (int32_t)LANE_S16(b, 2 * k);
          int32_t b1 = (int32_t)LANE_S16(b, 2 * k + 1);
          int32_t c  = LANE_S32(cv, k);
          int32_t result = (int32_t)((uint32_t)c + (uint32_t)(a0 * b0) +
                                     (uint32_t)(a1 * b1));
          SET_LANE_U32(r, k, (uint32_t)result);
        }
      } else if (va_xo == 38) {
        // vmsumuhm VT,VA,VB,VC : same as vmsumshm but unsigned halfwords.
        //   VT.u32[k] = VC.u32[k] + VA.u16[2k]*VB.u16[2k]
        //                         + VA.u16[2k+1]*VB.u16[2k+1]
        // Used by wasm i32x4.extadd_pairwise_i16x8_u with VB = splat(1)
        // and VC = 0.
        for (int k = 0; k < 4; k++) {
          uint32_t a0 = (uint32_t)LANE_U16(a, 2 * k);
          uint32_t a1 = (uint32_t)LANE_U16(a, 2 * k + 1);
          uint32_t b0 = (uint32_t)LANE_U16(b, 2 * k);
          uint32_t b1 = (uint32_t)LANE_U16(b, 2 * k + 1);
          uint32_t c  = LANE_U32(cv, k);
          uint32_t result = c + a0 * b0 + a1 * b1;
          SET_LANE_U32(r, k, result);
        }
      } else if (va_xo == 42) {
        // vsel VT,VA,VB,VC : VT[i] = (VC[i] & VB[i]) | (~VC[i] & VA[i])
        for (int i = 0; i < 16; i++) {
          r[i] = (uint8_t)((cv[i] & b[i]) | (~cv[i] & a[i]));
        }
      } else {
        // vperm VT,VA,VB,VC; empirical LE:
        //   r[LE_i] = (VC[LE_i] < 16) ? VA[LE_(15-VC[i])]
        //                             : VB[LE_(31-VC[i])]
        for (int i = 0; i < 16; i++) {
          uint8_t idx = cv[i] & 0x1F;
          r[i] = (idx < 16) ? a[15 - idx] : b[31 - idx];
        }
      }
      setVRBytes(vrt, r);
      goto vmx_done;
    }
  }

  switch (xo) {
    // === Integer add (modulo) ===
    case 0:    // vaddubm
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, LANE_U8(a, i) + LANE_U8(b, i));
      }
      setVRBytes(vrt, r); break;
    case 64:   // vadduhm
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, LANE_U16(a, i) + LANE_U16(b, i));
      }
      setVRBytes(vrt, r); break;
    case 128:  // vadduwm
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, LANE_U32(a, i) + LANE_U32(b, i));
      }
      setVRBytes(vrt, r); break;
    case 192:  // vaddudm
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i, LANE_U64(a, i) + LANE_U64(b, i));
      }
      setVRBytes(vrt, r); break;

    // === Integer sub (modulo) ===
    case 1024: // vsububm
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, LANE_U8(a, i) - LANE_U8(b, i));
      }
      setVRBytes(vrt, r); break;
    case 1088: // vsubuhm
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, LANE_U16(a, i) - LANE_U16(b, i));
      }
      setVRBytes(vrt, r); break;
    case 1152: // vsubuwm
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, LANE_U32(a, i) - LANE_U32(b, i));
      }
      setVRBytes(vrt, r); break;
    case 1216: // vsubudm
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i, LANE_U64(a, i) - LANE_U64(b, i));
      }
      setVRBytes(vrt, r); break;

    // === Integer add (saturating, signed) ===
    case 768:  // vaddsbs
      for (int i = 0; i < 16; i++) {
        int s = (int)LANE_S8(a, i) + (int)LANE_S8(b, i);
        if (s > INT8_MAX) s = INT8_MAX;
        if (s < INT8_MIN) s = INT8_MIN;
        SET_LANE_U8(r, i, (uint8_t)s);
      }
      setVRBytes(vrt, r); break;
    case 832:  // vaddshs
      for (int i = 0; i < 8; i++) {
        int s = (int)LANE_S16(a, i) + (int)LANE_S16(b, i);
        if (s > INT16_MAX) s = INT16_MAX;
        if (s < INT16_MIN) s = INT16_MIN;
        SET_LANE_U16(r, i, (uint16_t)s);
      }
      setVRBytes(vrt, r); break;
    case 896:  // vaddsws
      for (int i = 0; i < 4; i++) {
        int64_t s = (int64_t)LANE_S32(a, i) + (int64_t)LANE_S32(b, i);
        if (s > INT32_MAX) s = INT32_MAX;
        if (s < INT32_MIN) s = INT32_MIN;
        SET_LANE_U32(r, i, (uint32_t)s);
      }
      setVRBytes(vrt, r); break;

    // === Integer add (saturating, unsigned) ===
    case 512:  // vaddubs
      for (int i = 0; i < 16; i++) {
        unsigned s = (unsigned)LANE_U8(a, i) + (unsigned)LANE_U8(b, i);
        if (s > UINT8_MAX) s = UINT8_MAX;
        SET_LANE_U8(r, i, (uint8_t)s);
      }
      setVRBytes(vrt, r); break;
    case 576:  // vadduhs
      for (int i = 0; i < 8; i++) {
        unsigned s = (unsigned)LANE_U16(a, i) + (unsigned)LANE_U16(b, i);
        if (s > UINT16_MAX) s = UINT16_MAX;
        SET_LANE_U16(r, i, (uint16_t)s);
      }
      setVRBytes(vrt, r); break;
    case 640:  // vadduws
      for (int i = 0; i < 4; i++) {
        uint64_t s = (uint64_t)LANE_U32(a, i) + (uint64_t)LANE_U32(b, i);
        if (s > UINT32_MAX) s = UINT32_MAX;
        SET_LANE_U32(r, i, (uint32_t)s);
      }
      setVRBytes(vrt, r); break;

    // === Integer sub (saturating, signed) ===
    case 1792: // vsubsbs
      for (int i = 0; i < 16; i++) {
        int s = (int)LANE_S8(a, i) - (int)LANE_S8(b, i);
        if (s > INT8_MAX) s = INT8_MAX;
        if (s < INT8_MIN) s = INT8_MIN;
        SET_LANE_U8(r, i, (uint8_t)s);
      }
      setVRBytes(vrt, r); break;
    case 1856: // vsubshs
      for (int i = 0; i < 8; i++) {
        int s = (int)LANE_S16(a, i) - (int)LANE_S16(b, i);
        if (s > INT16_MAX) s = INT16_MAX;
        if (s < INT16_MIN) s = INT16_MIN;
        SET_LANE_U16(r, i, (uint16_t)s);
      }
      setVRBytes(vrt, r); break;

    // === Integer sub (saturating, unsigned) ===
    case 1536: // vsububs
      for (int i = 0; i < 16; i++) {
        int s = (int)LANE_U8(a, i) - (int)LANE_U8(b, i);
        if (s < 0) s = 0;
        SET_LANE_U8(r, i, (uint8_t)s);
      }
      setVRBytes(vrt, r); break;
    case 1600: // vsubuhs
      for (int i = 0; i < 8; i++) {
        int s = (int)LANE_U16(a, i) - (int)LANE_U16(b, i);
        if (s < 0) s = 0;
        SET_LANE_U16(r, i, (uint16_t)s);
      }
      setVRBytes(vrt, r); break;

    // === Average unsigned (rounded: (a+b+1)>>1) ===
    case 1026: // vavgub
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i,
                    ((unsigned)LANE_U8(a, i) + LANE_U8(b, i) + 1) >> 1);
      }
      setVRBytes(vrt, r); break;
    case 1090: // vavguh
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i,
                     ((unsigned)LANE_U16(a, i) + LANE_U16(b, i) + 1) >> 1);
      }
      setVRBytes(vrt, r); break;

    // === Vector multiply per-lane (i32x4.mul) ===
    case 137: { // vmuluwm: per-lane i32 multiply (low 32 bits)
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, LANE_U32(a, i) * LANE_U32(b, i));
      }
      setVRBytes(vrt, r); break;
    }

    // === POWER10 vmulld: per-lane i64 multiply (low 64 bits) ===
    case 457: {
      for (int i = 0; i < 2; i++) {
        uint64_t av = 0, bv = 0;
        for (int j = 0; j < 8; j++) {
          av |= ((uint64_t)a[i * 8 + j]) << (j * 8);
          bv |= ((uint64_t)b[i * 8 + j]) << (j * 8);
        }
        uint64_t prod = av * bv;  // low 64 bits, modulo wrap
        for (int j = 0; j < 8; j++) {
          r[i * 8 + j] = (uint8_t)(prod >> (j * 8));
        }
      }
      setVRBytes(vrt, r); break;
    }

    // === vmule/vmulo* (multiply even/odd lanes, widening) ===
    //
    // All XO values below were verified by disassembling the
    // PPC_vmule*/PPC_vmulo* constants from Assembler-ppc64.h with
    // `as -mppc64 -mlittle` + `objdump -Mpower9 -d`. The previous
    // version had all 12 XO labels swapped with each other's semantic
    // pair (so the JIT's vmulesb was decoded as vmulosb and vice
    // versa), causing i8x16→i16x8 extmul to produce wrong halfwords.
    //
    //   PPC_vmuloub = 0x10000008 → XO=8     vmuloub (LE even-byte pairs)
    //   PPC_vmulouh = 0x10000048 → XO=72    vmulouh
    //   PPC_vmulouw = 0x10000088 → XO=136   vmulouw
    //   PPC_vmulosb = 0x10000108 → XO=264   vmulosb
    //   PPC_vmulosh = 0x10000148 → XO=328   vmulosh
    //   PPC_vmulosw = 0x10000188 → XO=392   vmulosw
    //   PPC_vmuleub = 0x10000208 → XO=520   vmuleub (LE odd-byte pairs)
    //   PPC_vmuleuh = 0x10000248 → XO=584   vmuleuh
    //   PPC_vmuleuw = 0x10000288 → XO=648   vmuleuw
    //   PPC_vmulesb = 0x10000308 → XO=776   vmulesb
    //   PPC_vmulesh = 0x10000348 → XO=840   vmulesh
    //   PPC_vmulesw = 0x10000388 → XO=904   vmulesw
    //
    // Lane indexing on LE storage: "BE-even byte i" is stored at LE
    // byte index (15 - 2i); since our LANE_S8 uses LE byte index, the
    // "BE-even" = "LE-odd" mapping gives `2*i + 1` for vmule, `2*i`
    // for vmulo. The JIT's extmul helpers emit `vmulesb + vmulosb +
    // vmrglh` to pack both halves; getting the semantics swapped here
    // produces the right result register but with the halves in the
    // wrong merge order, breaking extmul.
    case 776: { // vmulesb: signed BE-even byte → halfword (8 results)
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i,
                     (int16_t)LANE_S8(a, 2 * i + 1) *
                     (int16_t)LANE_S8(b, 2 * i + 1));
      }
      setVRBytes(vrt, r); break;
    }
    case 520: { // vmuleub: unsigned BE-even byte → halfword
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i,
                     (uint16_t)LANE_U8(a, 2 * i + 1) *
                     (uint16_t)LANE_U8(b, 2 * i + 1));
      }
      setVRBytes(vrt, r); break;
    }
    case 840: { // vmulesh: signed BE-even halfword → word
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i,
                     (int32_t)LANE_S16(a, 2 * i + 1) *
                     (int32_t)LANE_S16(b, 2 * i + 1));
      }
      setVRBytes(vrt, r); break;
    }
    case 584: { // vmuleuh: unsigned BE-even halfword → word
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i,
                     (uint32_t)LANE_U16(a, 2 * i + 1) *
                     (uint32_t)LANE_U16(b, 2 * i + 1));
      }
      setVRBytes(vrt, r); break;
    }
    case 904: { // vmulesw: signed BE-even word → dword (POWER8)
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i,
                     (int64_t)LANE_S32(a, 2 * i + 1) *
                     (int64_t)LANE_S32(b, 2 * i + 1));
      }
      setVRBytes(vrt, r); break;
    }
    case 648: { // vmuleuw: unsigned BE-even word → dword (POWER8)
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i,
                     (uint64_t)LANE_U32(a, 2 * i + 1) *
                     (uint64_t)LANE_U32(b, 2 * i + 1));
      }
      setVRBytes(vrt, r); break;
    }
    case 264: { // vmulosb: signed BE-odd byte → halfword
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i,
                     (int16_t)LANE_S8(a, 2 * i) *
                     (int16_t)LANE_S8(b, 2 * i));
      }
      setVRBytes(vrt, r); break;
    }
    case 8: { // vmuloub: unsigned BE-odd byte
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i,
                     (uint16_t)LANE_U8(a, 2 * i) *
                     (uint16_t)LANE_U8(b, 2 * i));
      }
      setVRBytes(vrt, r); break;
    }
    case 328: { // vmulosh: signed BE-odd halfword → word
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i,
                     (int32_t)LANE_S16(a, 2 * i) *
                     (int32_t)LANE_S16(b, 2 * i));
      }
      setVRBytes(vrt, r); break;
    }
    case 72: { // vmulouh: unsigned BE-odd halfword → word
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i,
                     (uint32_t)LANE_U16(a, 2 * i) *
                     (uint32_t)LANE_U16(b, 2 * i));
      }
      setVRBytes(vrt, r); break;
    }
    case 392: { // vmulosw: signed BE-odd word
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i,
                     (int64_t)LANE_S32(a, 2 * i) *
                     (int64_t)LANE_S32(b, 2 * i));
      }
      setVRBytes(vrt, r); break;
    }
    case 136: { // vmulouw: unsigned BE-odd word
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i,
                     (uint64_t)LANE_U32(a, 2 * i) *
                     (uint64_t)LANE_U32(b, 2 * i));
      }
      setVRBytes(vrt, r); break;
    }

    // === Per-lane rotate left (vrl{b,h,w,d}) ===
    case 4:    // vrlb
      for (int i = 0; i < 16; i++) {
        uint8_t v = LANE_U8(a, i);
        uint32_t s = LANE_U8(b, i) & 7;
        SET_LANE_U8(r, i, (uint8_t)((v << s) | (v >> ((8 - s) & 7))));
      }
      setVRBytes(vrt, r); break;
    case 68:   // vrlh
      for (int i = 0; i < 8; i++) {
        uint16_t v = LANE_U16(a, i);
        uint32_t s = LANE_U16(b, i) & 15;
        SET_LANE_U16(r, i, (uint16_t)((v << s) | (v >> ((16 - s) & 15))));
      }
      setVRBytes(vrt, r); break;
    case 132:  // vrlw
      for (int i = 0; i < 4; i++) {
        uint32_t v = LANE_U32(a, i);
        uint32_t s = LANE_U32(b, i) & 31;
        SET_LANE_U32(r, i, (v << s) | (v >> ((32 - s) & 31)));
      }
      setVRBytes(vrt, r); break;
    case 196:  // vrld
      for (int i = 0; i < 2; i++) {
        uint64_t v = LANE_U64(a, i);
        uint32_t s = LANE_U64(b, i) & 63;
        SET_LANE_U64(r, i, (v << s) | (v >> ((64 - s) & 63)));
      }
      setVRBytes(vrt, r); break;

    // === Min / Max signed ===
    case 258:  // vmaxsb
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, std::max(LANE_S8(a, i), LANE_S8(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 322:  // vmaxsh
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, std::max(LANE_S16(a, i), LANE_S16(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 386:  // vmaxsw
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, std::max(LANE_S32(a, i), LANE_S32(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 450:  // vmaxsd
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i, std::max(LANE_S64(a, i), LANE_S64(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 770:  // vminsb
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, std::min(LANE_S8(a, i), LANE_S8(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 834:  // vminsh
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, std::min(LANE_S16(a, i), LANE_S16(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 898:  // vminsw
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, std::min(LANE_S32(a, i), LANE_S32(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 962:  // vminsd
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i, std::min(LANE_S64(a, i), LANE_S64(b, i)));
      }
      setVRBytes(vrt, r); break;

    // === Min / Max unsigned ===
    case 2:    // vmaxub
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, std::max(LANE_U8(a, i), LANE_U8(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 66:   // vmaxuh
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, std::max(LANE_U16(a, i), LANE_U16(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 130:  // vmaxuw
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, std::max(LANE_U32(a, i), LANE_U32(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 194:  // vmaxud
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i, std::max(LANE_U64(a, i), LANE_U64(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 514:  // vminub
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, std::min(LANE_U8(a, i), LANE_U8(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 578:  // vminuh
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, std::min(LANE_U16(a, i), LANE_U16(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 642:  // vminuw
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, std::min(LANE_U32(a, i), LANE_U32(b, i)));
      }
      setVRBytes(vrt, r); break;
    case 706:  // vminud
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i, std::min(LANE_U64(a, i), LANE_U64(b, i)));
      }
      setVRBytes(vrt, r); break;

    // === Vector compare (eq, gt signed, gt unsigned, ne POWER9) ===
    //
    // All vcmp* ops set per-lane all-1s on true, all-0s on false. The
    // record form (Rc=1, XO MSB bit set; XO_rec = XO_base + 1024) must
    // additionally write CR6:
    //   CR6.LT = 1 iff ALL lanes are true;
    //   CR6.GT = 0 (always);
    //   CR6.EQ = 1 iff NO lane is true;
    //   CR6.SO = 0 (always).
    // `i8x16.all_true` etc. in wasm rely on CR6.EQ via `mfocrf cr6`; the
    // previous simulator implementation left CR6 untouched, so the
    // predicate was always wrong.
    //
    // Helper: count true lanes by looking at byte 0 of each lane (all
    // bytes within a "true" lane are 0xFF so byte 0 is a sound proxy).
    #define VCMP_DONE(lanes_, lane_bytes_)                                \
      do {                                                                \
        setVRBytes(vrt, r);                                                \
        if (xo >= 1024) {                                                  \
          int numTrue_ = 0;                                                \
          for (int i_ = 0; i_ < (lanes_); i_++) {                          \
            if (r[i_ * (lane_bytes_)] == 0xFF) numTrue_++;                 \
          }                                                                \
          uint8_t field_ = 0;                                              \
          if (numTrue_ == (lanes_)) field_ |= kCRFieldLT;                  \
          if (numTrue_ == 0) field_ |= kCRFieldEQ;                         \
          setCRField(6, field_);                                           \
        }                                                                  \
      } while (0)

    case 6:    // vcmpequb (Rc=0)
    case 1030: // vcmpequb. (record, CR6 updated)
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, LANE_U8(a, i) == LANE_U8(b, i) ? 0xFF : 0);
      }
      VCMP_DONE(16, 1); break;
    case 70:   // vcmpequh
    case 1094: // vcmpequh.
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, LANE_U16(a, i) == LANE_U16(b, i) ? 0xFFFF : 0);
      }
      VCMP_DONE(8, 2); break;
    case 134:  // vcmpequw
    case 1158: // vcmpequw.
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i,
                     LANE_U32(a, i) == LANE_U32(b, i) ? 0xFFFFFFFFu : 0);
      }
      VCMP_DONE(4, 4); break;
    case 199:  // vcmpequd
    case 1223: // vcmpequd.
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i,
                     LANE_U64(a, i) == LANE_U64(b, i)
                         ? UINT64_MAX
                         : 0);
      }
      VCMP_DONE(2, 8); break;

    // === Compare greater-than signed ===
    case 774:  // vcmpgtsb
    case 1798: // vcmpgtsb.
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, LANE_S8(a, i) > LANE_S8(b, i) ? 0xFF : 0);
      }
      VCMP_DONE(16, 1); break;
    case 838:  // vcmpgtsh
    case 1862: // vcmpgtsh.
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, LANE_S16(a, i) > LANE_S16(b, i) ? 0xFFFF : 0);
      }
      VCMP_DONE(8, 2); break;
    case 902:  // vcmpgtsw
    case 1926: // vcmpgtsw.
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i,
                     LANE_S32(a, i) > LANE_S32(b, i) ? 0xFFFFFFFFu : 0);
      }
      VCMP_DONE(4, 4); break;
    case 967:  // vcmpgtsd
    case 1991: // vcmpgtsd.
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i,
                     LANE_S64(a, i) > LANE_S64(b, i) ? UINT64_MAX : 0);
      }
      VCMP_DONE(2, 8); break;

    // === Compare greater-than unsigned ===
    case 518:  // vcmpgtub
    case 1542: // vcmpgtub.
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, LANE_U8(a, i) > LANE_U8(b, i) ? 0xFF : 0);
      }
      VCMP_DONE(16, 1); break;
    case 582:  // vcmpgtuh
    case 1606: // vcmpgtuh.
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, LANE_U16(a, i) > LANE_U16(b, i) ? 0xFFFF : 0);
      }
      VCMP_DONE(8, 2); break;
    case 646:  // vcmpgtuw
    case 1670: // vcmpgtuw.
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i,
                     LANE_U32(a, i) > LANE_U32(b, i) ? 0xFFFFFFFFu : 0);
      }
      VCMP_DONE(4, 4); break;
    case 711:  // vcmpgtud
    case 1735: // vcmpgtud.
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i,
                     LANE_U64(a, i) > LANE_U64(b, i) ? UINT64_MAX : 0);
      }
      VCMP_DONE(2, 8); break;

    // === Splat from immediate (5-bit signed splat into all lanes) ===
    // ISA defines UIM in BE element numbering. For LE storage, BE element i = LE element (N-1-i).
    case 524:  // vspltb: VRT[*] = VRB[BE-byte-UIM]; uimm from VRA field (bits 11..15)
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, LANE_U8(b, 15 - (uimm & 0xF)));
      }
      setVRBytes(vrt, r); break;
    case 588:  // vsplth
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, LANE_U16(b, 7 - (uimm & 0x7)));
      }
      setVRBytes(vrt, r); break;
    case 652:  // vspltw
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, LANE_U32(b, 3 - (uimm & 0x3)));
      }
      setVRBytes(vrt, r); break;

    // === Splat 5-bit signed immediate to all byte lanes ===
    case 780: {  // vspltisb VRT, SIMM5
      int32_t simm5 = (int32_t)((instr->instructionBits() >> 16) & 0x1F);
      if (simm5 & 0x10) simm5 |= ~0x1F;
      uint8_t b = (uint8_t)(int8_t)simm5;
      memset(r, b, 16);
      setVRBytes(vrt, r); break;
    }

    // === Splat 5-bit signed immediate to all halfword lanes ===
    case 844: {  // vspltish VRT, SIMM5
      // SIMM5 occupies bits 11..15 of the instruction (VRA field). It
      // is sign-extended to 16 bits and replicated across all 8 halfword
      // lanes of VRT. Range: [-16, 15].
      int32_t simm5 = (int32_t)((instr->instructionBits() >> 16) & 0x1F);
      if (simm5 & 0x10) simm5 |= ~0x1F;  // sign-extend bit 4
      int16_t hw = (int16_t)simm5;
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, (uint16_t)hw);
      }
      setVRBytes(vrt, r); break;
    }

    // === Splat 5-bit signed immediate to all word lanes ===
    case 908: {  // vspltisw VRT, SIMM5
      int32_t simm5 = (int32_t)((instr->instructionBits() >> 16) & 0x1F);
      if (simm5 & 0x10) simm5 |= ~0x1F;
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, (uint32_t)simm5);
      }
      setVRBytes(vrt, r); break;
    }

    // === Merge (interleave) ===
    //
    // The ISA defines vmrgh* / vmrgl* in BE numbering; the
    // empirical LE storage behaviour is:
    //   vmrgh* VT,VA,VB: for i in 0..N/2-1,
    //     VT.lane_LE[2i]   = VB.lane_LE[(N/2) + i]
    //     VT.lane_LE[2i+1] = VA.lane_LE[(N/2) + i]
    //   vmrgl* VT,VA,VB: for i in 0..N/2-1,
    //     VT.lane_LE[2i]   = VB.lane_LE[i]
    //     VT.lane_LE[2i+1] = VA.lane_LE[i]
    // i.e. the VB operand goes to the even result positions (reversed
    // from what a naïve BE reading would suggest) and the "high" form
    // selects the upper-half of LE storage.
    //
    // Previous implementation had both the operand order swapped AND
    // the high/low halves swapped (consistent with each other, so
    // JIT-only-visible ops that round-tripped through vmrg* happened
    // to produce the right answer, but wasm-visible extmul exposed
    // the bug).
    case 12:   // vmrghb
      for (int i = 0; i < 8; i++) {
        SET_LANE_U8(r, 2 * i, LANE_U8(b, 8 + i));
        SET_LANE_U8(r, 2 * i + 1, LANE_U8(a, 8 + i));
      }
      setVRBytes(vrt, r); break;
    case 76:   // vmrghh
      for (int i = 0; i < 4; i++) {
        SET_LANE_U16(r, 2 * i, LANE_U16(b, 4 + i));
        SET_LANE_U16(r, 2 * i + 1, LANE_U16(a, 4 + i));
      }
      setVRBytes(vrt, r); break;
    case 140:  // vmrghw
      for (int i = 0; i < 2; i++) {
        SET_LANE_U32(r, 2 * i, LANE_U32(b, 2 + i));
        SET_LANE_U32(r, 2 * i + 1, LANE_U32(a, 2 + i));
      }
      setVRBytes(vrt, r); break;
    case 268:  // vmrglb
      for (int i = 0; i < 8; i++) {
        SET_LANE_U8(r, 2 * i, LANE_U8(b, i));
        SET_LANE_U8(r, 2 * i + 1, LANE_U8(a, i));
      }
      setVRBytes(vrt, r); break;
    case 332:  // vmrglh
      for (int i = 0; i < 4; i++) {
        SET_LANE_U16(r, 2 * i, LANE_U16(b, i));
        SET_LANE_U16(r, 2 * i + 1, LANE_U16(a, i));
      }
      setVRBytes(vrt, r); break;
    case 396:  // vmrglw
      for (int i = 0; i < 2; i++) {
        SET_LANE_U32(r, 2 * i, LANE_U32(b, i));
        SET_LANE_U32(r, 2 * i + 1, LANE_U32(a, i));
      }
      setVRBytes(vrt, r); break;

    // === Per-lane shift left (count from VRB, low N bits per element) ===
    case 260:  // vslb
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, LANE_U8(a, i) << (LANE_U8(b, i) & 7));
      }
      setVRBytes(vrt, r); break;
    case 324:  // vslh
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, LANE_U16(a, i) << (LANE_U16(b, i) & 15));
      }
      setVRBytes(vrt, r); break;
    case 388:  // vslw
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, LANE_U32(a, i) << (LANE_U32(b, i) & 31));
      }
      setVRBytes(vrt, r); break;
    case 1476: // vsld
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i, LANE_U64(a, i) << (LANE_U64(b, i) & 63));
      }
      setVRBytes(vrt, r); break;

    // === Per-lane shift right unsigned ===
    case 516:  // vsrb
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, LANE_U8(a, i) >> (LANE_U8(b, i) & 7));
      }
      setVRBytes(vrt, r); break;
    case 580:  // vsrh
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, LANE_U16(a, i) >> (LANE_U16(b, i) & 15));
      }
      setVRBytes(vrt, r); break;
    case 644:  // vsrw
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, LANE_U32(a, i) >> (LANE_U32(b, i) & 31));
      }
      setVRBytes(vrt, r); break;
    case 1732: // vsrd
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i, LANE_U64(a, i) >> (LANE_U64(b, i) & 63));
      }
      setVRBytes(vrt, r); break;

    // === Per-lane shift right algebraic (signed) ===
    case 772:  // vsrab
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i,
                    (uint8_t)(LANE_S8(a, i) >> (LANE_U8(b, i) & 7)));
      }
      setVRBytes(vrt, r); break;
    case 836:  // vsrah
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i,
                     (uint16_t)(LANE_S16(a, i) >> (LANE_U16(b, i) & 15)));
      }
      setVRBytes(vrt, r); break;
    case 900:  // vsraw
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i,
                     (uint32_t)(LANE_S32(a, i) >> (LANE_U32(b, i) & 31)));
      }
      setVRBytes(vrt, r); break;
    case 964:  // vsrad
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i,
                     (uint64_t)(LANE_S64(a, i) >> (LANE_U64(b, i) & 63)));
      }
      setVRBytes(vrt, r); break;

    // === POWER9 per-lane integer negate (subop in VRA field) ===
    // PPC_vnegw = 0x10060602 → XO=0x602=1538, VRA=6
    // PPC_vnegd = 0x10070602 → XO=0x602=1538, VRA=7
    case 1538:
      if (vra == 6) {  // vnegw
        for (int i = 0; i < 4; i++) {
          SET_LANE_U32(r, i, (uint32_t)(-LANE_S32(b, i)));
        }
      } else if (vra == 7) {  // vnegd
        for (int i = 0; i < 2; i++) {
          SET_LANE_U64(r, i, (uint64_t)(-LANE_S64(b, i)));
        }
      } else {
        MOZ_CRASH_UNSAFE_PRINTF("decodeVMX XO=1538: unknown subop %u", vra);
      }
      setVRBytes(vrt, r); break;

    // === POWER10 vextract{b,h,w,d}m (XO=1602=0x642) ===
    // RT (GPR) gets the wasm-spec bitmask in low 16/8/4/2 bits. UIM at
    // bits 11..15 (= sim `vra`) selects lane width: 8=byte, 9=halfword,
    // 10=word, 11=doubleword.
    case 1602: {
      uint64_t result = 0;
      switch (vra) {
        case 8:  // vextractbm: 16 byte lanes
          for (int i = 0; i < 16; i++) {
            if (b[i] & 0x80) result |= (1ULL << i);
          }
          break;
        case 9:  // vextracthm: 8 halfword lanes; MSB lives at byte 2i+1
          for (int i = 0; i < 8; i++) {
            if (b[2 * i + 1] & 0x80) result |= (1ULL << i);
          }
          break;
        case 10:  // vextractwm: 4 word lanes; MSB at byte 4i+3
          for (int i = 0; i < 4; i++) {
            if (b[4 * i + 3] & 0x80) result |= (1ULL << i);
          }
          break;
        case 11:  // vextractdm: 2 dword lanes; MSB at byte 8i+7
          for (int i = 0; i < 2; i++) {
            if (b[8 * i + 7] & 0x80) result |= (1ULL << i);
          }
          break;
        default:
          MOZ_CRASH_UNSAFE_PRINTF("decodeVMX XO=1602: unknown UIM %u", vra);
      }
      // vrt is the GPR target (RT field at bits 6..10).
      setRegister(int(vrt), int64_t(result));
      goto vmx_done;  // Skip the trailing setVRBytes used by VR-targeting ops.
    }

    // === POWER9 vinsertb (XO=781) / vinserth (XO=845) ===
    // Insert byte/halfword from a VR (NOT a GPR) at an immediate byte
    // position UIM (BE).
    //   vinsertb: VRT.byte[UIM]   (BE) ← VRB.byte[7] (BE)
    //   vinserth: VRT.byte[UIM]   (BE) ← VRB.byte[6] (BE)
    //             VRT.byte[UIM+1] (BE) ← VRB.byte[7] (BE)
    // BE byte i ↔ LE byte (15-i). So VRB.byte[6] (BE) = LE byte 9 of
    // VRB, VRB.byte[7] (BE) = LE byte 8. (Byte-pair order matters.)
    case 781:    // vinsertb
    case 845: {  // vinserth
      getVRBytes(vrt, r);  // start from current VRT
      if (xo == 845) {
        // vinserth: copy 2-byte halfword (BE bytes 6..7 of VRB).
        r[15 - uimm]     = b[9];  // BE byte UIM   ← VRB BE byte 6
        r[14 - uimm]     = b[8];  // BE byte UIM+1 ← VRB BE byte 7
      } else {
        // vinsertb: copy a single byte (BE byte 7 of VRB).
        r[15 - uimm]     = b[8];  // BE byte UIM   ← VRB BE byte 7
      }
      setVRBytes(vrt, r); break;
    }

    // === POWER9 vextractub (XO=525) / vextractuh (XO=589) ===
    // Extract one byte/halfword from VRB at immediate BE position UIM
    // and place it at BE byte 7 of VRT, with all other bytes of VRT
    // zeroed. Companion to vinsertb/h; chooses an immediate BE position
    // and lands the result at the low byte of VRT (= low byte of mfvsrd).
    //   vextractub: VRT.byte[7] (BE) ← VRB.byte[UIM] (BE), rest = 0
    //   vextractuh: VRT.byte[6] (BE) ← VRB.byte[UIM]   (BE)
    //               VRT.byte[7] (BE) ← VRB.byte[UIM+1] (BE), rest = 0
    case 525:    // vextractub
    case 589: {  // vextractuh
      memset(r, 0, sizeof(r));
      if (xo == 589) {
        r[9] = b[15 - uimm];  // VRT BE byte 6 ← VRB BE byte UIM
        r[8] = b[14 - uimm];  // VRT BE byte 7 ← VRB BE byte UIM+1
      } else {
        r[8] = b[15 - uimm];  // VRT BE byte 7 ← VRB BE byte UIM
      }
      setVRBytes(vrt, r); break;
    }

    // === POWER10 vinsbrx (XO=783) / vinshrx (XO=847) ===
    // Right-indexed (LE-natural) byte/halfword insert from GPR. RA's
    // low 4 bits supply the byte position (mod 16); for vinshrx the
    // position is also masked to even (& 0xE) so the halfword is
    // 2-byte aligned. RB's low 8 / 16 bits are inserted; other bytes
    // of VRT are unchanged. RA and RB are GPRs (NOT VRs) — sim's
    // pre-fetched `a` and `b` from getVRBytes are unused here.
    case 783:    // vinsbrx
    case 847: {  // vinshrx
      uint64_t ra_val = U64(getRegister(int(vra)));
      uint64_t rb_val = U64(getRegister(int(vrb)));
      getVRBytes(vrt, r);  // start from current VRT
      const bool isHalf = (xo == 847);
      const uint32_t pos = isHalf ? uint32_t(ra_val & 0xEULL)
                                  : uint32_t(ra_val & 0xFULL);
      r[pos] = (uint8_t)(rb_val & 0xFFULL);
      if (isHalf) {
        r[pos + 1] = (uint8_t)((rb_val >> 8) & 0xFFULL);
      }
      setVRBytes(vrt, r); break;
    }

    // === POWER10 vinsw (XO=207) / vinsd (XO=463) ===
    // VRT[UIM*8:UIM*8+N-1] (BE bits) ← RB low N bits, where N = 32 or 64.
    // RB is a GPR (the `vrb` field at sim bits 15..11). UIM is at sim
    // bits 20..16 (= the `uimm` / `vra` decode). Other bytes of VRT are
    // unchanged, so we read VRT first then patch UIM..UIM+(N/8-1).
    case 207:    // vinsw
    case 463: {  // vinsd
      uint64_t rb_val = U64(getRegister(int(vrb)));
      getVRBytes(vrt, r);  // start from current VRT
      const int width = (xo == 463) ? 8 : 4;  // bytes
      // BE byte UIM+i of VRT = LE byte (15 - UIM - i).
      // For vinsd, RB.dword[0] (BE) = bits 56..63 of rb_val (host LSB end
      // of the GPR — recall U64() puts the canonical 64-bit value in a
      // host uint64_t with bit 63 = MSB).
      // For vinsw, source is RB[32:63] = low 32 bits of rb_val.
      uint64_t src = (width == 8) ? rb_val : (rb_val & 0xFFFFFFFFULL);
      const int srcMsbShift = (width * 8) - 8;  // 56 or 24
      for (int i = 0; i < width; i++) {
        r[15 - uimm - i] = (uint8_t)(src >> (srcMsbShift - 8 * i));
      }
      setVRBytes(vrt, r); break;
    }

    // === POWER8+ vbpermq (XO=1356=0x54C): per-byte bit permute ===
    // For each i in 0..15, take VRB BE-byte i (= sim b[15-i]); if its
    // high bit is set, perm[i]=0; else perm[i] = bit at BE position
    // (low 7 bits) of VRA. ISA says perm[0..15] go into VRT.dw[1] low
    // 16 bits, but on real LE silicon the bitmap is observable in dw[0]
    // low 16 bits — i.e., recoverable via mfvsrd. Match that observable
    // behaviour: write the bitmap into sim bytes[8..9] (where mfvsrd
    // reads dw[0] from), zero the rest.
    case 1356: {
      uint8_t perm[16];
      for (int k = 0; k < 16; k++) {
        uint8_t ctl = b[15 - k];
        if (ctl & 0x80) {
          perm[k] = 0;
        } else {
          int p = ctl & 0x7F;
          int le_idx = 15 - (p / 8);
          int bit_in_byte = 7 - (p % 8);
          perm[k] = (a[le_idx] >> bit_in_byte) & 1;
        }
      }
      uint8_t lo = 0, hi = 0;
      for (int k = 0; k < 8; k++) hi = (hi << 1) | perm[k];
      for (int k = 8; k < 16; k++) lo = (lo << 1) | perm[k];
      for (int i = 0; i < 16; i++) r[i] = 0;
      r[8] = lo;
      r[9] = hi;
      setVRBytes(vrt, r); break;
    }

    // VA-form ops vmladduhm (XO=34), vsel (XO=42), vperm (XO=43) are
    // peeled off in the pre-dispatch above (see "VA-form pre-dispatch"
    // comment near the top of this function), since the 11-bit XO
    // mask conflates VRC into the case label.

    // === Unpack high signed (BE-numbering = LE indices 8..15) ===
    // vupkhsb: VRT[i] = sign_extend_to_16(VRA[i+0..7]). On LE storage with
    // BE-named "high" being the low-indexed bytes, vupkhsb sign-extends the
    // low 8 bytes of VRA into 8 halfwords. PPC64LE wasm calls these the
    // "high" lanes per PPC convention; the JIT compensates internally via
    // the vupklsb/vupkhsb swap documented in MacroAssembler-ppc64-inl.h.
    case 526:  // vupkhsb (high signed byte → halfword)
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, (uint16_t)(int16_t)LANE_S8(b, 8 + i));
      }
      setVRBytes(vrt, r); break;
    case 590:  // vupkhsh (high signed halfword → word)
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, (uint32_t)(int32_t)LANE_S16(b, 4 + i));
      }
      setVRBytes(vrt, r); break;
    case 1614: // vupkhsw (high signed word → dword) POWER8+
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i, (uint64_t)(int64_t)LANE_S32(b, 2 + i));
      }
      setVRBytes(vrt, r); break;
    case 654:  // vupklsb (low signed byte → halfword) — PPC LE: takes high lanes
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, (uint16_t)(int16_t)LANE_S8(b, i));
      }
      setVRBytes(vrt, r); break;
    case 718:  // vupklsh (low signed halfword → word)
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i, (uint32_t)(int32_t)LANE_S16(b, i));
      }
      setVRBytes(vrt, r); break;
    case 1742: // vupklsw (low signed word → dword)
      for (int i = 0; i < 2; i++) {
        SET_LANE_U64(r, i, (uint64_t)(int64_t)LANE_S32(b, i));
      }
      setVRBytes(vrt, r); break;

    // === Pack (saturate or modulo) ===
    //
    // vpk* definitions are BE-specified:
    // VT.byte[0..7] = saturate(VA.halfword[0..7]), VT.byte[8..15] =
    // saturate(VB.halfword[0..7]) (BE-numbered throughout). On
    // PPC64LE register storage that inverts to: LE bytes 0-7 = VB's
    // saturated halfwords, LE bytes 8-15 = VA's.
    //
    //   vpkshus = XO 270   (s16 → u8 sat)
    //   vpkshss = XO 398   (s16 → s8 sat)
    //   vpkswus = XO 334   (s32 → u16 sat)
    //   vpkswss = XO 462   (s32 → s16 sat)
    // The sim previously had three of these four labels rotated
    // (270=vpkshss, 334=vpkshus, 398=vpkswus) so every i8x16/i16x8
    // narrow_* call silently used the wrong saturation kind or
    // lane width — vpkshss was completely absent.
    case 398: { // vpkshss (signed halfword → signed byte)
      for (int i = 0; i < 8; i++) {
        int v = LANE_S16(b, i);
        if (v > INT8_MAX) v = INT8_MAX;
        if (v < INT8_MIN) v = INT8_MIN;
        SET_LANE_U8(r, i, (uint8_t)(int8_t)v);
      }
      for (int i = 0; i < 8; i++) {
        int v = LANE_S16(a, i);
        if (v > INT8_MAX) v = INT8_MAX;
        if (v < INT8_MIN) v = INT8_MIN;
        SET_LANE_U8(r, 8 + i, (uint8_t)(int8_t)v);
      }
      setVRBytes(vrt, r); break;
    }
    case 462: { // vpkswss (signed word → signed halfword)
      for (int i = 0; i < 4; i++) {
        int64_t v = LANE_S32(b, i);
        if (v > INT16_MAX) v = INT16_MAX;
        if (v < INT16_MIN) v = INT16_MIN;
        SET_LANE_U16(r, i, (uint16_t)(int16_t)v);
      }
      for (int i = 0; i < 4; i++) {
        int64_t v = LANE_S32(a, i);
        if (v > INT16_MAX) v = INT16_MAX;
        if (v < INT16_MIN) v = INT16_MIN;
        SET_LANE_U16(r, 4 + i, (uint16_t)(int16_t)v);
      }
      setVRBytes(vrt, r); break;
    }
    case 270: { // vpkshus (signed halfword → unsigned byte, sat)
      for (int i = 0; i < 8; i++) {
        int v = LANE_S16(b, i);
        if (v > UINT8_MAX) v = UINT8_MAX;
        if (v < 0) v = 0;
        SET_LANE_U8(r, i, (uint8_t)v);
      }
      for (int i = 0; i < 8; i++) {
        int v = LANE_S16(a, i);
        if (v > UINT8_MAX) v = UINT8_MAX;
        if (v < 0) v = 0;
        SET_LANE_U8(r, 8 + i, (uint8_t)v);
      }
      setVRBytes(vrt, r); break;
    }
    case 334: { // vpkswus (signed word → unsigned halfword, sat)
      for (int i = 0; i < 4; i++) {
        int64_t v = LANE_S32(b, i);
        if (v > UINT16_MAX) v = UINT16_MAX;
        if (v < 0) v = 0;
        SET_LANE_U16(r, i, (uint16_t)v);
      }
      for (int i = 0; i < 4; i++) {
        int64_t v = LANE_S32(a, i);
        if (v > UINT16_MAX) v = UINT16_MAX;
        if (v < 0) v = 0;
        SET_LANE_U16(r, 4 + i, (uint16_t)v);
      }
      setVRBytes(vrt, r); break;
    }

    // === POWER9 compare not-equal (vcmpne{b,h,w}) — Rc=0 and Rc=1 ===
    case 7:    // vcmpneb
    case 1031: // vcmpneb.
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, LANE_U8(a, i) != LANE_U8(b, i) ? 0xFF : 0);
      }
      VCMP_DONE(16, 1); break;
    case 71:   // vcmpneh
    case 1095: // vcmpneh.
      for (int i = 0; i < 8; i++) {
        SET_LANE_U16(r, i, LANE_U16(a, i) != LANE_U16(b, i) ? 0xFFFF : 0);
      }
      VCMP_DONE(8, 2); break;
    case 135:  // vcmpnew
    case 1159: // vcmpnew.
      for (int i = 0; i < 4; i++) {
        SET_LANE_U32(r, i,
                     LANE_U32(a, i) != LANE_U32(b, i) ? 0xFFFFFFFFu : 0);
      }
      VCMP_DONE(4, 4); break;
    #undef VCMP_DONE

    // === Population count per byte (POWER8) ===
    case 1795: { // vpopcntb (XO 0x703 = 1795). VRA field unused.
      for (int i = 0; i < 16; i++) {
        SET_LANE_U8(r, i, (uint8_t)__builtin_popcount(LANE_U8(b, i)));
      }
      setVRBytes(vrt, r); break;
    }

    // === vsldoi: VRT = (VRA || VRB) shifted left by SH bytes (SH at bits 22..25) ===
    case 44: case 45: case 46: case 47: {
      // SH is at bits 22..25 (PPC) → LSB bits 6..9 of the instruction →
      // (instructionBits >> 6) & 0xF. Our XO mask already bottoms-out at
      // bit 0, so extract from the raw instruction.
      uint32_t sh = (instr->instructionBits() >> 6) & 0xF;
      uint8_t cat[32];
      memcpy(cat, a, 16);
      memcpy(cat + 16, b, 16);
      for (int i = 0; i < 16; i++) {
        r[i] = cat[sh + i];
      }
      setVRBytes(vrt, r); break;
    }


    default:
      MOZ_CRASH_UNSAFE_PRINTF(
          "decodeVMX: unimplemented XO=%u (instruction 0x%08x)", xo,
          instr->instructionBits());
  }

vmx_done:
  #undef LANE_U8
  #undef LANE_S8
  #undef LANE_U16
  #undef LANE_S16
  #undef LANE_U32
  #undef LANE_S32
  #undef LANE_U64
  #undef LANE_S64
  #undef SET_LANE_U8
  #undef SET_LANE_U16
  #undef SET_LANE_U32
  #undef SET_LANE_U64
  ;  // empty stmt for label
}

// -----------------------------------------------------------------------------
// decodeVSX: Major opcode 60 (XX1-form, XX2-form)
// mfvsrd, mtvsrd, mtvsrwz, mtvsrws, xscvdpsp, xscvdpspn, xscvspdp,
// xscvspdpn, xxbrd

void Simulator::decodeVSX(SimInstruction* instr) {
  // VSX major opcode 60 covers XX1/XX2/XX3/XX4 forms. We dispatch XX4
  // (xxsel) first because its XO is only 2 bits (at ISA 26-27 = sim
  // bits 5-4), and the XC register field at ISA 21-25 would otherwise
  // produce 32 different 9-bit XO values to enumerate in the switch.
  // Peel off any instruction with XX4 XO=3 (xxsel). No XX2/XX3 op currently
  // emitted by the JIT has sim bits (5,4) == 3.
  if (instr->bits(5, 4) == 3) {
    // xxsel XT,XA,XB,XC  (VA-like XX4-form).
    //   XT[i] = (XA[i] & ~XC[i]) | (XB[i] & XC[i])
    // Register fields: XA/XB/XT per-byte; XC at ISA bits 21-25 (sim
    // bits 10-6) with CX extension at ISA bit 28 (sim bit 3).
    int xa = int(instr->raValue() | (instr->bit(2) << 5));
    int xb = int(instr->rbValue() | (instr->bit(1) << 5));
    int xt = int(instr->rtValue() | (instr->bit(0) << 5));
    int xc = int(instr->bits(10, 6) | (instr->bit(3) << 5));
    uint8_t ab[16], bb[16], cb[16], result[16];
    getVSR128(xa, ab);
    getVSR128(xb, bb);
    getVSR128(xc, cb);
    for (int i = 0; i < 16; i++) {
      result[i] = (uint8_t)((ab[i] & ~cb[i]) | (bb[i] & cb[i]));
    }
    setVSR128(xt, result);
    return;
  }

  // The remaining forms (XX1/XX2/XX3) share a 9-bit XO at ISA bits
  // 21-29 (sim bits 10-2). For XX3 this is (8-bit XO << 1) | AX; for
  // XX2 the full 9 bits are the XO (no AX field).
  uint32_t xo = instr->bits(10, 2);
  uint32_t rt = instr->rtValue();
  uint32_t rb = instr->rbValue();

  switch (xo) {
    // xscvdpsp / xscvdpspn / xscvspdp / xscvspdpn / xxbrd are
    // XX2-form: XT/XB are each 6-bit (5-bit field + TX/BX extension at
    // sim bits 0/1). Post-Phase-2 the JIT emits these with Simd128
    // targets (encoding 32-63), which require the extension bit to
    // select VR-space instead of FPR-space. The previous code used
    // only the 5-bit field, so any VR-space target silently clobbered
    // FPR 0..31 and the post-splat fbits in splatX4 never reached the
    // vector lanes.
    case 265: {
      // xscvdpsp: double→single with sNaN quieting. The ISA says
      // result lands at XT[0:31] (BE word 0 = LE bytes 12..15) and
      // XT[32:127] is "undefined". Real POWER9 silicon actually
      // duplicates the result into BE word 1 as well, so the bytes
      // at LE 8..11 hold the same single. The JIT's
      // replaceLaneFloat32x4 lowering depends on this: it follows
      // xscvdpspn with `xxinsertw …, 12`, which reads XB.word[1]
      // (LE bytes 8..11). Zeroing those bytes here would silently
      // lose the single under sim. Mirror HW.
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16];
      getVSR128(xb, bb);
      // Source double at BE DW0 = LE bytes 8..15 of xb.
      uint64_t dbits = 0;
      for (int i = 0; i < 8; i++) dbits |= ((uint64_t)bb[8 + i]) << (i * 8);
      double frb;
      memcpy(&frb, &dbits, sizeof(frb));
      float result = demoteDoublePreservingNaN(frb);
      uint32_t fbits;
      memcpy(&fbits, &result, sizeof(fbits));
      if ((fbits & 0x7F800000u) == 0x7F800000u && (fbits & 0x007FFFFFu) != 0) {
        fbits |= 0x00400000u;
      }
      uint8_t out[16];
      memset(out, 0, 8);
      // BE word 1 (LE 8..11) and BE word 0 (LE 12..15) both = fbits.
      for (int off : {8, 12}) {
        out[off]     = (uint8_t)(fbits);
        out[off + 1] = (uint8_t)(fbits >> 8);
        out[off + 2] = (uint8_t)(fbits >> 16);
        out[off + 3] = (uint8_t)(fbits >> 24);
      }
      setVSR128(xt, out);
      break;
    }
    case 267: {
      // xscvdpspn: same as xscvdpsp but non-signaling. Same HW-observed
      // word-1 duplication (see xscvdpsp comment above).
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16];
      getVSR128(xb, bb);
      uint64_t dbits = 0;
      for (int i = 0; i < 8; i++) dbits |= ((uint64_t)bb[8 + i]) << (i * 8);
      double frb;
      memcpy(&frb, &dbits, sizeof(frb));
      float result = demoteDoublePreservingNaN(frb);
      uint32_t fbits;
      memcpy(&fbits, &result, sizeof(fbits));
      uint8_t out[16];
      memset(out, 0, 8);
      for (int off : {8, 12}) {
        out[off]     = (uint8_t)(fbits);
        out[off + 1] = (uint8_t)(fbits >> 8);
        out[off + 2] = (uint8_t)(fbits >> 16);
        out[off + 3] = (uint8_t)(fbits >> 24);
      }
      setVSR128(xt, out);
      break;
    }
    case 393: {
      // xvcvdpsp: convert two doubles to two singles, replicating each
      // result across its dword. BE words = [s(BE_dw0), s(BE_dw0),
      // s(BE_dw1), s(BE_dw1)]. SIGNALING form per ISA: sNaN inputs are
      // quieted (high-order fraction bit set in result).
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16], out[16];
      getVSR128(xb, bb);
      uint32_t fbits[2];
      // BE_dw0 = LE bytes 8..15, BE_dw1 = LE bytes 0..7.
      for (int dw = 0; dw < 2; dw++) {
        int leOff = (dw == 0) ? 8 : 0;
        uint64_t dbits = 0;
        for (int i = 0; i < 8; i++) {
          dbits |= ((uint64_t)bb[leOff + i]) << (i * 8);
        }
        double frb;
        memcpy(&frb, &dbits, sizeof(frb));
        float result = demoteDoublePreservingNaN(frb);
        memcpy(&fbits[dw], &result, sizeof(uint32_t));
        if ((fbits[dw] & 0x7F800000u) == 0x7F800000u &&
            (fbits[dw] & 0x007FFFFFu) != 0) {
          fbits[dw] |= 0x00400000u;  // quiet sNaN result
        }
      }
      // LE words: [s(dw1), s(dw1), s(dw0), s(dw0)]
      // (LE word 0 = BE word 3 = s(dw1); LE word 3 = BE word 0 = s(dw0)).
      uint32_t leWords[4] = {fbits[1], fbits[1], fbits[0], fbits[0]};
      for (int w = 0; w < 4; w++) {
        out[w * 4]     = (uint8_t)leWords[w];
        out[w * 4 + 1] = (uint8_t)(leWords[w] >> 8);
        out[w * 4 + 2] = (uint8_t)(leWords[w] >> 16);
        out[w * 4 + 3] = (uint8_t)(leWords[w] >> 24);
      }
      setVSR128(xt, out);
      break;
    }
    case 216:    // xvcvdpsxws: double → signed word, saturating, RTZ (vector)
    case 200: {  // xvcvdpuxws: double → unsigned word, saturating, RTZ (vector)
      //   src1 := XB.dword_BE[0]; src2 := XB.dword_BE[1]
      //   r1 := ConvertDPtoSat(src1); r2 := ConvertDPtoSat(src2)
      //   XT.word_BE[0] := r1; XT.word_BE[1] := r1 (replicated)
      //   XT.word_BE[2] := r2; XT.word_BE[3] := r2 (replicated)
      // Saturation: signed clamps to [INT32_MIN, INT32_MAX] with NaN→INT32_MIN;
      //             unsigned clamps to [0, UINT32_MAX] with NaN→0 and neg→0.
      // BE_dw0 = LE bytes 8..15; BE_dw1 = LE bytes 0..7.
      bool isSigned = (xo == 216);
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16], out[16];
      getVSR128(xb, bb);
      const int srcOffsets[2] = {8, 0};   // BE_dw0 (LE 8..15), BE_dw1 (LE 0..7)
      uint32_t results[2];
      for (int lane = 0; lane < 2; lane++) {
        uint64_t dbits = 0;
        for (int j = 0; j < 8; j++) {
          dbits |= ((uint64_t)bb[srcOffsets[lane] + j]) << (j * 8);
        }
        double dval;
        memcpy(&dval, &dbits, sizeof(dval));
        if (std::isnan(dval)) {
          results[lane] = isSigned ? 0x80000000u : 0u;
        } else if (isSigned) {
          if (dval >= 2147483647.0) {
            results[lane] = 0x7FFFFFFFu;
          } else if (dval <= -2147483648.0) {
            results[lane] = 0x80000000u;
          } else {
            results[lane] = (uint32_t)(int32_t)dval;  // RTZ
          }
        } else {  // unsigned
          if (dval <= 0.0) {
            results[lane] = 0u;
          } else if (dval >= 4294967295.0) {
            results[lane] = 0xFFFFFFFFu;
          } else {
            results[lane] = (uint32_t)dval;  // RTZ
          }
        }
      }
      // Replicated layout: BE words [r1, r1, r2, r2]; in LE bytes
      // [r2, r2, r1, r1] (LE word 0 = BE word 3 = r2, LE word 3 = BE word 0 = r1).
      uint32_t leWords[4] = {results[1], results[1], results[0], results[0]};
      for (int w = 0; w < 4; w++) {
        out[w * 4]     = (uint8_t)leWords[w];
        out[w * 4 + 1] = (uint8_t)(leWords[w] >> 8);
        out[w * 4 + 2] = (uint8_t)(leWords[w] >> 16);
        out[w * 4 + 3] = (uint8_t)(leWords[w] >> 24);
      }
      setVSR128(xt, out);
      break;
    }
    case 248:    // xvcvsxwdp: signed word → double (vector)
    case 232: {  // xvcvuxwdp: unsigned word → double (vector)
      //   src1 := XB.word_BE[0]; src2 := XB.word_BE[2]
      //   XT.dword_BE[0] := Convert(src1); XT.dword_BE[1] := Convert(src2)
      // BE word 0 = LE bytes 12..15; BE word 2 = LE bytes 4..7.
      // Output BE dword 0 = LE bytes 8..15; BE dword 1 = LE bytes 0..7.
      // No NaN handling needed (integer source).
      bool isSigned = (xo == 248);
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16], out[16];
      getVSR128(xb, bb);
      const int srcOffsets[2] = {12, 4};
      const int dstOffsets[2] = {8, 0};
      for (int lane = 0; lane < 2; lane++) {
        uint32_t bits = (uint32_t)bb[srcOffsets[lane]] |
                        ((uint32_t)bb[srcOffsets[lane] + 1] << 8) |
                        ((uint32_t)bb[srcOffsets[lane] + 2] << 16) |
                        ((uint32_t)bb[srcOffsets[lane] + 3] << 24);
        double dval = isSigned ? (double)(int32_t)bits : (double)bits;
        uint64_t dbits;
        memcpy(&dbits, &dval, sizeof(dbits));
        for (int i = 0; i < 8; i++) {
          out[dstOffsets[lane] + i] = (uint8_t)(dbits >> (i * 8));
        }
      }
      setVSR128(xt, out);
      break;
    }
    case 457: {
      // xvcvspdp: convert two singles to two doubles. SIGNALING form
      // per ISA: sNaN inputs are quieted in the result (bit 51 set).
      //   src1 := XB.word_BE[0]; src2 := XB.word_BE[2]
      //   XT.dword_BE[0] := ConvertSPtoDP(src1)
      //   XT.dword_BE[1] := ConvertSPtoDP(src2)
      // BE word 0 = LE bytes 12..15; BE word 2 = LE bytes 4..7.
      // Output BE dword 0 = LE bytes 8..15; BE dword 1 = LE bytes 0..7.
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16], out[16];
      getVSR128(xb, bb);
      // src1 from BE word 0 (LE 12..15), output dword at LE 8..15.
      // src2 from BE word 2 (LE 4..7),   output dword at LE 0..7.
      const int srcOffsets[2] = {12, 4};   // LE byte offsets of word_BE[0], word_BE[2]
      const int dstOffsets[2] = {8, 0};    // LE byte offsets of dword_BE[0], dword_BE[1]
      for (int lane = 0; lane < 2; lane++) {
        uint32_t fbits = (uint32_t)bb[srcOffsets[lane]] |
                         ((uint32_t)bb[srcOffsets[lane] + 1] << 8) |
                         ((uint32_t)bb[srcOffsets[lane] + 2] << 16) |
                         ((uint32_t)bb[srcOffsets[lane] + 3] << 24);
        float fval;
        memcpy(&fval, &fbits, sizeof(fval));
        double dval = promoteFloatPreservingNaN(fval);
        uint64_t dbits;
        memcpy(&dbits, &dval, sizeof(dbits));
        if ((dbits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL &&
            (dbits & 0x000FFFFFFFFFFFFFULL) != 0) {
          dbits |= 0x0008000000000000ULL;  // quiet sNaN result
        }
        for (int i = 0; i < 8; i++) {
          out[dstOffsets[lane] + i] = (uint8_t)(dbits >> (i * 8));
        }
      }
      setVSR128(xt, out);
      break;
    }
    case 329: {
      // xscvspdp: single→double from BE word 0 of XB. SIGNALING form;
      // an sNaN input yields a qNaN result with the high-order
      // fraction bit (quiet bit) set.
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16];
      getVSR128(xb, bb);
      // BE word 0 = LE bytes 12..15 of xb.
      uint32_t fbits = (uint32_t)bb[12] |
                       ((uint32_t)bb[13] << 8) |
                       ((uint32_t)bb[14] << 16) |
                       ((uint32_t)bb[15] << 24);
      float fval;
      memcpy(&fval, &fbits, sizeof(fval));
      double dval = promoteFloatPreservingNaN(fval);
      uint64_t dbits;
      memcpy(&dbits, &dval, sizeof(dbits));
      // Quiet any NaN result (signaling form): set bit 51 of mantissa.
      if ((dbits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL &&
          (dbits & 0x000FFFFFFFFFFFFFULL) != 0) {
        dbits |= 0x0008000000000000ULL;
      }
      uint8_t out[16];
      memset(out, 0, 8);
      for (int i = 0; i < 8; i++) out[8 + i] = (uint8_t)(dbits >> (i * 8));
      setVSR128(xt, out);
      break;
    }
    case 331: {
      // xscvspdpn: non-signaling variant of xscvspdp.
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16];
      getVSR128(xb, bb);
      uint32_t fbits = (uint32_t)bb[12] |
                       ((uint32_t)bb[13] << 8) |
                       ((uint32_t)bb[14] << 16) |
                       ((uint32_t)bb[15] << 24);
      float fval;
      memcpy(&fval, &fbits, sizeof(fval));
      double dval = promoteFloatPreservingNaN(fval);
      uint64_t dbits;
      memcpy(&dbits, &dval, sizeof(dbits));
      uint8_t out[16];
      memset(out, 0, 8);
      for (int i = 0; i < 8; i++) out[8 + i] = (uint8_t)(dbits >> (i * 8));
      setVSR128(xt, out);
      break;
    }
    case 347: {
      // POWER9 XX2-form ops sharing XO=347; disambiguated by the 5-bit
      // A immediate (sim bits 20..16):
      //   A=0  -> xsxexpdp  (extract biased exponent into 11 LSBs of XT.dw0)
      //   A=16 -> xscvhpdp  (FP16 -> FP64)
      //   A=17 -> xscvdphp  (FP64 -> FP16)
      // Half placement: the FP16 value lives at LE bytes 8..9 of
      // the VSR (= BE bits 48..63 of
      // dword[0]), with the rest of dword[0] zeroed. This matches the
      // lxsihzx layout already used by the JIT.
      uint32_t aImm = (instr->instructionBits() >> 16) & 0x1F;
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16], out[16];
      getVSR128(xb, bb);
      memset(out, 0, 16);
      if (aImm == 17) {
        // xscvdphp: read FP64 from BE 0..63 of XB (LE bytes 8..15),
        // convert to FP16, place at LE bytes 8..9 of XT.
        double d;
        memcpy(&d, bb + 8, 8);
        uint16_t h = js::float16(d).toRawBits();
        out[8] = (uint8_t)(h & 0xFF);
        out[9] = (uint8_t)((h >> 8) & 0xFF);
      } else if (aImm == 16) {
        // xscvhpdp: read FP16 from LE bytes 8..9 of XB, convert to FP64,
        // place at LE bytes 8..15 of XT.
        uint16_t h = (uint16_t)bb[8] | ((uint16_t)bb[9] << 8);
        double d = static_cast<double>(js::float16::fromRawBits(h));
        memcpy(out + 8, &d, 8);
      } else if (aImm == 0) {
        // xsxexpdp: read FP64 from LE bytes 8..15 of XB, extract biased
        // exponent (bits 1..11 of the IEEE-754 double = bits 52..62 of
        // the 64-bit pattern), place into XT.dw0 with rest zeroed.
        uint64_t bits = 0;
        for (int i = 0; i < 8; i++) bits |= uint64_t(bb[8 + i]) << (i * 8);
        uint64_t exp = (bits >> 52) & 0x7FF;
        for (int i = 0; i < 8; i++) out[8 + i] = (uint8_t)(exp >> (i * 8));
      } else {
        MOZ_CRASH_UNSAFE_PRINTF(
            "decodeVSX XO=347 with unexpected A=%u (instr 0x%08x)",
            aImm, instr->instructionBits());
      }
      setVSR128(xt, out);
      break;
    }
    case 475: {
      // xxbrd: byte-reverse each doubleword.
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16], out[16];
      getVSR128(xb, bb);
      for (int i = 0; i < 8; i++) out[i] = bb[7 - i];
      for (int i = 0; i < 8; i++) out[8 + i] = bb[15 - i];
      setVSR128(xt, out);
      break;
    }

    // === XX3-form scalar: xsmaxjdp / xsminjdp (POWER9) ===
    //
    // xs{max,min}jdp XT, XA, XB. Scalar inputs at BE bits 0..63 of
    // XA / XB (= LE bytes 8..15); result lands at BE 0..63 of XT
    // (upper bits "undefined" per ISA).
    //
    // Semantics match ECMA-262 Math.{max,min} / wasm f64.{max,min}:
    //   - NaN: if A is NaN return A; else if B is NaN return B. sNaN
    //     payload preserved bit-for-bit (NOT quieted).
    //   - ±0 tie: signed-zero ordering. xsmaxjdp returns +0 for any
    //     mix of (-0, +0); xsminjdp returns -0.
    //   - Otherwise: standard IEEE max / min.
    case 288: case 289:    // xsmaxjdp  (XO8=144 → 9-bit 288/289)
    case 304: case 305: {  // xsminjdp  (XO8=152 → 9-bit 304/305)
      int xa = int(instr->raValue() | (instr->bit(2) << 5));
      int xb = int(instr->rbValue() | (instr->bit(1) << 5));
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t ab[16], bb[16], out[16];
      getVSR128(xa, ab);
      getVSR128(xb, bb);
      double a, b;
      memcpy(&a, ab + 8, 8);
      memcpy(&b, bb + 8, 8);
      bool isMax = (xo >> 1) == 144;
      double r;
      if (std::isnan(a)) {
        r = a;
      } else if (std::isnan(b)) {
        r = b;
      } else if (a == 0.0 && b == 0.0) {
        // Signed-zero ordering: max picks +0, min picks -0.
        if (isMax) {
          r = std::signbit(a) ? b : a;
        } else {
          r = std::signbit(a) ? a : b;
        }
      } else {
        r = isMax ? std::max(a, b) : std::min(a, b);
      }
      memset(out, 0, 8);
      memcpy(out + 8, &r, 8);
      setVSR128(xt, out);
      break;
    }

    // --- VSX XX3-form: xxpermdi ---
    //
    // xxpermdi XT, XA, XB, DM:
    //   XT.DW0 = XA.DW(DM[0])
    //   XT.DW1 = XB.DW(DM[1])
    // In BE, DW0 is MSB-side, DW1 is LSB-side. On PPC64LE register
    // storage, DW0 = LE bytes 8-15 and DW1 = LE bytes 0-7. The sim's
    // previous implementation used the reversed "DW0 = LE 0-7"
    // convention which cancelled for self-swap round-trips but
    // produced wrong halves when chained with ISA-correct ops
    // (mtvsrd, xxspltw, mfvsrd).
    case 20: case 21:       // xxpermdi DM=0
    case 84: case 85:       // xxpermdi DM=1
    case 148: case 149:     // xxpermdi DM=2 (= xxswapd when XA==XB)
    case 212: case 213: {   // xxpermdi DM=3
      uint8_t dm_hi = (xo >> 7) & 1;  // DM[0]
      uint8_t dm_lo = (xo >> 6) & 1;  // DM[1]
      int xa = int(instr->raValue() | (instr->bit(2) << 5));
      int xb = int(instr->rbValue() | (instr->bit(1) << 5));
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t xa_bytes[16], xb_bytes[16], result[16];
      getVSR128(xa, xa_bytes);
      getVSR128(xb, xb_bytes);
      // DW0 in LE storage is bytes 8-15; DW1 is bytes 0-7.
      //   XT.DW0 (result[8..15]) = XA.DW(dm_hi)
      //   XT.DW1 (result[0..7])  = XB.DW(dm_lo)
      // DW(0) is at LE 8, DW(1) is at LE 0.
      memcpy(result + 8, xa_bytes + (dm_hi ? 0 : 8), 8);
      memcpy(result,     xb_bytes + (dm_lo ? 0 : 8), 8);
      setVSR128(xt, result);
      break;
    }

    // --- VSX logical (XX3-form, primary opcode 60) ---
    //
    // Each takes two 6-bit VSR sources XA/XB and writes 6-bit VSR
    // destination XT. 8-bit ISA XO at bits 21-28; our
    // 9-bit XO extraction (bits 10:2) includes the AX bit at position 0,
    // so each op appears as two consecutive values (AX=0 and AX=1).
    //
    //   xxland XT,XA,XB     XO=130  (9-bit: 260, 261)  XT = XA & XB
    //   xxlandc XT,XA,XB    XO=138  (276, 277)         XT = XA & ~XB
    //   xxlor XT,XA,XB      XO=146  (292, 293)         XT = XA | XB
    //   xxlxor XT,XA,XB     XO=154  (308, 309)         XT = XA ^ XB
    //   xxlnor XT,XA,XB     XO=162  (324, 325)         XT = ~(XA | XB)
    //   xxlorc XT,XA,XB     XO=170  (340, 341)         XT = XA | ~XB
    //   xxlnand XT,XA,XB    XO=178  (356, 357)         XT = ~(XA & XB)
    //   xxleqv XT,XA,XB     XO=186  (372, 373)         XT = ~(XA ^ XB)
    //
    // The encoding constants in Assembler-ppc64.h match: PPC_xxlor=0xF0000490
    // has bits 4,7,10 set in its base (XO=146 in the 8-bit field), which
    // under the simulator's 9-bit extraction gives 2*146=292 (AX=0 default).
    case 260: case 261:  // xxland
    case 276: case 277:  // xxlandc
    case 292: case 293:  // xxlor
    case 308: case 309:  // xxlxor
    case 324: case 325:  // xxlnor
    case 340: case 341:  // xxlorc
    case 356: case 357:  // xxlnand
    case 372: case 373:  // xxleqv
    {
      int xa = int(instr->raValue() | (instr->bit(2) << 5));
      int xb = int(instr->rbValue() | (instr->bit(1) << 5));
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t a_bytes[16], b_bytes[16], result[16];
      getVSR128(xa, a_bytes);
      getVSR128(xb, b_bytes);
      // Dispatch on the 8-bit ISA XO (ignoring AX bit at position 0).
      uint32_t xo8 = xo >> 1;
      for (int i = 0; i < 16; i++) {
        uint8_t a = a_bytes[i], b = b_bytes[i];
        switch (xo8) {
          case 130: result[i] = a & b;        break;  // xxland
          case 138: result[i] = a & ~b;       break;  // xxlandc
          case 146: result[i] = a | b;        break;  // xxlor
          case 154: result[i] = a ^ b;        break;  // xxlxor
          case 162: result[i] = (uint8_t)~(a | b);  break;  // xxlnor
          case 170: result[i] = a | (uint8_t)~b;    break;  // xxlorc
          case 178: result[i] = (uint8_t)~(a & b);  break;  // xxlnand
          case 186: result[i] = (uint8_t)~(a ^ b);  break;  // xxleqv
        }
      }
      setVSR128(xt, result);
      break;
    }

    // === XX2-form: xxspltw (splat word from VRB[UIM] to all 4 lanes) ===
    //
    // xxspltw: UIM selects one of four words in BE numbering. UIM=0
    // → BE word 0 (MSB side of the 128 bits). On PPC64LE register
    // storage that maps to LE word (3 - UIM). With the input
    // {0x11111111, 0x22222222, 0x33333333, 0x44444444}: UIM=0
    // splats 0x44444444 (= LE word 3), UIM=3 splats 0x11111111
    // (= LE word 0). The JIT emits xxspltw UIM=1 after mtvsrd on the
    // POWER8 splatX4 path — mtvsrd puts the GPR's low 32 bits in BE
    // word 1 (= LE word 2 on HW), so xxspltw UIM=1 picks up exactly
    // that word and splats it to every lane.
    case 164: {  // xxspltw
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint32_t uim = (instr->instructionBits() >> 16) & 0x3;
      uint32_t leIdx = 3 - uim;  // BE word UIM → LE word (3-UIM)
      uint8_t bb[16], result[16];
      getVSR128(xb, bb);
      uint32_t word = (uint32_t)bb[leIdx * 4] |
                      ((uint32_t)bb[leIdx * 4 + 1] << 8) |
                      ((uint32_t)bb[leIdx * 4 + 2] << 16) |
                      ((uint32_t)bb[leIdx * 4 + 3] << 24);
      for (int i = 0; i < 4; i++) {
        result[i * 4]     = (uint8_t)(word & 0xFF);
        result[i * 4 + 1] = (uint8_t)((word >> 8) & 0xFF);
        result[i * 4 + 2] = (uint8_t)((word >> 16) & 0xFF);
        result[i * 4 + 3] = (uint8_t)((word >> 24) & 0xFF);
      }
      setVSR128(xt, result);
      break;
    }

    // === XX2-form: xxextractuw (extract word at BE byte UIM, place at BE word 1) ===
    //
    // xxextractuw XT, XB, UIM:
    //   Bytes [4:7] of XT receive bytes [UIM:UIM+3] of XB. Bytes [0:3]
    //   and [8:15] of XT are set to zero.
    // UIM ∈ {0, 4, 8, 12} (caller responsible for alignment).
    // BE byte i ↔ LE byte (15-i), so the word at XB BE bytes UIM..UIM+3
    // sits at XB LE bytes (12-UIM)..(15-UIM), and lands at XT LE bytes
    // 8..11 (= XT BE word 1).
    case 165: {  // xxextractuw
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint32_t uim = (instr->instructionBits() >> 16) & 0xF;
      uint8_t bb[16], result[16];
      getVSR128(xb, bb);
      memset(result, 0, sizeof(result));
      // result.LE[8..11] = XB.LE[(12-UIM)..(15-UIM)] (preserves byte order).
      memcpy(result + 8, bb + (12 - uim), 4);
      setVSR128(xt, result);
      break;
    }

    case 180: {
      // xxspltib XT, IMM8 (POWER9, ISA 3.0): splat 8-bit immediate to
      // all 16 bytes of XT. The encoder writes `imm8 << 11`, so IMM8
      // occupies LE bits 11..18; TX bit at LE bit 0 selects upper VSR.
      uint32_t imm8 = instr->bits(18, 11);
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      uint8_t xt_bytes[16];
      memset(xt_bytes, (uint8_t)imm8, 16);
      setVSR128(xt, xt_bytes);
      break;
    }
    case 181: {
      // xxinsertw XT, XB, UIM (POWER9, ISA 3.0): copy XB[32..63] (the
      // low 32 bits of XB's BE doubleword 0, which lives at LE bytes
      // 8-11 of XB) into XT at BE byte position UIM. UIM ∈ {0,4,8,12};
      // dest occupies XT LE bytes (12-UIM)..(15-UIM). Other bytes of
      // XT are preserved. UIM at PPC bits 11-15 = LE bits 16-20; TX/BX
      // at LE bits 0/1.
      uint32_t uim = instr->bits(20, 16);
      int xt = int(instr->rtValue() | (instr->bit(0) << 5));
      int xb = int(instr->rbValue() | (instr->bit(1) << 5));
      uint8_t xb_bytes[16], xt_bytes[16];
      getVSR128(xb, xb_bytes);
      getVSR128(xt, xt_bytes);
      memcpy(xt_bytes + (12 - uim), xb_bytes + 8, 4);
      setVSR128(xt, xt_bytes);
      break;
    }

    // === XX2-form: xvabssp / xvabsdp (vector absolute value) ===
    case 408: case 409: case 410: case 411: {  // xvabssp + AX/BX bits
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16], result[16];
      getVSR128(xb, bb);
      for (int i = 0; i < 4; i++) {
        uint32_t bits = (uint32_t)bb[i * 4] |
                        ((uint32_t)bb[i * 4 + 1] << 8) |
                        ((uint32_t)bb[i * 4 + 2] << 16) |
                        ((uint32_t)bb[i * 4 + 3] << 24);
        bits &= 0x7FFFFFFFu;  // clear sign bit
        result[i * 4]     = (uint8_t)(bits & 0xFF);
        result[i * 4 + 1] = (uint8_t)((bits >> 8) & 0xFF);
        result[i * 4 + 2] = (uint8_t)((bits >> 16) & 0xFF);
        result[i * 4 + 3] = (uint8_t)((bits >> 24) & 0xFF);
      }
      setVSR128(xt, result);
      break;
    }
    case 472: case 473: case 474: {            // xvabsdp (475 used by xxbrd)
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16], result[16];
      getVSR128(xb, bb);
      for (int i = 0; i < 2; i++) {
        uint64_t bits = 0;
        for (int k = 0; k < 8; k++) bits |= ((uint64_t)bb[i * 8 + k]) << (k * 8);
        bits &= 0x7FFFFFFFFFFFFFFFULL;
        for (int k = 0; k < 8; k++) result[i * 8 + k] = (uint8_t)((bits >> (k * 8)) & 0xFF);
      }
      setVSR128(xt, result);
      break;
    }

    // === XX2-form unary vector float ops (single XB operand, no AX) ===
    //
    // Encoding: opcode 60, bits 6-10=XT, 11-15 reserved, 16-20=XB,
    // 21-29 = 9-bit XO (full field), 30=BX, 31=TX. Extraction gives us
    // xo = XO9 directly (no AX bit). Every op below has a unique XO9.
    //
    //   xvsqrtsp  XO9=139  PPC_xvsqrtsp=0xF000022C
    //   xvsqrtdp  XO9=203  PPC_xvsqrtdp=0xF000032C
    //   xvnegsp   XO9=441  PPC_xvnegsp=0xF00006E4
    //   xvnegdp   XO9=505  PPC_xvnegdp=0xF00007E4
    //   xvrspip   XO9=169  PPC_xvrspip=0xF00002A4   (round +inf = ceil)
    //   xvrspiz   XO9=153  PPC_xvrspiz=0xF0000264   (round toward 0 = trunc)
    //   xvrspim   XO9=185  PPC_xvrspim=0xF00002E4   (round -inf = floor)
    //   xvrspic   XO9=171  PPC_xvrspic=0xF00002AC   (round per FPSCR)
    //   xvrdpip   XO9=233  PPC_xvrdpip=0xF00003A4
    //   xvrdpiz   XO9=217  PPC_xvrdpiz=0xF0000364
    //   xvrdpim   XO9=249  PPC_xvrdpim=0xF00003E4
    //   xvrdpic   XO9=235  PPC_xvrdpic=0xF00003AC
    //   xvcvspsxws XO9=152 PPC_xvcvspsxws=0xF0000260  (f32 → s32, sat)
    //   xvcvspuxws XO9=136 PPC_xvcvspuxws=0xF0000220  (f32 → u32, sat)
    //   xvcvsxwsp XO9=184  PPC_xvcvsxwsp=0xF00002E0   (s32 → f32)
    //   xvcvuxwsp XO9=168  PPC_xvcvuxwsp=0xF00002A0   (u32 → f32)
    case 139: case 203:     // xvsqrtsp / xvsqrtdp
    case 441: case 505:     // xvnegsp / xvnegdp
    case 169: case 233:     // xvrspip / xvrdpip (ceil)
    case 153: case 217:     // xvrspiz / xvrdpiz (trunc)
    case 185: case 249:     // xvrspim / xvrdpim (floor)
    case 171: case 235:     // xvrspic / xvrdpic (round-to-nearest)
    case 136: case 152:     // xvcvspuxws / xvcvspsxws
    case 168: case 184: {   // xvcvuxwsp / xvcvsxwsp
      int xt = int(rt | (instr->bit(0) << 5));
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t bb[16], result[16];
      getVSR128(xb, bb);
      bool isSp = (xo == 139 || xo == 441 || xo == 169 || xo == 153 ||
                   xo == 185 || xo == 171 || xo == 136 || xo == 152 ||
                   xo == 168 || xo == 184);
      auto getF32 = [](uint8_t* buf, int i) -> float {
        uint32_t b = (uint32_t)buf[i * 4] |
                     ((uint32_t)buf[i * 4 + 1] << 8) |
                     ((uint32_t)buf[i * 4 + 2] << 16) |
                     ((uint32_t)buf[i * 4 + 3] << 24);
        float f; memcpy(&f, &b, sizeof(f)); return f;
      };
      auto setF32 = [](uint8_t* buf, int i, float f) {
        uint32_t b; memcpy(&b, &f, sizeof(b));
        buf[i*4]=(uint8_t)b; buf[i*4+1]=(uint8_t)(b>>8);
        buf[i*4+2]=(uint8_t)(b>>16); buf[i*4+3]=(uint8_t)(b>>24);
      };
      auto getF64 = [](uint8_t* buf, int i) -> double {
        uint64_t b = 0;
        for (int k=0;k<8;k++) b |= ((uint64_t)buf[i*8+k])<<(k*8);
        double d; memcpy(&d, &b, sizeof(d)); return d;
      };
      auto setF64 = [](uint8_t* buf, int i, double d) {
        uint64_t b; memcpy(&b, &d, sizeof(b));
        for (int k=0;k<8;k++) buf[i*8+k]=(uint8_t)(b>>(k*8));
      };
      // Integer lane read/write (used by conversion ops).
      auto setU32 = [](uint8_t* buf, int i, uint32_t v) {
        buf[i*4]=(uint8_t)v; buf[i*4+1]=(uint8_t)(v>>8);
        buf[i*4+2]=(uint8_t)(v>>16); buf[i*4+3]=(uint8_t)(v>>24);
      };
      // Saturated float→int conversion per Power ISA v3.0B: input NaN maps
      // to 0; out-of-range saturates to the extreme of the destination type.
      auto fp2sxw = [](double f) -> uint32_t {
        if (std::isnan(f)) return 0;
        if (f >= (double)INT32_MAX) return (uint32_t)INT32_MAX;
        if (f <= (double)INT32_MIN) return (uint32_t)INT32_MIN;
        return (uint32_t)(int32_t)std::trunc(f);
      };
      auto fp2uxw = [](double f) -> uint32_t {
        if (std::isnan(f)) return 0;
        if (f >= (double)UINT32_MAX) return UINT32_MAX;
        if (f <= 0.0) return 0;
        return (uint32_t)std::trunc(f);
      };

      if (isSp) {
        for (int i = 0; i < 4; i++) {
          float v = getF32(bb, i);
          float out = 0.0f;
          uint32_t iout = 0;
          bool isInt = false;
          switch (xo) {
            case 139: out = std::sqrt(v); break;                // xvsqrtsp
            case 441: out = -v; break;                          // xvnegsp
            case 169: out = std::ceil(v); break;                // xvrspip
            case 153: out = std::trunc(v); break;               // xvrspiz
            case 185: out = std::floor(v); break;               // xvrspim
            case 171: out = std::nearbyint(v); break;           // xvrspic
            case 152: iout = fp2sxw(v); isInt = true; break;    // xvcvspsxws
            case 136: iout = fp2uxw(v); isInt = true; break;    // xvcvspuxws
            case 184: {                                          // xvcvsxwsp
              uint32_t bits = (uint32_t)bb[i*4] |
                              ((uint32_t)bb[i*4+1]<<8) |
                              ((uint32_t)bb[i*4+2]<<16) |
                              ((uint32_t)bb[i*4+3]<<24);
              out = (float)(int32_t)bits;
              break;
            }
            case 168: {                                          // xvcvuxwsp
              uint32_t bits = (uint32_t)bb[i*4] |
                              ((uint32_t)bb[i*4+1]<<8) |
                              ((uint32_t)bb[i*4+2]<<16) |
                              ((uint32_t)bb[i*4+3]<<24);
              out = (float)(uint32_t)bits;
              break;
            }
          }
          if (isInt) setU32(result, i, iout);
          else setF32(result, i, out);
        }
      } else {
        for (int i = 0; i < 2; i++) {
          double v = getF64(bb, i);
          double out = 0.0;
          switch (xo) {
            case 203: out = std::sqrt(v); break;                // xvsqrtdp
            case 505: out = -v; break;                          // xvnegdp
            case 233: out = std::ceil(v); break;                // xvrdpip
            case 217: out = std::trunc(v); break;               // xvrdpiz
            case 249: out = std::floor(v); break;               // xvrdpim
            case 235: out = std::nearbyint(v); break;           // xvrdpic
          }
          setF64(result, i, out);
        }
      }
      setVSR128(xt, result);
      break;
    }

    // === XX3-form vector float compare (eq, gt, ge) ===
    // The wasm SIMD compares emit these and use the result as a bitmask.
    // Per Power ISA: result is all-1s for true lanes, all-0s for false
    // (for the non-recording form; bit 0 of XO selects record form which
    // we don't model — wasm doesn't read CR6 here).
    // Encodings:
    //   0xF0000218 xvcmpeqsp (XO8=67) → XO9 = 134/135 (+AX).
    //   0xF0000258 xvcmpgtsp (XO8=75) → XO9 = 150/151.
    //   0xF0000298 xvcmpgesp (XO8=83) → XO9 = 166/167.
    //   0xF0000318 xvcmpeqdp (XO8=99) → XO9 = 198/199.
    //   0xF0000358 xvcmpgtdp (XO8=107) → XO9 = 214/215.
    //   0xF0000398 xvcmpgedp (XO8=115) → XO9 = 230/231.
    // Rc=1 record form flips ISA bit 21 (sim bit 10), yielding XO9+256
    // (not adjacent to the Rc=0 slot). wasm never emits the record form.
    case 134: case 135:    // xvcmpeqsp (XO8=67)
    case 198: case 199:    // xvcmpeqdp (XO8=99)
    case 150: case 151:    // xvcmpgtsp (XO8=75)
    case 214: case 215:    // xvcmpgtdp (XO8=107)
    case 166: case 167:    // xvcmpgesp (XO8=83)
    case 230: case 231: {  // xvcmpgedp (XO8=115)
      int xt = int(rt | (instr->bit(0) << 5));
      uint32_t ra = instr->raValue();
      int xa = int(ra | ((instr->instructionBits() >> 2) & 1) << 5);
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t ab[16], bb[16], result[16];
      getVSR128(xa, ab);
      getVSR128(xb, bb);
      uint32_t op8 = xo >> 1;  // canonical 8-bit XO
      bool isF32 = (op8 == 67 || op8 == 75 || op8 == 83);
      bool isEq  = (op8 == 67 || op8 == 99);
      bool isGt  = (op8 == 75 || op8 == 107);
      bool isGe  = (op8 == 83 || op8 == 115);
      (void)isGe;
      auto cmpF32 = [&](int i) -> bool {
        uint32_t aBits = (uint32_t)ab[i * 4] |
                         ((uint32_t)ab[i * 4 + 1] << 8) |
                         ((uint32_t)ab[i * 4 + 2] << 16) |
                         ((uint32_t)ab[i * 4 + 3] << 24);
        uint32_t bBits = (uint32_t)bb[i * 4] |
                         ((uint32_t)bb[i * 4 + 1] << 8) |
                         ((uint32_t)bb[i * 4 + 2] << 16) |
                         ((uint32_t)bb[i * 4 + 3] << 24);
        float fa, fb;
        memcpy(&fa, &aBits, sizeof(fa));
        memcpy(&fb, &bBits, sizeof(fb));
        if (isEq) return fa == fb;
        if (isGt) return fa > fb;
        return fa >= fb;
      };
      auto cmpF64 = [&](int i) -> bool {
        uint64_t aBits = 0, bBits = 0;
        for (int k = 0; k < 8; k++) aBits |= ((uint64_t)ab[i * 8 + k]) << (k * 8);
        for (int k = 0; k < 8; k++) bBits |= ((uint64_t)bb[i * 8 + k]) << (k * 8);
        double fa, fb;
        memcpy(&fa, &aBits, sizeof(fa));
        memcpy(&fb, &bBits, sizeof(fb));
        if (isEq) return fa == fb;
        if (isGt) return fa > fb;
        return fa >= fb;
      };
      if (isF32) {
        for (int i = 0; i < 4; i++) {
          uint32_t mask = cmpF32(i) ? 0xFFFFFFFFu : 0;
          for (int k = 0; k < 4; k++) {
            result[i * 4 + k] = (uint8_t)((mask >> (k * 8)) & 0xFF);
          }
        }
      } else {
        for (int i = 0; i < 2; i++) {
          uint64_t mask = cmpF64(i) ? UINT64_MAX : 0;
          for (int k = 0; k < 8; k++) {
            result[i * 8 + k] = (uint8_t)((mask >> (k * 8)) & 0xFF);
          }
        }
      }
      setVSR128(xt, result);
      break;
    }

    // === XX3-form vector float arithmetic ===
    // Encoding: bits 6-10=XT, 11-15=XA, 16-20=XB, 21-28=XO (8 bits), 29=AX,
    // 30=BX, 31=TX. We dispatched above using `bits(10, 2)` which is bits
    // 21-29 (9 bits) — that includes the AX register-extension bit, which
    // changes for every XA in {0..31} vs {32..63}. To match all 4
    // (AX,BX) combinations of an XX3 op we use `case xo3 | 0|1|2|3` where
    // xo3 = (8-bit XO) << 1 (because XO occupies bits 1..8 of our 9-bit
    // extraction). Helper macro: each case covers four labels.
    #define XX3_CASE_BASE(name) \
      case ((name) | 0): case ((name) | 1):
    case 128:  case 129:  // xvaddsp: 4 × f32 add (XO=64 → bits 1..8 = 128)
    case 192:  case 193:  // xvadddp
    case 144:  case 145:  // xvsubsp
    case 208:  case 209:  // xvsubdp
    case 160:  case 161:  // xvmulsp
    case 224:  case 225:  // xvmuldp
    case 176:  case 177:  // xvdivsp
    case 240:  case 241:  // xvdivdp
    case 384:  case 385:  // xvmaxsp
    case 448:  case 449:  // xvmaxdp
    case 400:  case 401:  // xvminsp
    case 464:  case 465:  // xvmindp
    {
      // Re-extract the canonical 8-bit XX3 XO.
      uint32_t xo3 = (xo >> 1);
      (void)xo3;
      int xt = int(rt | (instr->bit(0) << 5));
      uint32_t ra = instr->raValue();
      int xa = int(ra | ((instr->instructionBits() >> 2) & 1) << 5);
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t ab[16], bb[16], rb_bytes[16];
      getVSR128(xa, ab);
      getVSR128(xb, bb);

      auto getF32 = [](uint8_t* buf, int i) -> float {
        uint32_t bits = (uint32_t)buf[i * 4] |
                        ((uint32_t)buf[i * 4 + 1] << 8) |
                        ((uint32_t)buf[i * 4 + 2] << 16) |
                        ((uint32_t)buf[i * 4 + 3] << 24);
        float f;
        memcpy(&f, &bits, sizeof(f));
        return f;
      };
      auto setF32 = [](uint8_t* buf, int i, float f) {
        uint32_t bits;
        memcpy(&bits, &f, sizeof(bits));
        buf[i * 4]     = (uint8_t)(bits & 0xFF);
        buf[i * 4 + 1] = (uint8_t)((bits >> 8) & 0xFF);
        buf[i * 4 + 2] = (uint8_t)((bits >> 16) & 0xFF);
        buf[i * 4 + 3] = (uint8_t)((bits >> 24) & 0xFF);
      };
      auto getF64 = [](uint8_t* buf, int i) -> double {
        uint64_t bits = 0;
        for (int k = 0; k < 8; k++) bits |= ((uint64_t)buf[i * 8 + k]) << (k * 8);
        double d;
        memcpy(&d, &bits, sizeof(d));
        return d;
      };
      auto setF64 = [](uint8_t* buf, int i, double d) {
        uint64_t bits;
        memcpy(&bits, &d, sizeof(bits));
        for (int k = 0; k < 8; k++) buf[i * 8 + k] = (uint8_t)((bits >> (k * 8)) & 0xFF);
      };

      // Dispatch on the canonical 8-bit XX3 XO (bits 21..28 PPC = xo>>1).
      switch (xo3) {
        case 64:  for (int i = 0; i < 4; i++) setF32(rb_bytes, i, getF32(ab, i) + getF32(bb, i)); break;  // xvaddsp
        case 96:  for (int i = 0; i < 2; i++) setF64(rb_bytes, i, getF64(ab, i) + getF64(bb, i)); break;  // xvadddp
        case 72:  for (int i = 0; i < 4; i++) setF32(rb_bytes, i, getF32(ab, i) - getF32(bb, i)); break;  // xvsubsp
        case 104: for (int i = 0; i < 2; i++) setF64(rb_bytes, i, getF64(ab, i) - getF64(bb, i)); break;  // xvsubdp
        case 80:  for (int i = 0; i < 4; i++) setF32(rb_bytes, i, getF32(ab, i) * getF32(bb, i)); break;  // xvmulsp
        case 112: for (int i = 0; i < 2; i++) setF64(rb_bytes, i, getF64(ab, i) * getF64(bb, i)); break;  // xvmuldp
        case 88:  for (int i = 0; i < 4; i++) setF32(rb_bytes, i, getF32(ab, i) / getF32(bb, i)); break;  // xvdivsp
        case 120: for (int i = 0; i < 2; i++) setF64(rb_bytes, i, getF64(ab, i) / getF64(bb, i)); break;  // xvdivdp
        // xvmin{sp,dp} / xvmax{sp,dp}:
        //   If both operands are NaN, result is the NaN from XA.
        //   If exactly one operand is NaN, result is the NON-NaN operand.
        //   For 0 / -0, treat -0 < +0 (signed-zero ordering): xvminsp(+0,-0)
        //   = -0, xvmaxsp(+0,-0) = +0, in either operand order.
        //   Otherwise, result is IEEE min/max(a, b).
        // This differs from IEEE 754 (which propagates NaN) and is
        // relied upon by wasm relaxed_min/max (bug1946618.js) and by
        // wasm f32x4.min(0,-0) returning -0 (simd_f32x4.wast.js).
        #define XV_MAX(T, a, b) [](T a_, T b_) -> T {                          \
          bool an = std::isnan(a_), bn = std::isnan(b_);                       \
          if (an && bn) return a_;                                              \
          if (an) return b_;                                                    \
          if (bn) return a_;                                                    \
          if (a_ == 0.0 && b_ == 0.0) {                                         \
            /* -0 is smaller than +0; max picks +0. */                          \
            return std::signbit(a_) ? b_ : a_;                                  \
          }                                                                     \
          return std::max(a_, b_);                                              \
        }(a, b)
        #define XV_MIN(T, a, b) [](T a_, T b_) -> T {                          \
          bool an = std::isnan(a_), bn = std::isnan(b_);                       \
          if (an && bn) return a_;                                              \
          if (an) return b_;                                                    \
          if (bn) return a_;                                                    \
          if (a_ == 0.0 && b_ == 0.0) {                                         \
            /* -0 is smaller than +0; min picks -0. */                          \
            return std::signbit(a_) ? a_ : b_;                                  \
          }                                                                     \
          return std::min(a_, b_);                                              \
        }(a, b)
        case 192: for (int i = 0; i < 4; i++) {  // xvmaxsp
          float a = getF32(ab, i), b = getF32(bb, i);
          setF32(rb_bytes, i, XV_MAX(float, a, b));
        } break;
        case 224: for (int i = 0; i < 2; i++) {  // xvmaxdp
          double a = getF64(ab, i), b = getF64(bb, i);
          setF64(rb_bytes, i, XV_MAX(double, a, b));
        } break;
        case 200: for (int i = 0; i < 4; i++) {  // xvminsp
          float a = getF32(ab, i), b = getF32(bb, i);
          setF32(rb_bytes, i, XV_MIN(float, a, b));
        } break;
        case 232: for (int i = 0; i < 2; i++) {  // xvmindp
          double a = getF64(ab, i), b = getF64(bb, i);
          setF64(rb_bytes, i, XV_MIN(double, a, b));
        } break;
        #undef XV_MAX
        #undef XV_MIN
        default:
          MOZ_CRASH_UNSAFE_PRINTF(
              "xv float dispatch missing 8-bit XO=%u (instr 0x%08x)",
              xo3, instr->instructionBits());
      }
      setVSR128(xt, rb_bytes);
      break;
    }

    // === XX3-form fused multiply-add (3-source: XT is also input) ===
    //
    //   xvmaddasp XT,XA,XB:  XT = (XA * XB) + XT       (fused madd)
    //   xvmaddadp XT,XA,XB:  same for f64
    //   xvnmsubasp XT,XA,XB: XT = -((XA * XB) - XT) = XT - (XA * XB)
    //   xvnmsubadp XT,XA,XB: same for f64
    //
    // Encodings (each +AX): XO8 → XO9 pairs
    //   xvmaddasp  PPC_xvmaddasp=0xF0000208   XO8=65  → XO9 130/131
    //   xvmaddadp  PPC_xvmaddadp=0xF0000308   XO8=97  → XO9 194/195
    //   xvnmsubasp PPC_xvnmsubasp=0xF0000688  XO8=209 → XO9 418/419
    //   xvnmsubadp PPC_xvnmsubadp=0xF0000788  XO8=241 → XO9 482/483
    // std::fma gives IEEE-correct single-rounding behaviour matching the
    // Power ISA definition of these fused forms.
    case 130: case 131:      // xvmaddasp
    case 194: case 195:      // xvmaddadp
    case 418: case 419:      // xvnmsubasp
    case 482: case 483: {    // xvnmsubadp
      int xt = int(rt | (instr->bit(0) << 5));
      uint32_t ra = instr->raValue();
      int xa = int(ra | ((instr->instructionBits() >> 2) & 1) << 5);
      int xb = int(rb | ((instr->instructionBits() >> 1) & 1) << 5);
      uint8_t ab[16], bb[16], tb[16];
      getVSR128(xa, ab);
      getVSR128(xb, bb);
      getVSR128(xt, tb);  // XT is also an input (accumulator).
      bool isSp = (xo == 130 || xo == 131 || xo == 418 || xo == 419);
      bool isNmsub = (xo == 418 || xo == 419 || xo == 482 || xo == 483);
      auto rdF32 = [](uint8_t* buf, int i) -> float {
        uint32_t b = (uint32_t)buf[i * 4] |
                     ((uint32_t)buf[i * 4 + 1] << 8) |
                     ((uint32_t)buf[i * 4 + 2] << 16) |
                     ((uint32_t)buf[i * 4 + 3] << 24);
        float f; memcpy(&f, &b, sizeof(f)); return f;
      };
      auto wrF32 = [](uint8_t* buf, int i, float f) {
        uint32_t b; memcpy(&b, &f, sizeof(b));
        buf[i*4]=(uint8_t)b; buf[i*4+1]=(uint8_t)(b>>8);
        buf[i*4+2]=(uint8_t)(b>>16); buf[i*4+3]=(uint8_t)(b>>24);
      };
      auto rdF64 = [](uint8_t* buf, int i) -> double {
        uint64_t b = 0;
        for (int k=0;k<8;k++) b |= ((uint64_t)buf[i*8+k])<<(k*8);
        double d; memcpy(&d, &b, sizeof(d)); return d;
      };
      auto wrF64 = [](uint8_t* buf, int i, double d) {
        uint64_t b; memcpy(&b, &d, sizeof(b));
        for (int k=0;k<8;k++) buf[i*8+k]=(uint8_t)(b>>(k*8));
      };
      uint8_t result[16];
      if (isSp) {
        for (int i = 0; i < 4; i++) {
          float a = rdF32(ab, i), b = rdF32(bb, i), t = rdF32(tb, i);
          // madd:  t + a*b ;  nmsub: -(a*b - t) = t - a*b = std::fma(a,b,-t) negated.
          float out = isNmsub ? -std::fma(a, b, -t)
                              :  std::fma(a, b, t);
          wrF32(result, i, out);
        }
      } else {
        for (int i = 0; i < 2; i++) {
          double a = rdF64(ab, i), b = rdF64(bb, i), t = rdF64(tb, i);
          double out = isNmsub ? -std::fma(a, b, -t)
                               :  std::fma(a, b, t);
          wrF64(result, i, out);
        }
      }
      setVSR128(xt, result);
      break;
    }

    default:
      MOZ_CRASH_UNSAFE_PRINTF(
          "decodeVSX: unimplemented XO=%u (instruction 0x%08x)", xo,
          instr->instructionBits());
  }
}

// =============================================================================
// Power ISA v3.1 prefixed instructions (POWER10).
// =============================================================================
//
// A prefixed instruction is 8 bytes: a 4-byte prefix word (primary opcode 1)
// followed by a 4-byte suffix word. Prefix and suffix must lie in the same
// 64-byte aligned block — the JIT must guarantee this when emitting; the sim
// asserts.
//
// Prefix word layout (BE bit numbering):
//   [0..5]   primary opcode = 1
//   [6..7]   Type (00 = 8LS, 10 = MLS — only forms we implement)
//   [8..10]  reserved (must be 0)
//   [11]     R (1 = PC-relative; RA must be 0)
//   [12..13] reserved (must be 0)
//   [14..31] d0 (high 18 bits of the 34-bit signed immediate)
//
// Suffix word (MLS/8LS form, GPR-target instructions like paddi/pld):
//   [0..5]   suffix primary opcode (selects the actual instruction)
//   [6..10]  RT (or RS for stores)
//   [11..15] RA
//   [16..31] d1 (low 16 bits of immediate)
//
// Suffix word (8LS plxv quirk): the suffix opcode field is only 5 bits
// wide and bit [5] holds TX, the high bit of the 6-bit XT VSR number:
//   [0..4]   plxv suffix opcode = 11001 (= 25)
//   [5]      TX
//   [6..10]  T
//   [11..15] RA
//   [16..31] d1
// Combined: XT = (TX << 5) | T. (Equivalent: full 6-bit field at [0..5]
// is 0b11001(TX) — values 50 or 51 in our LE bits 31..26.)
//
// Combined immediate: SI = sign_extend((d0 << 16) | d1, 34).
// EA when R=1: address-of-prefix + SI. (RA must be 0.)
// EA when R=0: (RA == 0 ? 0 : GPR[RA]) + SI.
//
// Suffix opcodes implemented here:
//   MLS (Type 2) / suffix=14  paddi
//   MLS (Type 2) / suffix=48  plfs   (load FP single, widens to double)
//   MLS (Type 2) / suffix=50  plfd   (load FP double)
//   8LS (Type 0) / suffix=57  pld
//   8LS (Type 0) / 5-bit suffix=25, bit 26 = TX  plxv
//
// Verification recipe when adding more: assemble with `gcc -mcpu=power10
// -c` (or clang) and compare the emitted bytes against the encoder; encode
// in a small inline-asm program and step through under this sim.

void Simulator::decodePrefixed(SimInstruction* prefix) {
  // Prefix and suffix must reside in the same 64-byte block.
  uint64_t prefixAddr = reinterpret_cast<uint64_t>(prefix);
  MOZ_ASSERT((prefixAddr & 63) <= 56,
             "POWER10 prefixed instruction crosses 64-byte boundary");

  SimInstruction* suffix = reinterpret_cast<SimInstruction*>(
      reinterpret_cast<uint8_t*>(prefix) + SimInstruction::kInstrSize);

  uint32_t type = prefix->bits(25, 24);
  uint32_t R = prefix->bit(20);
  uint32_t d0 = prefix->bits(17, 0);  // 18 bits
  uint32_t suffixOp6 = suffix->bits(31, 26);  // 6-bit form (paddi, pld)
  uint32_t suffixOp5 = suffix->bits(31, 27);  // 5-bit form (plxv)
  uint32_t plxvTX = suffix->bit(26);
  uint32_t rt = suffix->rtValue();
  uint32_t ra = suffix->raValue();
  uint32_t d1 = suffix->uimm16Value();

  // Reassemble 34-bit signed displacement.
  int64_t imm34 = (static_cast<int64_t>(d0) << 16) | d1;
  imm34 = (imm34 << 30) >> 30;  // sign-extend from bit 33

  // R=1 forms require RA=0 per the ISA.
  MOZ_ASSERT(!R || ra == 0,
             "POWER10 prefixed R=1 form requires RA=0");

  // Type 2 = MLS, Type 0 = 8LS. Other types are reserved here.
  if (type == 2 && suffixOp6 == 14) {
    // paddi RT, RA, SI, R (MLS)
    int64_t base = R ? static_cast<int64_t>(prefixAddr)
                     : (ra == 0 ? 0 : getRegister(ra));
    setRegister(rt, base + imm34);
  } else if (type == 0 && suffixOp6 == 57) {
    // pld RT, D(RA), R (8LS)
    uint64_t ea = R ? prefixAddr + static_cast<uint64_t>(imm34)
                    : (ra == 0 ? 0 : getRegister(ra)) +
                          static_cast<uint64_t>(imm34);
    if (!handleWasmSegFault(ea, 8)) {
      setRegister(rt, readDW(ea, prefix));
    }
  } else if (type == 2 && suffixOp6 == 50) {
    // plfd FRT, D(RA), R (MLS) — load 8-byte double into FPR.
    uint64_t ea = R ? prefixAddr + static_cast<uint64_t>(imm34)
                    : (ra == 0 ? 0 : getRegister(ra)) +
                          static_cast<uint64_t>(imm34);
    if (!handleWasmSegFault(ea, 8)) {
      setFpuRegisterDouble(rt, readD(ea, prefix));
    }
  } else if (type == 2 && suffixOp6 == 48) {
    // plfs FRT, D(RA), R (MLS) — load 4-byte single, widen NaN-preserving.
    uint64_t ea = R ? prefixAddr + static_cast<uint64_t>(imm34)
                    : (ra == 0 ? 0 : getRegister(ra)) +
                          static_cast<uint64_t>(imm34);
    if (!handleWasmSegFault(ea, 4)) {
      float val = *reinterpret_cast<float*>(ea);
      setFpuRegisterDouble(rt, promoteFloatPreservingNaN(val));
    }
  } else if (type == 0 && suffixOp5 == 25) {
    // plxv XT, D(RA), R (8LS) — XT = (TX << 5) | T, TX at suffix bit 26.
    int xt = static_cast<int>(rt | (plxvTX << 5));
    uint64_t ea = R ? prefixAddr + static_cast<uint64_t>(imm34)
                    : (ra == 0 ? 0 : getRegister(ra)) +
                          static_cast<uint64_t>(imm34);
    if (!handleWasmSegFault(ea, 16)) {
      uint8_t buf[16];
      memcpy(buf, reinterpret_cast<const void*>(ea), 16);
      setVSR128(xt, buf);
    }
  } else if (type == 0 && suffixOp6 == 61) {
    // pstd RS, D(RA), R (8LS) — store doubleword.
    uint64_t ea = R ? prefixAddr + static_cast<uint64_t>(imm34)
                    : (ra == 0 ? 0 : getRegister(ra)) +
                          static_cast<uint64_t>(imm34);
    if (!handleWasmSegFault(ea, 8)) {
      writeDW(ea, getRegister(rt), prefix);
    }
  } else if (type == 2 && suffixOp6 == 54) {
    // pstfd FRS, D(RA), R (MLS) — store double.
    uint64_t ea = R ? prefixAddr + static_cast<uint64_t>(imm34)
                    : (ra == 0 ? 0 : getRegister(ra)) +
                          static_cast<uint64_t>(imm34);
    if (!handleWasmSegFault(ea, 8)) {
      writeD(ea, getFpuRegisterDouble(rt), prefix);
    }
  } else if (type == 2 && suffixOp6 == 52) {
    // pstfs FRS, D(RA), R (MLS) — store single (narrow from double in FPR).
    uint64_t ea = R ? prefixAddr + static_cast<uint64_t>(imm34)
                    : (ra == 0 ? 0 : getRegister(ra)) +
                          static_cast<uint64_t>(imm34);
    if (!handleWasmSegFault(ea, 4)) {
      double dval = getFpuRegisterDouble(rt);
      *reinterpret_cast<float*>(ea) = demoteDoublePreservingNaN(dval);
    }
  } else if (type == 0 && suffixOp5 == 27) {
    // pstxv XS, D(RA), R (8LS) — XS = (SX << 5) | S, SX at suffix bit 26.
    int xs = static_cast<int>(rt | (plxvTX << 5));
    uint64_t ea = R ? prefixAddr + static_cast<uint64_t>(imm34)
                    : (ra == 0 ? 0 : getRegister(ra)) +
                          static_cast<uint64_t>(imm34);
    if (!handleWasmSegFault(ea, 16)) {
      uint8_t buf[16];
      getVSR128(xs, buf);
      memcpy(reinterpret_cast<void*>(ea), buf, 16);
    }
  } else {
    MOZ_CRASH_UNSAFE_PRINTF(
        "decodePrefixed: unimplemented type=%u "
        "(prefix 0x%08x, suffix 0x%08x)",
        type, prefix->instructionBits(), suffix->instructionBits());
  }

  // Advance past the full 8-byte prefixed instruction unless a handler
  // already redirected the PC. The caller (instructionDecode) returns
  // immediately after us, so its 4-byte trailing advance is skipped.
  if (!pc_modified_) {
    set_pc(static_cast<int64_t>(prefixAddr) + 2 * SimInstruction::kInstrSize);
  }
}

// =============================================================================
// Top-level instruction decoder.
// =============================================================================

void Simulator::instructionDecode(SimInstruction* instr) {
  if (!SimulatorProcess::ICacheCheckingDisableCount) {
    AutoLockSimulatorCache als;
    SimulatorProcess::checkICacheLocked(instr);
  }
  pc_modified_ = false;

  uint32_t instrBits = instr->instructionBits();

  // Check for kCallRedirInstr first (PPC_stop = 0x4C0002E4).
  if (instrBits == kCallRedirInstr) {
    softwareInterrupt(instr);
    if (!pc_modified_) {
      set_pc(reinterpret_cast<int64_t>(instr) + SimInstruction::kInstrSize);
    }
    return;
  }

  // Check for PPC_trap (0x7FE00008).
  if (instrBits == 0x7FE00008) {
    softwareInterrupt(instr);
    if (!pc_modified_) {
      set_pc(reinterpret_cast<int64_t>(instr) + SimInstruction::kInstrSize);
    }
    return;
  }

  uint32_t opcode = instr->opcode();

  // Power ISA v3.1 prefixed instructions: primary opcode 1 marks a
  // 4-byte prefix word followed by a 4-byte suffix word. decodePrefixed
  // advances the PC by the full 8 bytes (or leaves it modified for
  // PC-relative side-effects).
  if (opcode == 1) {
    decodePrefixed(instr);
    return;
  }

  switch (opcode) {
    // D-form ALU
    case 3:   // twi
    case 7:   // mulli
    case 8:   // subfic
    case 10:  // cmpli
    case 11:  // cmpi
    case 12:  // addic
    case 13:  // addic.
    case 14:  // addi
    case 15:  // addis
    case 24:  // ori
    case 25:  // oris
    case 26:  // xori
    case 27:  // xoris
    case 28:  // andi.
    case 29:  // andis.
      decodeDFormALU(instr);
      break;

    // D-form loads
    case 32:  // lwz
    case 33:  // lwzu
    case 34:  // lbz
    case 35:  // lbzu
    case 40:  // lhz
    case 41:  // lhzu
    case 42:  // lha
    case 43:  // lhau
    case 48:  // lfs
    case 49:  // lfsu
    case 50:  // lfd
    case 51:  // lfdu
      decodeDFormLoad(instr);
      break;

    // D-form stores
    case 36:  // stw
    case 38:  // stb
    case 39:  // stbu
    case 44:  // sth
    case 45:  // sthu
    case 52:  // stfs
    case 53:  // stfsu
    case 54:  // stfd
    case 55:  // stfdu
      decodeDFormStore(instr);
      break;

    // DS-form
    case 58:  // ld, ldu, lwa
    case 62:  // std, stdu
      decodeDSForm(instr);
      break;

    // B-form conditional branch
    case 16:
      decodeBranch(instr);
      break;

    // SC (system call) - unused in JIT
    case 17:
      MOZ_CRASH("Simulator: sc instruction not supported");
      break;

    // I-form unconditional branch
    case 18:
      decodeBranch(instr);
      break;

    // XL-form (branch to LR/CTR, CR operations)
    case 19:
      decodeBranch(instr);
      break;

    // M-form / MD-form rotate/mask
    case 20:  // rlwimi
    case 21:  // rlwinm
    case 23:  // rlwnm
    case 30:  // rldicl, rldicr, rldic, rldimi, rldcl, rldcr
      decodeRotateMask(instr);
      break;

    // VMX (AltiVec) — primary opcode 4. Vector arithmetic / compare / shift /
    // splat / merge / pack / unpack on VR0-VR31. The wasm SIMD lowering
    // emits these directly (Simd128 lives in the VR namespace).
    case 4:
      decodeVMX(instr);
      break;

    // X-form / XO-form
    case 31:
      decodeXForm(instr);
      break;

    // FP single (A-form)
    case 59:
      decodeFP(instr);
      break;

    // VSX (XX1-form)
    case 60:
      decodeVSX(instr);
      break;

    // FP double (X-form / A-form)
    case 63:
      decodeFP(instr);
      break;

    default:
      MOZ_CRASH_UNSAFE_PRINTF(
          "instructionDecode: unsupported opcode %u (instruction 0x%08x)",
          opcode, instrBits);
  }

  if (!pc_modified_) {
    set_pc(reinterpret_cast<int64_t>(instr) + SimInstruction::kInstrSize);
  }
}

// =============================================================================
// Single-stepping / execute loop.
// =============================================================================

void Simulator::enable_single_stepping(SingleStepCallback cb, void* arg) {
  single_stepping_ = true;
  single_step_callback_ = cb;
  single_step_callback_arg_ = arg;
  single_step_callback_(single_step_callback_arg_, this, (void*)get_pc());
}

void Simulator::disable_single_stepping() {
  if (!single_stepping_) {
    return;
  }
  single_step_callback_(single_step_callback_arg_, this, (void*)get_pc());
  single_stepping_ = false;
  single_step_callback_ = nullptr;
  single_step_callback_arg_ = nullptr;
}

template <bool enableStopSimAt>
void Simulator::execute() {
  if (single_stepping_ && getenv("PPC64_TRACE_SIM")) {
    fprintf(stderr, "[sim] enter execute pc=0x%lx lr=0x%lx fp=0x%lx sp=0x%lx\n",
            (long)get_pc(), (long)getLR(), (long)getRegister(fp),
            (long)getRegister(sp));
  }
  if (single_stepping_) {
    single_step_callback_(single_step_callback_arg_, this, nullptr);
  }

  int64_t program_counter = get_pc();

  while (program_counter != end_sim_pc) {
    if (enableStopSimAt && (icount_ == Simulator::StopSimAt)) {
      ppc64Debugger dbg(this);
      dbg.debug();
    } else {
      if (single_stepping_) {
        if (getenv("PPC64_TRACE_SIM")) {
          fprintf(stderr,
                  "[sim] step icount=%llu pc=0x%lx instr=0x%08x lr=0x%lx fp=0x%lx sp=0x%lx\n",
                  (unsigned long long)icount_, (long)program_counter,
                  *(uint32_t*)program_counter, (long)getLR(),
                  (long)getRegister(fp), (long)getRegister(sp));
        }
        single_step_callback_(single_step_callback_arg_, this,
                              (void*)program_counter);
      }
      SimInstruction* instr =
          reinterpret_cast<SimInstruction*>(program_counter);
      instructionDecode(instr);
      icount_++;
    }
    program_counter = get_pc();
  }

  if (single_stepping_) {
    single_step_callback_(single_step_callback_arg_, this, nullptr);
  }
}

// =============================================================================
// callInternal / call.
// =============================================================================

void Simulator::callInternal(uint8_t* entry) {
  // Prepare to execute the code at entry.
  setRegister(pc, reinterpret_cast<int64_t>(entry));
  // The simulation stops when returning to this call point (LR == end_sim_pc).
  setLR(end_sim_pc);

  // Remember the values of callee-saved registers (r14-r31 in ELFv2).
  int64_t r14_val = getRegister(r14);
  int64_t r15_val = getRegister(r15);
  int64_t r16_val = getRegister(r16);
  int64_t r17_val = getRegister(r17);
  int64_t r18_val = getRegister(r18);
  int64_t r19_val = getRegister(r19);
  int64_t r20_val = getRegister(r20);
  int64_t r21_val = getRegister(r21);
  int64_t r22_val = getRegister(r22);
  int64_t r23_val = getRegister(r23);
  int64_t r24_val = getRegister(r24);
  int64_t r25_val = getRegister(r25);
  int64_t r26_val = getRegister(r26);
  int64_t r27_val = getRegister(r27);
  int64_t r28_val = getRegister(r28);
  int64_t r29_val = getRegister(r29);
  int64_t r30_val = getRegister(r30);
  int64_t r31_val = getRegister(r31);
  int64_t sp_val = getRegister(sp);

#ifdef DEBUG
  // Set up callee-saved registers with a known value to detect clobbers.
  // DEBUG-only: in release this would silently corrupt every JS-jit-entry
  // stub frame, since the stub saves r14-r31 to its stack early on. Any
  // single-step-profiling sample taken later (or any unwind through the
  // stub's saved CSR area) then dereferences `icount_` as a frame
  // pointer and crashes — see e.g. wasm/profiling.js, ion-error-*.js,
  // ion-lazy-tables.js, ion-callerfp-tag.js, return-call-profiling.js,
  // externref-global-postbarrier.js, builtin-modules/i8vecmul.js,
  // asm.js/testBug1357053.js (all single-step-profiling tests). In
  // debug builds the value collides with the same callsites but the
  // MOZ_ASSERTs below catch any actual ABI violation, which is the
  // entire point.
  int64_t callee_saved_value = icount_;
  setRegister(r14, callee_saved_value);
  setRegister(r15, callee_saved_value);
  setRegister(r16, callee_saved_value);
  setRegister(r17, callee_saved_value);
  setRegister(r18, callee_saved_value);
  setRegister(r19, callee_saved_value);
  setRegister(r20, callee_saved_value);
  setRegister(r21, callee_saved_value);
  setRegister(r22, callee_saved_value);
  setRegister(r23, callee_saved_value);
  setRegister(r24, callee_saved_value);
  setRegister(r25, callee_saved_value);
  setRegister(r26, callee_saved_value);
  setRegister(r27, callee_saved_value);
  setRegister(r28, callee_saved_value);
  setRegister(r29, callee_saved_value);
  setRegister(r30, callee_saved_value);
  setRegister(r31, callee_saved_value);
#endif

  // Start the simulation.
  if (Simulator::StopSimAt != -1) {
    execute<true>();
  } else {
    execute<false>();
  }

#ifdef DEBUG
  // Check that the callee-saved registers have been preserved.
  MOZ_ASSERT(callee_saved_value == getRegister(r14));
  MOZ_ASSERT(callee_saved_value == getRegister(r15));
  MOZ_ASSERT(callee_saved_value == getRegister(r16));
  MOZ_ASSERT(callee_saved_value == getRegister(r17));
  MOZ_ASSERT(callee_saved_value == getRegister(r18));
  MOZ_ASSERT(callee_saved_value == getRegister(r19));
  MOZ_ASSERT(callee_saved_value == getRegister(r20));
  MOZ_ASSERT(callee_saved_value == getRegister(r21));
  MOZ_ASSERT(callee_saved_value == getRegister(r22));
  MOZ_ASSERT(callee_saved_value == getRegister(r23));
  MOZ_ASSERT(callee_saved_value == getRegister(r24));
  MOZ_ASSERT(callee_saved_value == getRegister(r25));
  MOZ_ASSERT(callee_saved_value == getRegister(r26));
  MOZ_ASSERT(callee_saved_value == getRegister(r27));
  MOZ_ASSERT(callee_saved_value == getRegister(r28));
  MOZ_ASSERT(callee_saved_value == getRegister(r29));
  MOZ_ASSERT(callee_saved_value == getRegister(r30));
  MOZ_ASSERT(callee_saved_value == getRegister(r31));
#endif

  // Restore callee-saved registers.
  setRegister(r14, r14_val);
  setRegister(r15, r15_val);
  setRegister(r16, r16_val);
  setRegister(r17, r17_val);
  setRegister(r18, r18_val);
  setRegister(r19, r19_val);
  setRegister(r20, r20_val);
  setRegister(r21, r21_val);
  setRegister(r22, r22_val);
  setRegister(r23, r23_val);
  setRegister(r24, r24_val);
  setRegister(r25, r25_val);
  setRegister(r26, r26_val);
  setRegister(r27, r27_val);
  setRegister(r28, r28_val);
  setRegister(r29, r29_val);
  setRegister(r30, r30_val);
  setRegister(r31, r31_val);
  setRegister(sp, sp_val);
}

int64_t Simulator::call(uint8_t* entry, int argument_count, ...) {
  va_list parameters;
  va_start(parameters, argument_count);

  int64_t original_stack = getRegister(sp);
  // Compute position of stack on entry to generated code.
  int64_t entry_stack = original_stack;
  if (argument_count > kCArgSlotCount) {
    entry_stack = entry_stack - argument_count * sizeof(int64_t);
  } else {
    entry_stack = entry_stack - kCArgsSlotsSize;
  }

  entry_stack &= ~U64(ABIStackAlignment - 1);

  intptr_t* stack_argument = reinterpret_cast<intptr_t*>(entry_stack);

  // PPC64 ELFv2: first 8 integer args go in r3-r10.
  for (int i = 0; i < argument_count; i++) {
    js::jit::Register argReg;
    if (GetIntArgReg(i, &argReg)) {
      setRegister(argReg.code(), va_arg(parameters, int64_t));
    } else {
      stack_argument[i] = va_arg(parameters, int64_t);
    }
  }

  va_end(parameters);
  setRegister(sp, entry_stack);

  callInternal(entry);

  MOZ_ASSERT(entry_stack == getRegister(sp));
  setRegister(sp, original_stack);

  int64_t result = getRegister(r3);
  return result;
}

uintptr_t Simulator::pushAddress(uintptr_t address) {
  int64_t new_sp = getRegister(sp) - sizeof(uintptr_t);
  uintptr_t* stack_slot = reinterpret_cast<uintptr_t*>(new_sp);
  *stack_slot = address;
  setRegister(sp, new_sp);
  return new_sp;
}

uintptr_t Simulator::popAddress() {
  int64_t current_sp = getRegister(sp);
  uintptr_t* stack_slot = reinterpret_cast<uintptr_t*>(current_sp);
  uintptr_t address = *stack_slot;
  setRegister(sp, current_sp + sizeof(uintptr_t));
  return address;
}

}  // namespace jit
}  // namespace js

js::jit::Simulator* JSContext::simulator() const { return simulator_; }
