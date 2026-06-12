/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef jit_ppc64_Simulator_ppc64_h
#define jit_ppc64_Simulator_ppc64_h

#ifdef JS_SIMULATOR_PPC64

#  include "mozilla/Atomics.h"

#  include "jit/IonTypes.h"
#  include "js/ProfilingFrameIterator.h"
#  include "threading/Thread.h"
#  include "vm/MutexIDs.h"
#  include "wasm/WasmSignalHandlers.h"

namespace js {
namespace jit {

class JitActivation;
class Simulator;
class Redirection;
class CachePage;
class AutoLockSimulator;

typedef void (*SingleStepCallback)(void* arg, Simulator* sim, void* pc);

const intptr_t kPointerAlignment = 8;
const intptr_t kPointerAlignmentMask = kPointerAlignment - 1;
const intptr_t kDoubleAlignment = 8;
const intptr_t kDoubleAlignmentMask = kDoubleAlignment - 1;

const int kNumGPRegisters = 32;
const int kPCRegister = 32;
const int kNumFPURegisters = 32;
const int kNumVRRegisters = 32;  // VR0-VR31 (Altivec/VMX; = VSR32-63 in VSX)

// PPC64 Condition Register: 8 fields of 4 bits each.
// Each field: bit3=LT, bit2=GT, bit1=EQ, bit0=SO (in PPC big-endian numbering
// within a field, but stored in little-endian nibble order in our uint32_t).
const int kNumCRFields = 8;

// CR field bit positions (within a 4-bit field).
const uint8_t kCRFieldLT = 0x8;
const uint8_t kCRFieldGT = 0x4;
const uint8_t kCRFieldEQ = 0x2;
const uint8_t kCRFieldSO = 0x1;

// XER register bit positions.
const int kXERSOBit = 31;
const int kXEROVBit = 30;
const int kXERCABit = 29;
const int kXEROV32Bit = 19;
const int kXERCA32Bit = 18;

// FPSCR rounding mode bits (bits 62:63, stored in low bits of our uint64_t).
const uint64_t kFPSCRRNMask = 0x3;

// FPU rounding modes matching PPC64 FPSCR RN field.
enum FPURoundingMode {
  RN = 0,  // Round to Nearest (ties to even)
  RZ = 1,  // Round toward Zero
  RP = 2,  // Round toward +Infinity
  RM = 3,  // Round toward -Infinity
};

// FPU invalid result constants.
const uint32_t kFPUInvalidResult = static_cast<uint32_t>(1 << 31) - 1;
const int32_t kFPUInvalidResultNegative = static_cast<int32_t>(1u << 31);
const uint64_t kFPU64InvalidResult =
    static_cast<uint64_t>(static_cast<uint64_t>(1) << 63) - 1;
const int64_t kFPU64InvalidResultNegative =
    static_cast<int64_t>(static_cast<uint64_t>(1) << 63);

// Breakpoint/stop code ranges.
const uint32_t kMaxWatchpointCode = 31;
const uint32_t kMaxStopCode = 127;
const uint32_t kWasmTrapCode = 6;

// Redirection instruction: PPC_stop (0x4C0002E4).
// Distinct from PPC_trap (0x7FE00008) used for wasm traps.
const uint32_t kCallRedirInstr = 0x4C0002E4;

typedef uint32_t Instr;
class SimInstruction;

class Simulator {
  friend class ppc64Debugger;

 public:
  enum Register {
    no_reg = -1,
    r0 = 0,
    r1,
    r2,
    r3,
    r4,
    r5,
    r6,
    r7,
    r8,
    r9,
    r10,
    r11,
    r12,
    r13,
    r14,
    r15,
    r16,
    r17,
    r18,
    r19,
    r20,
    r21,
    r22,
    r23,
    r24,
    r25,
    r26,
    r27,
    r28,
    r29,
    r30,
    r31,
    pc,
    kNumSimuRegisters,
    // Aliases
    sp = r1,
    fp = r31,
  };

  enum FPURegister {
    f0 = 0,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10,
    f11,
    f12,
    f13,
    f14,
    f15,
    f16,
    f17,
    f18,
    f19,
    f20,
    f21,
    f22,
    f23,
    f24,
    f25,
    f26,
    f27,
    f28,
    f29,
    f30,
    f31,
    kNumFPURegisters
  };

  static Simulator* Create();
  static void Destroy(Simulator* simulator);

  Simulator();
  ~Simulator();

  static Simulator* Current();

  static inline uintptr_t StackLimit() {
    return Simulator::Current()->stackLimit();
  }

  uintptr_t* addressOfStackLimit();

  // GPR accessors.
  void setRegister(int reg, int64_t value);
  int64_t getRegister(int reg) const;

  // FPR accessors.
  void setFpuRegister(int fpureg, int64_t value);
  void setFpuRegisterWord(int fpureg, int32_t value);
  void setFpuRegisterFloat(int fpureg, float value);
  void setFpuRegisterDouble(int fpureg, double value);
  int64_t getFpuRegister(int fpureg) const;
  int32_t getFpuRegisterWord(int fpureg) const;
  int32_t getFpuRegisterSignedWord(int fpureg) const;
  float getFpuRegisterFloat(int fpureg) const;
  double getFpuRegisterDouble(int fpureg) const;

  // VR accessors (Altivec/VMX registers VR0-VR31). The bytes array is the
  // ground truth: bytes[0] is the most-significant-byte on PPC64 big-endian
  // numbering, i.e., VSR[MSB..LSB] mapped as bytes[0..15]. Callers that want
  // typed views (lane 0 etc.) should extract from the bytes array according
  // to the ISA's lane numbering for that instruction.
  void setVRBytes(int vreg, const uint8_t bytes[16]);
  void getVRBytes(int vreg, uint8_t bytes[16]) const;

  // VSR (Vector-Scalar Register) accessors: unified 64-register namespace
  // where VSR 0-31 aliases FPR 0-31 (DW0 is the FPR value, DW1 is
  // architecturally undefined — we model it as zero on read, ignored on
  // write) and VSR 32-63 aliases VR 0-31. Used by VSX instructions
  // (xxpermdi, xxlor, xxlxor, mtvsrd, mfvsrd, ...).
  void getVSR128(int vsr, uint8_t bytes[16]) const;
  void setVSR128(int vsr, const uint8_t bytes[16]);

  // SPR accessors.
  int64_t getLR() const { return LR_; }
  void setLR(int64_t value) { LR_ = value; }
  int64_t getCTR() const { return CTR_; }
  void setCTR(int64_t value) { CTR_ = value; }
  uint32_t getCR() const { return CR_; }
  void setCR(uint32_t value) { CR_ = value; }
  uint64_t getXER() const { return XER_; }
  void setXER(uint64_t value) { XER_ = value; }
  uint64_t getFPSCR() const { return FPSCR_; }
  void setFPSCR(uint64_t value) { FPSCR_ = value; }

  // CR field accessors: field 0 is the most significant nibble (bits 31:28).
  uint8_t getCRField(int field) const {
    return (CR_ >> (4 * (7 - field))) & 0xF;
  }
  void setCRField(int field, uint8_t val) {
    uint32_t shift = 4 * (7 - field);
    CR_ = (CR_ & ~(0xFu << shift)) | ((val & 0xFu) << shift);
  }

  // XER bit accessors.
  bool getXERSO() const { return (XER_ >> kXERSOBit) & 1; }
  void setXERSO(bool v) {
    XER_ = (XER_ & ~(1ull << kXERSOBit)) | ((uint64_t)v << kXERSOBit);
  }
  bool getXEROV() const { return (XER_ >> kXEROVBit) & 1; }
  void setXEROV(bool v) {
    XER_ = (XER_ & ~(1ull << kXEROVBit)) | ((uint64_t)v << kXEROVBit);
    // Mirror to OV32. Real POWER9 silicon sets OV32 == OV for both 32-bit
    // and 64-bit overflow ops: mulldo(2, 2^62) produces OV=OV32=1;
    // mulldo(2^30, 4) produces OV=OV32=0. The JIT's
    // POWER9 Overflow path is `mulldo + mcrxrx + bc Overflow`, where
    // mcrxrx places OV32 in the GT slot and the Overflow condition tests
    // GT — so OV32 must be live or no-overflow is reported even when
    // OV=1. Without this mirror, BigInt fast-path mul silently wraps.
    XER_ = (XER_ & ~(1ull << kXEROV32Bit)) | ((uint64_t)v << kXEROV32Bit);
    if (v) setXERSO(true);
  }
  bool getXERCA() const { return (XER_ >> kXERCABit) & 1; }
  void setXERCA(bool v) {
    XER_ = (XER_ & ~(1ull << kXERCABit)) | ((uint64_t)v << kXERCABit);
  }

  // PC accessors.
  void set_pc(int64_t value);
  int64_t get_pc() const;

  template <typename T>
  T get_pc_as() const {
    return reinterpret_cast<T>(get_pc());
  }

  void enable_single_stepping(SingleStepCallback cb, void* arg);
  void disable_single_stepping();

  uintptr_t stackLimit() const;
  bool overRecursed(uintptr_t newsp = 0) const;
  bool overRecursedWithExtra(uint32_t extra) const;

  template <bool enableStopSimAt>
  void execute();

  int64_t call(uint8_t* entry, int argument_count, ...);

  uintptr_t pushAddress(uintptr_t address);
  uintptr_t popAddress();

  void setLastDebuggerInput(char* input);
  char* lastDebuggerInput() { return lastDebuggerInput_; }

  bool has_bad_pc() const;

  // Update CR field 0 from a 64-bit result.
  void updateCR0(int64_t result) {
    uint8_t field = kCRFieldSO * getXERSO();
    if (result < 0)
      field |= kCRFieldLT;
    else if (result > 0)
      field |= kCRFieldGT;
    else
      field |= kCRFieldEQ;
    setCRField(0, field);
  }

  // Update CR field 0 from a 32-bit result (sign-extended comparison).
  void updateCR0_32(int32_t result) {
    uint8_t field = kCRFieldSO * getXERSO();
    if (result < 0)
      field |= kCRFieldLT;
    else if (result > 0)
      field |= kCRFieldGT;
    else
      field |= kCRFieldEQ;
    setCRField(0, field);
  }

  // Compare and set an arbitrary CR field.
  void setCRFieldCmp(int field, int64_t lhs, int64_t rhs) {
    uint8_t val = kCRFieldSO * getXERSO();
    if (lhs < rhs)
      val |= kCRFieldLT;
    else if (lhs > rhs)
      val |= kCRFieldGT;
    else
      val |= kCRFieldEQ;
    setCRField(field, val);
  }

  void setCRFieldCmpU(int field, uint64_t lhs, uint64_t rhs) {
    uint8_t val = kCRFieldSO * getXERSO();
    if (lhs < rhs)
      val |= kCRFieldLT;
    else if (lhs > rhs)
      val |= kCRFieldGT;
    else
      val |= kCRFieldEQ;
    setCRField(field, val);
  }

 private:
  enum SpecialValues {
    // PPC64 masks the low 2 bits of branch targets, so these must be
    // 4-byte aligned to survive the & ~3 mask in blr/bcctr.
    bad_ra = -4,
    end_sim_pc = -8,
    Unpredictable = 0xbadbeaf
  };

  bool init();

  void format(SimInstruction* instr, const char* format);

  // Memory access.
  inline uint8_t readBU(uint64_t addr);
  inline int8_t readB(uint64_t addr);
  inline void writeB(uint64_t addr, uint8_t value);
  inline void writeB(uint64_t addr, int8_t value);

  inline uint16_t readHU(uint64_t addr, SimInstruction* instr);
  inline int16_t readH(uint64_t addr, SimInstruction* instr);
  inline void writeH(uint64_t addr, uint16_t value, SimInstruction* instr);
  inline void writeH(uint64_t addr, int16_t value, SimInstruction* instr);

  inline uint32_t readWU(uint64_t addr, SimInstruction* instr);
  inline int32_t readW(uint64_t addr, SimInstruction* instr);
  inline void writeW(uint64_t addr, uint32_t value, SimInstruction* instr);
  inline void writeW(uint64_t addr, int32_t value, SimInstruction* instr);

  inline int64_t readDW(uint64_t addr, SimInstruction* instr);
  inline void writeDW(uint64_t addr, int64_t value, SimInstruction* instr);

  inline double readD(uint64_t addr, SimInstruction* instr);
  inline void writeD(uint64_t addr, double value, SimInstruction* instr);

  inline uint8_t loadLinkedB(uint64_t addr, SimInstruction* instr);
  inline int storeConditionalB(uint64_t addr, uint8_t value,
                               SimInstruction* instr);
  inline uint16_t loadLinkedH(uint64_t addr, SimInstruction* instr);
  inline int storeConditionalH(uint64_t addr, uint16_t value,
                               SimInstruction* instr);
  inline int32_t loadLinkedW(uint64_t addr, SimInstruction* instr);
  inline int storeConditionalW(uint64_t addr, int32_t value,
                               SimInstruction* instr);
  inline int64_t loadLinkedD(uint64_t addr, SimInstruction* instr);
  inline int storeConditionalD(uint64_t addr, int64_t value,
                               SimInstruction* instr);

  // Instruction decoders.
  void decodeDFormALU(SimInstruction* instr);
  void decodeDFormLoad(SimInstruction* instr);
  void decodeDFormStore(SimInstruction* instr);
  void decodeDSForm(SimInstruction* instr);
  void decodeXForm(SimInstruction* instr);
  void decodeRotateMask(SimInstruction* instr);
  void decodeBranch(SimInstruction* instr);
  void decodeFP(SimInstruction* instr);
  void decodeVSX(SimInstruction* instr);
  void decodeVMX(SimInstruction* instr);
  // Power ISA v3.1 prefixed instructions. `prefix` points at the
  // 4-byte prefix word; the suffix is read from `prefix + 4`.
  void decodePrefixed(SimInstruction* prefix);

  void softwareInterrupt(SimInstruction* instr);

  // Stop/breakpoint helpers.
  bool isWatchpoint(uint32_t code);
  void printWatchpoint(uint32_t code);
  void handleStop(uint32_t code, SimInstruction* instr);
  bool isStopInstruction(SimInstruction* instr);
  bool isEnabledStop(uint32_t code);
  void enableStop(uint32_t code);
  void disableStop(uint32_t code);
  void increaseStopCounter(uint32_t code);
  void printStopInfo(uint32_t code);

  JS::ProfilingFrameIterator::RegisterState registerState();

  bool MOZ_ALWAYS_INLINE handleWasmSegFault(uint64_t addr, unsigned numBytes) {
    if (MOZ_LIKELY(!js::wasm::CodeExists)) {
      return false;
    }
    uint8_t* newPC;
    if (!js::wasm::MemoryAccessTraps(registerState(), (uint8_t*)addr, numBytes,
                                     &newPC)) {
      return false;
    }
    LLBit_ = false;
    set_pc(int64_t(newPC));
    return true;
  }

  void instructionDecode(SimInstruction* instr);

 public:
  static int64_t StopSimAt;

  static void* RedirectNativeFunction(void* nativeFunction,
                                      ABIFunctionType type);

 private:
  void setCallResultDouble(double result);
  void setCallResultFloat(float result);
  void setCallResult(int64_t res);
#  ifdef XP_DARWIN
  void setCallResult(intptr_t res);
#  endif
  void setCallResult(__int128 res);

  void callInternal(uint8_t* entry);

  // Architecture state.
  int64_t registers_[kNumSimuRegisters];
  int64_t FPUregisters_[kNumFPURegisters];
  // VR namespace (Altivec/VMX registers VR0-VR31 == VSR32-63). Stored as
  // 16 raw bytes per register to preserve exact architectural byte order
  // independent of host endianness. Accessors defined below; the bytes
  // array is the ground truth.
  uint8_t VRregisters_[kNumVRRegisters][16];

  // PPC64 Special Purpose Registers.
  int64_t LR_;
  int64_t CTR_;
  uint32_t CR_;
  uint64_t XER_;
  uint64_t FPSCR_;

  // Atomics.
  bool LLBit_;
  uintptr_t LLAddr_;
  int64_t lastLLValue_;

  // Simulator support.
  char* stack_;
  uintptr_t stackLimit_;
  bool pc_modified_;
  int64_t icount_;
  int64_t break_count_;

  char* lastDebuggerInput_;

  SimInstruction* break_pc_;
  Instr break_instr_;

  bool single_stepping_;
  SingleStepCallback single_step_callback_;
  void* single_step_callback_arg_;

  static const uint32_t kNumOfWatchedStops = 256;
  static const uint32_t kStopDisabledBit = 1U << 31;

  struct StopCountAndDesc {
    uint32_t count_;
    char* desc_;
  };
  StopCountAndDesc watchedStops_[kNumOfWatchedStops];
};

// Process-wide simulator state.
class SimulatorProcess {
  friend class Redirection;
  friend class AutoLockSimulatorCache;

 private:
  struct ICacheHasher {
    typedef void* Key;
    typedef void* Lookup;
    static HashNumber hash(const Lookup& l);
    static bool match(const Key& k, const Lookup& l);
  };

 public:
  typedef HashMap<void*, CachePage*, ICacheHasher, SystemAllocPolicy> ICacheMap;

  static mozilla::Atomic<size_t, mozilla::ReleaseAcquire>
      ICacheCheckingDisableCount;
  static void FlushICache(void* start, size_t size);
  static void checkICacheLocked(SimInstruction* instr);

  static bool initialize() {
    singleton_ = js_new<SimulatorProcess>();
    return singleton_;
  }
  static void destroy() {
    js_delete(singleton_);
    singleton_ = nullptr;
  }

  SimulatorProcess();
  ~SimulatorProcess();

 private:
  static SimulatorProcess* singleton_;

  Mutex cacheLock_;
  Redirection* redirection_;
  ICacheMap icache_;

 public:
  static ICacheMap& icache() {
    singleton_->cacheLock_.assertOwnedByCurrentThread();
    return singleton_->icache_;
  }

  static Redirection* redirection() {
    singleton_->cacheLock_.assertOwnedByCurrentThread();
    return singleton_->redirection_;
  }

  static void setRedirection(js::jit::Redirection* redirection) {
    singleton_->cacheLock_.assertOwnedByCurrentThread();
    singleton_->redirection_ = redirection;
  }
};

}  // namespace jit
}  // namespace js

#endif /* JS_SIMULATOR_PPC64 */

#endif /* jit_ppc64_Simulator_ppc64_h */
