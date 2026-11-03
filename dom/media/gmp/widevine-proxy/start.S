/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

# Minimal x86_64 _start: the same hand-off to __libc_start_main as glibc's crt1.
	.text
	.globl _start
	.type _start, @function
_start:
	xorl %ebp, %ebp
	movq %rdx, %r9
	popq %rsi
	movq %rsp, %rdx
	andq $-16, %rsp
	pushq %rax
	pushq %rsp
	xorl %r8d, %r8d
	xorl %ecx, %ecx
	leaq main(%rip), %rdi
	call *__libc_start_main@GOTPCREL(%rip)
	hlt
	.section .note.GNU-stack,"",@progbits
