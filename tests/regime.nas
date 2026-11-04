; regime.nas - prove that a program can rewrite the class nibble of the next
; BOOL instruction, then execute it. The slot starts as BOOL.0 (constant 0)
; and is rewritten from an input word to BOOL.6 (XOR). Expected output: PASS.
.org 0
main:
MOVI r13,1
; payload for BOOL d=3, a=1, b=2  =>  0x00310002
MOVI r5,0x0002
MOVHI r5,0x0031
; slot address 4096
MOVI r6,0x1000
MOVHI r6,0
; initialize slot to class 0 (BOOL.0)
MOVI r7,0
STW r5,r6,r7
; operands: a=0xCCCCCCCC b=0xAAAAAAAA
MOVI r1,0xCCCC
MOVHI r1,0xCCCC
MOVI r2,0xAAAA
MOVHI r2,0xAAAA
; read the desired regime from data (6 = XOR)
LD r7,r0,@regime
; rewrite the instruction word at slot 4096 with class 6
STW r5,r6,r7
; execute it
JMPR r6
back:
; expect 0x66666666
MOVI r8,0x6666
MOVHI r8,0x6666
CMP r3,r8
JMP[NE] @fail
MOVI r1,80
PUTC r1
MOVI r1,65
PUTC r1
MOVI r1,83
PUTC r1
MOVI r1,83
PUTC r1
MOVI r1,10
PUTC r1
HALT
fail:
MOVI r1,70
PUTC r1
MOVI r1,65
PUTC r1
MOVI r1,73
PUTC r1
MOVI r1,76
PUTC r1
MOVI r1,10
PUTC r1
HALT
regime:
.word 6
.org 4096
.word 0x00310002
.org 4097
JMP @back
