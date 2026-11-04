; regime_all.nas - self-modify through all 16 truth tables at runtime.
; For each n, write class n into a BOOL slot, execute it, and check
; BOOL.n(0xCCCCCCCC, 0xAAAAAAAA) == n * 0x11111111.
.org 0
main:
MOVI r13,1
; operands
MOVI r1,0xCCCC
MOVHI r1,0xCCCC
MOVI r2,0xAAAA
MOVHI r2,0xAAAA
; instruction payload: BOOL d=3, a=1, b=2  => 0x00310002
MOVI r5,0x0002
MOVHI r5,0x0031
; slot address 4096
MOVI r6,0x1000
MOVHI r6,0
; n = 0
MOVI r7,0
loop:
MOVI r8,16
CMP r7,r8
JMP[GE] @pass
; rewrite the slot with class n and execute it
STW r5,r6,r7
JMPR r6
back:
; expected = n * 0x11111111
MOVI r8,0x1111
MOVHI r8,0x1111
MUL r9,r7,r8
CMP r3,r9
JMP[NE] @fail
ADD r7,r7,r13
JMP @loop
pass:
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
.org 4096
.word 0x00310002
.org 4097
JMP @back
