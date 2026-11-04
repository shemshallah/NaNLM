; layers.nas - one payload, three interpretations.
; Layer A: 32-bit scalar add.
; Layer B: 32-lane Boolean complement (BOOL.6).
; Layer C: byte-lane mask (BOOL.8).
; Layer D: byte-lane OR (BOOL.14).
; Prints PASS if every interpretation is exact.
.org 0
main:
MOVI r13,1
; w = 0x00010203
MOVI r1,0x0203
MOVHI r1,0x0001
; A: scalar add 0x01010101 -> 0x01020304
MOVI r2,0x0101
MOVHI r2,0x0101
ADD r3,r1,r2
MOVI r4,0x0304
MOVHI r4,0x0102
CMP r3,r4
JMP[NE] @fail
; B: complement -> 0xFFFEFDFC
MOVI r5,0xFFFF
MOVHI r5,0xFFFF
BOOL.6 r3,r1,r5
MOVI r4,0xFDFC
MOVHI r4,0xFFFE
CMP r3,r4
JMP[NE] @fail
; C: 0x00010203 & 0x00FF00FF -> 0x00010003
MOVI r5,0x00FF
MOVHI r5,0x00FF
BOOL.8 r3,r1,r5
MOVI r4,0x0003
MOVHI r4,0x0001
CMP r3,r4
JMP[NE] @fail
; D: 0x00010203 | 0x80808080 -> 0x80818283
MOVI r5,0x8080
MOVHI r5,0x8080
BOOL.14 r3,r1,r5
MOVI r4,0x8283
MOVHI r4,0x8081
CMP r3,r4
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
