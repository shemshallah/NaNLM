/* nanlm_slot8.h - 8-byte / 16-slot simplex slot student.
 *
 * This is the same slot model as nanlm_slot.h, compiled with an 8-byte
 * context: eight byte slots plus eight character-type slots. It consumes the
 * 8-byte contexts already emitted by the teacher format.
 */
#ifndef NANLM_SLOT8_H
#define NANLM_SLOT8_H

#define NANLM_SLOT_CTX_BYTES 8u
#include "nanlm_slot.h"

#endif /* NANLM_SLOT8_H */
