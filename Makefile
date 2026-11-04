CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Werror -Iinclude
LDFLAGS ?=

BUILD := build

.PHONY: all test clean model model-check model-ctx model-ctx-check model-lm model-lm-check model-slot model-slot-check slot8-check shard-check shard2-check mixer-check eval-check pack-check plasticity-check shard-slot-check teacher-check teacher-slot-check continuous-check regime-check layers-check lut-check lane-check manifest

all: test

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_word: tests/test_word.c include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tests/test_word.c -o $@ $(LDFLAGS)

$(BUILD)/test_vec: tests/test_vec.c src/nanlm_vec.c include/nanlm_vec.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tests/test_vec.c src/nanlm_vec.c -o $@ $(LDFLAGS)

$(BUILD)/test_model: tests/test_model.c src/nanlm_model.c src/nanlm_vec.c include/nanlm_model.h include/nanlm_vec.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tests/test_model.c src/nanlm_model.c src/nanlm_vec.c -o $@ $(LDFLAGS)

$(BUILD)/test_fast: tests/test_fast.c src/nanlm_fast.c include/nanlm_fast.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tests/test_fast.c src/nanlm_fast.c -o $@ $(LDFLAGS)

$(BUILD)/nanlm_train: tools/nanlm_train.c src/nanlm_model.c src/nanlm_vec.c include/nanlm_model.h include/nanlm_vec.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_train.c src/nanlm_model.c src/nanlm_vec.c -o $@ $(LDFLAGS)

$(BUILD)/test_ctx: tests/test_ctx.c src/nanlm_ctx.c src/nanlm_vec.c include/nanlm_ctx.h include/nanlm_vec.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tests/test_ctx.c src/nanlm_ctx.c src/nanlm_vec.c -o $@ $(LDFLAGS)

$(BUILD)/nanlm_ctx_train: tools/nanlm_ctx_train.c src/nanlm_ctx.c src/nanlm_vec.c include/nanlm_ctx.h include/nanlm_vec.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_ctx_train.c src/nanlm_ctx.c src/nanlm_vec.c -o $@ $(LDFLAGS)

$(BUILD)/nanlm_lm_train: tools/nanlm_lm_train.c src/nanlm_lm.c src/nanlm_vec.c include/nanlm_lm.h include/nanlm_vec.h include/nanlm_runtime.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_lm_train.c src/nanlm_lm.c src/nanlm_ctx.c src/nanlm_vec.c -o $@ $(LDFLAGS) -lm

$(BUILD)/nanlm_asm: tools/nanlm_asm.c src/nanlm_asm.c include/nanlm_asm.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_asm.c src/nanlm_asm.c -o $@ $(LDFLAGS)

$(BUILD)/nanlm_vm: tools/nanlm_vm.c src/nanlm_vm.c include/nanlm_vm.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_vm.c src/nanlm_vm.c -o $@ $(LDFLAGS)

$(BUILD)/nanlm_slot_train: tools/nanlm_slot_train.c src/nanlm_slot.c src/nanlm_fast.c include/nanlm_slot.h include/nanlm_fast.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_slot_train.c src/nanlm_slot.c src/nanlm_fast.c -o $@ $(LDFLAGS) -lm

$(BUILD)/nanlm_slot8_train: tools/nanlm_slot8_train.c src/nanlm_slot8.c src/nanlm_teacher.c src/nanlm_fast.c include/nanlm_slot8.h include/nanlm_slot.h include/nanlm_teacher.h include/nanlm_fast.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_slot8_train.c src/nanlm_slot8.c src/nanlm_teacher.c src/nanlm_fast.c -o $@ $(LDFLAGS) -lm

$(BUILD)/nanlm_shard: tools/nanlm_shard.c src/nanlm_shard.c include/nanlm_shard.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_shard.c src/nanlm_shard.c -o $@ $(LDFLAGS)

$(BUILD)/nanlm_shard2: tools/nanlm_shard2.c src/nanlm_shard2.c include/nanlm_shard2.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_shard2.c src/nanlm_shard2.c -o $@ $(LDFLAGS)

$(BUILD)/nanlm_mixer: tools/nanlm_mixer.c src/nanlm_mixer.c include/nanlm_mixer.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_mixer.c src/nanlm_mixer.c -o $@ $(LDFLAGS) -lm

$(BUILD)/nanlm_teacher: tools/nanlm_teacher.c src/nanlm_teacher.c include/nanlm_teacher.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_teacher.c src/nanlm_teacher.c -o $@ $(LDFLAGS) -lm

$(BUILD)/nanlm_shard_slot: tools/nanlm_shard_slot.c src/nanlm_shard.c src/nanlm_slot.c src/nanlm_fast.c include/nanlm_shard.h include/nanlm_slot.h include/nanlm_fast.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_shard_slot.c src/nanlm_shard.c src/nanlm_slot.c src/nanlm_fast.c -o $@ $(LDFLAGS) -lm

$(BUILD)/nanlm_teacher_slot: tools/nanlm_teacher_slot.c src/nanlm_slot.c src/nanlm_teacher.c src/nanlm_fast.c include/nanlm_slot.h include/nanlm_teacher.h include/nanlm_fast.h | $(BUILD)
	$(CC) $(CFLAGS) tools/nanlm_teacher_slot.c src/nanlm_slot.c src/nanlm_teacher.c src/nanlm_fast.c -o $@ $(LDFLAGS) -lm

test: $(BUILD)/test_word $(BUILD)/test_vec $(BUILD)/test_model $(BUILD)/test_ctx $(BUILD)/test_fast
	./$(BUILD)/test_word
	./$(BUILD)/test_vec
	./$(BUILD)/test_model
	./$(BUILD)/test_ctx
	./$(BUILD)/test_fast

model: $(BUILD)/nanlm_train
	./$(BUILD)/nanlm_train --out models/nanlm0.nas --expected models/nanlm0.expected --manifest models/nanlm0.manifest

lut-check: $(BUILD)/nanlm_asm $(BUILD)/nanlm_vm
	sh tools/build_lut.sh

layers-check: $(BUILD)/nanlm_asm $(BUILD)/nanlm_vm
	./$(BUILD)/nanlm_asm tests/layers.nas $(BUILD)/layers.nan
	./$(BUILD)/nanlm_vm --img $(BUILD)/layers.nan --steps 100000 > $(BUILD)/layers.out 2> $(BUILD)/layers.err
	grep -q "rc=3, .*trap=''" $(BUILD)/layers.err
	grep -qx PASS $(BUILD)/layers.out

regime-check: $(BUILD)/nanlm_asm $(BUILD)/nanlm_vm
	./$(BUILD)/nanlm_asm tests/regime.nas $(BUILD)/regime.nan
	./$(BUILD)/nanlm_vm --img $(BUILD)/regime.nan --steps 100000 > $(BUILD)/regime.out 2> $(BUILD)/regime.err
	grep -q "rc=3, .*trap=''" $(BUILD)/regime.err
	grep -qx PASS $(BUILD)/regime.out
	./$(BUILD)/nanlm_asm tests/regime_all.nas $(BUILD)/regime_all.nan
	./$(BUILD)/nanlm_vm --img $(BUILD)/regime_all.nan --steps 100000 > $(BUILD)/regime_all.out 2> $(BUILD)/regime_all.err
	grep -q "rc=3, .*trap=''" $(BUILD)/regime_all.err
	grep -qx PASS $(BUILD)/regime_all.out

lane-check: $(BUILD)/nanlm_asm $(BUILD)/nanlm_vm
	./$(BUILD)/nanlm_asm tests/lane.nas $(BUILD)/lane.nan
	./$(BUILD)/nanlm_vm --img $(BUILD)/lane.nan --steps 100000 > $(BUILD)/lane.out 2> $(BUILD)/lane.err
	grep -q "rc=3, .*trap=''" $(BUILD)/lane.err
	grep -qx PASS $(BUILD)/lane.out

model-check: $(BUILD)/nanlm_train
	sh tools/build_model.sh

model-ctx: $(BUILD)/nanlm_ctx_train
	./$(BUILD)/nanlm_ctx_train --out models/nanlm_ctx.nas --expected models/nanlm_ctx.expected --manifest models/nanlm_ctx.manifest

model-ctx-check: $(BUILD)/nanlm_ctx_train
	sh tools/build_ctx.sh

model-lm: $(BUILD)/nanlm_lm_train
	./$(BUILD)/nanlm_lm_train --out models/nanlm_lm.nas --expected models/nanlm_lm.expected --manifest models/nanlm_lm.manifest

model-lm-check: $(BUILD)/nanlm_lm_train
	sh tools/build_lm.sh

model-slot: $(BUILD)/nanlm_slot_train
	./$(BUILD)/nanlm_slot_train --out models/nanlm_slot.nas --expected models/nanlm_slot.expected --manifest models/nanlm_slot.manifest

model-slot-check: $(BUILD)/nanlm_slot_train
	sh tools/build_slot.sh

slot8-check: $(BUILD)/nanlm_slot8_train
	sh tools/build_slot8.sh

shard-check: $(BUILD)/nanlm_shard
	sh tools/build_shard.sh

shard2-check: $(BUILD)/nanlm_shard2
	sh tools/build_shard2.sh

mixer-check: $(BUILD)/nanlm_mixer
	sh tools/build_mixer.sh

eval-check: $(BUILD)/nanlm_slot_train
	sh tools/build_eval.sh

pack-check:
	sh tools/build_pack.sh

plasticity-check: $(BUILD)/nanlm_asm $(BUILD)/nanlm_vm
	sh tools/build_plasticity.sh

shard-slot-check: $(BUILD)/nanlm_shard_slot
	sh tools/build_shard_slot.sh

continuous-check: $(BUILD)/nanlm_shard $(BUILD)/nanlm_shard_slot $(BUILD)/nanlm_asm $(BUILD)/nanlm_vm
	sh tools/build_continuous.sh

teacher-check: $(BUILD)/nanlm_teacher
	sh tools/build_teacher.sh

teacher-slot-check: $(BUILD)/nanlm_teacher_slot
	sh tools/build_teacher_slot.sh

manifest:
	sh tools/manifest.sh

clean:
	rm -rf $(BUILD) models/nanlm0.nas models/nanlm0.nan models/nanlm0.expected models/nanlm0.manifest models/nanlm_ctx.nas models/nanlm_ctx.nan models/nanlm_ctx.expected models/nanlm_ctx.manifest models/nanlm_lm.nas models/nanlm_lm.nan models/nanlm_lm.expected models/nanlm_lm.manifest models/nanlm_slot.nas models/nanlm_slot.nan models/nanlm_slot.expected models/nanlm_slot.manifest
