CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Werror -Iinclude
LDFLAGS ?=

BUILD := build

.PHONY: all test clean model model-check model-ctx model-ctx-check model-lm model-lm-check lane-check

all: test

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_word: tests/test_word.c include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tests/test_word.c -o $@ $(LDFLAGS)

$(BUILD)/test_vec: tests/test_vec.c src/nanlm_vec.c include/nanlm_vec.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tests/test_vec.c src/nanlm_vec.c -o $@ $(LDFLAGS)

$(BUILD)/test_model: tests/test_model.c src/nanlm_model.c src/nanlm_vec.c include/nanlm_model.h include/nanlm_vec.h include/nan16_word.h | $(BUILD)
	$(CC) $(CFLAGS) tests/test_model.c src/nanlm_model.c src/nanlm_vec.c -o $@ $(LDFLAGS)

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

test: $(BUILD)/test_word $(BUILD)/test_vec $(BUILD)/test_model $(BUILD)/test_ctx
	./$(BUILD)/test_word
	./$(BUILD)/test_vec
	./$(BUILD)/test_model
	./$(BUILD)/test_ctx

model: $(BUILD)/nanlm_train
	./$(BUILD)/nanlm_train --out models/nanlm0.nas --expected models/nanlm0.expected --manifest models/nanlm0.manifest

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

clean:
	rm -rf $(BUILD) models/nanlm0.nas models/nanlm0.nan models/nanlm0.expected models/nanlm0.manifest models/nanlm_ctx.nas models/nanlm_ctx.nan models/nanlm_ctx.expected models/nanlm_ctx.manifest models/nanlm_lm.nas models/nanlm_lm.nan models/nanlm_lm.expected models/nanlm_lm.manifest
