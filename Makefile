# Makefile for the pickle GGUF engine — host (POSIX) build.
# Copyright (c) 2026 lestramk.org / Lee Muriihi Kingori
#
# The SAME src/pickle.c / src/pickle_softfp.c / src/pickle.h source also
# compiles into the lestraOS kernel (with -DPICKLE_KERNEL). Here we build
# it with -UPICKLE_KERNEL so it uses <stdio.h>/<stdlib.h>/<string.h>
# instead of lestra kernel headers, plus a small POSIX shim
# (src/pickle_host.c) and a CLI frontend (src/pickle_cli.c).
#
# NOTE on layout: the spec asked for source under `pickle/` and the CLI
# binary at `./pickle`, but those two paths collide in POSIX (a directory
# entry name is unique — you cannot have both a `pickle/` directory and a
# `pickle` file at the same level). Source therefore lives under `src/`
# and the binaries (`./pickle`, `./pickle_selftest`) live at the repo
# root, so the spec's verification commands (`./pickle selftest`,
# `./pickle info …`, `./pickle infer …`, `./pickle dequant …`) work
# exactly as written.
#
# Targets:
#   make / make all        build ./pickle (CLI) and ./pickle_selftest
#   make pickle            build only ./pickle
#   make pickle_selftest   build only ./pickle_selftest
#   make clean             remove binaries and *.o
#   make test              build + run ./pickle_selftest and ./pickle selftest

CC      ?= cc
CFLAGS  ?= -O2
CFLAGS  += -Wall -Wextra \
           -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function \
           -Wno-strict-aliasing -Wno-aggressive-loop-optimizations \
           -UPICKLE_KERNEL -Isrc
LDFLAGS ?=
LDLIBS  ?= -lm

# ---- object files --------------------------------------------------
PICKLE_CORE_OBJS = \
	src/pickle.o \
	src/pickle_softfp.o \
	src/pickle_demo_gguf.o

CLI_OBJS = \
	$(PICKLE_CORE_OBJS) \
	src/pickle_host.o \
	src/pickle_cli.o

SELFTEST_OBJS = \
	$(PICKLE_CORE_OBJS) \
	src/pickle_selftest_main.o

# ---- top-level targets ---------------------------------------------
.PHONY: all clean test pickle pickle_selftest

all: pickle pickle_selftest

pickle: $(CLI_OBJS)
	$(CC) $(LDFLAGS) -o $@ $(CLI_OBJS) $(LDLIBS)

pickle_selftest: $(SELFTEST_OBJS)
	$(CC) $(LDFLAGS) -o $@ $(SELFTEST_OBJS) $(LDLIBS)

# ---- generic compile rule ------------------------------------------
src/%.o: src/%.c src/pickle.h
	$(CC) $(CFLAGS) -c -o $@ $<

# ---- test ----------------------------------------------------------
test: pickle_selftest pickle
	@echo "--- ./pickle_selftest ---"
	./pickle_selftest
	@echo "--- ./pickle selftest ---"
	./pickle selftest

# ---- clean ---------------------------------------------------------
# `rm -f pickle` would fail when `pickle/` is a directory; we use find
# with -type f so only the root-level *regular files* named `pickle`
# and `pickle_selftest` are deleted. Object files live under src/.
clean:
	rm -f src/*.o
	find . -maxdepth 1 -type f \( -name 'pickle' -o -name 'pickle_selftest' \) -delete
