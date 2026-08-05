/*
 * pickle_selftest_main.c — tiny main() for the standalone selftest binary.
 * Copyright (c) 2026 lestramk.org / Lee Muriihi Kingori
 *
 * Just calls pickle_selftest() on the embedded demo GGUF and prints the
 * result. Mirrors the in-kernel boot-time selftest path so a host user
 * can verify the build with `./pickle_selftest`.
 */
#include "pickle.h"

#include <stdio.h>

int main(void) {
    int32_t t = -1;
    int rc = pickle_selftest(&t);
    printf("selftest rc=%d token=%d\n", rc, (int)t);
    return rc;
}
