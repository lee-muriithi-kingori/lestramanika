/*
 * version.h — single source of truth for the pickle version string.
 * Copyright (c) 2026 lestramk.org / Lee Muriithi Kingori
 *
 * Included by the CLI so `pickle --version` / `pickle version` reports the
 * same version the release was cut at. Bump these three macros in lockstep
 * when cutting a release; PICKLE_VERSION_STRING is composed from them so it
 * can never drift out of sync.
 *
 * Release policy (semantic versioning):
 *   MAJOR  incompatible API change to pickle_core / pickle_host
 *   MINOR  new feature, new subcommand, new tensor type — backwards compatible
 *   PATCH  bug fix or perf improvement — no behaviour change for valid input
 */
#ifndef PICKLE_VERSION_H
#define PICKLE_VERSION_H

#define PICKLE_VERSION_MAJOR  1
#define PICKLE_VERSION_MINOR  0
#define PICKLE_VERSION_PATCH  0

/* "pickle 1.0.0" — the form `pickle --version` prints. */
#define PICKLE_VERSION_STRING \
    "pickle " PICKLE_STR_(PICKLE_VERSION_MAJOR) "." \
                PICKLE_STR_(PICKLE_VERSION_MINOR) "." \
                PICKLE_STR_(PICKLE_VERSION_PATCH)

/* Numeric form for programmatic comparison, e.g. PICKLE_VERSION_INT >= 0x10000. */
#define PICKLE_VERSION_INT \
    ((PICKLE_VERSION_MAJOR << 16) | (PICKLE_VERSION_MINOR << 8) | \
     PICKLE_VERSION_PATCH)

#define PICKLE_STR_(x)  PICKLE_STR2_(x)
#define PICKLE_STR2_(x) #x

#endif /* PICKLE_VERSION_H */
