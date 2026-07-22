/*
 * Single-header inline build of the vendored xxHash (see xxhash.h,
 * v0.8.2, BSD 2-Clause, Copyright (C) 2012-2021 Yann Collet).
 *
 * Defining XXH_INLINE_ALL before the include makes every xxHash symbol
 * `static`, so the whole implementation is compiled directly into the kernel's
 * translation unit (_index.c): no libxxhash to link, and the exact XXH3 version
 * is pinned to the vendored header for digest stability.  XXH3 has been frozen
 * since v0.8.0, so this matches the pip `xxhash` package (which bundles the same
 * upstream 0.8.2) bit-for-bit — asserted by _index's import-time self-check.
 */
#ifndef POKER_AI_CORE_XXH3_H
#define POKER_AI_CORE_XXH3_H

#define XXH_INLINE_ALL
#include "xxhash.h"

#endif /* POKER_AI_CORE_XXH3_H */
