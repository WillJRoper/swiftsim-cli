/* Storage classes, INLINE macros and attributes, as written in SWIFT. */

void plain(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

static void static_fn(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

static INLINE double static_inline(const double *table, const double x) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  return table[0] * x;
}

__attribute__((always_inline)) INLINE static int attr_inline(
    struct part *restrict p, const int n) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  return n;
}

__attribute__((const)) static INLINE double attr_const(const double a) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  return a;
}

extern void extern_fn(void) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}
