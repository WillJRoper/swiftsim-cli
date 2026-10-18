/* Function headers built by macros (runner_doiact_functions_*.h style). */

#define PASTE(x, y) x##_##y
#define _DOPAIR1(f) PASTE(runner_dopair1, f)
#define DOPAIR1 _DOPAIR1(FUNCTION)
#define DEFINE_TIMED(name) void name(struct runner *r)

void DOPAIR1(struct runner *r, struct cell *ci, struct cell *cj) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}

DEFINE_TIMED(runner_macro_header) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}
