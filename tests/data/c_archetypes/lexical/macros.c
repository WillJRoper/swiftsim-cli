/* Preprocessor definitions with line continuations, and a long line. */

#define TIMER_TOC(t)                                       \
  do {                                                     \
    if (verbose) {                                         \
      message("took %.3f %s.", clocks_from_ticks(getticks() - t), \
              clocks_getunit());                           \
  } while (0)

#define OPEN_BRACE {
#define CLOSE_BRACE }

void after_macros(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose)
    message("A very long label that goes on and on well past any sensible line length because some messages in SWIFT are not wrapped by clang-format at all took %.3f %s.", clocks_from_ticks(getticks() - tic), clocks_getunit());
}
