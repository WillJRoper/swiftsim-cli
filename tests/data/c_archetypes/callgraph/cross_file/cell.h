/* A static inline function defined in a header. */

void cell_split(struct cell *c);

__attribute__((always_inline)) INLINE static void cell_split_inline(
    struct cell *c) {
  cell_split(c);
}

void cell_split(struct cell *c) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}
