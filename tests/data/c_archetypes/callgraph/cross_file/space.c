/* The callee of engine.c, and its own callee in a header. */

#include "cell.h"

void space_rebuild(struct space *s) {
  const ticks tic = getticks();
  cell_split(s->cells_top);
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}
