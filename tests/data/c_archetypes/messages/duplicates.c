/* Repeated messages in one function. */

void space_split(struct space *s, int a) {
  ticks tic = getticks();
  if (a) {
    if (s->e->verbose)
      message("Splitting took %.3f %s.", clocks_from_ticks(getticks() - tic),
              clocks_getunit());
  } else {
    tic = getticks();
    if (s->e->verbose)
      message("Splitting took %.3f %s.", clocks_from_ticks(getticks() - tic),
              clocks_getunit());
  }
  if (s->e->verbose)
    message("Splitting took: %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
  if (s->e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
  if (s->e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}
