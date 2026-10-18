/* A static helper whose name is reused in engine_b.c. */

static void make_tasks(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

static void link_tasks(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose)
    message("Linking A took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}
