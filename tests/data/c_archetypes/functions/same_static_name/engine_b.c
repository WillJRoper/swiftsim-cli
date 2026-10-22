/* Same static names as engine_a.c: identical lines are indistinguishable. */

static void make_tasks(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}

static void link_tasks(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose)
    message("Linking B took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}
