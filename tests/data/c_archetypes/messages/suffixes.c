/* Text after the time (engine.c, space.c). */

void engine_prepare(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose)
    message("took %.3f %s (including unskip, rebuild and reweight).",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void engine_marktasks(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose)
    message("took %.3f %s to find active cells.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
  if (e->verbose)
    message("took %.3f %s to duplicate the list.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
  if (e->verbose)
    message("Marking took %.3f %s (%d cells).",
            clocks_from_ticks(getticks() - tic), clocks_getunit(), 5);
}
