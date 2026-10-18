/* Calls across files resolve by name. */

void engine_rebuild(struct engine *e) {
  const ticks tic = getticks();
  space_rebuild(e->s);
  if (e->verbose)
    message("Space rebuild took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}
