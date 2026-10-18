/* Comments: commented-out timers and comments that look like timers.
 * message("took %.3f %s.", clocks_from_ticks(getticks() - tic), ...); */

void commented(struct engine *e) {
  const ticks tic = getticks();
  // message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
  //         clocks_getunit());
  /* message("Block took %.3f %s.", clocks_from_ticks(getticks() - tic),
             clocks_getunit()); */
  /* The format is "took %.3f %s." { */
  if (e->verbose)
    message(/* the format */ "Live took %.3f %s.", // trailing comment }
            clocks_from_ticks(getticks() - tic) /* ms */, clocks_getunit());
}

/* } A stray closing brace in a comment between functions. */

void next_function(struct engine *e) {
  const ticks tic = getticks();
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}
