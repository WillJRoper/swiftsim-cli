/* Calls that mention "took" but are not timers. */

#define REPORT_TIME(tic) \
  message("Macro took %.3f %s.", clocks_from_ticks(getticks() - (tic)), \
          clocks_getunit())

void not_timers(struct engine *e, const char *fmt) {
  const ticks tic = getticks();
  message("This took a while.");
  message("Nothing took place in %s.", "space");
  message(fmt, clocks_from_ticks(getticks() - tic), clocks_getunit());
  error("Rebuild took %.3f %s which is too long.",
        clocks_from_ticks(getticks() - tic), clocks_getunit());
  printf("Printing took %.3f %s.\n", clocks_from_ticks(getticks() - tic),
         clocks_getunit());
  fprintf(stderr, "took %.3f %s.\n", 1.0, "ms");
  pretty_message("took %.3f %s.", 1.0, "ms");
  REPORT_TIME(tic);
  message("Undertook nothing %.3f %s.", 1.0, "ms");
}
