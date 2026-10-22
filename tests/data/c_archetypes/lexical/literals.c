/* Braces and quotes inside string and character literals. */

void literal_braces(struct engine *e) {
  const ticks tic = getticks();
  const char open = '{';
  const char close = '}';
  const char quote = '"';
  const char *s = "}}} a closing run {";
  const char *t = "escaped \" quote } and \\";
  if (e->verbose)
    message("Literal {braces} took %.3f %s.",
            clocks_from_ticks(getticks() - tic), clocks_getunit());
}

void after_literals(struct engine *e) {
  const ticks tic = getticks();
  const char *url = "http://www.swiftsim.com/* not a comment */";
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
}
