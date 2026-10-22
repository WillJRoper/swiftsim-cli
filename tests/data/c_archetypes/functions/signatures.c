/* Parameter lists and layout: qualifiers, callbacks, comments, spacing. */

void restrict_params(struct part *restrict parts,
                     const struct xpart *restrict xparts, const int count) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}

void callback_param(void (*map_function)(void *map_data, int num_elements,
                                         void *extra_data),
                    void *data) {
  const ticks tic = getticks();
  map_function(data, 1, 0);
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}

void multi_line_commented(
    /* The engine. */ struct engine *e,
    // The number of things {
    const int n /* , int unused) { */) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}

void
odd_spacing ( int a ,
              int b )
{
  const ticks tic = getticks ( ) ;
  message ( "took %.3f %s." , clocks_from_ticks ( getticks ( ) - tic ) ,
            clocks_getunit ( ) ) ;
}

int knr_style(a, b)
int a;
int b;
{
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  return a + b;
}

void array_params(const double dim[3], int cdim[3], double mat[][3]) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}

void variadic(const char *fmt, ...) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}
