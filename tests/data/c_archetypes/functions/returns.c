/* Return types: pointers, double pointers, structs, function pointers. */

struct cell *pointer_return(struct space *s) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  return s->cells_top;
}

struct cell **double_pointer_return(struct space *s) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  return &s->cells_top;
}

const char *const_char_return(int i) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  return "name";
}

struct gravity_tensors struct_by_value(const struct cell *c) {
  const ticks tic = getticks();
  struct gravity_tensors m = c->grav.multipole[0];
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  return m;
}

void (*function_pointer_return(int which))(void *, int, void *) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  return 0;
}

static INLINE struct part *restrict *restrict_pointer_return(void) {
  const ticks tic = getticks();
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
  return 0;
}
