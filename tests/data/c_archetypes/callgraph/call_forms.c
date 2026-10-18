/* Where calls appear: statements, conditions, loops, returns, arguments. */

int engine_step(struct engine *e) {
  const ticks tic = getticks();
  direct(e);
  if (in_condition(e)) e->step++;
  for (int i = 0; in_loop_condition(e, i); i++) in_loop_body(e);
  while (in_while(e)) {
  }
  const int x = in_initialiser(e) ? in_ternary(e) : 0;
  nested_outer(nested_inner(e));
  e->callback = assigned_pointer;
  memcpy(e, e, sizeof(*e));
  MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  engine_step(e);
  if (e->verbose)
    message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
            clocks_getunit());
  return in_return(e);
}

void direct(struct engine *e) {}
int in_condition(struct engine *e) { return 1; }
int in_loop_condition(struct engine *e, int i) { return i < 1; }
void in_loop_body(struct engine *e) {}
int in_while(struct engine *e) { return 0; }
int in_initialiser(struct engine *e) { return 0; }
int in_ternary(struct engine *e) { return 0; }
int nested_inner(struct engine *e) { return 0; }
void nested_outer(int x) {}
void assigned_pointer(void) {}
int in_return(struct engine *e) { return 0; }
