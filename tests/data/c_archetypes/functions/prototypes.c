/* Declarations that are not definitions must not create functions. */

void proto_only(struct engine *e);
static INLINE void static_proto(int n);
extern struct cell *extern_proto(const struct space *s);
void (*function_pointer_variable)(int) = 0;
typedef void (*mapper_t)(void *, int, void *);
struct ops {
  void (*fn)(int);
};

void defined_caller(struct engine *e) {
  const ticks tic = getticks();
  proto_only(e);
  static_proto(1);
  function_pointer_variable(2);
  message("took %.3f %s.", clocks_from_ticks(getticks() - tic),
          clocks_getunit());
}
