/* Synthetic excerpt of SWIFT's src/task.c for the name-parser tests. */
#include "task.h"

/* Task type names. */
const char *taskID_names[task_type_count] = {
    "none", "sort", "self", "pair", "drift_part",
    "send", "recv", "my_new_task",
};

/* Sub-task type names. */
const char *subtaskID_names[task_subtype_count] = {
    "none",     "density", "gradient", "force",
    "grav",     "progeny", "my_new_subtype",
};

const char *task_category_names[task_category_count] = {"drift", "sorts"};

void task_unlock(struct task *t) { message("%s", taskID_names[t->type]); }
