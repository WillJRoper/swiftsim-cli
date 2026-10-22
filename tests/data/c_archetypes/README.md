# Archetypal C syntax for the timer parser

Small C files covering the syntax SWIFT actually writes (harvested from
`src/`, `swift.c` and `swift_fof.c`), each with the results
`swiftsim_cli.timing.timer_db` must produce. They are tested by
`tests/unit/test_timer_db_archetypes.py`, which discovers every case.

## Layout

```
c_archetypes/<category>/<case>.yaml   # expectations (one per case)
c_archetypes/<category>/<case>.c      # the source, or for multi-file cases
c_archetypes/<category>/<case>/       # a directory of .c/.h files
```

Categories: `functions/` (definition forms), `messages/` (timer message
forms, units, labels, tics), `lexical/` (comments, literals, preprocessor),
`callgraph/` (call edges and reachability).

Adding a case = adding `<case>.c` (or `<case>/`) plus `<case>.yaml`. The
sources are copied into an empty directory and parsed with
`build_timer_db`, so site ids are `<file>:<line>` with `<file>` relative to
that directory (`<case>.c:12`, or `engine.c:7` inside `<case>/`).

## Expectation keys

All keys except `description` are optional; each one present becomes one
parametrised test for the case.

| Key | Checks |
| --- | --- |
| `description` | What the case covers (required). |
| `functions` | Exact map of function definitions to `[first, last]` line (a list of spans if defined more than once). Keys are names, or `file:name` for multi-file cases. Only tree-sitter definitions: functions found by the brace-counting fallback are not listed. |
| `sites` | Exact set of timer site ids, each mapped to a subset of `function`, `kind` (`function`/`operation`), `label`, `tic` (start variable, `null` if none), `start` (line of the tic) and `format` (unescaped, concatenated format string). |
| `timers` | Exact grouping of sites into logical timers (list of site lists). |
| `calls` | Per listed caller, the exact callees: a list, or a map `callee: [call-site lines]` (lines are only recorded for functions with operation timers). `caller: []` asserts no edges. |
| `reach` | Per function, the exact set of reachable functions. |
| `op_reach` | Per operation site, the exact set reachable from calls in its `[tic, message]` range. |
| `lines` | Per site, `match: [{body, ms, key?}]` log-line bodies (text after `function: `) with the parsed milliseconds and display key, and `reject: [body]` near misses. |
| `log` | `text` (a log excerpt, with `[rank]` and `[elapsed]` prefixes) run through `scan_log`; `records: [[rank, function, key, ms]]` in log order and `unmatched: {"function: label": count}`. |
| `xfail` | `{check: reason}` marks that check of the case as a known limitation (strict xfail: it fails loudly once fixed). Checks: the keys above plus `render`. |

Every case also gets `render`: each site's format is rendered printf-style
(values for each conversion, units `ms`/`s`/`us`) and must parse back to
the rendered time in ms and the expected key; a copy with the number
replaced, and (for labelled timers) one without the label, must not match.

## Known limitations (xfail)

- `functions/macro_named` (`log`): function names built by macro token
  pasting (`runner_doiact_functions_*.h`) are recorded as the macro token,
  not the expanded name the log prints.
- `messages/tic_untracked` (`op_reach`): start times kept in struct members
  (`e->tic_step = getticks()`) are not tics, so the operation's range is
  just the message line.

Documented behaviour that is not a bug (asserted as is):

- The tic is chosen lexically (the last assignment before the message of a
  variable the message uses), not by control flow (`messages/duplicates`).
- An integer conversion after "took" is the time when no float follows, so
  `took %lld ticks` reads ticks as ms (`messages/units_specs`).
- "took" may end a word (`IDstook`, `Undertook`) (`messages/not_timers`).
- Messages inside `#define` bodies, non-`message` calls (`error`,
  `printf`) and non-literal formats are not timers (`messages/not_timers`).
- Functions assigned to pointers are not call edges; functions passed as
  arguments (with or without `&`) are (`callgraph/`).
