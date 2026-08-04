#!/bin/bash
# Submit a pattern sweep whose job name, logs and RESULT_TAG all agree.
#
#     bash scripts/submit_patterns.sh <tag> [qsub flags] [VAR=VALUE ...]
#
#     bash scripts/submit_patterns.sh luma -t 1-5 \
#         LABELS=/home/ucabim3/Scratch/luma_labels_253.csv LABEL_COL=label \
#         TASK=auto AB_SKIP=1 ARMS="pw-radius@gin+deg hg-radius@deepsets2"
#
# qstat then shows ps_luma instead of pat_seed, and the logs land in
# logs/ps_luma.<jobid>.<task>.{out,err}.
#
# SGE reads the "#$" directives out of the script before any shell runs, so
# -N cannot reference $RESULT_TAG from inside run_patterns_array.sh. Passing it
# on the command line is the only way to name a job after what it is running.
#
# Anything starting with "-" is forwarded to qsub; anything of the form VAR=VAL
# becomes part of -v. RESULT_TAG is set from <tag>, so do not pass it twice.

set -uo pipefail

TAG="${1:?usage: submit_patterns.sh <tag> [qsub flags] [VAR=VALUE ...]}"
shift

# qstat's default name column is 10 characters wide. A longer name is not an
# error but is displayed truncated, which defeats the point of naming it.
case "$TAG" in
    *[!a-zA-Z0-9_]*) echo "FATAL: tag must be alphanumeric/underscore: $TAG" >&2; exit 1 ;;
    [!a-zA-Z]*)      echo "FATAL: tag must start with a letter: $TAG" >&2; exit 1 ;;
esac
NAME="ps_$TAG"
[ ${#NAME} -le 10 ] || echo "note: qstat truncates to 10 chars, will show ${NAME:0:10}"

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_patterns_array.sh"
[ -f "$SCRIPT" ] || { echo "FATAL: missing $SCRIPT" >&2; exit 1; }

QSUB_ARGS=()
VARS=("RESULT_TAG=$TAG")
# "-l h_rt=2:0:0" and "AB_SKIP=1" both look like <flag> <thing containing =>, so
# a flag's value cannot be told from a variable by inspection -- the arity of
# each flag has to be known. Anything not listed is assumed to take no value;
# "--" forces everything after it to be treated as variables.
while [ $# -gt 0 ]; do
    case "$1" in
        RESULT_TAG=*) echo "FATAL: RESULT_TAG comes from <tag>, drop it" >&2; exit 1 ;;
        --)           shift
                      while [ $# -gt 0 ]; do VARS+=("$1"); shift; done ;;
        -pe)          QSUB_ARGS+=("$1" "${2:?-pe needs an environment}" \
                                       "${3:?-pe needs a slot count}"); shift 3 ;;
        -t|-tc|-l|-hold_jid|-q|-P|-p|-m|-M|-ac|-jc|-a|-A)
                      QSUB_ARGS+=("$1" "${2:?$1 needs a value}"); shift 2 ;;
        -*)           QSUB_ARGS+=("$1"); shift ;;
        *=*)          VARS+=("$1"); shift ;;
        *)            echo "FATAL: unrecognised argument: $1" >&2; exit 1 ;;
    esac
done

# Joined with commas for -v. $JOB_ID/$TASK_ID stay single-quoted so SGE expands
# them at dispatch rather than the shell expanding them to nothing here.
V=$(IFS=,; echo "${VARS[*]}")
LOGS=/home/ucabim3/Scratch/logs

echo "name    $NAME"
echo "vars    $V"
echo "qsub    ${QSUB_ARGS[*]:-<none>}"
echo
[ -n "${DRYRUN:-}" ] && { echo "DRYRUN set, not submitting"; exit 0; }

qsub -N "$NAME" \
     -o "$LOGS/$NAME.\$JOB_ID.\$TASK_ID.out" \
     -e "$LOGS/$NAME.\$JOB_ID.\$TASK_ID.err" \
     "${QSUB_ARGS[@]}" -v "$V" "$SCRIPT"
