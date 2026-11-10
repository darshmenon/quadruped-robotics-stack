#!/bin/bash
# usage: watch_load.sh <pid> [logfile] [limit]; SIGTERMs the run if 1-min load >= limit (default 16) for 3 checks in a row
P=$1; LOG=${2:-logs/run_logs/watch_load.log}; LIM=${3:-16}; hi=0
while kill -0 $P 2>/dev/null; do
  l=$(cut -d' ' -f1 /proc/loadavg); if [ "${l%.*}" -ge $LIM ]; then hi=$((hi+1)); else hi=0; fi
  echo "$(date +%T) load=$l hi=$hi" >> $LOG
  if [ $hi -ge 3 ]; then pkill -TERM -P $P; kill -TERM $P; echo "$(date +%T) STOPPED run (load $l)" >> $LOG; exit 0; fi
  sleep 30
done
echo "$(date +%T) run ended" >> $LOG
