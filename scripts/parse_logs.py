"""Collect RESULT lines from job logs into one CSV per experiment.

Every executable prints its measurements as
    RESULT exp=E3 key=value key=value ...
so this script needs no knowledge of individual experiments. Training curves
(EPOCH lines), MoE load over training (LOAD lines, --log-every) and the
test-set routing per class and confusion matrix (ROUTE / CONF lines, counts
separated by ':') go to <exp>_epoch.csv, <exp>_load.csv, <exp>_route.csv and
<exp>_conf.csv, tagged with the log name. Logs stay in
results/raw/ (not submitted, L12 p36); only the CSVs feed the report.
"""
import argparse
import collections
import csv
import pathlib


def parse_line(line):
    fields = {}
    for token in line.split()[1:]:
        key, _, value = token.partition("=")
        fields[key] = value
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", default="results/raw")
    parser.add_argument("--out", default="results/tables")
    args = parser.parse_args()

    rows = collections.defaultdict(list)
    for log in sorted(pathlib.Path(args.raw).rglob("*")):
        if not log.is_file():
            continue
        # superseded or failed runs (e.g. wrong placement) are kept for the record, never reported
        if any(part.startswith(("_superseded", "_failed")) for part in log.relative_to(args.raw).parts):
            continue
        for line in log.read_text(errors="replace").splitlines():
            for tag, suffix in (("RESULT ", ""), ("EPOCH ", "_epoch"), ("LOAD ", "_load"), ("ROUTE ", "_route"),
                                ("CONF ", "_conf")):
                if line.startswith(tag):
                    fields = parse_line(line)
                    fields["log"] = log.name
                    # results/raw/<platform>/<exp>/<run>.log: which cluster produced it
                    parts = log.relative_to(args.raw).parts
                    # results/raw/local_<exp>/: laptop development runs, never reported
                    fields["platform"] = parts[0] if len(parts) >= 3 else ("local" if parts[0].startswith("local") else "unknown")
                    rows[fields.get("exp", "unknown") + suffix].append(fields)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for exp, records in sorted(rows.items()):
        columns = list(dict.fromkeys(k for r in records for k in r))
        with open(out / f"{exp}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        print(f"{exp}: {len(records)} rows -> {out / f'{exp}.csv'}")


if __name__ == "__main__":
    main()
