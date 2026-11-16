"""Number tables and figures in report/report.md and build the PDF.

The source refers to tables and figures by label, "Table [e3]" and "Figure [e3]",
so sections can move freely. This script numbers each kind in order of first
appearance, writes report/build/report.md (image paths adjusted) and calls the
write-report skill's build_pdf.py on it.

    python scripts/build_report.py              # report only
    python scripts/build_report.py --appendix   # report + README + source code (submission)
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "report" / "report.md"
OUT = ROOT / "report" / "build" / "report.md"
BUILD_PDF = pathlib.Path.home() / ".claude/skills/write-report/scripts/build_pdf.py"


# source files in the code appendix, in reading order (README first, then core -> parallel -> mains -> tests)
APPENDIX = ["README.md", "Makefile",
            "src/core/config.hpp", "src/core/args.hpp", "src/core/options.hpp", "src/core/timer.hpp", "src/core/rng.hpp",
            "src/core/memory.hpp", "src/core/tensor.hpp", "src/core/data.hpp", "src/core/data.cpp",
            "src/backend/backend.hpp", "src/backend/cpu_omp.cpp", "src/backend/cuda/cuda_backend.cu",
            "src/core/layers.hpp", "src/core/layers.cpp", "src/core/moe.hpp", "src/core/moe.cpp",
            "src/core/model.hpp", "src/core/model.cpp", "src/core/optim.hpp", "src/core/train.hpp", "src/core/train.cpp",
            "src/parallel/allreduce.hpp", "src/parallel/allreduce.cpp", "src/parallel/alltoall.hpp", "src/parallel/alltoall.cpp",
            "src/serial/main_serial.cpp", "src/mpi/main_parallel.cpp",
            "bench/pingpong.cpp", "bench/gemm_bench.cpp", "bench/allreduce_bench.cpp",
            "tests/test_gemm.cpp", "tests/test_gradcheck.cpp", "tests/test_allreduce.cpp", "tests/test_alltoall.cpp",
            "tests/compare_params.cpp", "tests/test_equivalence.sh",
            "scripts/get_data.py", "scripts/sweep.py", "scripts/parse_logs.py", "scripts/plot.py", "scripts/cost_model.py"]
LANG = {".cpp": "cpp", ".hpp": "cpp", ".cu": "cpp", ".py": "python", ".sh": "bash", "Makefile": "makefile"}


def appendix() -> str:
    parts = ["\n\n## Appendix: README and Source Code\n",
             "The complete source of the submission, file by file. Build and run instructions are in the README "
             "(first file). Job logs are not included.\n"]
    for rel in APPENDIX:
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"appendix file missing: {rel}")
        body = path.read_text()
        lang = LANG.get(path.suffix, LANG.get(path.name, ""))
        lines = body.rstrip().count("\n") + 1
        parts.append(f"\n### {rel} ({lines} lines)\n")
        parts.append(f"```{lang}\n{body.rstrip()}\n```\n" if rel != "README.md" else f"```text\n{body.rstrip()}\n```\n")
    # code may break across pages and is set smaller than the report text
    return ('<div class="code" markdown="1">\n' + "".join(parts) + "\n</div>\n"
            "<style>.code pre { page-break-inside: auto; font-size: 6.8pt; line-height: 1.25; }</style>\n")


def main() -> None:
    text = SRC.read_text()
    if "--appendix" in sys.argv:
        text += appendix()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)  # drafting notes never reach the PDF
    for kind in ("Table", "Figure"):
        numbers = {}
        for label in re.findall(rf"{kind} \[([A-Za-z0-9-]+)\]", text):
            numbers.setdefault(label, len(numbers) + 1)
        text = re.sub(rf"{kind} \[([A-Za-z0-9-]+)\]", lambda m: f"{kind} {numbers[m.group(1)]}", text)
        missing = [lab for lab in numbers if text.count(f"{kind} {numbers[lab]}") < 2]
        if missing:
            print(f"{kind}s referenced only once (caption without a reference in the text): {missing}")
    text = text.replace("](../results/", "](../../results/")
    # long tables (wrapped in <div class="long">) may break across pages, with the header repeated
    text = ("<style>.long table { page-break-inside: auto; } .long tr { page-break-inside: avoid; } "
            "thead { display: table-header-group; } "
            # captions ("**Table n.** ...") stay on the page of their table
            "p:has(> strong:first-child) { break-after: avoid; page-break-after: avoid; } "
            # figure captions stay on the page of their figure
            "p:has(> img) { break-after: avoid; page-break-after: avoid; } "
            "p:has(> img) + p { break-before: avoid; page-break-before: avoid; } "
            # a figure caption (after its image) may be followed by a page break
            "p:has(> img) + p:has(> strong:first-child) { break-after: auto; page-break-after: auto; } "
            # display equations: centred, number at the right margin
            "p.eq { text-align: center; position: relative; margin: 0.5em 0; break-inside: avoid; } "
            "p.eq .eqn { position: absolute; right: 0; } "
            # the sentence that introduces an equation stays on its page
            "p:has(+ p.eq) { break-after: avoid; page-break-after: avoid; }</style>\n\n") + text
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    python = ROOT / ".venv-report" / "bin" / "python"  # has python-markdown
    subprocess.run([str(python if python.exists() else sys.executable), str(BUILD_PDF), str(OUT)], check=True)


if __name__ == "__main__":
    main()
