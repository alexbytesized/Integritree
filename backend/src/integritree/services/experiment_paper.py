"""Fill the retained experiment template and convert a temporary copy to PDF."""

import logging
import os
import subprocess
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from zipfile import BadZipFile, ZipFile

TEMPLATE = (
    Path(__file__).resolve().parents[1] / "templates" / "Experiment-Paper-Template.docx"
)
PHILIPPINE_TIME = timezone(timedelta(hours=8))
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
METRICS = ("precision", "recall", "f1", "mcc", "pr_auc")
LABELS = ("Precision", "Recall", "F1-score", "MCC", "PR-AUC")
LOG = logging.getLogger(__name__)


class PaperExportError(Exception):
    """An actionable error safe to show in the export status."""


def paper_filename(exported_at: datetime) -> str:
    """Name the paper using the export's Philippine calendar date."""
    return f"Experiment-Paper_{exported_at.astimezone(PHILIPPINE_TIME):%Y-%m-%d}.pdf"


def children(node, name):
    return [
        item
        for item in node.childNodes
        if item.nodeType == item.ELEMENT_NODE
        and item.namespaceURI == W
        and item.localName == name
    ]


def text_of(node):
    return "".join(
        child.data
        for item in node.getElementsByTagNameNS(W, "t")
        for child in item.childNodes
        if child.nodeType == child.TEXT_NODE
    )


def add_text(paragraph, value):
    """Add a run without replacing the paragraph/cell's existing properties."""
    doc = paragraph.ownerDocument
    run = doc.createElementNS(W, "w:r")
    props = children(paragraph, "pPr")
    run_props = children(props[0], "rPr") if props else []
    if run_props:
        run.appendChild(run_props[0].cloneNode(True))
    text = doc.createElementNS(W, "w:t")
    text.appendChild(doc.createTextNode(value))
    run.appendChild(text)
    paragraph.appendChild(run)


def fill_answer(cell, value, placeholder=None):
    """Replace only a recognized slot; keep its surrounding Word structure."""
    existing = text_of(cell).strip()
    if existing and existing != placeholder:
        raise ValueError(
            "answer cells must be empty or contain their expected placeholder"
        )
    if existing:
        texts = list(cell.getElementsByTagNameNS(W, "t"))
        for index, text in enumerate(texts):
            while text.firstChild:
                text.removeChild(text.firstChild)
            if index == 0:
                text.appendChild(cell.ownerDocument.createTextNode(value))
    else:
        add_text(children(cell, "p")[0], value)
    # Retain the requested uniform answer style even when an empty slot's
    # paragraph mark is bold or its new placeholder has different formatting.
    for run in cell.getElementsByTagNameNS(W, "r"):
        if not text_of(run):
            continue
        properties = children(run, "rPr")
        props = (
            properties[0]
            if properties
            else cell.ownerDocument.createElementNS(W, "w:rPr")
        )
        if not properties:
            run.insertBefore(props, run.firstChild)
        for name, setting in (("b", "0"), ("bCs", "0"), ("i", "1"), ("iCs", "1")):
            for old in children(props, name):
                props.removeChild(old)
            flag = cell.ownerDocument.createElementNS(W, f"w:{name}")
            flag.setAttributeNS(W, "w:val", setting)
            props.appendChild(flag)


def p_value(value):
    return f"{value:.3e}" if 0 < abs(value) < 0.00005 else f"{value:.4f}"


def table_values(evaluation):
    """Only format existing results; never recompute the evaluation."""
    notes = []
    tables = []
    for model in ("rf", "rf_smote"):
        counts = evaluation["models"][model]["confusion_matrix"]
        tables.append(
            [[str(counts[k]) for k in row] for row in (("tp", "fn"), ("fp", "tn"))]
        )

    metrics = []
    for model, name in (("rf", "Random Forest"), ("rf_smote", "Random Forest-SMOTE")):
        row = []
        for key, label in zip(METRICS, LABELS):
            item = evaluation["models"][model]["metrics"][key]
            if item["value"] is None:
                row.append("N/A")
                notes.append(
                    f"Table 3, {name}, {label}: {reason_text(item['reason'])}."
                )
            else:
                row.append(
                    f"{item['value']:.4f}"
                    if key == "mcc"
                    else f"{100 * item['value']:.2f}"
                )
        metrics.append(row)
    differences = []
    for key, label in zip(METRICS, LABELS):
        item = evaluation["descriptive_comparisons"][key]
        if item["value"] is None:
            differences.append("N/A")
            notes.append(f"Table 3, {label} difference: {reason_text(item['reason'])}.")
        elif item["units"] == "coefficient":
            differences.append(f"{item['value']:.4f}")
            notes.append(
                "Table 3, MCC difference: a negative model MCC requires the signed "
                "coefficient difference (RF-SMOTE minus RF), not a percentage."
            )
        else:
            differences.append(f"{item['value']:.2f}")
    tables.append(metrics + [differences])
    test = evaluation["statistical_test"]
    tables.append([[str(value) for value in row] for row in test["table"]["matrix"]])
    statistic = test["statistic"]
    tables.append(
        [["N/A" if statistic is None else f"{statistic:.4f}", p_value(test["p_value"])]]
    )
    if test["discordant_pairs"] == 0:
        notes.append(
            "Table 5: no discordant pairs; the chi-squared statistic is unavailable "
            "and the p-value is 1 by the evaluation convention."
        )
    if test.get("approximation_caution"):
        notes.append(
            "Table 5: only 1 to 24 discordant pairs were observed, so the "
            "chi-squared approximation should be interpreted cautiously."
        )
    supplement = test.get("supplementary")
    if supplement:
        notes.append(
            "Supplementary exact two-sided binomial McNemar p-value: "
            f"{p_value(supplement['p_value'])}. Table 5 retains the primary "
            "continuity-corrected test."
        )
    return tables, notes


def reason_text(reason):
    return {
        "no_predicted_positives": "no transactions were predicted fraudulent",
        "no_actual_positives": "no actual fraudulent transactions were present",
        "no_actual_or_predicted_positives": "no actual or predicted fraudulent transactions were present",
        "zero_mcc_denominator": "the MCC denominator is zero",
        "undefined_input": "at least one model's metric is unavailable",
        "zero_denominator": "the comparison denominator is zero",
    }.get(reason, str(reason or "unavailable").replace("_", " "))


def fill_template(evaluation, destination, template=TEMPLATE):
    """Change only answer slots and Notes in document.xml; retain other parts."""
    try:
        with ZipFile(template) as source:
            document = minidom.parseString(source.read("word/document.xml"))
            body = document.getElementsByTagNameNS(W, "body")[0]
            tables = children(body, "tbl")
            expected = ([3, 3, 3], [3, 3, 3], [2, 6, 6, 6, 6], [3, 3, 3], [2, 2])
            if len(tables) != 5:
                raise ValueError("expected five result tables")
            grids = []
            for table, shape in zip(tables, expected):
                grid = [children(row, "tc") for row in children(table, "tr")]
                if [len(row) for row in grid] != shape:
                    raise ValueError("unexpected table dimensions")
                grids.append(grid)
            # Check labels as well as dimensions so a rearranged template cannot
            # silently receive the wrong model or confusion-matrix orientation.
            for grid in grids[:2]:
                if [text_of(cell).strip() for cell in grid[0][1:]] != [
                    "Predicted Fraudulent",
                    "Predicted Legitimate",
                ] or [text_of(row[0]).strip() for row in grid[1:]] != [
                    "Actual Fraudulent",
                    "Actual Legitimate",
                ]:
                    raise ValueError("unexpected confusion matrix labels")
            if [text_of(row[0]).strip() for row in grids[2][2:]] != [
                "Random Forest",
                "Random Forest-SMOTE",
                "Percentage Difference",
            ] or [text_of(cell).strip() for cell in grids[2][1][1:]] != [
                "Precision(%)",
                "Recall(%)",
                "F1-score(%)",
                "MCC",
                "PR-AUC(%)",
            ]:
                raise ValueError("unexpected metric labels")
            if (
                [text_of(cell).strip() for cell in grids[3][0][1:]]
                != ["RF-SMOTE Correct", "RF-SMOTE Incorrect"]
                or [text_of(row[0]).strip() for row in grids[3][1:]]
                != ["RF Model Correct", "RF Model Incorrect"]
                or [text_of(cell).strip() for cell in grids[4][0]]
                != ["Chi-Squared", "p-value"]
            ):
                raise ValueError("unexpected statistical table labels")
            paragraphs = children(body, "p")
            headings = [p for p in paragraphs if text_of(p).strip() == "Notes:"]
            if len(headings) != 1:
                raise ValueError("expected a final Notes: section")
            heading = headings[0]
            following = paragraphs[paragraphs.index(heading) + 1 :]
            if not following or any(text_of(p).strip() for p in following):
                raise ValueError("expected empty paragraphs after Notes:")
            if list(body.childNodes).index(heading) < list(body.childNodes).index(
                tables[-1]
            ):
                raise ValueError("Notes must follow all result tables")

            values, notes = table_values(evaluation)
            for index, (grid, data) in enumerate(zip(grids, values)):
                start_row = 2 if index == 2 else 1
                start_col = 0 if index == 4 else 1
                for row_index, row in enumerate(data, start_row):
                    for col_index, value in enumerate(row, start_col):
                        cell = grid[row_index][col_index]
                        placeholder = None
                        if index in (0, 1):
                            placeholder = (("TP", "FN"), ("FP", "TN"))[row_index - 1][
                                col_index - 1
                            ]
                        elif index == 3:
                            placeholder = (("A", "B"), ("C", "D"))[row_index - 1][
                                col_index - 1
                            ]
                        fill_answer(cell, value, placeholder)
            note_slot = following[0]
            empty_note = deepcopy(note_slot)
            for index, note in enumerate(notes):
                paragraph = note_slot if index == 0 else deepcopy(empty_note)
                if index != 0:
                    body.insertBefore(
                        paragraph,
                        following[1]
                        if len(following) > 1
                        else children(body, "sectPr")[0],
                    )
                add_text(paragraph, note)
            with ZipFile(destination, "w") as output:
                for part in source.infolist():
                    output.writestr(
                        part,
                        document.toxml(encoding="UTF-8")
                        if part.filename == "word/document.xml"
                        else source.read(part.filename),
                    )
    except (OSError, BadZipFile, ValueError, IndexError, KeyError, ExpatError) as exc:
        raise PaperExportError(
            "Experiment paper template is missing or incompatible. Restore the "
            "five-table template with empty answer cells or correctly placed "
            "TP/FN/FP/TN and A/B/C/D placeholders, and its final Notes: section, then retry."
        ) from exc


def convert_pdf(document, converter, output_dir):
    converter = Path(converter)
    if not converter.is_file():
        raise PaperExportError(
            "PDF converter is missing. Run scripts/setup_research_pdf.ps1 or set "
            "INTEGRITREE_RESEARCH_PDF_CONVERTER to the LibreOffice executable, "
            "restart the backend, and retry."
        )
    # LibreOffice on Windows still encounters MAX_PATH in profile internals.
    # Keep names short because session and analysis directories already nest.
    profile = output_dir
    args = [
        str(converter),
        f"-env:UserInstallation={profile.as_uri()}",
        "--headless",
        "--nologo",
        "--nodefault",
        "--norestore",
        "--convert-to",
        "pdf:writer_pdf_Export",
        "--outdir",
        str(output_dir),
        str(document),
    ]
    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        try:
            stdout, stderr = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    timeout=10,
                )
            else:
                import signal

                os.killpg(process.pid, signal.SIGKILL)
            process.kill()
            process.communicate()
            raise PaperExportError(
                "PDF conversion timed out after 120 seconds. Retry the download."
            )
        pdf = output_dir / (document.stem + ".pdf")
        if process.returncode != 0 or not pdf.is_file():
            LOG.error(
                "LibreOffice conversion failed (%s): %s %s",
                process.returncode,
                stdout.decode(errors="replace"),
                stderr.decode(errors="replace"),
            )
            raise PaperExportError(
                "PDF conversion failed. Check the LibreOffice runtime and retry the download."
            )
        result = pdf.read_bytes()
        if not result.startswith(b"%PDF-") or b"%%EOF" not in result[-1024:]:
            raise PaperExportError(
                "PDF conversion produced an invalid file. Retry the download."
            )
        return result
    except OSError as exc:
        raise PaperExportError(
            "Could not start the PDF converter. Check INTEGRITREE_RESEARCH_PDF_CONVERTER and retry."
        ) from exc


def generate_paper(evaluation, folder, converter):
    # Work directly under research_sessions, avoiding the additional session/job
    # nesting. The session_ prefix also lets startup remove interrupted exports.
    with TemporaryDirectory(prefix="session_", dir=folder) as temporary:
        directory = Path(temporary).resolve()
        document = directory / "paper.docx"
        fill_template(evaluation, document)
        return convert_pdf(document, converter, directory)
