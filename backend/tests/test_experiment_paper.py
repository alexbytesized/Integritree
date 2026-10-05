"""Synthetic export checks; no saved model or PaySim data required."""

import hashlib
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from xml.dom import minidom
from zipfile import ZipFile

import pytest

from integritree.config import load_experiment
from integritree.ml.evaluation import evaluate_pair
from integritree.services import experiment_paper as paper
from integritree.settings import BACKEND_ROOT, Settings


@pytest.fixture
def evaluation():
    policy = load_experiment(BACKEND_ROOT / "configs/experiment.yaml").evaluation
    return evaluate_pair(
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0.9, 0.8, 0.7, 0.1, 0.9, 0.8, 0.1, 0.2, 0.3, 0.4],
        [0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
        0.5,
        policy,
    )


def read_document(path):
    with ZipFile(path) as archive:
        return minidom.parseString(archive.read("word/document.xml"))


def tables(document):
    body = document.getElementsByTagNameNS(paper.W, "body")[0]
    return [
        [
            [paper.text_of(cell) for cell in paper.children(row, "tc")]
            for row in paper.children(table, "tr")
        ]
        for table in paper.children(body, "tbl")
    ]


def test_fills_all_tables_and_preserves_template(evaluation, tmp_path):
    before = hashlib.sha256(paper.TEMPLATE.read_bytes()).hexdigest()
    result = tmp_path / "paper.docx"
    paper.fill_template(evaluation, result)
    document = read_document(result)
    filled = tables(document)
    assert [row[1:] for row in filled[0][1:]] == [["3", "1"], ["2", "4"]]
    assert [row[1:] for row in filled[1][1:]] == [["1", "3"], ["1", "5"]]
    assert filled[2][2][1:4] == ["60.00", "75.00", "66.67"]
    assert (
        filled[2][2][4]
        == f"{evaluation['models']['rf']['metrics']['mcc']['value']:.4f}"
    )
    assert filled[2][3][1:4] == ["50.00", "25.00", "33.33"]
    assert filled[2][4][1] == "-18.18"
    assert [row[1:] for row in filled[3][1:]] == [
        [str(v) for v in row]
        for row in evaluation["statistical_test"]["table"]["matrix"]
    ]
    assert filled[4][1] == ["0.0000", "1.0000"]
    # All 29 answer slots inherit the revised template's italic, non-bold style.
    answer_count = 0
    for index, table in enumerate(document.getElementsByTagNameNS(paper.W, "tbl")):
        rows = paper.children(table, "tr")[2 if index == 2 else 1 :]
        for row in rows:
            for cell in paper.children(row, "tc")[0 if index == 4 else 1 :]:
                run = cell.getElementsByTagNameNS(paper.W, "r")[0]
                assert run.getElementsByTagNameNS(paper.W, "i")
                assert not run.getElementsByTagNameNS(paper.W, "b")
                answer_count += 1
    assert answer_count == 29
    assert hashlib.sha256(paper.TEMPLATE.read_bytes()).hexdigest() == before
    with ZipFile(paper.TEMPLATE) as source, ZipFile(result) as output:
        assert source.namelist() == output.namelist()
        for name in source.namelist():
            if name != "word/document.xml":
                assert source.read(name) == output.read(name), name
    original = read_document(paper.TEMPLATE)
    # Every pre-existing text node and equation remains, in the same order.
    original_text = [
        paper.text_of(p) for p in original.getElementsByTagNameNS(paper.W, "p")
    ]
    final_text = [
        paper.text_of(p) for p in document.getElementsByTagNameNS(paper.W, "p")
    ]
    assert [t for t in original_text if t] == [
        t for t in final_text if t in original_text and t
    ]
    math_ns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    assert [n.toxml() for n in original.getElementsByTagNameNS(math_ns, "oMath")] == [
        n.toxml() for n in document.getElementsByTagNameNS(math_ns, "oMath")
    ]
    notes_index = next(i for i, t in enumerate(final_text) if t.strip() == "Notes:")
    assert any("Supplementary exact" in t for t in final_text[notes_index + 1 :])
    assert not any("Supplementary exact" in t for t in final_text[:notes_index])


def test_edge_values_are_explained_only_in_notes(evaluation, tmp_path):
    edge = deepcopy(evaluation)
    edge["models"]["rf"]["metrics"]["precision"] = {
        "value": None,
        "reason": "no_predicted_positives",
    }
    edge["descriptive_comparisons"]["precision"]["value"] = None
    edge["descriptive_comparisons"]["precision"]["reason"] = "undefined_input"
    edge["descriptive_comparisons"]["mcc"] = {"value": -0.32123, "units": "coefficient"}
    edge["statistical_test"].update(
        statistic=None,
        p_value=1.0,
        discordant_pairs=0,
        approximation_caution=False,
        supplementary=None,
    )
    values, notes = paper.table_values(edge)
    assert values[2][0][0] == "N/A"
    assert values[2][2][0] == "N/A"
    assert values[2][2][3] == "-0.3212"
    assert values[4] == [["N/A", "1.0000"]]
    assert any("signed coefficient difference" in note for note in notes)
    assert any("no discordant pairs" in note for note in notes)
    result = tmp_path / "edge.docx"
    paper.fill_template(edge, result)
    document = read_document(result)
    paragraphs = [
        paper.text_of(p) for p in document.getElementsByTagNameNS(paper.W, "p")
    ]
    index = next(i for i, text in enumerate(paragraphs) if text.strip() == "Notes:")
    for note in notes:
        assert note in paragraphs[index + 1 :]
        assert note not in paragraphs[:index]


@pytest.mark.parametrize(
    "value,expected",
    [(0, "0.0000"), (1e-12, "1.000e-12"), (0.00001, "1.000e-05"), (0.123456, "0.1235")],
)
def test_small_p_values(value, expected):
    assert paper.p_value(value) == expected


def test_no_edge_notes_for_ordinary_results(evaluation):
    evaluation["statistical_test"].update(
        discordant_pairs=40, approximation_caution=False, supplementary=None
    )
    assert paper.table_values(evaluation)[1] == []


@pytest.mark.parametrize("change", ["missing_notes", "filled_cell", "swapped_axes"])
def test_rejects_incompatible_template(evaluation, tmp_path, change):
    with ZipFile(paper.TEMPLATE) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    xml = parts["word/document.xml"]
    if change == "missing_notes":
        document = minidom.parseString(xml)
        heading = next(
            p
            for p in document.getElementsByTagNameNS(paper.W, "p")
            if paper.text_of(p).strip() == "Notes:"
        )
        heading.parentNode.removeChild(heading)
        xml = document.toxml(encoding="UTF-8")
    elif change == "swapped_axes":
        xml = xml.replace(b">Actual Fraudulent<", b">Actual Legitimate<", 1)
    else:
        document = minidom.parseString(xml)
        table = document.getElementsByTagNameNS(paper.W, "tbl")[0]
        cell = paper.children(paper.children(table, "tr")[1], "tc")[1]
        paper.add_text(paper.children(cell, "p")[0], "123")
        xml = document.toxml(encoding="UTF-8")
    parts["word/document.xml"] = xml
    template = tmp_path / "bad.docx"
    with ZipFile(template, "w") as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    with pytest.raises(
        paper.PaperExportError, match="template is missing or incompatible"
    ):
        paper.fill_template(evaluation, tmp_path / "result.docx", template)


def test_missing_converter_cleans_working_files(evaluation, tmp_path):
    with pytest.raises(paper.PaperExportError, match="PDF converter is missing"):
        paper.generate_paper(evaluation, tmp_path, tmp_path / "missing.exe")
    assert not list(tmp_path.glob("session_*"))


def test_conversion_uses_isolated_profile_and_checks_output(tmp_path, monkeypatch):
    converter = tmp_path / "soffice.com"
    converter.touch()
    document = tmp_path / "paper.docx"
    document.touch()
    process = Mock(returncode=0)
    process.communicate.return_value = (b"", b"")
    start = Mock(return_value=process)
    monkeypatch.setattr(paper.subprocess, "Popen", start)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfixture\n%%EOF")
    assert paper.convert_pdf(document, converter, tmp_path).startswith(b"%PDF-")
    args = start.call_args.args[0]
    assert f"-env:UserInstallation={tmp_path.as_uri()}" in args
    assert "pdf:writer_pdf_Export" in args
    process.communicate.assert_called_once_with(timeout=120)
    pdf.write_bytes(b"not a pdf")
    with pytest.raises(paper.PaperExportError, match="invalid file"):
        paper.convert_pdf(document, converter, tmp_path)
    pdf.unlink()
    with pytest.raises(paper.PaperExportError, match="conversion failed"):
        paper.convert_pdf(document, converter, tmp_path)


def test_timeout_stops_converter(tmp_path, monkeypatch):
    converter = tmp_path / "converter.exe"
    converter.touch()
    process = Mock(pid=123)
    process.communicate.side_effect = [
        subprocess.TimeoutExpired("converter", 120),
        (b"", b""),
    ]
    monkeypatch.setattr(paper.subprocess, "Popen", Mock(return_value=process))
    stop = Mock()
    monkeypatch.setattr(paper.subprocess, "run", stop)
    if paper.os.name != "nt":
        monkeypatch.setattr(paper.os, "killpg", stop)
    with pytest.raises(paper.PaperExportError, match="timed out"):
        paper.convert_pdf(tmp_path / "input.docx", converter, tmp_path)
    assert stop.called
    process.kill.assert_called_once()


def test_converter_setting_is_relative_to_backend(tmp_path):
    settings = Settings(
        backend_root=tmp_path, research_pdf_converter=Path("tools/soffice.com")
    )
    assert settings.research_pdf_converter == tmp_path / "tools/soffice.com"


@pytest.mark.parametrize(
    "instant,date",
    [
        ("2026-10-04T15:59:59+00:00", "2026-10-04"),
        ("2026-10-04T16:00:00+00:00", "2026-10-05"),
        ("2026-12-31T16:00:00+00:00", "2027-01-01"),
        ("2026-10-05T00:00:00+08:00", "2026-10-05"),
    ],
)
def test_paper_filename_uses_philippine_date(instant, date):
    assert (
        paper.paper_filename(datetime.fromisoformat(instant))
        == f"Experiment-Paper_{date}.pdf"
    )
