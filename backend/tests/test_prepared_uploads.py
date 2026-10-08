"""Prepared CSV contract, parity, and unchanged downstream researcher workflows."""

import csv
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from test_research_api import (
    CSV,
    PREFIX,
    session,
    upload,
    wait_for,
)
from test_research_api import (
    bundle as bundle,  # noqa: PLC0414 -- explicitly re-export pytest fixtures
)
from test_research_api import (
    client as client,  # noqa: PLC0414
)
from test_research_api import (
    experiment as experiment,  # noqa: PLC0414
)
from test_research_api import (
    test_explanation_failure_retry_cache_and_export_coverage as explanation_flow,
)

from integritree.ml.features import FEATURE_COLUMNS, SOURCE_COLUMNS
from integritree.ml.preprocessing import FittedPreprocessor
from integritree.services import research


def prepared_frame(bundle):
    raw = pd.read_csv(io.StringIO(CSV))
    frame = bundle.preprocessor.transform(raw[SOURCE_COLUMNS])
    frame["isFraud"] = raw.isFraud
    return frame


def forbid_preprocessing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared uploads must not preprocess or fit")

    for name in ("fit", "fit_batches", "transform", "transform_engineered"):
        monkeypatch.setattr(FittedPreprocessor, name, forbidden)
    monkeypatch.setattr(research, "engineer_features", forbidden)
    monkeypatch.setattr("integritree.ml.preprocessing.engineer_features", forbidden)


def test_prepared_parity_details_and_export(client, bundle, monkeypatch):
    owner = session(client)
    raw_id, raw_job = upload(client, owner)
    expected = [
        client.get(PREFIX + f"/analyses/{raw_id}/records/{i}", headers=owner).json()
        for i in (1, 2)
    ]
    frame = prepared_frame(bundle)
    content = frame.loc[:, list(reversed(frame.columns))].to_csv(index=False)
    forbid_preprocessing(monkeypatch)
    # Exercise multi-batch upload while preserving source order.
    monkeypatch.setattr(research, "BATCH_SIZE", 1)
    identifier, job = upload(client, owner, content)
    assert raw_job["input_format"] == "raw"
    assert job["status"] == "complete", job
    assert job["input_format"] == "prepared"
    assert job["evaluation"] == raw_job["evaluation"]
    path = PREFIX + f"/analyses/{identifier}"
    for i, raw in enumerate(expected, 1):
        detail = client.get(path + f"/records/{i}", headers=owner).json()
        assert detail["input_format"] == "prepared"
        assert detail["actual_label"] == raw["actual_label"]
        assert detail["rf"] == raw["rf"] and detail["rf_smote"] == raw["rf_smote"]
        assert list(detail["model_inputs"]) == FEATURE_COLUMNS
        for field in ("model_inputs", "derived"):
            assert detail[field] == pytest.approx(raw[field])
        assert "step" not in detail["original"]
    client.post(path + "/exports", headers=owner)
    exported = wait_for(
        lambda: client.get(path, headers=owner).json(),
        lambda j: j["export"]["status"] in ("complete", "failed"),
    )
    assert exported["export"]["status"] == "complete", exported
    metadata = exported["export"]["metadata"]
    assert metadata["input_format"] == "prepared"
    assert metadata["preprocessing_applied"] is False
    assert metadata["held_out_membership_verified"] is False
    response = client.get(path + "/exports/download", headers=owner)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert len(archive.namelist()) == 2
        name = next(n for n in archive.namelist() if n.startswith("Raw-Results_"))
        rows = list(csv.DictReader(io.StringIO(archive.read(name).decode())))
        assert list(rows[0]) == [
            "transaction_id",
            "upload_row_number",
            *reversed(frame.columns),
            "rf_score",
            "rf_predicted_label",
            "rf_smote_score",
            "rf_smote_predicted_label",
        ]
        source = list(csv.DictReader(io.StringIO(content)))
        for row, original in zip(rows, source):
            assert {key: row[key] for key in original} == original


@pytest.mark.parametrize(
    "column,value",
    [
        ("log_amount", ""),
        ("hour_of_day", "bad"),
        ("day_of_week", "NaN"),
        ("log_amount", "inf"),
        ("log_amount", "-inf"),
        ("is_zero_amount", "0.5"),
        ("is_merchant_origin", "-1"),
        ("is_merchant_dest", "2"),
        ("type_TRANSFER", "0.2"),
        ("isFraud", ""),
        ("isFraud", "2"),
    ],
)
def test_invalid_prepared_values(client, bundle, column, value):
    frame = prepared_frame(bundle).astype(str)
    frame.loc[0, column] = value
    _, job = upload(client, session(client), frame.to_csv(index=False))
    assert job["status"] == "failed"
    assert job["issues"][column]["rows"] == [1]
    assert "layout_issues" not in job


@pytest.mark.parametrize(
    "change",
    ["missing_feature", "missing_label", "wrong_case", "mixed", "extra", "duplicate"],
)
def test_invalid_prepared_layouts(client, bundle, change):
    frame = prepared_frame(bundle)
    if change == "missing_feature":
        frame = frame.drop(columns="log_amount")
    elif change == "missing_label":
        frame = frame.drop(columns="isFraud")
    elif change == "wrong_case":
        frame = frame.rename(columns={"isFraud": "IsFraud"})
    elif change == "mixed":
        for column in SOURCE_COLUMNS:
            frame[column] = "1"
    elif change == "extra":
        frame["source_row_number"] = [1, 2]
    else:
        frame = pd.concat([frame, frame[["log_amount"]]], axis=1)
    _, job = upload(client, session(client), frame.to_csv(index=False))
    assert job["status"] == "failed"
    assert job["error_code"] == "unsupported_csv_layout"
    assert "Accepted layouts: raw:" in job["error"] and "prepared:" in job["error"]
    assert job["layout_issues"]["formats"]["prepared"]["missing_columns"] == (
        ["log_amount"]
        if change == "missing_feature"
        else ["isFraud"]
        if change in ("missing_label", "wrong_case")
        else []
    )
    assert job["layout_issues"]["duplicate_columns"] == (
        ["log_amount"] if change == "duplicate" else []
    )


@pytest.mark.parametrize(
    "headers,missing,unsupported,duplicates",
    [
        (SOURCE_COLUMNS, ["isFraud"], [], []),
        (SOURCE_COLUMNS + ["isFraud", "Action"], [], ["Action"], []),
        (
            SOURCE_COLUMNS + ["isFraud", "amount", "amount", "type"],
            [],
            [],
            ["amount", "type"],
        ),
        (
            [
                "Action",
                "Actor Code",
                "Actor Email",
                "Date and Time",
                "Entity Code",
                "Entity Email",
                "New Status",
                "Previous Status",
            ],
            sorted(SOURCE_COLUMNS + ["isFraud"]),
            [
                "Action",
                "Actor Code",
                "Actor Email",
                "Date and Time",
                "Entity Code",
                "Entity Email",
                "New Status",
                "Previous Status",
            ],
            [],
        ),
        (
            ["<img src=x>", "Column, with: punctuation.", ""],
            sorted(SOURCE_COLUMNS + ["isFraud"]),
            ["", "<img src=x>", "Column, with: punctuation."],
            [],
        ),
    ],
)
def test_raw_layout_diagnostics_preserve_header_names(
    client, headers, missing, unsupported, duplicates
):
    content = io.StringIO()
    csv.writer(content).writerow(headers)
    _, job = upload(client, session(client), content.getvalue())
    assert job["status"] == "failed"
    assert job["error_code"] == "unsupported_csv_layout"
    assert job["issues"] is None
    assert job["layout_issues"]["duplicate_columns"] == duplicates
    assert job["layout_issues"]["formats"]["raw"] == {
        "missing_columns": missing,
        "unsupported_columns": unsupported,
    }
    assert set(job["layout_issues"]["formats"]) == {"raw", "prepared"}


def test_prepared_indicators_and_unbounded_scaling(client, bundle):
    frame = prepared_frame(bundle)
    frame.loc[0, "type_PAYMENT"] = 1
    owner = session(client)
    _, job = upload(client, owner, frame.to_csv(index=False))
    assert job["issues"]["transaction_type_indicators"]["rows"] == [1]
    frame.loc[:, [c for c in FEATURE_COLUMNS if c.startswith("type_")]] = 0
    frame["log_amount"] = [-0.25, 1.25]
    identifier, job = upload(client, owner, frame.to_csv(index=False))
    assert job["status"] == "complete"
    detail = client.get(
        PREFIX + f"/analyses/{identifier}/records/1", headers=owner
    ).json()
    assert detail["model_inputs"]["log_amount"] == -0.25
    path = PREFIX + f"/analyses/{identifier}"
    client.post(path + "/exports", headers=owner)
    wait_for(
        lambda: client.get(path, headers=owner).json(),
        lambda j: j["export"]["status"] == "complete",
    )
    with zipfile.ZipFile(
        io.BytesIO(client.get(path + "/exports/download", headers=owner).content)
    ) as archive:
        name = next(n for n in archive.namelist() if n.startswith("Raw-Results_"))
        rows = list(csv.DictReader(io.StringIO(archive.read(name).decode())))
        assert rows[0]["log_amount"] == "-0.25"


def test_prepared_explanation_retry_cache_and_export(
    client, bundle, monkeypatch, tmp_path
):
    # Reuse the complete downstream explanation contract with prepared input.
    import test_research_api

    content = prepared_frame(bundle).to_csv(index=False)
    original_upload = upload
    monkeypatch.setattr(test_research_api, "CSV", content)
    monkeypatch.setattr(
        test_research_api,
        "upload",
        lambda client, owner: original_upload(client, owner, content),
    )
    forbid_preprocessing(monkeypatch)
    explanation_flow(client, tmp_path)


def test_prepared_display_round_trip(bundle):
    raw = pd.DataFrame(
        [
            {
                "step": 150,
                "type": "unknown",
                "amount": 321.5,
                "nameOrig": "M_A",
                "nameDest": "C_B",
            }
        ]
    )
    features = bundle.preprocessor.transform(raw).iloc[0].to_dict()
    derived = research.prepared_display_features(features, bundle.preprocessor.state)
    assert derived["hour_of_day"] == 5
    assert derived["day_of_week"] == 6
    assert derived["log_amount"] == pytest.approx(np.log1p(321.5))
    assert features != derived
