"""越界剂量预算：窗末累计剂量、剩余额度与预算耗尽精确时刻。

纯算法单测（``dimension_budget``）+ 接口集成测试（内存 SQLite）。
"""

from __future__ import annotations

import datetime as dt
import math
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine, init_db
from app.excursions import HUMIDITY, TEMPERATURE, dimension_budget
from app.main import app
from app.segmentation import Sample

BASE = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


def _sample(minutes: float, t: float, h: float = 50.0) -> Sample:
    return Sample(BASE + dt.timedelta(minutes=minutes), t, h)


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


# ---------- 纯算法：累计剂量与耗尽时刻反解 ----------


def test_accumulated_dose_matches_triangle_area():
    # 20→30°C 线性上升，上限 25：交点在中点，三角形剂量 750
    res = dimension_budget(
        [_sample(0, 20.0), _sample(10, 30.0)],
        TEMPERATURE,
        "max",
        25.0,
        budget=10_000.0,
        max_gap_seconds=600.0,
    )
    assert res.dose == pytest.approx(750.0)
    assert res.remaining == pytest.approx(10_000.0 - 750.0)
    assert res.exhausted is False
    assert res.exhausted_ts is None


def test_budget_hit_inside_rising_segment_solved_by_linear_excess():
    # 交点 t=300s（e=0）线性上升到 t=600s（e=5）；累计 A(u)=750u²
    res = dimension_budget(
        [_sample(0, 20.0), _sample(10, 30.0)],
        TEMPERATURE,
        "max",
        25.0,
        budget=500.0,
        max_gap_seconds=600.0,
    )
    assert res.exhausted is True
    assert res.remaining == pytest.approx(-250.0)
    u = math.sqrt(2.0 / 3.0)
    expect = BASE + dt.timedelta(seconds=300.0 + 300.0 * u)
    assert res.exhausted_ts == expect
    # 必须落在越界线段（交点 → 末读数）之内
    assert BASE + dt.timedelta(seconds=300) < res.exhausted_ts < BASE + dt.timedelta(minutes=10)


def test_budget_hit_inside_falling_segment():
    # 30→20°C：起点即越界（e=5），中点回到阈值；A(u)=1500u-750u²
    res = dimension_budget(
        [_sample(0, 30.0), _sample(10, 20.0)],
        TEMPERATURE,
        "max",
        25.0,
        budget=250.0,
        max_gap_seconds=600.0,
    )
    # 越界片段为 [起点, 阈值交点]（300s）：e 5→0，片段内剂量
    # 300·(5u - 2.5u²) = 250 → 稳定根式 u = 2·target/(5+√disc)
    target = 250.0 / 300.0
    disc = 25.0 + 2.0 * -5.0 * target
    expect_u = 2.0 * target / (5.0 + math.sqrt(disc))
    expect = BASE + dt.timedelta(seconds=300.0 * expect_u)
    assert res.exhausted_ts == expect
    assert BASE < res.exhausted_ts < BASE + dt.timedelta(minutes=5)


def test_rectangular_fragment_hit_linear_in_time():
    # 恒定越界 3°C × 600s = 1800；预算 600 → 恰好 1/3 时刻
    res = dimension_budget(
        [_sample(0, 28.0), _sample(10, 28.0)],
        TEMPERATURE,
        "max",
        25.0,
        budget=600.0,
        max_gap_seconds=600.0,
    )
    assert res.dose == pytest.approx(1800.0)
    assert res.exhausted_ts == BASE + dt.timedelta(seconds=200.0)


def test_budget_exactly_one_fragment_hits_at_fragment_end():
    res = dimension_budget(
        [_sample(0, 20.0), _sample(10, 30.0), _sample(20, 30.0)],
        TEMPERATURE,
        "max",
        25.0,
        budget=750.0,  # 第一个三角片段的完整面积
        max_gap_seconds=600.0,
    )
    assert res.exhausted is True
    assert res.exhausted_ts == BASE + dt.timedelta(minutes=10)


def test_zero_budget_exhausted_at_first_violation_start():
    # 零预算：首次越界起点（阈值交点，幅度为零处）即耗尽
    res = dimension_budget(
        [_sample(0, 20.0), _sample(10, 30.0), _sample(20, 30.0)],
        TEMPERATURE,
        "max",
        25.0,
        budget=0.0,
        max_gap_seconds=600.0,
    )
    assert res.dose == pytest.approx(750.0 + 5.0 * 600.0)
    assert res.exhausted is True
    assert res.exhausted_ts == BASE + dt.timedelta(minutes=5)


def test_zero_budget_no_violation_not_exhausted():
    res = dimension_budget(
        [_sample(0, 20.0), _sample(10, 24.0)],
        TEMPERATURE,
        "max",
        25.0,
        budget=0.0,
        max_gap_seconds=600.0,
    )
    assert res.dose == 0.0
    assert res.remaining == 0.0
    assert res.exhausted is False
    assert res.exhausted_ts is None


def test_hit_in_second_fragment_after_accumulation():
    # 两段矩形越界：各 3°C × 600s = 1800；预算 2000，落在第二段内
    res = dimension_budget(
        [_sample(0, 28.0), _sample(10, 28.0), _sample(20, 28.0)],
        TEMPERATURE,
        "max",
        25.0,
        budget=2000.0,
        max_gap_seconds=600.0,
    )
    assert res.dose == pytest.approx(3600.0)
    assert res.exhausted_ts == BASE + dt.timedelta(seconds=600.0 + 200.0 / 3.0)


def test_gap_breaks_accumulation():
    # 0→10 分钟剂量 750；10→300 分钟断档，孤立高读数两侧未知，不产生片段
    res = dimension_budget(
        [_sample(0, 20.0), _sample(10, 30.0), _sample(300, 34.0)],
        TEMPERATURE,
        "max",
        25.0,
        budget=800.0,
        max_gap_seconds=600.0,
    )
    assert res.dose == pytest.approx(750.0)
    assert res.exhausted is False
    assert res.exhausted_ts is None
    # 放大 max_gap 后断档接续，剂量包含整段梯形（08:10 读数 30，阈值 25），
    # 预算在断档后续段内耗尽
    connected = dimension_budget(
        [_sample(0, 20.0), _sample(10, 30.0), _sample(300, 34.0)],
        TEMPERATURE,
        "max",
        25.0,
        budget=800.0,
        max_gap_seconds=20_000.0,
    )
    assert connected.dose == pytest.approx(
        750.0 + 0.5 * (5.0 + 9.0) * 290.0 * 60.0
    )
    assert connected.exhausted is True
    assert connected.exhausted_ts > BASE + dt.timedelta(minutes=10)


def test_humidity_min_branch():
    res = dimension_budget(
        [_sample(0, 20.0, 50.0), _sample(10, 20.0, 30.0)],
        HUMIDITY,
        "min",
        40.0,
        budget=100.0,
        max_gap_seconds=600.0,
    )
    assert res.dose == pytest.approx(1500.0)  # 0.5×10×300
    # 三角片段 e 0→10，A(u)=1500u²，预算 100 → u=sqrt(1/15)
    u = math.sqrt(1.0 / 15.0)
    assert res.exhausted_ts == BASE + dt.timedelta(seconds=300.0 + 300.0 * u)


def test_budget_unsorted_input_deterministic():
    samples = [
        _sample(0, 28.0),
        _sample(10, 28.0),
        _sample(20, 20.0),
        _sample(30, 20.0),
        _sample(40, 28.0),
        _sample(50, 28.0),
    ]
    a = dimension_budget(samples, TEMPERATURE, "max", 25.0, 2000.0, 600.0)
    b = dimension_budget(samples[::-1], TEMPERATURE, "max", 25.0, 2000.0, 600.0)
    assert a == b


def test_empty_samples_zero_dose():
    res = dimension_budget([], TEMPERATURE, "max", 25.0, 100.0, 600.0)
    assert res.dose == 0.0
    assert res.remaining == 100.0
    assert res.exhausted is False
    assert res.exhausted_ts is None


# ---------- 接口集成 ----------


def _ingest(client, sn, rows):
    r = client.post("/api/v1/readings/batch", json={"readings": rows})
    assert r.status_code == 200, r.text


def test_exposure_budget_basic_multi_dimension(client):
    _ingest(client, "CASE-B1", [
        _reading("CASE-B1", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-B1", "2026-09-01T08:10:00+08:00", 30.0, 30.0),
        _reading("CASE-B1", "2026-09-01T08:20:00+08:00", 20.0, 50.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-B1/exposure-budget",
        params={"temp_max": 25, "temp_max_budget": 1000,
                "humidity_min": 40, "humidity_min_budget": 5000},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["device_sn"] == "CASE-B1"
    assert body["max_gap_seconds"] == 600.0
    assert body["sample_count"] == 3
    assert body["temp_max"] == 25.0
    assert body["temp_max_budget"] == 1000.0
    assert body["temp_min_budget"] is None
    # 固定顺序：temp_max 在 humidity_min 之前
    dims = body["dimensions"]
    assert [d["dimension"] + ":" + d["threshold_kind"] for d in dims] == [
        "temperature:max",
        "humidity:min",
    ]

    tdim, hdim = dims
    assert tdim["accumulated_dose"] == pytest.approx(1500.0)  # 两个三角
    assert tdim["remaining"] == pytest.approx(-500.0)
    assert tdim["exhausted"] is True
    # 首段（交点 00:05 → 00:10，e 0→5）内累计 750；预算 1000 落在第二段
    assert _ts(tdim["exhausted_ts"]) > _ts("2026-09-01T00:10:00+00:00")
    assert _ts(tdim["exhausted_ts"]) < _ts("2026-09-01T00:15:00+00:00")

    assert hdim["accumulated_dose"] == pytest.approx(3000.0)
    assert hdim["remaining"] == pytest.approx(2000.0)
    assert hdim["exhausted"] is False
    assert hdim["exhausted_ts"] is None


def test_exposure_budget_zero_budget_hits_first_crossing(client):
    _ingest(client, "CASE-B2", [
        _reading("CASE-B2", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-B2", "2026-09-01T08:10:00+08:00", 30.0, 50.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-B2/exposure-budget",
        params={"temp_max": 25, "temp_max_budget": 0},
    )
    body = r.json()
    dim = body["dimensions"][0]
    assert dim["budget"] == 0.0
    assert dim["exhausted"] is True
    assert dim["remaining"] == pytest.approx(-750.0)
    assert _ts(dim["exhausted_ts"]) == _ts("2026-09-01T00:05:00+00:00")


def test_exposure_budget_window_scoping(client):
    _ingest(client, "CASE-B3", [
        _reading("CASE-B3", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-B3", "2026-09-01T08:10:00+08:00", 30.0, 50.0),
        _reading("CASE-B3", "2026-09-01T08:20:00+08:00", 20.0, 50.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-B3/exposure-budget",
        params={
            "temp_max": 25,
            "temp_max_budget": 10000,
            "start": "2026-09-01T08:10:00+08:00",
        },
    )
    body = r.json()
    assert body["sample_count"] == 2
    assert body["start"].startswith("2026-09-01T00:10:00")
    # 窗口内曲线仅从 08:10 开始：只剩下降三角 750
    assert body["dimensions"][0]["accumulated_dose"] == pytest.approx(750.0)
    assert body["dimensions"][0]["exhausted"] is False


def test_exposure_budget_gap_breaks_accumulation(client):
    _ingest(client, "CASE-B4", [
        _reading("CASE-B4", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-B4", "2026-09-01T08:10:00+08:00", 30.0, 50.0),
        _reading("CASE-B4", "2026-09-01T10:00:00+08:00", 34.0, 50.0),
    ])
    body = client.get(
        "/api/v1/cases/CASE-B4/exposure-budget",
        params={"temp_max": 25, "temp_max_budget": 1000},
    ).json()
    dim = body["dimensions"][0]
    assert dim["accumulated_dose"] == pytest.approx(750.0)
    assert dim["exhausted"] is False
    assert dim["remaining"] == pytest.approx(250.0)

    body = client.get(
        "/api/v1/cases/CASE-B4/exposure-budget",
        params={"temp_max": 25, "temp_max_budget": 1000, "max_gap": 7200},
    ).json()
    dim = body["dimensions"][0]
    assert dim["accumulated_dose"] == pytest.approx(46_950.0)
    assert dim["exhausted"] is True


def test_exposure_budget_empty_range_zero_dimensions_dose(client):
    _ingest(client, "CASE-B5", [
        _reading("CASE-B5", "2026-09-01T08:00:00+08:00", 30.0, 70.0),
        _reading("CASE-B5", "2026-09-01T08:10:00+08:00", 30.0, 70.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-B5/exposure-budget",
        params={
            "temp_max": 25,
            "temp_max_budget": 100,
            "start": "2030-01-01T00:00:00Z",
            "end": "2030-02-01T00:00:00Z",
        },
    )
    assert r.status_code == 200
    dim = r.json()["dimensions"][0]
    assert dim["accumulated_dose"] == 0.0
    assert dim["remaining"] == pytest.approx(100.0)
    assert dim["exhausted"] is False
    assert dim["exhausted_ts"] is None


def test_exposure_budget_unknown_case_404(client):
    r = client.get(
        "/api/v1/cases/NOPE/exposure-budget",
        params={"temp_max": 25, "temp_max_budget": 100},
    )
    assert r.status_code == 404


def test_exposure_budget_validation_errors(client):
    _ingest(client, "CASE-B6", [
        _reading("CASE-B6", "2026-09-01T08:00:00+08:00", 20.0, 50.0),
        _reading("CASE-B6", "2026-09-01T08:10:00+08:00", 20.0, 50.0),
    ])
    url = "/api/v1/cases/CASE-B6/exposure-budget"
    # 完全没有“阈值+预算”组合
    assert client.get(url).status_code == 422
    # 只有阈值没有预算
    assert client.get(url, params={"temp_max": 25}).status_code == 422
    # 只有预算没有阈值
    assert client.get(url, params={"temp_max_budget": 100}).status_code == 422
    # 阈值与预算错配（温度阈值 + 湿度预算）
    assert client.get(
        url, params={"temp_max": 25, "humidity_max_budget": 100}
    ).status_code == 422
    # 预算为负
    assert client.get(
        url, params={"temp_max": 25, "temp_max_budget": -1}
    ).status_code == 422
    # start 晚于 end
    assert client.get(
        url,
        params={
            "temp_max": 25,
            "temp_max_budget": 100,
            "start": "2030-01-01T00:00:00Z",
            "end": "2029-01-01T00:00:00Z",
        },
    ).status_code == 422
    # 上下限自相矛盾
    assert client.get(
        url,
        params={"temp_min": 30, "temp_max": 20, "temp_max_budget": 100},
    ).status_code == 422
    # max_gap 非正 / 超范围
    assert client.get(
        url, params={"temp_max": 25, "temp_max_budget": 100, "max_gap": 0}
    ).status_code == 422
    # 阈值超出量程
    assert client.get(
        url, params={"temp_max": 150, "temp_max_budget": 100}
    ).status_code == 422


def test_exposure_budget_threshold_without_budget_silently_ignored(client):
    # temp_max 提供了阈值但没给预算；humidity_max 组合完整 → 只返回湿度维度
    _ingest(client, "CASE-B7", [
        _reading("CASE-B7", "2026-09-01T08:00:00+08:00", 30.0, 70.0),
        _reading("CASE-B7", "2026-09-01T08:10:00+08:00", 30.0, 70.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-B7/exposure-budget",
        params={"temp_max": 25, "humidity_max": 60, "humidity_max_budget": 5000},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["temp_max_budget"] is None
    dims = body["dimensions"]
    assert len(dims) == 1
    assert dims[0]["dimension"] == "humidity"
    assert dims[0]["threshold_kind"] == "max"
    assert dims[0]["threshold"] == 60.0
    assert dims[0]["accumulated_dose"] == pytest.approx(10.0 * 600.0)


def test_exposure_budget_min_and_max_same_dimension(client):
    # 同一维度上、下限各自独立预算
    _ingest(client, "CASE-B8", [
        _reading("CASE-B8", "2026-09-01T08:00:00+08:00", 16.0, 50.0),
        _reading("CASE-B8", "2026-09-01T08:10:00+08:00", 26.0, 50.0),
    ])
    r = client.get(
        "/api/v1/cases/CASE-B8/exposure-budget",
        params={
            "temp_min": 18, "temp_min_budget": 0,
            "temp_max": 24, "temp_max_budget": 1_000_000,
        },
    )
    assert r.status_code == 200, r.text
    dims = r.json()["dimensions"]
    assert [d["threshold_kind"] for d in dims] == ["max", "min"]
    lo = dims[1]
    # 下限越界：起点 16（e=2）线性回到阈值 18，三角 0.5×2×t
    assert lo["exhausted"] is True
    assert lo["budget"] == 0.0
    # 零预算 → 首次越界起点即读数起点（该读数已在阈值外）
    assert _ts(lo["exhausted_ts"]) == _ts("2026-09-01T00:00:00+00:00")


def test_exposure_budget_deterministic(client):
    rows = [
        _reading("CASE-B9", f"2026-09-01T08:{i:02d}:00+08:00", 20.0 + (i % 3) * 4, 50.0)
        for i in range(0, 60, 10)
    ]
    _ingest(client, "CASE-B9", rows)
    params = {"temp_max": 25, "temp_max_budget": 300, "humidity_min": 45,
              "humidity_min_budget": 0, "max_gap": 900}
    first = client.get("/api/v1/cases/CASE-B9/exposure-budget", params=params).json()
    second = client.get("/api/v1/cases/CASE-B9/exposure-budget", params=params).json()
    assert first == second
