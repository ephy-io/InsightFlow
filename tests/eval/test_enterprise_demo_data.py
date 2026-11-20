import sqlite3

import pytest

from example.enterprise.generate_demo_db import _FixedRandom, build_database


@pytest.mark.parametrize(
    "values, weights",
    [([], []), (["a"], []), (["a", "b"], [0, 0]), (["a"], [-1]), (["a"], [float("inf")])],
)
def test_fixed_random_rejects_invalid_choice_weights(values, weights):
    with pytest.raises(ValueError):
        _FixedRandom(1).choices(values, weights)


def test_demo_data_is_reproducible_relational_and_covers_continuous_dates(tmp_path):
    db_path = tmp_path / "demo.sqlite"
    build_database(db_path)
    second_db_path = tmp_path / "demo-second.sqlite"
    build_database(second_db_path)
    first_snapshot = {}
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 5000
        assert connection.execute("SELECT COUNT(DISTINCT order_date) FROM orders").fetchone()[0] == 547
        assert connection.execute("SELECT MIN(order_date), MAX(order_date) FROM orders").fetchone() == (
            "2025-01-01",
            "2026-07-01",
        )
        status_counts = dict(connection.execute("SELECT status, COUNT(*) FROM orders GROUP BY status"))
        assert set(status_counts) == {"已完成", "已取消", "处理中"}
        assert status_counts["已完成"] / 5000 == pytest.approx(0.88, abs=0.02)
        assert status_counts["已取消"] / 5000 == pytest.approx(0.06, abs=0.02)
        assert status_counts["处理中"] / 5000 == pytest.approx(0.06, abs=0.02)
        for table in ("regions", "sales_channels", "customers", "products", "orders", "order_items"):
            first_snapshot[table] = connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
    with sqlite3.connect(second_db_path) as connection:
        second_snapshot = {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in ("regions", "sales_channels", "customers", "products", "orders", "order_items")
        }
    assert first_snapshot == second_snapshot


def test_demo_generator_never_overwrites_without_explicit_opt_in(tmp_path):
    db_path = tmp_path / "demo.sqlite"
    db_path.write_bytes(b"user data")
    with pytest.raises(FileExistsError, match="不覆盖用户文件"):
        build_database(db_path)
    assert db_path.read_bytes() == b"user data"
