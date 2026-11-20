"""生成固定种子的中文电商经营分析演示数据库（模拟数据，非真实生产数据）。"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).with_name("insightflow_ecommerce.sqlite")
SEED = 20261116


class _FixedRandom:
    """跨运行保持一致的轻量 LCG，避免演示数据依赖随机库实现细节。"""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def randrange(self, stop: int) -> int:
        return self._next() % stop

    def randint(self, start: int, stop: int) -> int:
        return start + self.randrange(stop - start + 1)

    def uniform(self, start: float, stop: float) -> float:
        return start + (stop - start) * (self._next() / 2**32)

    def choice(self, values):
        return values[self.randrange(len(values))]

    def choices(self, values, weights):
        if len(values) != len(weights) or not values:
            raise ValueError("values 与 weights 必须等长且不能为空")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("抽样权重必须为有限的非负数")
        total_weight = sum(weights)
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("抽样权重总和必须大于 0")
        target = (self._next() / 2**32) * total_weight
        for value, weight in zip(values, weights, strict=True):
            if target < weight:
                return [value]
            target -= weight
        return [values[-1]]  # 浮点舍入的兜底分支。


def build_database(path: Path = OUT, *, overwrite: bool = False) -> None:
    """创建包含 18 个月连续日期和完整外键关系的演示数据。"""
    rng = _FixedRandom(SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"目标文件已存在，不覆盖用户文件：{path}。如需重建请显式传入 overwrite=True。")
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
        CREATE TABLE regions(region_id INTEGER PRIMARY KEY, region_name TEXT NOT NULL UNIQUE);
        CREATE TABLE sales_channels(channel_id INTEGER PRIMARY KEY, channel_name TEXT NOT NULL UNIQUE);
        CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, customer_name TEXT NOT NULL, region_id INTEGER NOT NULL REFERENCES regions(region_id), created_at TEXT NOT NULL);
        CREATE TABLE products(product_id INTEGER PRIMARY KEY, product_name TEXT NOT NULL, category TEXT NOT NULL, unit_cost REAL NOT NULL);
        CREATE TABLE orders(order_id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(customer_id), channel_id INTEGER NOT NULL REFERENCES sales_channels(channel_id), order_date TEXT NOT NULL, status TEXT NOT NULL);
        CREATE TABLE order_items(order_item_id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL REFERENCES orders(order_id), product_id INTEGER NOT NULL REFERENCES products(product_id), quantity INTEGER NOT NULL CHECK(quantity > 0), unit_price REAL NOT NULL CHECK(unit_price >= 0), discount_rate REAL NOT NULL CHECK(discount_rate BETWEEN 0 AND 1));
        CREATE INDEX idx_orders_date ON orders(order_date);
        CREATE INDEX idx_orders_customer ON orders(customer_id);
        CREATE INDEX idx_items_order ON order_items(order_id);
        """)
    regions = ["华北", "华东", "华南", "华中", "西南", "西北", "东北"]
    channels = ["直营网店", "电商平台", "线下门店", "企业团购"]
    conn.executemany("INSERT INTO regions VALUES (?, ?)", enumerate(regions, 1))
    conn.executemany("INSERT INTO sales_channels VALUES (?, ?)", enumerate(channels, 1))
    conn.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?)",
        [(i, f"客户{i:04d}", rng.randint(1, len(regions)), f"2024-{rng.randint(1, 12):02d}-01") for i in range(1, 301)],
    )
    product_rows = []
    for i in range(1, 81):
        category = ["食品", "家居", "数码", "服饰", "美妆", "运动", "母婴", "图书"][((i - 1) // 10) % 8]
        product_rows.append((i, f"{category}商品{i:03d}", category, round(rng.uniform(8, 300), 2)))
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", product_rows)

    # 按天循环生成 18 个月订单，分布固定、日期连续，支持趋势与同比类查询。
    start = date(2025, 1, 1)
    order_rows, item_rows = [], []
    item_id = 1
    for order_id in range(1, 5001):
        day = start + timedelta(days=(order_id * 7 + rng.randrange(31)) % 547)
        order_rows.append(
            (
                order_id,
                rng.randint(1, 300),
                rng.randint(1, 4),
                day.isoformat(),
                rng.choices(["已完成", "已取消", "处理中"], [0.88, 0.06, 0.06])[0],
            )
        )
        for _ in range(rng.randint(1, 4)):
            product = product_rows[rng.randrange(len(product_rows))]
            item_rows.append(
                (
                    item_id,
                    order_id,
                    product[0],
                    rng.randint(1, 5),
                    round(product[3] * rng.uniform(1.2, 2.4), 2),
                    rng.choice([0, 0, 0.05, 0.1, 0.15]),
                )
            )
            item_id += 1
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", order_rows)
    conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)", item_rows)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成中文电商模拟 SQLite 数据库")
    parser.add_argument("--overwrite", action="store_true", help="明确允许覆盖现有演示数据库")
    args = parser.parse_args()
    build_database(overwrite=args.overwrite)
    print(f"模拟经营分析数据库已生成：{OUT}")
