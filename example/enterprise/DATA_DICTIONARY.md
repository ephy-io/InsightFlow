# 中文电商模拟数据字典

本数据集通过 [generate_demo_db.py](generate_demo_db.py) 以固定种子 `20261116` 生成，覆盖 2025-01-01 至 2026-07-01 的连续日期范围（547 天）。数据完全为模拟数据，不含真实企业或个人信息。

| 表 | 主键 | 主要字段 | 关系与用途 |
|---|---|---|---|
| `regions` | `region_id` | `region_name` | 地区维表；客户通过 `region_id` 关联 |
| `sales_channels` | `channel_id` | `channel_name` | 销售渠道维表；订单通过 `channel_id` 关联 |
| `customers` | `customer_id` | `customer_name`, `region_id`, `created_at` | 客户维表；关联地区 |
| `products` | `product_id` | `product_name`, `category`, `unit_cost` | 商品维表 |
| `orders` | `order_id` | `customer_id`, `channel_id`, `order_date`, `status` | 订单事实表；关联客户和渠道 |
| `order_items` | `order_item_id` | `order_id`, `product_id`, `quantity`, `unit_price`, `discount_rate` | 订单明细事实表；关联订单和商品 |

## 指标口径

- 订单数：`COUNT(DISTINCT orders.order_id)`。
- 销售额：`SUM(order_items.quantity * order_items.unit_price * (1 - order_items.discount_rate))`。
- 已完成销售额：在销售额公式上增加 `orders.status = '已完成'`。
- 商品件数：`SUM(order_items.quantity)`。
- 订单日期范围使用左闭右开区间，字段以 ISO 日期文本存储。

演示库使用 SQLite 外键约束，生成完成后启用 `PRAGMA foreign_keys=ON` 校验。Schema Linking 元数据存放在同目录的 `catalog/`。
