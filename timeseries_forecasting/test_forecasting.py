#!/usr/bin/env python3
"""test_forecasting.py:用于计时器预测服务的测试脚本。"""

import time
from datetime import datetime, timedelta

import numpy as np
import requests
from requests.exceptions import RequestException


class TimeseriesForecastingTester:
    """Timer 预测服务的测试类。"""

    def __init__(self, base_url="http://localhost:8765"):
        """初始化测试器。"""
        self.base_url = base_url
        self.predictions_endpoint = f"{base_url}/predict"
        self.health_endpoint = f"{base_url}/health"

    def generate_sample_data(self, length=100, frequency="H"):
        """生成样本时间序列数据进行测试。"""
        # 生成带有趋势和季节性的合成时间序列
        t = np.arange(length)

        # 添加趋势
        trend = 0.1 * t

        # 添加季节性(小时数据的每日模式)
        if frequency == "H":
            seasonality = 5 * np.sin(2 * np.pi * t / 24)
        else:
            seasonality = 3 * np.sin(2 * np.pi * t / 7)  # 每日数据的每周模式

        # 添加噪声
        noise = np.random.normal(0, 1, length)

        # 组合组件
        values = 100 + trend + seasonality + noise

        return values.tolist()

    def test_basic_forecasting(self):
        """测试基本时间序列预测。"""
        print("\n=== Testing Basic Forecasting ===")

        # 生成样本数据
        sample_data = self.generate_sample_data(length=168, frequency="H")  # 1周小时数据

        # 构造请求数据
        payload = {
            "input": sample_data,
            "forecast_window": 24,  # 接下来24小时的预报
            "frequency": "H",
            "input_len": 168,  # 使用上周的数据
        }

        # 发送请求
        try:
            response = requests.post(
                self.predictions_endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print("✓ Basic forecasting successful")
                print(f"  - Input length: {len(sample_data)}")
                print(f"  - Forecast Window: {payload['forecast_window']}")
                print(f"  - Predictions length: {len(result.get('predictions', []))}")
                print(f"  - Sample predictions: {result.get('predictions', [])[:5]}")
                return True
            else:
                print(f"✗ Request failed with status: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {str(e)}")
            return False

    def test_structured_data(self):
        """以结构化数据(时间戳+值)进行测试预测。"""
        print("\n=== Testing Structured Data Forecasting ===")

        # 用时间戳生成结构化数据
        start_time = datetime.now() - timedelta(days=7)
        structured_data = []

        for i in range(168):  # 1周小时数据
            timestamp = start_time + timedelta(hours=i)
            value = 100 + 0.1 * i + 5 * np.sin(2 * np.pi * i / 24) + np.random.normal(0, 1)

            structured_data.append({"timestamp": timestamp.isoformat(), "value": value})

        # 构造请求数据
        payload = {
            "input": structured_data,
            "forecast_window": 48,  # 接下来48小时的预测
            "frequency": "H",
            "target_column": "value",
        }

        # 发送请求
        try:
            response = requests.post(
                self.predictions_endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print("✓ Structured data forecasting successful")
                print(f"  - Input records: {len(structured_data)}")
                print(f"  - Forecast Window: {payload['forecast_window']}")
                print(f"  - Predictions length: {len(result.get('predictions', []))}")
                return True
            else:
                print(f"✗ Request failed with status: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {str(e)}")
            return False

    def test_different_windows(self):
        """使用不同预测窗口的测试预测。"""
        print("\n=== Testing Different Forecast Horizons ===")

        sample_data = self.generate_sample_data(length=100)
        windows = [1, 12, 24, 48, 72]

        for window in windows:
            payload = {"input": sample_data, "forecast_window": window, "frequency": "H"}

            try:
                response = requests.post(
                    self.predictions_endpoint, json=payload, headers={"Content-Type": "application/json"}, timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    predictions_len = len(result.get("predictions", []))
                    print(f"✓ Window {window}: {predictions_len} predictions")
                else:
                    print(f"✗ Window {window}: Failed with status {response.status_code}")
                    return False

            except requests.exceptions.RequestException as e:
                print(f"✗ Window {window}: Request failed - {str(e)}")
                return False
        return True

    def test_error_handling(self):
        """用无效输入进行测试错误处理。"""
        print("\n=== Testing Error Handling ===")

        # 测试空输入
        try:
            response = requests.post(
                self.predictions_endpoint, json={"input": []}, headers={"Content-Type": "application/json"}, timeout=10
            )
            print(f"Empty input: Status {response.status_code}")
            if response.status_code != 400:
                print("✗ Empty input: Expected 400 status code")
                return False
        except RequestException:
            print("Empty input: exception occurred not expected")
            return False

        # 测试无效的 JSON
        try:
            response = requests.post(
                self.predictions_endpoint, data="invalid json", headers={"Content-Type": "application/json"}, timeout=10
            )
            print(f"Invalid JSON: Status {response.status_code}")
            if response.status_code != 422:
                print("✗ Empty input: Expected 422 status code")
                return False
        except RequestException:
            print("Empty input: exception occurred not expected")
            return False
        return True

    def test_health_check(self):
        """检测服务健康检查。"""
        print("\n=== Testing Service Health ===")

        try:
            # 测试健康终点
            response = requests.get(self.health_endpoint, timeout=5)

            if response.status_code == 200:
                result = response.json()
                print("✓ Service health check passed")
                print(f"  - Model initialized: {result.get('model_initialized', 'Unknown')}")
                print(f"  - Uptime: {result.get('uptime_seconds', 'Unknown')} seconds")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False

        except RequestException as e:
            print(f"Health check failed: {str(e)}")
            return False

    def run_all_tests(self):
        """进行所有测试。"""
        print("=" * 50)
        print("TIMER FORECASTING SERVICE TESTS")
        print("=" * 50)

        # 等服务准备好了
        print("Waiting for service to be ready...")
        for _i in range(30):  # 等30秒钟
            try:
                response = requests.get(self.health_endpoint, timeout=2)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("model_initialized", False):
                        print("✓ Service is ready")
                        break
            except RequestException:
                pass
            time.sleep(1)
        else:
            print("✗ Service not ready after 30 seconds")
            return False

        # 运行测试
        tests = [
            self.test_health_check,
            self.test_basic_forecasting,
            self.test_structured_data,
            self.test_different_windows,
            self.test_error_handling,
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"✗ Test {test.__name__} failed with exception: {str(e)}")

        print("\n" + "=" * 50)
        print(f"TESTS COMPLETED: {passed}/{total} passed")
        print("=" * 50)

        return passed == total


def main():
    """主要职能。"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Timer forecasting service")
    parser.add_argument(
        "--url", default="http://localhost:8765", help="Base URL of the service (default: http://localhost:8080)"
    )

    args = parser.parse_args()

    tester = TimeseriesForecastingTester(args.url)
    success = tester.run_all_tests()

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
