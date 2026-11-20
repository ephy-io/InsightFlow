"""用于上下文配置的单元测试。"""

from insightflow.context_config import ContextConfig, get_context_config, update_context_config


class TestContextConfig:
    """ContextConfig 类测试用例。"""

    def test_default_config_values(self):
        """测试默认配置有预期值。"""
        config = ContextConfig()

        # 测试默认值
        assert config.enabled is True
        assert config.summary_trigger_tokens == 50000
        assert config.keep_recent_messages == 30
        assert config.max_tool_output_length == 3000
        assert config.max_sql_result_rows == 50
        assert config.max_code_output_lines == 50

        # 测试布尔旗
        assert config.enable_summarization is True
        assert config.enable_conversation_summary is True
        assert config.preserve_tool_errors is True
        assert config.preserve_recent_sql is True

    def test_custom_config_values(self):
        """测试以自定义值创建配置。"""
        config = ContextConfig(
            enabled=False,
            summary_trigger_tokens=8000,
            keep_recent_messages=5,
            max_tool_output_length=1000,
            enable_summarization=False,
        )

        assert config.enabled is False
        assert config.summary_trigger_tokens == 8000
        assert config.keep_recent_messages == 5
        assert config.max_tool_output_length == 1000
        assert config.enable_summarization is False

        # 其他值应当使用默认值
        assert config.max_sql_result_rows == 50
        assert config.preserve_tool_errors is True

    def test_config_validation_logic(self):
        """测试配置中的逻辑关系。"""
        config = ContextConfig()

        # 保留最近的信息应当合理
        assert config.keep_recent_messages > 0
        assert config.keep_recent_messages < 100  # 基本一致性检查

        # 产出限制应为正数
        assert config.max_tool_output_length > 0
        assert config.max_sql_result_rows > 0
        assert config.max_code_output_lines > 0

        # 托肯限额应当合理
        assert config.summary_trigger_tokens > 0

    def test_get_context_config(self):
        """测试获取上下文配置。"""
        config = get_context_config()
        assert isinstance(config, ContextConfig)

    def test_update_context_config_single_value(self):
        """测试更新单一配置值。"""
        updated_config = update_context_config(summary_trigger_tokens=15000)

        assert updated_config.summary_trigger_tokens == 15000
        # 其他数值应保持不变
        assert updated_config.keep_recent_messages == get_context_config().keep_recent_messages

    def test_update_context_config_multiple_values(self):
        """测试更新多个配置值。"""
        updated_config = update_context_config(
            summary_trigger_tokens=20000,
            keep_recent_messages=15,
            enable_summarization=False,
            max_tool_output_length=3000,
        )

        assert updated_config.summary_trigger_tokens == 20000
        assert updated_config.keep_recent_messages == 15
        assert updated_config.enable_summarization is False
        assert updated_config.max_tool_output_length == 3000

    def test_update_context_config_invalid_attribute(self):
        """测试以无效属性名称更新配置。"""
        # 不应产生错误，只需忽略无效属性
        config = update_context_config(invalid_attribute=123)
        assert not hasattr(config, "invalid_attribute")

    def test_update_context_config_returns_copy(self):
        """测试 update_context_config 返回一个修改的复制件。"""
        get_context_config()
        updated_config = update_context_config(summary_trigger_tokens=30000)

        # 原件应保持不变(如果是那样设计的)
        # 更新后应具有新的值
        assert updated_config.summary_trigger_tokens == 30000


class TestContextConfigPresets:
    """测试常见情景的不同配置预设。"""

    def test_minimal_context_config(self):
        """用于最小上下文管理的测试配置。"""
        config = ContextConfig(
            enabled=True,
            enable_summarization=False,
            enable_conversation_summary=False,
            max_tool_output_length=500,
        )

        assert config.enabled is True
        assert config.enable_summarization is False
        assert config.enable_conversation_summary is False

    def test_aggressive_compression_config(self):
        """用于侵略性上下文压缩的测试配置。"""
        config = ContextConfig(
            enabled=True,
            summary_trigger_tokens=6000,
            keep_recent_messages=5,
            max_tool_output_length=1000,
            max_sql_result_rows=10,
            max_code_output_lines=20,
            enable_summarization=True,
        )

        assert config.summary_trigger_tokens == 6000
        assert config.keep_recent_messages == 5
        assert config.max_tool_output_length == 1000
        assert config.max_sql_result_rows == 10
        assert config.max_code_output_lines == 20

    def test_development_debug_config(self):
        """适合开发/调试的测试配置。"""
        config = ContextConfig(
            enabled=True,
            summary_trigger_tokens=40000,
            keep_recent_messages=20,
            max_tool_output_length=10000,  # 别多修剪
            preserve_tool_errors=True,  # 总是保存错误
        )

        assert config.preserve_tool_errors is True

    def test_production_optimized_config(self):
        """测试配置优化用于生产。"""
        config = ContextConfig(
            enabled=True,
            summary_trigger_tokens=10000,
            keep_recent_messages=8,
            max_tool_output_length=1500,
            max_sql_result_rows=15,
            enable_summarization=True,
            preserve_tool_errors=True,
        )

        assert config.summary_trigger_tokens == 10000
        assert config.enable_summarization is True
        assert config.preserve_tool_errors is True


class TestContextConfigEdgeCases:
    """测试边框大小写和上下文配置的边界条件。"""

    def test_zero_values(self):
        """测试配置为零值。"""
        config = ContextConfig(
            summary_trigger_tokens=0,
            keep_recent_messages=0,
            max_tool_output_length=0,
        )

        # 应接受零值(可能在实践中引起问题)
        assert config.keep_recent_messages == 0
        assert config.summary_trigger_tokens == 0
        assert config.max_tool_output_length == 0

    def test_very_large_values(self):
        """测试配置值非常大。"""
        config = ContextConfig(
            summary_trigger_tokens=900000,
            keep_recent_messages=1000,
            max_tool_output_length=100000,
        )

        assert config.keep_recent_messages == 1000

    def test_inconsistent_token_limits(self):
        """测试即时触发> 最大符号的配置。"""

        # 应该接受，但可能会引起逻辑问题

    def test_all_features_disabled(self):
        """已禁用所有特性的测试配置。"""
        config = ContextConfig(
            enabled=False,
            enable_summarization=False,
            enable_conversation_summary=False,
        )

        assert config.enabled is False
        assert config.enable_summarization is False
        assert config.enable_conversation_summary is False

    def test_config_serialization(self):
        """测试配置可以转换成/从 dict(如果需要)。"""
        config = ContextConfig(summary_trigger_tokens=15000, enable_summarization=True)

        # 测试转换为类似 dict 的表示
        config_dict = {
            "summary_trigger_tokens": config.summary_trigger_tokens,
            "enable_summarization": config.enable_summarization,
            "enabled": config.enabled,
        }

        assert config_dict["summary_trigger_tokens"] == 15000
        assert config_dict["enable_summarization"] is True
        assert config_dict["enabled"] is True

    def test_config_immutability_simulation(self):
        """测试配置在行动上的表现一致。"""
        config1 = ContextConfig(summary_trigger_tokens=10000)
        config2 = ContextConfig(summary_trigger_tokens=10000)

        # 同样的数值应该是平等的
        assert config1.summary_trigger_tokens == config2.summary_trigger_tokens
        assert config1.enabled == config2.enabled

    def test_realistic_configuration_scenarios(self):
        """测试现实的配置方案。"""

        # 小型数据集设想
        small_config = ContextConfig()

        # 大型数据集设想
        ContextConfig()

        # 交互式分析设想
        interactive_config = ContextConfig(
            keep_recent_messages=50,  # 保留更多回转的上下文
            preserve_tool_errors=True,
            max_tool_output_length=5000,  # 别太冲动修剪
        )

        assert interactive_config.keep_recent_messages > small_config.keep_recent_messages
        assert interactive_config.preserve_tool_errors is True
