import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# ============================================================================
# Mock设置 - 使用fixture避免全局污染
# ============================================================================

@pytest.fixture(scope="function", autouse=True)
def mock_mindspore_modules():
    """在每个测试中mock MindSpore模块，测试结束后恢复"""
    from unittest.mock import patch
    
    with patch.dict('sys.modules', {
        'mindspore': MagicMock(),
        'mindformers': MagicMock(),
        'mindformers.tools': MagicMock(),
        'mindformers.tools.register': MagicMock(),
        'mindformers.dataset': MagicMock(),
        'mindformers.dataset.handler': MagicMock(),
        'mindformers.dataset.handler.base_handler': MagicMock(),
        'mindformers.models.tokenization_utils': MagicMock(),
        'mindformers.models.tokenization_utils_base': MagicMock(),
        'mindformers.tools.utils': MagicMock(),
    }):
        # 设置Mock装饰器
        mock_register = MagicMock()
        # 加强mock：记录注册调用
        registered_classes = []
        
        def register_decorator(module_type):
            def decorator(cls):
                registered_classes.append((module_type, cls))
                return cls
            return decorator
        
        mock_register.register = register_decorator
        sys.modules['mindformers.tools.register'].MindFormerRegister = mock_register
        sys.modules['mindformers.tools.register'].MindFormerModuleType = MagicMock()
        
        # 创建Mock基类，匹配真实API
        class MockBaseInstructDataHandler:
            """Mock基类，匹配BaseInstructDataHandler的API"""
            def __init__(self, config, **kwargs):
                self.config = config
                self.tokenizer = kwargs.get('tokenizer', MagicMock())
                self.ignore_token_id = kwargs.get('ignore_token_id', -100)
            
            def format_func(self, example):
                """格式化函数 - 子类必须实现"""
                raise NotImplementedError("Subclasses must implement format_func")
            
            def tokenize_func(self, messages):
                """分词函数 - 子类必须实现"""
                raise NotImplementedError("Subclasses must implement tokenize_func")
        
        sys.modules['mindformers.dataset.handler.base_handler'].BaseInstructDataHandler = MockBaseInstructDataHandler
        
        yield {
            'registered_classes': registered_classes,
            'mock_register': mock_register
        }


# ============================================================================
# 常量测试
# ============================================================================

class TestOpenR1Math220kDataHandlerConstants:
    """数据处理器常量测试类"""
    
    def test_prompt_input_constant(self):
        """测试PROMPT_INPUT常量"""
        from modules.openr1_data_handler import PROMPT_INPUT
        expected = r"Please reason step by step, and put your final answer within \boxed{}."
        assert PROMPT_INPUT == expected
    
    def test_max_token_length_constant(self):
        """测试MAX_TOKEN_LENGTH常量"""
        from modules.openr1_data_handler import MAX_TOKEN_LENGTH
        assert MAX_TOKEN_LENGTH == 20480
        assert isinstance(MAX_TOKEN_LENGTH, int)


# ============================================================================
# 初始化测试
# ============================================================================

class TestOpenR1Math220kDataHandlerInit:
    """数据处理器初始化测试类"""
    
    def test_handler_initialization(self):
        """测试处理器初始化"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler
        
        mock_config = MagicMock()
        handler = OpenR1Math220kDataHandler(mock_config)
        
        assert handler is not None
        assert handler.config == mock_config
    
    def test_handler_with_kwargs(self):
        """测试带额外参数的初始化"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler
        
        mock_config = MagicMock()
        mock_tokenizer = MagicMock()
        
        handler = OpenR1Math220kDataHandler(
            mock_config,
            tokenizer=mock_tokenizer
        )
        
        assert handler.tokenizer == mock_tokenizer


# ============================================================================
# format_func测试
# ============================================================================

class TestFormatFunc:
    """format_func功能测试类"""
    
    @pytest.fixture
    def handler(self):
        """创建测试用的处理器"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler
        mock_config = MagicMock()
        return OpenR1Math220kDataHandler(mock_config)
    
    def test_format_func_adds_system_prompt(self, handler):
        """测试format_func添加系统提示"""
        example = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4."}
            ]
        }
        
        result = handler.format_func(example)
        
        assert len(result) == 3  # system + user + assistant
        assert result[0]["role"] == "system"
        assert "boxed" in result[0]["content"]
    
    def test_format_func_preserves_messages(self, handler):
        """测试format_func保留原始消息"""
        original_messages = [
            {"role": "user", "content": "Test question"},
            {"role": "assistant", "content": "Test answer"}
        ]
        example = {"messages": original_messages.copy()}
        
        result = handler.format_func(example)
        
        # 检查原始消息被保留
        assert result[1] == original_messages[0]
        assert result[2] == original_messages[1]
    
    def test_format_func_empty_messages(self, handler):
        """测试空消息列表"""
        example = {"messages": []}
        
        result = handler.format_func(example)
        
        assert len(result) == 1
        assert result[0]["role"] == "system"
    
    def test_format_func_missing_messages_key(self, handler):
        """测试缺少messages键 - 加强断言"""
        from modules.openr1_data_handler import PROMPT_INPUT
        
        example = {}
        
        result = handler.format_func(example)
        
        # 应该有一个系统消息
        assert len(result) >= 1
        # 验证系统消息的内容和角色
        assert result[0]["role"] == "system"
        assert result[0]["content"] == PROMPT_INPUT
        # 如果没有messages，应该只有系统消息
        assert len(result) == 1
    
    def test_format_func_with_thinking_tags(self, handler):
        """测试包含thinking标签的消息"""
        example = {
            "messages": [
                {"role": "user", "content": "Solve x + 1 = 2"},
                {"role": "assistant", "content": "<think>Let me think...</think>\nx = 1"}
            ]
        }
        
        result = handler.format_func(example)
        
        assert "<think>" in result[2]["content"] or "<think>" in result[2]["content"]
    
    def test_format_func_multiple_turns(self, handler):
        """测试多轮对话"""
        example = {
            "messages": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2"}
            ]
        }
        
        result = handler.format_func(example)
        
        assert len(result) == 5  # 1 system + 4 messages
    
    def test_format_func_system_prompt_content(self, handler):
        """测试系统提示的具体内容"""
        from modules.openr1_data_handler import PROMPT_INPUT
        
        example = {"messages": [{"role": "user", "content": "test"}]}
        result = handler.format_func(example)
        
        assert result[0]["content"] == PROMPT_INPUT


# ============================================================================
# tokenize_func测试
# ============================================================================

class TestTokenizeFunc:
    """tokenize_func功能测试类"""
    
    @pytest.fixture
    def handler_with_tokenizer(self):
        """创建带模拟分词器的处理器"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler
        
        mock_config = MagicMock()
        handler = OpenR1Math220kDataHandler(mock_config)
        
        # 配置模拟分词器
        mock_tokenizer = MagicMock()
        # 模拟apply_chat_template返回token ids
        # 151644是<|im_start|>, 77091是assistant token id
        # 结构: [system_tokens, user_tokens, assistant_tokens]
        mock_tokenizer.apply_chat_template.return_value = [
            151644, 100, 101, 102,  # system: im_start + content
            151645,  # im_end
            151644, 103, 104, 105,  # user: im_start + content
            151645,  # im_end
            151644, 77091, 106,  # assistant: im_start + assistant_token + content_start
            200, 201, 202, 203,  # assistant content
            151645  # im_end
        ]
        
        handler.tokenizer = mock_tokenizer
        handler.ignore_token_id = -100
        
        return handler
    
    def test_tokenize_func_returns_dict(self, handler_with_tokenizer):
        """测试tokenize_func返回字典"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant response"}
        ]
        
        result = handler_with_tokenizer.tokenize_func(messages)
        
        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "labels" in result
    
    def test_tokenize_func_input_ids_and_labels_same_length(self, handler_with_tokenizer):
        """测试input_ids和labels长度相同"""
        messages = [
            {"role": "system", "content": "Test"},
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Test"}
        ]
        
        result = handler_with_tokenizer.tokenize_func(messages)
        
        assert len(result["input_ids"]) == len(result["labels"])
    
    def test_tokenize_func_labels_ignore_system_user(self, handler_with_tokenizer):
        """测试labels忽略system和user部分 - 精确验证"""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Assistant"}
        ]
        
        result = handler_with_tokenizer.tokenize_func(messages)
        
        input_ids = result["input_ids"]
        labels = result["labels"]
        
        # 根据mock的token序列结构:
        # [0-3]: system (151644, 100, 101, 102)
        # [4]: im_end (151645)
        # [5-8]: user (151644, 103, 104, 105)
        # [9]: im_end (151645)
        # [10-12]: assistant start (151644, 77091, 106)
        # [13-16]: assistant content (200, 201, 202, 203)
        # [17]: im_end (151645)
        
        # 找到assistant开始位置（77091之后）
        assistant_start_idx = None
        for i, token_id in enumerate(input_ids):
            if token_id == 77091:  # assistant token
                assistant_start_idx = i + 1  # 下一个token开始是assistant内容
                break
        
        assert assistant_start_idx is not None, "Should find assistant token"
        
        # 验证system和user部分的labels都是ignore_token_id
        # system部分: 0到assistant_start_idx-1
        for i in range(assistant_start_idx):
            assert labels[i] == -100, f"Position {i} should be ignored (system/user part)"
        
        # 验证assistant部分的labels不是ignore_token_id
        assistant_labels = labels[assistant_start_idx:]
        assert any(label != -100 for label in assistant_labels), "Assistant part should have valid labels"
        
        # 验证assistant内容token的labels正确
        # assistant内容从assistant_start_idx开始
        for i in range(assistant_start_idx, len(labels)):
            if input_ids[i] != 151645:  # 不是结束token
                assert labels[i] == input_ids[i], f"Position {i}: label should match input_id for assistant content"
    
    def test_tokenize_func_calls_apply_chat_template(self, handler_with_tokenizer):
        """测试调用apply_chat_template"""
        messages = [
            {"role": "system", "content": "Test"},
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Test"}
        ]
        
        handler_with_tokenizer.tokenize_func(messages)
        
        handler_with_tokenizer.tokenizer.apply_chat_template.assert_called_once()
        # 验证调用参数
        call_args = handler_with_tokenizer.tokenizer.apply_chat_template.call_args
        assert call_args[0][0] == messages  # 第一个参数是messages


# ============================================================================
# 截断测试
# ============================================================================

class TestTokenizeFuncTruncation:
    """tokenize_func截断功能测试类"""
    
    @pytest.fixture
    def handler_for_truncation(self):
        """创建用于截断测试的处理器"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler, MAX_TOKEN_LENGTH
        
        mock_config = MagicMock()
        handler = OpenR1Math220kDataHandler(mock_config)
        
        # 创建超长token序列
        # 结构: [start_tokens] + [long_content] + [end_tokens]
        start_tokens = [151644, 77091, 106]
        end_tokens = [151645, 151643]  # im_end, eos
        long_content = list(range(1000, 1000 + MAX_TOKEN_LENGTH + 100))
        long_tokens = start_tokens + long_content + end_tokens
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = long_tokens
        
        handler.tokenizer = mock_tokenizer
        handler.ignore_token_id = -100
        
        return handler, MAX_TOKEN_LENGTH, start_tokens, end_tokens
    
    def test_truncation_applied_for_long_sequences(self, handler_for_truncation):
        """测试长序列被截断 - 精确检查"""
        handler, max_length, start_tokens, end_tokens = handler_for_truncation
        
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Very long assistant response..."}
        ]
        
        result = handler.tokenize_func(messages)
        
        input_ids = result["input_ids"]
        # 精确检查：长度应该 <= MAX_TOKEN_LENGTH
        assert len(input_ids) <= max_length, \
            f"Sequence length {len(input_ids)} should not exceed MAX_TOKEN_LENGTH {max_length}"
        # 应该接近MAX_TOKEN_LENGTH（考虑截断逻辑）
        assert len(input_ids) >= max_length - 10, \
            f"Sequence length {len(input_ids)} should be close to MAX_TOKEN_LENGTH {max_length}"
    
    def test_truncation_preserves_end_tokens(self, handler_for_truncation):
        """测试截断保留结束token - 验证最后tokens"""
        handler, max_length, start_tokens, end_tokens = handler_for_truncation
        
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"}
        ]
        
        result = handler.tokenize_func(messages)
        
        input_ids = result["input_ids"]
        assert len(input_ids) > 0
        
        # 验证最后两个token是原始序列的最后两个（end_tokens）
        # 根据实现，截断应该保留最后两个token
        assert input_ids[-2:] == end_tokens, \
            f"Last two tokens {input_ids[-2:]} should be end tokens {end_tokens}"
        
        # 验证包含结束标记
        assert 151645 in input_ids, "Should contain im_end token"
        assert 151643 in input_ids, "Should contain eos token"


# ============================================================================
# 边界条件测试
# ============================================================================

class TestTokenizeFuncEdgeCases:
    """tokenize_func边界条件测试类"""
    
    @pytest.fixture
    def edge_handler(self):
        """创建边界测试处理器"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler
        
        mock_config = MagicMock()
        handler = OpenR1Math220kDataHandler(mock_config)
        
        mock_tokenizer = MagicMock()
        handler.tokenizer = mock_tokenizer
        handler.ignore_token_id = -100
        
        return handler, mock_tokenizer
    
    def test_empty_assistant_content(self, edge_handler):
        """测试空的assistant内容"""
        handler, mock_tokenizer = edge_handler
        
        mock_tokenizer.apply_chat_template.return_value = [
            151644, 100, 101, 151645,  # system
            151644, 101, 151645,  # user
            151644, 77091, 106, 151645  # assistant (minimal)
        ]
        
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": ""}
        ]
        
        result = handler.tokenize_func(messages)
        
        assert len(result["input_ids"]) == len(result["labels"])
    
    def test_very_short_sequence(self, edge_handler):
        """测试非常短的序列"""
        handler, mock_tokenizer = edge_handler
        
        mock_tokenizer.apply_chat_template.return_value = [
            151644, 77091, 106, 151645
        ]
        
        messages = [{"role": "assistant", "content": "Hi"}]
        
        result = handler.tokenize_func(messages)
        
        assert len(result["input_ids"]) > 0
    
    def test_only_system_message(self, edge_handler):
        """测试仅有系统消息"""
        handler, mock_tokenizer = edge_handler
        
        # 没有assistant token的序列
        mock_tokenizer.apply_chat_template.return_value = [
            151644, 100, 101, 151645
        ]
        
        messages = [{"role": "system", "content": "System only"}]
        
        # 这种情况下target_index会是0
        result = handler.tokenize_func(messages)
        
        # 应该不会崩溃
        assert "input_ids" in result
        assert "labels" in result


# ============================================================================
# 集成测试
# ============================================================================

class TestDataHandlerIntegration:
    """数据处理器集成测试类"""
    
    @pytest.fixture
    def integrated_handler(self):
        """创建集成测试处理器"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler
        
        mock_config = MagicMock()
        handler = OpenR1Math220kDataHandler(mock_config)
        
        def mock_apply_template(messages, **kwargs):
            # 模拟真实的模板应用
            tokens = []
            for msg in messages:
                if msg["role"] == "system":
                    tokens.extend([151644, 100, 101, 151645])
                elif msg["role"] == "user":
                    tokens.extend([151644, 102, 103, 151645])
                elif msg["role"] == "assistant":
                    tokens.extend([151644, 77091, 106, 200, 201, 151645])
            return tokens
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = mock_apply_template
        
        handler.tokenizer = mock_tokenizer
        handler.ignore_token_id = -100
        
        return handler
    
    def test_full_processing_pipeline(self, integrated_handler):
        """测试完整处理流程"""
        # 原始数据
        example = {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "<think>2 + 2 = 4</think>\n\\boxed{4}"}
            ]
        }
        
        # 格式化
        formatted = integrated_handler.format_func(example)
        assert len(formatted) == 3  # system added
        
        # 分词
        tokenized = integrated_handler.tokenize_func(formatted)
        assert "input_ids" in tokenized
        assert "labels" in tokenized
        assert len(tokenized["input_ids"]) == len(tokenized["labels"])
    
    def test_batch_processing(self, integrated_handler):
        """测试批量处理"""
        examples = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"}
                ]
            }
            for i in range(10)
        ]
        
        results = []
        for example in examples:
            formatted = integrated_handler.format_func(example)
            tokenized = integrated_handler.tokenize_func(formatted)
            results.append(tokenized)
        
        assert len(results) == 10
        for result in results:
            assert "input_ids" in result
            assert "labels" in result


# ============================================================================
# 真实数据测试
# ============================================================================

class TestDataHandlerWithRealData:
    """使用真实数据格式的测试类"""
    
    @pytest.fixture
    def real_data_handler(self):
        """创建真实数据测试处理器"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler
        
        mock_config = MagicMock()
        handler = OpenR1Math220kDataHandler(mock_config)
        
        def mock_apply_template(messages, **kwargs):
            tokens = []
            assistant_started = False
            for msg in messages:
                if msg["role"] == "system":
                    tokens.extend([151644, 8948, 100, 101, 151645])
                elif msg["role"] == "user":
                    tokens.extend([151644, 872, 200, 201, 151645])
                elif msg["role"] == "assistant":
                    assistant_started = True
                    tokens.extend([151644, 77091, 106])
                    # 添加内容token
                    content_len = min(len(msg["content"]), 100)
                    tokens.extend(list(range(300, 300 + content_len)))
                    tokens.append(151645)
            return tokens
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = mock_apply_template
        
        handler.tokenizer = mock_tokenizer
        handler.ignore_token_id = -100
        
        return handler
    
    def test_math_problem_format(self, real_data_handler):
        """测试数学问题格式"""
        example = {
            "messages": [
                {"role": "user", "content": "Solve: 3x + 5 = 20"},
                {"role": "assistant", "content": "<think>3x + 5 = 20\n3x = 15\nx = 5</think>\n\\boxed{5}"}
            ]
        }
        
        formatted = real_data_handler.format_func(example)
        tokenized = real_data_handler.tokenize_func(formatted)
        
        assert len(tokenized["input_ids"]) > 0
    
    def test_complex_math_notation(self, real_data_handler):
        """测试复杂数学符号"""
        example = {
            "messages": [
                {"role": "user", "content": "Calculate: ∫x²dx from 0 to 1"},
                {"role": "assistant", "content": "<think>∫x²dx = x³/3\nFrom 0 to 1: 1/3 - 0 = 1/3</think>\n\\boxed{\\frac{1}{3}}"}
            ]
        }
        
        formatted = real_data_handler.format_func(example)
        tokenized = real_data_handler.tokenize_func(formatted)
        
        assert len(tokenized["input_ids"]) > 0
    
    def test_multiline_thinking(self, real_data_handler):
        """测试多行思考过程"""
        example = {
            "messages": [
                {"role": "user", "content": "Prove that √2 is irrational"},
                {"role": "assistant", "content": """<think>
Assume √2 is rational.
Then √2 = p/q where p,q are coprime integers.
2 = p²/q²
2q² = p²
Therefore p² is even, so p is even.
Let p = 2k.
2q² = 4k²
q² = 2k²
Therefore q is also even.
This contradicts p,q being coprime.
</think>
By contradiction, √2 is irrational. \\boxed{\\text{Proven}}"""}
            ]
        }
        
        formatted = real_data_handler.format_func(example)
        tokenized = real_data_handler.tokenize_func(formatted)
        
        assert len(tokenized["input_ids"]) > 0


# ============================================================================
# 错误处理测试
# ============================================================================

class TestDataHandlerErrorHandling:
    """错误处理测试类"""
    
    @pytest.fixture
    def error_handler(self):
        """创建错误测试处理器"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler
        
        mock_config = MagicMock()
        handler = OpenR1Math220kDataHandler(mock_config)
        handler.ignore_token_id = -100
        
        return handler
    
    def test_none_messages(self, error_handler):
        """测试None消息"""
        example = {"messages": None}
        
        # 应该能处理或抛出明确的错误
        try:
            result = error_handler.format_func(example)
            # 如果返回结果，应该是有效的
            assert isinstance(result, list)
        except (TypeError, AttributeError):
            # 预期的错误类型
            pass
    
    def test_invalid_message_format(self, error_handler):
        """测试无效消息格式"""
        example = {"messages": "not a list"}
        
        try:
            result = error_handler.format_func(example)
        except (TypeError, AttributeError):
            pass
    
    def test_missing_role_in_message(self, error_handler):
        """测试消息缺少role字段"""
        example = {
            "messages": [
                {"content": "Missing role"}
            ]
        }
        
        # format_func不检查消息内部结构，应该能通过
        result = error_handler.format_func(example)
        assert len(result) == 2  # system + original message


# ============================================================================
# 性能测试 - 使用conftest中的fixtures
# ============================================================================

class TestDataHandlerPerformance:
    """性能测试类"""
    
    @pytest.fixture
    def perf_handler(self):
        """创建性能测试处理器"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler
        
        mock_config = MagicMock()
        handler = OpenR1Math220kDataHandler(mock_config)
        
        def fast_apply_template(messages, **kwargs):
            return [151644, 77091, 106] + list(range(1000)) + [151645]
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = fast_apply_template
        
        handler.tokenizer = mock_tokenizer
        handler.ignore_token_id = -100
        
        return handler
    
    @pytest.mark.slow
    def test_processing_speed(self, perf_handler, performance_timer):
        """测试处理速度 - 标记为慢速测试"""
        example = {
            "messages": [
                {"role": "user", "content": "Test question"},
                {"role": "assistant", "content": "Test answer"}
            ]
        }
        
        iterations = 1000
        
        with performance_timer() as timer:
            for _ in range(iterations):
                formatted = perf_handler.format_func(example)
        
        # 1000次format应该在1秒内完成
        assert timer.elapsed < 1.0
    
    @pytest.mark.slow
    def test_memory_usage(self, perf_handler, memory_tracker):
        """测试内存使用 - 标记为慢速测试"""
        example = {
            "messages": [
                {"role": "user", "content": "Test " * 1000},
                {"role": "assistant", "content": "Answer " * 1000}
            ]
        }
        
        with memory_tracker() as tracker:
            for _ in range(100):
                formatted = perf_handler.format_func(example)
                tokenized = perf_handler.tokenize_func(formatted)
        
        # 内存使用应该在合理范围内（小于100MB）
        assert tracker.peak < 100 * 1024 * 1024


# 运行测试的主函数
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
