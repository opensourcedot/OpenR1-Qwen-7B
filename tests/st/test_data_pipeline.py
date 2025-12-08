import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ============================================================================
# 常量定义
# ============================================================================

# 测试数据集大小
SAMPLE_DATASET_SIZE = 100
LARGE_DATASET_SIZE = 1000
INTEGRATION_DATASET_SIZE = 50

# Token ID 常量
PAD_TOKEN_ID = 151643
EOS_TOKEN_ID = 151643
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
ASSISTANT_TOKEN_ID = 77091
IGNORE_TOKEN_ID = -100

# 模型配置常量
VOCAB_SIZE = 152064
MAX_SEQ_LENGTH = 32768

# 性能测试常量
BATCH_SIZE = 100
PERFORMANCE_TEST_SIZE = 10000
MIN_THROUGHPUT = 1000  # 每秒至少处理1000个样本
MAX_IO_TIME = 5.0  # IO操作最大时间（秒）
MAX_MEMORY_MB = 500  # 最大内存使用（MB）


# ============================================================================
# Mock设置 - 使用fixture避免全局污染
# ============================================================================

@pytest.fixture(scope="function", autouse=True)
def mock_mindspore_modules():
    """在每个测试中mock MindSpore模块，测试结束后恢复"""
    with patch.dict('sys.modules', {
        'mindspore': MagicMock(),
        'mindformers': MagicMock(),
        'mindformers.tools': MagicMock(),
        'mindformers.tools.register': MagicMock(),
        'mindformers.dataset': MagicMock(),
        'mindformers.dataset.handler': MagicMock(),
        'mindformers.dataset.handler.base_handler': MagicMock(),
    }):
        # 设置Mock装饰器
        mock_register = MagicMock()
        mock_register.register = lambda x: lambda cls: cls
        sys.modules['mindformers.tools.register'].MindFormerRegister = mock_register
        sys.modules['mindformers.tools.register'].MindFormerModuleType = MagicMock()
        
        # 创建Mock基类
        class MockBaseInstructDataHandler:
            """Mock基类，匹配BaseInstructDataHandler的API"""
            def __init__(self, config, **kwargs):
                self.config = config
                self.tokenizer = kwargs.get('tokenizer', MagicMock())
                self.ignore_token_id = IGNORE_TOKEN_ID
        
        sys.modules['mindformers.dataset.handler.base_handler'].BaseInstructDataHandler = MockBaseInstructDataHandler
        
        yield


# ============================================================================
# 数据管道设置测试
# ============================================================================

class TestDataPipelineSetup:
    """数据管道设置测试类"""
    
    @pytest.fixture
    def sample_math_dataset(self) -> List[Dict[str, Any]]:
        """创建示例数学数据集"""
        return [
            {
                "problem": f"Calculate {i} + {i+1}",
                "messages": [
                    {"role": "user", "content": f"Calculate {i} + {i+1}"},
                    {"role": "assistant", "content": f"<think>{i} + {i+1} = {2*i+1}</think>\n\\boxed{{{2*i+1}}}"}
                ],
                "answer": str(2*i+1)
            }
            for i in range(SAMPLE_DATASET_SIZE)
        ]
    
    @pytest.fixture
    def dataset_file(self, tmp_path, sample_math_dataset) -> Path:
        """创建数据集文件"""
        file_path = tmp_path / "math_dataset.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in sample_math_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return file_path
    
    def test_dataset_file_creation(self, dataset_file):
        """测试数据集文件创建"""
        assert dataset_file.exists()
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        assert len(lines) == SAMPLE_DATASET_SIZE
    
    def test_dataset_loading(self, dataset_file):
        """测试数据集加载"""
        data = []
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        assert len(data) == SAMPLE_DATASET_SIZE
        assert "problem" in data[0]
        assert "messages" in data[0]
        assert "answer" in data[0]


# ============================================================================
# 数据格式化测试
# ============================================================================

class TestDataFormatting:
    """数据格式化测试类"""
    
    @pytest.fixture
    def data_handler(self):
        """创建数据处理器"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler
        
        mock_config = MagicMock()
        handler = OpenR1Math220kDataHandler(mock_config)
        
        return handler
    
    def test_format_single_example(self, data_handler):
        """测试格式化单个样本"""
        example = {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4"}
            ]
        }
        
        formatted = data_handler.format_func(example)
        
        assert len(formatted) == 3  # system + user + assistant
        assert formatted[0]["role"] == "system"
    
    def test_format_batch_examples(self, data_handler):
        """测试批量格式化"""
        examples = [
            {"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]}
            for i in range(10)
        ]
        
        formatted_batch = [data_handler.format_func(ex) for ex in examples]
        
        assert len(formatted_batch) == 10
        for formatted in formatted_batch:
            assert formatted[0]["role"] == "system"
    
    def test_format_preserves_content(self, data_handler):
        """测试格式化保留内容"""
        original_content = "This is a <think>thought</think> with \\boxed{answer}"
        example = {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": original_content}
            ]
        }
        
        formatted = data_handler.format_func(example)
        
        assert formatted[2]["content"] == original_content


# ============================================================================
# 分词测试
# ============================================================================

class TestTokenization:
    """分词测试类"""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """创建模拟分词器"""
        tokenizer = MagicMock()
        
        def mock_apply_template(messages, **kwargs):
            """模拟apply_chat_template方法"""
            base_tokens = [IM_START_TOKEN_ID, 100, 101, IM_END_TOKEN_ID]  # system
            for msg in messages:
                if msg["role"] == "user":
                    base_tokens.extend([IM_START_TOKEN_ID, 102, 103, IM_END_TOKEN_ID])
                elif msg["role"] == "assistant":
                    base_tokens.extend([IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, 106, 200, 201, 202, IM_END_TOKEN_ID])
            return base_tokens
        
        tokenizer.apply_chat_template = mock_apply_template
        tokenizer.pad_token_id = PAD_TOKEN_ID
        tokenizer.eos_token_id = EOS_TOKEN_ID
        
        return tokenizer
    
    @pytest.fixture
    def tokenizer_handler(self, mock_tokenizer):
        """创建带分词器的处理器"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler
        
        mock_config = MagicMock()
        handler = OpenR1Math220kDataHandler(mock_config)
        handler.tokenizer = mock_tokenizer
        handler.ignore_token_id = IGNORE_TOKEN_ID
        
        return handler
    
    def test_tokenize_single_example(self, tokenizer_handler):
        """测试单样本分词"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"}
        ]
        
        result = tokenizer_handler.tokenize_func(messages)
        
        assert "input_ids" in result
        assert "labels" in result
        assert len(result["input_ids"]) == len(result["labels"])
    
    def test_tokenize_batch(self, tokenizer_handler):
        """测试批量分词"""
        batch = [
            [
                {"role": "system", "content": "S"},
                {"role": "user", "content": f"Q{i}"},
                {"role": "assistant", "content": f"A{i}"}
            ]
            for i in range(10)
        ]
        
        results = [tokenizer_handler.tokenize_func(msgs) for msgs in batch]
        
        assert len(results) == 10
        for result in results:
            assert "input_ids" in result
            assert "labels" in result


# ============================================================================
# 数据打包测试
# ============================================================================

class TestDataPacking:
    """数据打包测试类"""
    
    def test_sequence_packing_basic(self):
        """测试基本序列打包"""
        sequences = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9]
        ]
        
        max_length = 10
        pad_token = PAD_TOKEN_ID
        
        # 简单打包：连接并填充
        packed = []
        current_pack = []
        
        for seq in sequences:
            if len(current_pack) + len(seq) <= max_length:
                current_pack.extend(seq)
            else:
                while len(current_pack) < max_length:
                    current_pack.append(pad_token)
                packed.append(current_pack)
                current_pack = seq.copy()
        
        if current_pack:
            while len(current_pack) < max_length:
                current_pack.append(pad_token)
            packed.append(current_pack)
        
        assert len(packed) == 2
        assert all(len(p) == max_length for p in packed)
    
    def test_sequence_packing_with_attention_mask(self):
        """测试带注意力掩码的打包"""
        sequences = [[1, 2, 3], [4, 5, 6, 7]]
        max_length = 10
        pad_token = PAD_TOKEN_ID
        
        packed_input_ids = []
        packed_attention_mask = []
        current_ids = []
        current_mask = []
        
        for seq in sequences:
            if len(current_ids) + len(seq) <= max_length:
                current_ids.extend(seq)
                current_mask.extend([1] * len(seq))
        
        while len(current_ids) < max_length:
            current_ids.append(pad_token)
            current_mask.append(0)
        
        packed_input_ids.append(current_ids)
        packed_attention_mask.append(current_mask)
        
        assert sum(packed_attention_mask[0]) == 7  # 3 + 4 真实token


# ============================================================================
# 数据验证测试
# ============================================================================

class TestDataValidation:
    """数据验证测试类"""
    
    def test_validate_input_ids_range(self):
        """测试input_ids范围验证"""
        # 有效的input_ids
        valid_ids = [100, 200, PAD_TOKEN_ID, IM_START_TOKEN_ID]
        assert all(0 <= token_id < VOCAB_SIZE for token_id in valid_ids)
        
        # 无效的input_ids
        invalid_ids = [200000, -1]
        assert not all(0 <= token_id < VOCAB_SIZE for token_id in invalid_ids)
    
    def test_validate_labels_format(self):
        """测试labels格式验证"""
        # 有效的labels
        labels = [IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID, 100, 200, PAD_TOKEN_ID]
        
        # 检查ignore_token_id之后有真实标签
        has_valid_labels = any(label != IGNORE_TOKEN_ID for label in labels)
        assert has_valid_labels
    
    def test_validate_sequence_length(self):
        """测试序列长度验证"""
        # 有效长度
        valid_seq = list(range(1000))
        assert len(valid_seq) <= MAX_SEQ_LENGTH
        
        # 超长序列
        long_seq = list(range(50000))
        assert len(long_seq) > MAX_SEQ_LENGTH
    
    def test_validate_data_integrity(self):
        """测试数据完整性验证"""
        def validate_example(example):
            required_keys = ["input_ids", "labels"]
            
            # 检查必需键
            if not all(k in example for k in required_keys):
                return False
            
            # 检查长度一致
            if len(example["input_ids"]) != len(example["labels"]):
                return False
            
            # 检查非空
            if len(example["input_ids"]) == 0:
                return False
            
            return True
        
        valid_example = {"input_ids": [1, 2, 3], "labels": [IGNORE_TOKEN_ID, 2, 3]}
        invalid_example = {"input_ids": [1, 2, 3], "labels": [IGNORE_TOKEN_ID, 2]}
        
        assert validate_example(valid_example) is True
        assert validate_example(invalid_example) is False


# ============================================================================
# 大规模数据处理测试
# ============================================================================

class TestLargeScaleProcessing:
    """大规模数据处理测试类"""
    
    @pytest.fixture
    def large_dataset(self) -> List[Dict[str, Any]]:
        """创建大规模数据集"""
        return [
            {
                "problem": f"Problem {i}",
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}" * 100}  # 较长的回答
                ],
                "answer": str(i)
            }
            for i in range(LARGE_DATASET_SIZE)
        ]
    
    def test_batch_processing(self, large_dataset):
        """测试批量处理"""
        num_batches = len(large_dataset) // BATCH_SIZE
        
        processed_count = 0
        for i in range(num_batches):
            batch = large_dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            processed_count += len(batch)
        
        assert processed_count == LARGE_DATASET_SIZE
    
    def test_streaming_processing(self, tmp_path, large_dataset):
        """测试流式处理"""
        file_path = tmp_path / "large_dataset.jsonl"
        
        # 写入数据
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in large_dataset:
                f.write(json.dumps(item) + '\n')
        
        # 流式读取
        processed_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                processed_count += 1
        
        assert processed_count == LARGE_DATASET_SIZE
    
    def test_memory_efficient_processing(self, large_dataset, memory_tracker):
        """测试内存高效处理"""
        with memory_tracker() as tracker:
            # 使用生成器处理
            def process_generator(dataset):
                for item in dataset:
                    # 模拟处理
                    yield {"processed": True, "id": item.get("problem")}
            
            processed_items = list(process_generator(large_dataset))
        
        assert len(processed_items) == LARGE_DATASET_SIZE
        # 内存使用应该在合理范围内
        assert tracker.peak < MAX_MEMORY_MB * 1024 * 1024


# ============================================================================
# 错误恢复测试
# ============================================================================

class TestErrorRecovery:
    """错误恢复测试类"""
    
    def test_skip_invalid_examples(self):
        """测试跳过无效样本"""
        dataset = [
            {"messages": [{"role": "user", "content": "Valid"}]},
            {"invalid": "data"},
            {"messages": []},
            {"messages": [{"role": "user", "content": "Another valid"}]}
        ]
        
        valid_examples = []
        for example in dataset:
            try:
                if "messages" in example and len(example["messages"]) > 0:
                    valid_examples.append(example)
            except Exception:
                continue
        
        assert len(valid_examples) == 2
    
    def test_handle_encoding_errors(self, tmp_path):
        """测试处理编码错误"""
        file_path = tmp_path / "data.jsonl"
        
        # 写入一些数据
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('{"text": "valid"}\n')
        
        # 追加可能有问题的数据（这里模拟）
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write('{"text": "also valid"}\n')
        
        # 读取时处理错误
        valid_data = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    valid_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        assert len(valid_data) == 2
    
    def test_resume_from_checkpoint(self, tmp_path):
        """测试从检查点恢复"""
        checkpoint_file = tmp_path / "checkpoint.json"
        data_file = tmp_path / "data.jsonl"
        
        # 创建数据
        with open(data_file, 'w', encoding='utf-8') as f:
            for i in range(100):
                f.write(json.dumps({"id": i}) + '\n')
        
        # 模拟处理到一半保存检查点
        checkpoint = {"last_processed_id": 49}
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f)
        
        # 从检查点恢复
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            saved_checkpoint = json.load(f)
        
        start_id = saved_checkpoint["last_processed_id"] + 1
        
        # 继续处理
        processed_count = 0
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data["id"] >= start_id:
                    processed_count += 1
        
        assert processed_count == 50


# ============================================================================
# 数据管道性能测试
# ============================================================================

class TestDataPipelinePerformance:
    """数据管道性能测试类"""
    
    def test_processing_throughput(self, performance_timer):
        """测试处理吞吐量"""
        dataset = [{"id": i, "text": f"Example {i}"} for i in range(PERFORMANCE_TEST_SIZE)]
        
        def simple_process(example):
            return {"processed_id": example["id"]}
        
        with performance_timer() as timer:
            results = [simple_process(ex) for ex in dataset]
        
        throughput = PERFORMANCE_TEST_SIZE / timer.elapsed
        
        # 应该能达到一定的吞吐量
        assert throughput > MIN_THROUGHPUT
    
    def test_io_performance(self, tmp_path, performance_timer):
        """测试IO性能"""
        file_path = tmp_path / "test_io.jsonl"
        
        # 写入性能
        with performance_timer() as write_timer:
            with open(file_path, 'w', encoding='utf-8') as f:
                for i in range(PERFORMANCE_TEST_SIZE):
                    f.write(json.dumps({"id": i}) + '\n')
        
        # 读取性能
        with performance_timer() as read_timer:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        
        assert write_timer.elapsed < MAX_IO_TIME
        assert read_timer.elapsed < MAX_IO_TIME
        assert len(data) == PERFORMANCE_TEST_SIZE


# ============================================================================
# 数据管道集成测试
# ============================================================================

class TestDataPipelineIntegration:
    """数据管道集成测试类"""
    
    @pytest.fixture
    def full_pipeline_setup(self, tmp_path):
        """设置完整管道"""
        # 创建原始数据
        raw_data_file = tmp_path / "raw_data.jsonl"
        processed_data_file = tmp_path / "processed_data.jsonl"
        
        raw_data = [
            {
                "problem": f"Calculate {i} * 2",
                "messages": [
                    {"role": "user", "content": f"Calculate {i} * 2"},
                    {"role": "assistant", "content": f"<think>{i} * 2 = {i*2}</think>\n\\boxed{{{i*2}}}"}
                ],
                "answer": str(i * 2)
            }
            for i in range(INTEGRATION_DATASET_SIZE)
        ]
        
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            for item in raw_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return {
            "raw_data_file": raw_data_file,
            "processed_data_file": processed_data_file,
            "tmp_path": tmp_path
        }
    
    def test_full_pipeline_execution(self, full_pipeline_setup):
        """测试完整管道执行"""
        from modules.openr1_data_handler import OpenR1Math220kDataHandler
        
        setup = full_pipeline_setup
        
        # 创建处理器
        mock_config = MagicMock()
        handler = OpenR1Math220kDataHandler(mock_config)
        
        # 配置模拟分词器
        def mock_apply_template(messages, **kwargs):
            return [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, 106] + list(range(100)) + [IM_END_TOKEN_ID]
        
        handler.tokenizer = MagicMock()
        handler.tokenizer.apply_chat_template = mock_apply_template
        handler.ignore_token_id = IGNORE_TOKEN_ID
        
        # 读取原始数据
        raw_data = []
        with open(setup["raw_data_file"], 'r', encoding='utf-8') as f:
            for line in f:
                raw_data.append(json.loads(line))
        
        # 处理数据
        processed_data = []
        for item in raw_data:
            formatted = handler.format_func(item)
            tokenized = handler.tokenize_func(formatted)
            processed_data.append({
                "input_ids": tokenized["input_ids"],
                "labels": tokenized["labels"],
                "original_problem": item["problem"]
            })
        
        # 保存处理后的数据
        with open(setup["processed_data_file"], 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item) + '\n')
        
        # 验证
        assert setup["processed_data_file"].exists()
        
        with open(setup["processed_data_file"], 'r', encoding='utf-8') as f:
            saved_data = [json.loads(line) for line in f]
        
        assert len(saved_data) == INTEGRATION_DATASET_SIZE
        for item in saved_data:
            assert "input_ids" in item
            assert "labels" in item
            assert len(item["input_ids"]) == len(item["labels"])


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

