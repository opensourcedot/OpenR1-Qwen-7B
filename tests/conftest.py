"""
Pytest配置文件和共享Fixtures

该文件包含所有测试共享的fixtures和配置。
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

# ============================================================================
# 常量定义 - 特殊Token映射
# ============================================================================

# Qwen2.5 特殊Token常量
SPECIAL_TOKENS = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|object_ref_start|>": 151646,
    "<|object_ref_end|>": 151647,
    "<|box_start|>": 151648,
    "<|box_end|>": 151649,
    "<|quad_start|>": 151650,
    "<|quad_end|>": 151651,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
    "<tool_call>": 151657,
    "</tool_call>": 151658,
    "<|fim_prefix|>": 151659,
    "<|fim_middle|>": 151660,
    "<|fim_suffix|>": 151661,
    "<|fim_pad|>": 151662,
    "<|repo_name|>": 151663,
    "<|file_sep|>": 151664,
    "<tool_response>": 151665,
    "</tool_response>": 151666,
    "<think>": 151667,
    "</think>": 151668,
}

# 必需的特殊Token
REQUIRED_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]

# Token ID 常量
PAD_TOKEN_ID = 151643
EOS_TOKEN_ID = 151643
BOS_TOKEN_ID = 151643
ASSISTANT_TOKEN_ID = 77091

# 模型配置常量
VOCAB_SIZE = 152064
MAX_SEQ_LENGTH = 32768
IGNORE_TOKEN_ID = -100


# ============================================================================
# 验证辅助函数
# ============================================================================

def validate_special_tokens(vocab: Dict[str, int]) -> Tuple[bool, List[str]]:
    """
    验证词汇表中是否包含所有必需的特殊Token
    
    Args:
        vocab: 词汇表字典
        
    Returns:
        (是否通过验证, 缺失的token列表)
    """
    missing = []
    for token in REQUIRED_SPECIAL_TOKENS:
        if token not in vocab:
            missing.append(token)
    return len(missing) == 0, missing


def validate_bpe_merges(merges: List[str]) -> Tuple[bool, List[str]]:
    """
    验证BPE合并规则的格式
    
    Args:
        merges: BPE合并规则列表
        
    Returns:
        (是否通过验证, 错误信息列表)
    """
    errors = []
    
    if not merges:
        errors.append("Merges list is empty")
        return False, errors
    
    # 检查版本头
    if not merges[0].startswith("#version:"):
        errors.append("Missing version header (should start with '#version:')")
    
    # 检查唯一性和格式
    seen_merges = set()
    for i, merge in enumerate(merges[1:], start=2):  # 跳过版本头
        # 检查格式：应该是两个由空格分隔的token
        parts = merge.split()
        if len(parts) != 2:
            errors.append(f"Line {i}: Invalid merge format '{merge}' (expected 'token1 token2')")
            continue
        
        # 检查唯一性
        if merge in seen_merges:
            errors.append(f"Line {i}: Duplicate merge rule '{merge}'")
        seen_merges.add(merge)
    
    return len(errors) == 0, errors


def validate_tokenizer_output_strict(
    output: Dict[str, Any],
    max_length: int = MAX_SEQ_LENGTH,
    vocab_size: int = VOCAB_SIZE
) -> Tuple[bool, List[str]]:
    """
    严格验证分词器输出
    
    Args:
        output: 分词器输出字典
        max_length: 最大序列长度
        vocab_size: 词汇表大小
        
    Returns:
        (是否通过验证, 错误信息列表)
    """
    errors = []
    
    # 检查必需键
    required_keys = ["input_ids", "labels"]
    for key in required_keys:
        if key not in output:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        return False, errors
    
    input_ids = output["input_ids"]
    labels = output["labels"]
    
    # 检查类型
    if not isinstance(input_ids, (list, tuple)):
        errors.append(f"input_ids should be list/tuple, got {type(input_ids)}")
    if not isinstance(labels, (list, tuple)):
        errors.append(f"labels should be list/tuple, got {type(labels)}")
    
    if errors:
        return False, errors
    
    # 检查长度一致
    if len(input_ids) != len(labels):
        errors.append(f"Length mismatch: input_ids={len(input_ids)}, labels={len(labels)}")
    
    # 检查非空
    if len(input_ids) == 0:
        errors.append("input_ids is empty")
    
    # 检查长度限制
    if len(input_ids) > max_length:
        errors.append(f"Sequence length {len(input_ids)} exceeds max_length {max_length}")
    
    # 检查token ID范围
    for i, token_id in enumerate(input_ids):
        if not (0 <= token_id < vocab_size):
            errors.append(f"input_ids[{i}]={token_id} out of range [0, {vocab_size})")
    
    # 检查labels中的特殊值
    has_valid_label = False
    for i, label in enumerate(labels):
        if label == IGNORE_TOKEN_ID:
            continue  # 忽略的位置
        has_valid_label = True
        if not (0 <= label < vocab_size):
            errors.append(f"labels[{i}]={label} out of range [0, {vocab_size}) or IGNORE_TOKEN_ID")
    
    if not has_valid_label:
        errors.append("No valid labels found (all are IGNORE_TOKEN_ID)")
    
    # 检查pad token位置的labels应该被忽略
    for i, (token_id, label) in enumerate(zip(input_ids, labels)):
        if token_id == PAD_TOKEN_ID and label != IGNORE_TOKEN_ID:
            errors.append(f"Position {i}: PAD token should have IGNORE_TOKEN_ID label, got {label}")
    
    return len(errors) == 0, errors


# ============================================================================
# 项目路径
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================================
# 基础配置Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """返回项目根目录"""
    return PROJECT_ROOT


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path) -> Path:
    """
    返回测试数据目录
    
    使用function作用域，每个测试使用独立目录
    """
    test_data = tmp_path / "test_data"
    test_data.mkdir(parents=True, exist_ok=True)
    return test_data


@pytest.fixture(scope="function")
def temp_dir(tmp_path: Path) -> Path:
    """
    创建并返回临时目录
    
    使用function作用域确保每个测试独立
    """
    return tmp_path


@pytest.fixture
def clean_temp_dir(tmp_path: Path) -> Path:
    """每个测试独立的临时目录"""
    return tmp_path


# ============================================================================
# Tokenizer相关Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def special_tokens() -> Dict[str, int]:
    """返回特殊Token映射常量"""
    return SPECIAL_TOKENS.copy()


@pytest.fixture(scope="session")
def sample_vocab() -> Dict[str, int]:
    """返回示例词汇表"""
    vocab = {
        "hello": 0,
        "world": 1,
        "Ġthe": 2,
        "Ġa": 3,
        "Ġis": 4,
        "Ġtest": 5,
        "Ġdata": 6,
        "Ġmodel": 7,
        "Ġtraining": 8,
        "Ġinference": 9,
        "Ġmath": 10,
        "Ġproblem": 11,
        "Ġsolution": 12,
        "Ġanswer": 13,
        "Ġquestion": 14,
    }
    # 添加更多基础token
    for i in range(100):
        vocab[f"token_{i}"] = 100 + i
    
    # 添加特殊token
    vocab.update(SPECIAL_TOKENS)
    
    return vocab


@pytest.fixture(scope="session")
def sample_merges() -> List[str]:
    """返回示例BPE合并规则（已验证格式）"""
    merges = [
        "#version: 0.2",
        "h e",
        "l l",
        "o w",
        "w o",
        "r l",
        "d s",
        "t h",
        "e s",
        "a n",
        "i s",
        "t e",
        "s t",
        "da ta",
        "mo de",
        "tr ai",
        "ni ng",
        "in fe",
        "re nc",
        "ma th",
        "pr ob",
        "le m",
        "so lu",
        "ti on",
        "an sw",
        "er s",
        "qu es",
    ]
    
    # 验证merges格式
    is_valid, errors = validate_bpe_merges(merges)
    assert is_valid, f"Invalid BPE merges: {errors}"
    
    return merges


@pytest.fixture
def vocab_file(clean_temp_dir: Path, sample_vocab: Dict[str, int]) -> Path:
    """创建临时词汇表文件"""
    vocab_path = clean_temp_dir / "vocab.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(sample_vocab, f, ensure_ascii=False, indent=2)
    return vocab_path


@pytest.fixture
def merges_file(clean_temp_dir: Path, sample_merges: List[str]) -> Path:
    """创建临时BPE合并文件（带末尾换行）"""
    merges_path = clean_temp_dir / "merges.txt"
    with open(merges_path, 'w', encoding='utf-8') as f:
        for merge in sample_merges:
            f.write(merge + '\n')
    return merges_path


@pytest.fixture
def merges_file_no_trailing_newline(clean_temp_dir: Path, sample_merges: List[str]) -> Path:
    """创建临时BPE合并文件（无末尾换行，用于边界测试）"""
    merges_path = clean_temp_dir / "merges_no_newline.txt"
    with open(merges_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sample_merges))  # 最后一行无换行
    return merges_path


@pytest.fixture
def vocab_file_with_bom(clean_temp_dir: Path, sample_vocab: Dict[str, int]) -> Path:
    """创建带BOM的词汇表文件（用于边界测试）"""
    vocab_path = clean_temp_dir / "vocab_bom.json"
    with open(vocab_path, 'w', encoding='utf-8-sig') as f:  # utf-8-sig 添加BOM
        json.dump(sample_vocab, f, ensure_ascii=False, indent=2)
    return vocab_path


# ============================================================================
# 数据处理相关Fixtures
# ============================================================================

@pytest.fixture
def sample_openr1_data() -> List[Dict[str, Any]]:
    """返回示例OpenR1数据"""
    return [
        {
            "messages": [
                {"role": "user", "content": "Solve the equation: 2x + 3 = 7"},
                {"role": "assistant", "content": "<think>Let me solve this step by step.\n2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2</think>\nThe answer is \\boxed{2}"}
            ],
            "answer": "2"
        },
        {
            "messages": [
                {"role": "user", "content": "What is 5 + 3?"},
                {"role": "assistant", "content": "<think>Simple addition.\n5 + 3 = 8</think>\nThe answer is \\boxed{8}"}
            ],
            "answer": "8"
        },
        {
            "messages": [
                {"role": "user", "content": "Calculate the area of a square with side length 4."},
                {"role": "assistant", "content": "<think>Area of square = side^2\n= 4^2\n= 16</think>\nThe answer is \\boxed{16}"}
            ],
            "answer": "16"
        },
    ]


@pytest.fixture
def sample_cot_data() -> List[Dict[str, Any]]:
    """返回示例CoT数据"""
    return [
        {
            "problem": "Solve: 3x - 5 = 10",
            "generations": [
                "<think>To solve 3x - 5 = 10:\n3x = 15\nx = 5</think>\n\\boxed{5}",
                "<think>Adding 5 to both sides:\n3x = 15\nDividing by 3:\nx = 5</think>\n\\boxed{5}"
            ],
            "answer": "5",
            "finish_reasons": ["stop", "stop"],
            "api_metadata": [
                {"prompt_tokens": 50, "completion_tokens": 100},
                {"prompt_tokens": 50, "completion_tokens": 120}
            ]
        },
        {
            "problem": "What is 12 * 8?",
            "generations": [
                "<think>12 * 8 = 96</think>\n\\boxed{96}",
                "<think>Computing: 12 * 8\n= 10*8 + 2*8\n= 80 + 16\n= 96</think>\n\\boxed{96}"
            ],
            "answer": "96",
            "finish_reasons": ["stop", "stop"],
            "api_metadata": [
                {"prompt_tokens": 30, "completion_tokens": 50},
                {"prompt_tokens": 30, "completion_tokens": 80}
            ]
        },
    ]


@pytest.fixture
def sample_jsonl_file(clean_temp_dir: Path, sample_cot_data: List[Dict]) -> Path:
    """创建示例JSONL文件（带末尾换行）"""
    file_path = clean_temp_dir / "sample_data.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in sample_cot_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return file_path


@pytest.fixture
def sample_jsonl_file_no_trailing_newline(clean_temp_dir: Path, sample_cot_data: List[Dict]) -> Path:
    """创建示例JSONL文件（无末尾换行，用于边界测试）"""
    file_path = clean_temp_dir / "sample_data_no_newline.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        lines = [json.dumps(item, ensure_ascii=False) for item in sample_cot_data]
        f.write('\n'.join(lines))  # 最后一行无换行
    return file_path


# ============================================================================
# 配置文件相关Fixtures
# ============================================================================

@pytest.fixture
def sample_finetune_config() -> Dict[str, Any]:
    """返回示例微调配置"""
    return {
        "seed": 42,
        "output_dir": "./output",
        "load_checkpoint": "/path/to/checkpoint",
        "load_ckpt_format": "safetensors",
        "auto_trans_ckpt": True,
        "run_mode": "finetune",
        "trainer": {
            "type": "CausalLanguageModelingTrainer",
            "model_name": "qwen2_5_7b"
        },
        "runner_config": {
            "epochs": 3,
            "batch_size": 1,
            "sink_mode": True,
            "sink_size": 1
        },
        "optimizer": {
            "type": "AdamW",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "learning_rate": 1e-6,
            "weight_decay": 0.01
        },
        "lr_schedule": {
            "type": "LinearWithWarmUpLR",
            "learning_rate": 5e-5,
            "warmup_ratio": 0.1,
            "total_steps": -1
        },
        "model": {
            "model_config": {
                "type": "LlamaConfig",
                "batch_size": 1,
                "seq_length": MAX_SEQ_LENGTH,
                "hidden_size": 3584,
                "num_layers": 28,
                "num_heads": 28,
                "n_kv_heads": 4,
                "vocab_size": VOCAB_SIZE,
            }
        }
    }


@pytest.fixture
def config_file(clean_temp_dir: Path, sample_finetune_config: Dict) -> Path:
    """创建临时配置文件"""
    config_path = clean_temp_dir / "config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample_finetune_config, f, default_flow_style=False)
    return config_path


@pytest.fixture
def sample_data_process_config() -> Dict[str, Any]:
    """返回示例数据处理配置"""
    return {
        "seed": 42,
        "output_dir": "./output",
        "train_dataset": {
            "input_columns": ["input_ids", "labels"],
            "batch_size": 2,
            "drop_remainder": True,
            "data_loader": {
                "type": "CommonDataLoader",
                "shuffle": True,
                "split": "train",
                "path": "parquet",
            }
        }
    }


# ============================================================================
# Mock对象Fixtures
# ============================================================================

@pytest.fixture
def mock_tokenizer():
    """
    返回Mock分词器
    
    apply_chat_template 契约:
    - 输入: List[Dict[str, str]] 格式的messages
    - 输出: List[int] 格式的token ids
    """
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    
    # apply_chat_template 返回 List[int]
    def mock_apply_chat_template(messages: List[Dict], **kwargs) -> List[int]:
        """
        模拟 apply_chat_template
        
        Args:
            messages: 对话消息列表
            **kwargs: 其他参数 (tokenize, add_generation_prompt, etc.)
            
        Returns:
            List[int]: token id 列表
        """
        # 基础token序列
        tokens = [SPECIAL_TOKENS["<|im_start|>"]]
        for msg in messages:
            if msg.get("role") == "assistant":
                tokens.append(ASSISTANT_TOKEN_ID)
            tokens.extend([100, 101, 102])
        tokens.append(SPECIAL_TOKENS["<|im_end|>"])
        return tokens
    
    tokenizer.apply_chat_template = mock_apply_chat_template
    tokenizer.vocab_size = VOCAB_SIZE
    tokenizer.pad_token_id = PAD_TOKEN_ID
    tokenizer.eos_token_id = EOS_TOKEN_ID
    tokenizer.bos_token_id = BOS_TOKEN_ID
    
    return tokenizer


@pytest.fixture
def mock_model():
    """
    返回Mock模型
    
    generate 契约:
    - 输入: input_ids (List[int] 或 List[List[int]])
    - 输出: List[List[int]] 格式的生成序列
    """
    model = MagicMock()
    
    def mock_generate(
        input_ids: List,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
        **kwargs
    ) -> List[List[int]]:
        """
        模拟 generate 方法
        
        Args:
            input_ids: 输入token ids
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            do_sample: 是否采样
            **kwargs: 其他参数
            
        Returns:
            List[List[int]]: 生成的token id序列
        """
        # 确保返回格式为 List[List[int]]
        if isinstance(input_ids[0], int):
            # 单个序列输入
            return [[*input_ids, 200, 201, 202, EOS_TOKEN_ID]]
        else:
            # 批量输入
            return [[*seq, 200, 201, 202, EOS_TOKEN_ID] for seq in input_ids]
    
    model.generate = mock_generate
    model.config.vocab_size = VOCAB_SIZE
    model.config.hidden_size = 3584
    model.config.num_layers = 28
    
    return model


@pytest.fixture
def mock_dataset():
    """返回Mock数据集"""
    dataset = MagicMock()
    dataset.__len__.return_value = 1000
    dataset.__iter__.return_value = iter([
        {"input_ids": [1, 2, 3], "labels": [IGNORE_TOKEN_ID, 2, 3]},
        {"input_ids": [4, 5, 6], "labels": [IGNORE_TOKEN_ID, 5, 6]},
    ])
    return dataset


@pytest.fixture
def mock_api_response():
    """返回Mock API响应"""
    return {
        "choices": [
            {
                "message": {
                    "content": "<think>Step by step solution</think>\n\\boxed{42}"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300
        }
    }


# ============================================================================
# 异步测试支持
# ============================================================================

@pytest.fixture
def event_loop():
    """创建事件循环"""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# 测试数据生成器
# ============================================================================

@pytest.fixture
def math_problem_generator():
    """数学问题生成器"""
    def generate(num_problems: int = 10) -> List[Dict[str, Any]]:
        problems = []
        for i in range(num_problems):
            a, b = i + 1, i + 2
            problems.append({
                "problem": f"What is {a} + {b}?",
                "answer": str(a + b),
                "uuid": f"problem_{i}"
            })
        return problems
    return generate


@pytest.fixture
def cot_response_generator():
    """CoT响应生成器"""
    def generate(problem: str, answer: str) -> str:
        return f"<think>Let me solve this problem step by step.\n{problem}\nThe answer is {answer}.</think>\n\\boxed{{{answer}}}"
    return generate


# ============================================================================
# 环境配置
# ============================================================================

@pytest.fixture(autouse=False)  # 不自动使用，需要时显式引用
def setup_test_environment(monkeypatch):
    """设置测试环境"""
    # 设置环境变量
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def suppress_warnings():
    """
    上下文管理器：在特定测试中抑制警告
    
    使用方式:
        def test_something(suppress_warnings):
            with suppress_warnings():
                # 可能产生警告的代码
    """
    import warnings
    from contextlib import contextmanager
    
    @contextmanager
    def _suppress():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            yield
    
    return _suppress


@pytest.fixture
def mock_mindspore():
    """Mock MindSpore模块"""
    with patch.dict('sys.modules', {
        'mindspore': MagicMock(),
        'mindspore.log': MagicMock(),
        'mindformers': MagicMock(),
        'mindformers.tools.register': MagicMock(),
        'mindformers.dataset.handler.base_handler': MagicMock(),
        'mindformers.models.tokenization_utils': MagicMock(),
        'mindformers.models.tokenization_utils_base': MagicMock(),
        'mindformers.tools.utils': MagicMock(),
    }):
        yield


# ============================================================================
# 性能测试支持
# ============================================================================

@pytest.fixture
def performance_timer():
    """性能计时器"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time
    
    return Timer


@pytest.fixture
def memory_tracker():
    """内存追踪器"""
    import tracemalloc
    
    class MemoryTracker:
        def __init__(self):
            self.peak = 0
            self.current = 0
        
        def __enter__(self):
            tracemalloc.start()
            return self
        
        def __exit__(self, *args):
            self.current, self.peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
    
    return MemoryTracker


# ============================================================================
# 验证Fixtures
# ============================================================================

@pytest.fixture
def validate_config():
    """配置验证函数"""
    def validator(config: Dict[str, Any], required_keys: List[str]) -> Tuple[bool, List[str]]:
        """
        验证配置是否包含必需键
        
        Returns:
            (是否通过, 缺失的键列表)
        """
        missing = [key for key in required_keys if key not in config]
        return len(missing) == 0, missing
    return validator


@pytest.fixture
def validate_tokenizer_output():
    """分词器输出验证函数（使用严格验证）"""
    return validate_tokenizer_output_strict


@pytest.fixture
def validate_special_tokens_fixture():
    """特殊Token验证函数"""
    return validate_special_tokens


@pytest.fixture
def validate_bpe_merges_fixture():
    """BPE Merges验证函数"""
    return validate_bpe_merges


# ============================================================================
# 边界值测试Fixtures
# ============================================================================

@pytest.fixture
def boundary_test_cases() -> Dict[str, Any]:
    """边界值测试用例"""
    return {
        "empty_input": [],
        "single_token": [100],
        "max_length_input": list(range(MAX_SEQ_LENGTH)),
        "over_max_length": list(range(MAX_SEQ_LENGTH + 100)),
        "all_pad_tokens": [PAD_TOKEN_ID] * 100,
        "all_ignore_labels": [IGNORE_TOKEN_ID] * 100,
        "min_token_id": [0],
        "max_token_id": [VOCAB_SIZE - 1],
        "out_of_range_token": [VOCAB_SIZE],  # 无效
        "negative_token": [-1],  # 无效
    }


# ============================================================================
# 文件读取辅助Fixtures
# ============================================================================

@pytest.fixture
def read_file_safely():
    """安全读取文件，处理BOM和编码"""
    def reader(file_path: Path, encoding: str = 'utf-8') -> str:
        """
        安全读取文件内容
        
        Args:
            file_path: 文件路径
            encoding: 编码格式
            
        Returns:
            文件内容（已移除BOM）
        """
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        # 移除BOM
        if content.startswith('\ufeff'):
            content = content[1:]
        
        return content
    
    return reader


@pytest.fixture
def read_jsonl_safely():
    """安全读取JSONL文件"""
    def reader(file_path: Path, encoding: str = 'utf-8') -> List[Dict]:
        """
        安全读取JSONL文件
        
        处理:
        - BOM
        - 末行无换行
        - 空行
        
        Args:
            file_path: 文件路径
            encoding: 编码格式
            
        Returns:
            解析后的数据列表
        """
        data = []
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        # 移除BOM
        if content.startswith('\ufeff'):
            content = content[1:]
        
        for line in content.splitlines():
            line = line.strip()
            if line:  # 跳过空行
                data.append(json.loads(line))
        
        return data
    
    return reader

