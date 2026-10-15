import torch

from configs.model.model import build_model
from scripts.validate_comparison import evaluation_config, normalize
from src.model.no_k_variant import NoKVariantImpl
from src.model.stability import RMSNormImpl


def test_scaled_architecture_and_exact_ranged_initialization():
    class Tokenizer:
        vocab_size = 8192

    model, entries = build_model(Tokenizer())
    assert sum(p.numel() for p in model.parameters()) == 29579776
    mixers = [m for m in model.modules() if isinstance(m, NoKVariantImpl)]
    assert len(mixers) == len(entries) == 9
    for m in mixers:
        assert m.features == 512 and m.num_heads == 4
        assert type(m.value_norm) is RMSNormImpl
        assert m.normalization_scope == "global"
        u = torch.arange(512) / 511
        torch.testing.assert_close(m.value_decay_gate.bias.sigmoid(), 0.90 + 0.095 * u)
        torch.testing.assert_close(m.write_gate.bias.sigmoid(), 0.20 - 0.18 * u)
        assert torch.count_nonzero(m.value_decay_gate.weight) == 0
        assert torch.count_nonzero(m.write_gate.weight) == 0
    assert all(
        e.key_sum.numel() == 0 and e.value_sum.shape == (0, 1, 512) for e in entries
    )


def test_pipeline_evaluation_stage_and_normalization():
    pre_data, pre_schedule, pre_checkpoint = evaluation_config(False)
    qa_data, qa_schedule, qa_checkpoint = evaluation_config(True)
    assert pre_data.context_length == qa_data.context_length == 512
    assert pre_schedule.total_steps == qa_schedule.total_steps == 6500
    assert pre_data.response_prefix == qa_data.response_prefix == "\nAnswer: "
    assert "pretraining/tide-30m-v1" in str(pre_checkpoint.directory)
    assert "sft/tide-30m-qa-v1" in str(qa_checkpoint.directory)
    assert normalize("The, BLUE! cat.") == "blue cat"
