from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonConfig:
    budget_seconds: int = 1800
    reserve_seconds: int = 180
    story_batches_per_qa: int = 3
    cpu_threads: int = 4
    benchmark_warmup: int = 10
    benchmark_steps: int = 50
    qa_generation_examples: int = 256
    qa_max_new_tokens: int = 64

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.reserve_seconds >= self.budget_seconds:
            raise ValueError("reserve must be less than budget")
