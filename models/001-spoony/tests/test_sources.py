from unittest.mock import patch

from src.data.sources import HuggingFaceSource


def test_source_is_lazy_and_can_be_read_again() -> None:
    source = HuggingFaceSource("test/dataset", "fixed-revision", "external-split", True)
    rows = [{"story": "A little fox."}]
    with patch("src.data.sources.load_dataset", return_value=rows) as load:
        records = source.read()
        load.assert_not_called()
        assert list(records) == rows
        assert list(source.read()) == rows
        assert load.call_count == 2
        load.assert_called_with(
            "test/dataset",
            revision="fixed-revision",
            split="external-split",
            streaming=True,
        )
