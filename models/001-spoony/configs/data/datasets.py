"""Declare selected datasets here. Importing does not read or download data."""

from src.data.pipeline import DataSetPipe

# Add DataSet(manifest=DocumentManifest(...), source=...) entries here.
# Each DataSet represents one split; no source is selected yet.
dataset_pipe = DataSetPipe(datasets=[])
