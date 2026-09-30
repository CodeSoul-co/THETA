"""Data loading utilities; split validation needs no numerical dependencies."""

__all__ = ["ETMDataset"]

def __getattr__(name):
    if name == "ETMDataset":
        from .dataloader import ETMDataset
        return ETMDataset
    raise AttributeError(name)
