"""Processed datasets and fixed cell ordering for fine-tuning and evaluation."""

from dataclasses import dataclass
import math
import os

from torch.utils.data import DataLoader, Dataset, Sampler

from .common import PeakDictionary, gene_meta, load_embeddings, load_pickle, peak_meta
from .gex_dataset import MUSSGEXDataset
from .unseen_seq import unseen_masks


@dataclass
class DataConfig:
    data: str
    sequences_bed: str
    unseen_folds: str = "fold3,fold4"
    max_cell_peaks_num: int = 10000
    seed: int = 1
    peak_embeddings: str = ""
    gene_embeddings: str = ""


class MUSSData:
    def __init__(self, config):
        self.cfg = config
        overview = load_pickle(os.path.join(config.data, "chunks", "overview.pkl"))
        self.peak_meta, self.gene_meta = peak_meta(overview), gene_meta(overview)
        self.n_peaks = self.peak_meta["n_peaks"]
        self.num_genes = self.gene_meta["n_genes"]
        self.gene_ids = self.gene_meta["gene_ids"]
        self.dictionary = PeakDictionary(self.n_peaks)
        self.unseen_peak_mask, self.unseen_gene_mask = unseen_masks(
            config.sequences_bed, set(config.unseen_folds.split(",")),
            self.peak_meta["Chromosome"], self.peak_meta["Start"], self.peak_meta["End"],
            self.gene_meta["Chromosome"], self.gene_meta["tss"],
        )
        if self.unseen_gene_mask.all():
            raise ValueError("The gene panel contains no training genes")
        peak_path = config.peak_embeddings or os.path.join(config.data, "peak_embeddings_alphagenome.h5")
        shared = os.path.join(os.path.dirname(config.data), "shared", "peak_embeddings_alphagenome.h5")
        if not config.peak_embeddings and not os.path.exists(peak_path) and os.path.exists(shared):
            peak_path = shared
        self.peak_emb_path = peak_path
        self.gene_emb_path = config.gene_embeddings or os.path.join(config.data, "gene_embeddings_alphagenome.h5")
        self.peak_emb_matrix = self.gene_emb_matrix = None
        self.datasets = {}

    def load_embeddings(self):
        if self.peak_emb_matrix is None:
            self.peak_emb_matrix = load_embeddings(self.peak_emb_path, "peak_embeddings")
            self.gene_emb_matrix = load_embeddings(self.gene_emb_path, "gene_embeddings")
            if self.peak_emb_matrix.shape[0] != self.n_peaks or self.gene_emb_matrix.shape[0] != self.num_genes:
                raise ValueError("Embedding rows do not match the peak and gene vocabularies")

    def dataset(self, split):
        if split not in self.datasets:
            self.datasets[split] = MUSSGEXDataset(
                self.cfg.data, split, pad_idx=0, n_special=self.dictionary.n_special,
                max_cell_peaks_num=self.cfg.max_cell_peaks_num, seed=self.cfg.seed,
                unseen_peak_mask=self.unseen_peak_mask, unseen_gene_mask=self.unseen_gene_mask,
            )
        return self.datasets[split]

    def loader(self, split):
        dataset = self.dataset(split)
        return DataLoader(dataset, batch_size=1, sampler=dataset.ordered_indices().tolist(),
                          collate_fn=dataset.collater, num_workers=0)


class CellSampler(Sampler):
    """Shard fixed cell order; reproduce the released runs' final-group normalization."""

    def __init__(self, dataset, world_size=1, rank=0, accumulation=1):
        if world_size < 1 or accumulation < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid world size, rank or accumulation")
        self.order = dataset.ordered_indices()
        self.world_size, self.rank, self.accumulation = world_size, rank, accumulation

    def __len__(self):
        return math.ceil(len(self.order) / self.world_size)

    def __iter__(self):
        n, world, accum = len(self.order), self.world_size, self.accumulation
        for local in range(len(self)):
            start = (local // accum) * accum
            count = min(accum, len(self) - start)
            # A rank ending in padding contributes zero to the original sample-size sum.
            active_ranks = min(world, n - (start + count - 1) * world)
            denominator = count * active_ranks
            position = local * world + self.rank
            dummy = position >= n
            index = int(self.order[0 if dummy else position])
            yield index, dummy, world / denominator


class CellView(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        index, dummy, grad_factor = key
        return self.dataset[index], dummy, grad_factor

    def collate(self, samples):
        (sample, dummy, factor), = samples
        batch = self.dataset.collater([sample])
        batch.update(is_dummy=dummy, grad_factor=factor)
        return batch
