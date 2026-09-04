"""Unit tests for the Hierarchical Attention Network (spec §7).

These tests use random embeddings, not SciBERT: the network mechanics
(padded collation, attention normalisation, unpadding) are what is under
test, and they must run fast and offline.
"""

import numpy as np
import pytest
import torch

from src.models.hierarchical_attention import (
    build_han_network,
    collate_documents,
)

pytest.importorskip("torch", reason="PyTorch not installed")


def _doc(n_sections, sentences_per_section, dim=16):
    """Build a document: n_sections sections of fixed sentence counts."""
    return [
        [np.random.rand(dim) for _ in range(sentences_per_section)]
        for _ in range(n_sections)
    ]


class TestCollateDocuments:
    def test_pads_to_uniform_shape(self):
        docs = [
            _doc(2, 3, dim=8),
            _doc(1, 1, dim=8),
        ]
        batch = collate_documents(docs)
        assert batch.embeddings.shape == (2, 2, 3, 8)
        assert batch.sentence_mask.shape == (2, 2, 3)
        assert batch.section_mask.shape == (2, 2)

    def test_masks_mark_real_entries(self):
        docs = [_doc(2, 2, dim=4), _doc(1, 1, dim=4)]
        batch = collate_documents(docs)
        assert batch.section_mask[0].all()
        assert bool(batch.section_mask[1, 0])
        assert not bool(batch.section_mask[1, 1])
        assert int(batch.sentence_mask[0, 0].sum()) == 2
        assert int(batch.sentence_mask[1, 0].sum()) == 1
        assert not batch.sentence_mask[1, 1].any()

    def test_empty_document_gets_placeholder_section(self):
        docs = [[]]
        batch = collate_documents(docs)
        assert batch.embeddings.shape[0] == 1
        assert not batch.section_mask.any()


class TestHANForward:
    def _network(self, dim=8, n_classes=3):
        torch.manual_seed(0)
        return build_han_network(
            input_dim=dim, sentence_hidden=8, section_hidden=8, n_classes=n_classes, dropout=0.0
        )

    def test_logits_shape_and_softmax(self):
        net = self._network(dim=8, n_classes=3)
        net.eval()
        docs = [_doc(2, 3, dim=8), _doc(3, 2, dim=8)]
        batch = collate_documents(docs)
        with torch.no_grad():
            logits = net(batch)
            probs = torch.softmax(logits, dim=-1)
        assert logits.shape == (2, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_attention_weights_are_real_and_normalised(self):
        """The headline property: weights come from the network and sum to 1."""
        net = self._network(dim=8, n_classes=3)
        net.eval()
        docs = [_doc(2, 4, dim=8), _doc(1, 2, dim=8)]
        batch = collate_documents(docs)
        with torch.no_grad():
            _, sentence_w, section_w = net(batch, return_attention=True)

        # Section attention: one weight per real section, summing to 1.
        for doc_index, doc in enumerate(docs):
            weights = section_w[doc_index]
            assert len(weights) == len(doc)
            assert abs(sum(weights) - 1.0) < 1e-5
            assert all(w > 0 for w in weights)

        # Sentence attention: one weight per real sentence per section.
        for doc_index, doc in enumerate(docs):
            for s, section in enumerate(doc):
                weights = sentence_w[doc_index][s]
                assert len(weights) == len(section)
                assert abs(sum(weights) - 1.0) < 1e-5

    def test_attention_changes_with_input(self):
        """Different documents produce different weights (not a constant)."""
        net = self._network(dim=8, n_classes=2)
        net.eval()
        torch.manual_seed(1)
        docs = [_doc(2, 3, dim=8), _doc(2, 3, dim=8)]
        # Make the two documents very different.
        for t in range(3):
            docs[1][0][t] = docs[1][0][t] * -3.0
        batch = collate_documents(docs)
        with torch.no_grad():
            _, _, section_w = net(batch, return_attention=True)
        assert section_w[0] != pytest.approx(section_w[1])

    def test_training_reduces_loss_on_separable_data(self):
        """A sanity check that gradients flow and the loss decreases."""
        torch.manual_seed(0)
        np.random.seed(0)
        net = build_han_network(
            input_dim=8, sentence_hidden=8, section_hidden=8, n_classes=2, dropout=0.0
        )
        class_a = [np.full((2, 8), 1.0) for _ in range(1)]
        class_b = [np.full((2, 8), -1.0) for _ in range(1)]
        docs = [
            [[np.full(8, 1.0), np.full(8, 1.0)]],
            [[np.full(8, -1.0), np.full(8, -1.0)]],
        ]
        targets = torch.tensor([0, 1])
        batch = collate_documents(docs)
        optimiser = torch.optim.AdamW(net.parameters(), lr=0.05)
        loss_fn = torch.nn.CrossEntropyLoss()

        net.train()
        first_loss = None
        for _ in range(15):
            optimiser.zero_grad()
            logits = net(batch)
            loss = loss_fn(logits, targets)
            if first_loss is None:
                first_loss = float(loss)
            loss.backward()
            optimiser.step()
        assert float(loss) < first_loss