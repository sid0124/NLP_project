"""Regenerate ``openalex_sample.jsonl``, the offline edge-case fixture.

Scope: this corpus exercises **parsing, validation, and deduplication** against
realistically messy payloads. It is deliberately small and is *not* used to
train anything — bulk corpora for split/training mechanics come from the seeded
synthetic generator in ``tests/conftest.py``.

The fixture is committed, so this script is not needed to run the tests. It
exists because the corpus encodes properties that must hold *exactly*:

* every abstract is **mutually distinct** except where a duplicate is intended,
  so deduplication removes only what a test asserts it should;
* one near-duplicate pair sits **comfortably above** the Jaccard threshold and
  one **comfortably below**, so the test is not boundary-brittle;
* one record per parse/validation failure mode.

Hand-editing JSONL would silently break these. ``verify()`` re-checks them on
every run.

Abstract inverted indexes are built here in the **forward** direction
(text -> index). The parser inverts them (index -> text). The two directions are
implemented independently, so the round-trip test is real, not tautological.

Usage:
    python tests/fixtures/generate_sample.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.dedup import jaccard, shingle_hashes  # noqa: E402
from src.preprocessing.text import normalize_for_matching  # noqa: E402

OUTPUT = Path(__file__).with_name("openalex_sample.jsonl")

#: Must match ``dedup.near_duplicate`` in ``configs/dataset.yaml``.
SHINGLE_SIZE = 5
JACCARD_THRESHOLD = 0.85
#: How far above the threshold an intended near-duplicate must score, so that a
#: small future edit to the wording cannot silently push it below.
MARGIN = 0.05

# --------------------------------------------------------------------------
# Payload builders mirroring the OpenAlex Works schema
# --------------------------------------------------------------------------


def inverted(text: str) -> dict[str, list[int]]:
    """Build an OpenAlex-style inverted index from plain text."""
    index: dict[str, list[int]] = {}
    for position, token in enumerate(text.split()):
        index.setdefault(token, []).append(position)
    return index


def topic(
    topic_id: str,
    name: str,
    subfield: str,
    subfield_id: str,
    *,
    field: str = "Computer Science",
    domain: str = "Physical Sciences",
    score: float = 0.99,
) -> dict[str, Any]:
    """Build a topic object carrying the full topic/subfield/field/domain chain."""
    return {
        "id": f"https://openalex.org/{topic_id}",
        "display_name": name,
        "score": score,
        "subfield": {
            "id": f"https://openalex.org/subfields/{subfield_id}",
            "display_name": subfield,
        },
        "field": {"id": "https://openalex.org/fields/17", "display_name": field},
        "domain": {"id": "https://openalex.org/domains/3", "display_name": domain},
    }


def author(
    name: str,
    author_id: str | None = None,
    institution: str | None = None,
    *,
    position: str = "first",
    orcid: str | None = None,
) -> dict[str, Any]:
    """Build one entry of an OpenAlex ``authorships`` array."""
    return {
        "author": {
            "id": f"https://openalex.org/{author_id}" if author_id else None,
            "display_name": name,
            "orcid": orcid,
        },
        "author_position": position,
        "raw_author_name": name,
        "institutions": (
            [{"id": "https://openalex.org/I1", "display_name": institution}] if institution else []
        ),
        "countries": [],
        "is_corresponding": position == "first",
        "raw_affiliation_strings": [],
    }


def work(
    work_id: str | None,
    title: str | None,
    abstract: str | None,
    primary: dict[str, Any] | None,
    topics: list[dict[str, Any]],
    *,
    doi: str | None = None,
    year: int | None = 2023,
    language: str | None = "en",
    work_type: str = "article",
    venue: str | None = "Journal of Applied Testing",
    authors: list[dict[str, Any]] | None = None,
    references: list[str] | None = None,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    """Build a single OpenAlex work, matching the configured ``select`` fields."""
    return {
        "id": f"https://openalex.org/{work_id}" if work_id else None,
        "doi": f"https://doi.org/{doi}" if doi else None,
        "title": title,
        "publication_year": year,
        "publication_date": f"{year}-06-15" if year else None,
        "language": language,
        "type": work_type,
        "primary_location": (
            {"source": {"id": "https://openalex.org/S1", "display_name": venue}} if venue else None
        ),
        "authorships": (
            authors if authors is not None else [author("Smith, John", "A1", "Example University")]
        ),
        "abstract_inverted_index": inverted(abstract) if abstract else None,
        "primary_topic": primary,
        "topics": topics,
        "keywords": [
            {
                "id": "https://openalex.org/keywords/" + keyword.lower().replace(" ", "-"),
                "display_name": keyword,
                "score": 0.8,
            }
            for keyword in (keywords or [])
        ],
        "referenced_works": [f"https://openalex.org/{ref}" for ref in (references or [])],
    }


# --------------------------------------------------------------------------
# Topics. Subfield ids match the CS targets in configs/dataset.yaml.
# --------------------------------------------------------------------------
AI = topic("T10181", "Natural Language Processing Techniques", "Artificial Intelligence", "1702")
AI_2 = topic("T10182", "Topic Modeling", "Artificial Intelligence", "1702", score=0.62)
CV = topic(
    "T10201", "Image Segmentation Methods", "Computer Vision and Pattern Recognition", "1707"
)
CV_2 = topic(
    "T10202", "Object Detection", "Computer Vision and Pattern Recognition", "1707", score=0.55
)
NET = topic("T10301", "Wireless Sensor Networks", "Computer Networks and Communications", "1705")
HCI = topic("T10401", "User Interface Design", "Human-Computer Interaction", "1709")
SIG = topic("T10501", "Adaptive Filtering", "Signal Processing", "1711")

# --------------------------------------------------------------------------
# Abstracts. Every one is on a distinct topic with distinct vocabulary, so
# pairwise shingle overlap is near zero unless a duplicate is intended.
# All are >= validation.min_abstract_chars (200) except SHORT.
# --------------------------------------------------------------------------

# The near-duplicate base is deliberately long: a longer text keeps Jaccard high
# under a realistic multi-word edit, so the pair clears the threshold with room
# to spare instead of landing on it.
NEAR_BASE = (
    "This paper introduces a hierarchical vision transformer for volumetric medical image "
    "segmentation of glioma subregions in multiparametric magnetic resonance scans. Shifted window "
    "attention is paired with a lightweight convolutional decoder so that fine boundary detail "
    "survives upsampling, and a boundary aware loss term penalises leakage across adjacent tissue "
    "classes. We further describe a patch sampling schedule that keeps class balance stable when "
    "tumour cores occupy only a small fraction of the imaged volume. An ablation isolates the "
    "contribution of the window shifting mechanism from that of the decoder skip connections, and "
    "a "
    "second ablation varies the depth of the encoder hierarchy to show where additional capacity "
    "stops paying for itself. We also report inter rater agreement on a held out subset in order "
    "to "
    "place the automatic scores in the context of ordinary human labelling variance. Across two "
    "public benchmarks the method attains a higher mean Dice coefficient than competitive fully "
    "convolutional architectures while remaining efficient enough for routine inference on a "
    "single "
    "accelerator."
)
# Realistic published-version edit: the closing clause is rewritten. The base
# text is deliberately long, because the pair is compared on title *and*
# abstract: a short abstract lets the one-word title difference below
# ("Transformer" -> "Transformers") drag the pair down toward the 0.85 boundary.
NEAR_VARIANT = NEAR_BASE.replace(
    "while remaining efficient enough for routine inference on a single accelerator.",
    "while remaining efficient enough for routine inference on one commodity accelerator.",
)

# Same research area as the pair above, but independently written: this is the
# negative control, expected well below the threshold.
NEAR_NEGATIVE = (
    "We evaluate uncertainty estimation for tumour delineation networks under distribution shift "
    "between scanner vendors. Deep ensembles and test time augmentation are compared against a "
    "single deterministic model using calibration error and selective prediction curves. Results "
    "indicate that ensembling improves calibration substantially but that neither approach removes "
    "the systematic contrast bias observed when transferring between acquisition protocols."
)

A_NLP = (
    "We present a transformer based approach for multilingual named entity recognition that "
    "leverages contextual subword embeddings together with a conditional random field decoder. The "
    "encoder is pretrained on a large scientific corpus and fine tuned on seven downstream "
    "benchmarks spanning newswire and biomedical text. Experiments show consistent gains in macro "
    "averaged F1 over strong recurrent baselines while requiring far fewer labelled sentences."
)
A_QA = (
    "This work investigates instruction tuning for scholarly question answering over full text "
    "articles. We assemble a corpus of expert written question answer pairs and analyse how "
    "retrieval granularity affects answer faithfulness. Ablations isolate the contribution of "
    "passage reranking and demonstrate that requiring every claim to cite a retrieved span sharply "
    "reduces unsupported generations without narrowing answer coverage."
)
A_SUMM = (
    "Abstractive summarisation of clinical trial reports demands strict factual consistency. We "
    "introduce an entailment guided decoding constraint that rejects candidate sentences "
    "unsupported by the source document, and pair it with a numeric consistency checker for dosage "
    "and cohort size. Human assessment by two clinicians finds markedly fewer factual errors at "
    "comparable readability to an unconstrained decoder."
)
A_RS = (
    "We revisit self supervised pretraining for dense prediction in multispectral satellite "
    "imagery. A masked autoencoding objective is adapted by reconstructing band specific "
    "statistics rather than raw pixel intensities, which avoids collapse when spectral bands "
    "differ "
    "in dynamic range. Transfer experiments on land cover mapping and change detection show that "
    "spectrally aware pretext tasks converge faster than supervised initialisation."
)
A_POSE = (
    "Occlusion remains the dominant failure mode in monocular human pose estimation for crowded "
    "scenes. We propose a part visibility head that predicts per joint occlusion likelihood and "
    "reweights the keypoint regression loss accordingly. Combined with synthetic occlusion "
    "augmentation drawn from segmented object masks, the approach reduces mean per joint position "
    "error on pedestrian benchmarks without additional inference cost."
)
A_ROUTING = (
    "We study energy aware routing for large scale wireless sensor deployments used in remote "
    "environmental monitoring. A distributed clustering procedure rotates cluster heads according "
    "to residual battery capacity and estimated link quality, avoiding the hotspot depletion that "
    "static hierarchies suffer. Simulation across varied node densities and duty cycles shows "
    "extended network lifetime and lower packet loss than established schemes."
)
A_CONGESTION = (
    "Congestion control for low latency datacentre fabrics must react within a few round trips. We "
    "propose a receiver driven credit scheme that paces senders using queue occupancy signals "
    "exported by programmable switches. A testbed evaluation with mixed message sizes shows lower "
    "tail completion times than delay based alternatives while sustaining high aggregate link "
    "utilisation under repeated incast."
)
A_P4 = (
    "Stateful packet processing in programmable data planes is constrained by limited register "
    "memory and per stage dependencies. We describe a compiler pass that folds sketch based "
    "counters into shared register arrays and proves the resulting layout satisfies pipeline stage "
    "limits. Evaluation on production trace replays indicates accurate heavy hitter detection at a "
    "third of the memory footprint."
)
A_TRIAGE = (
    "Through a mixed methods study with forty two participants we examine how researchers browse "
    "and triage large collections of scholarly literature. Screen recordings and think aloud "
    "protocols surface recurring navigation strategies together with points of friction in current "
    "discovery interfaces. We derive concrete design implications for tools intended to support "
    "exploratory search over academic corpora."
)
A_DIARY = (
    "Sensemaking during literature review is fragmented across reading, note taking, and reference "
    "management applications. We report a four week diary study of graduate researchers and "
    "characterise the artefacts they build to externalise emerging structure. Our findings "
    "motivate "
    "interfaces that treat annotation and organisation as one continuous activity rather than as "
    "separate application modes."
)
A_ACCESS = (
    "Screen reader users encounter systematic barriers when navigating mathematical notation on "
    "the "
    "web. We audit two hundred technical pages and catalogue the markup patterns that defeat "
    "assistive technology, then co design an alternative navigation model with eight blind "
    "participants. The resulting prototype reduces task completion time for equation exploration "
    "and was preferred by every participant."
)
A_FILTER = (
    "A low complexity adaptive filtering scheme is proposed for real time noise suppression in "
    "embedded audio pipelines operating under strict latency budgets. The method couples a "
    "normalised least mean squares update with a spectral gating stage tuned by online signal to "
    "noise estimation. Measurements on constrained microcontrollers confirm improved perceptual "
    "quality at a fraction of the arithmetic cost."
)
A_RADAR = (
    "Automotive radar suffers mutual interference as sensor density grows in dense traffic. We "
    "formulate interference mitigation as sparse recovery over the range Doppler map and solve it "
    "with an iterative thresholding scheme that requires no coordination between vehicles. Road "
    "trials indicate restored detection sensitivity for weak targets under simultaneous "
    "transmission from several nearby units."
)
A_COMPILER = (
    "Register allocation for vector extensions must reconcile variable length vectors with fixed "
    "physical register files. We present a live range splitting heuristic that accounts for vector "
    "length agnostic instructions and quantifies spill cost in terms of achieved memory bandwidth "
    "rather than instruction count. Benchmarks across kernels show reduced spill traffic and "
    "modest "
    "throughput improvements."
)
A_TESTING = (
    "Flaky tests erode trust in continuous integration and consume substantial engineering effort. "
    "We mine two years of build history from several large repositories and classify flakiness "
    "causes by their observable failure signatures. A lightweight rerun policy informed by these "
    "signatures identifies the majority of genuinely order dependent tests at a small fraction of "
    "the cost of exhaustive reruns."
)
A_BEAMFORM = (
    "Robust adaptive beamforming degrades sharply when the assumed steering vector is mismatched "
    "against the true array manifold. We derive a worst case optimisation over an uncertainty "
    "ellipsoid whose extent is estimated directly from received snapshots rather than assumed a "
    "priori. Anechoic chamber measurements with an eight element array confirm restored array gain "
    "and suppressed sidelobes under calibration error."
)
A_ACTIVE = (
    "Annotation budgets dominate the cost of building supervised classifiers for specialised "
    "domains. We compare acquisition functions for pool based active learning when the unlabelled "
    "pool is itself biased, and show that uncertainty sampling amplifies that bias unless coupled "
    "with a density term. A stratified acquisition variant recovers most of the accuracy "
    "attainable "
    "under fully random labelling at half the annotation volume."
)
SHORT = "We propose a method. It works well."

# --------------------------------------------------------------------------
# The corpus. Comments state what each record exercises.
# --------------------------------------------------------------------------
RECORDS: list[dict[str, Any]] = [
    # -- clean, labelable records: three per class, four classes ------------
    # Non-ASCII author names are a Windows cp1252 regression guard.
    work(
        "W1001",
        "Multilingual Named Entity Recognition with Transformers",
        A_NLP,
        AI,
        [AI, AI_2],
        doi="10.1000/ai.1001",
        references=["W9001", "W9002"],
        keywords=["Transformer", "Named entity recognition"],
        authors=[
            author("Jakubův, Jan", "A10", "Czech Technical University in Prague"),
            author("Müller, Sophie", "A11", "ETH Zürich", position="middle"),
            author("Zhang, Wei", "A12", "Tsinghua University", position="last"),
        ],
    ),
    work(
        "W1002",
        "Instruction Tuning for Scholarly Question Answering",
        A_QA,
        AI,
        [AI],
        doi="10.1000/ai.1002",
        year=2022,
        references=["W9001"],
        keywords=["Question answering"],
    ),
    work(
        "W1003",
        "Entailment Guided Decoding for Clinical Trial Summarisation",
        A_SUMM,
        AI,
        [AI, AI_2],
        doi="10.1000/ai.1003",
        year=2024,
    ),
    work(
        "W1004",
        "Hierarchical Vision Transformer for Glioma Segmentation",
        NEAR_BASE,
        CV,
        [CV, CV_2],
        doi="10.1000/cv.1004",
        references=["W9003"],
        keywords=["Image segmentation", "Medical imaging"],
    ),
    work(
        "W1005",
        "Spectrally Aware Masked Autoencoding for Satellite Imagery",
        A_RS,
        CV,
        [CV],
        doi="10.1000/cv.1005",
        year=2021,
    ),
    work(
        "W1006",
        "Part Visibility Modelling for Monocular Pose Estimation",
        A_POSE,
        CV,
        [CV, CV_2],
        doi="10.1000/cv.1006",
        year=2024,
    ),
    work(
        "W1007",
        "Energy Aware Routing for Wireless Sensor Deployments",
        A_ROUTING,
        NET,
        [NET],
        doi="10.1000/net.1007",
        references=["W9004", "W9005", "W9006"],
    ),
    work(
        "W1008",
        "Receiver Driven Congestion Control for Datacentre Fabrics",
        A_CONGESTION,
        NET,
        [NET],
        doi="10.1000/net.1008",
        year=2020,
    ),
    work(
        "W1009",
        "Folding Sketch Counters into Programmable Data Planes",
        A_P4,
        NET,
        [NET],
        doi="10.1000/net.1009",
        year=2025,
    ),
    work(
        "W1010",
        "How Researchers Triage Scholarly Literature",
        A_TRIAGE,
        HCI,
        [HCI],
        doi="10.1000/hci.1010",
    ),
    work(
        "W1011",
        "A Diary Study of Literature Review Sensemaking",
        A_DIARY,
        HCI,
        [HCI],
        doi="10.1000/hci.1011",
        year=2022,
    ),
    work(
        "W1012",
        "Screen Reader Navigation of Mathematical Notation",
        A_ACCESS,
        HCI,
        [HCI],
        doi="10.1000/hci.1012",
        year=2025,
    ),
    # -- INTENDED near-duplicate of W1004 -----------------------------------
    # Distinct id and DOI, abstract differing only in its closing clause, as a
    # published version differs from its preprint. Must be removed before
    # splitting, or W1004's content leaks across the train/test boundary.
    work(
        "W1013",
        "Hierarchical Vision Transformers for Glioma Segmentation",
        NEAR_VARIANT,
        CV,
        [CV],
        doi="10.1000/cv.1013",
        keywords=["Image segmentation"],
    ),
    # -- NEGATIVE control: same research area, independently written --------
    # Expected well below the Jaccard threshold; asserts dedup does not
    # over-remove merely topically related papers.
    work(
        "W1015",
        "Uncertainty Estimation for Tumour Delineation Under Shift",
        NEAR_NEGATIVE,
        CV,
        [CV],
        doi="10.1000/cv.1015",
        year=2022,
    ),
    # -- INTENDED exact duplicate of W1001: identical OpenAlex id -----------
    work(
        "W1001",
        "Multilingual Named Entity Recognition with Transformers",
        A_NLP,
        AI,
        [AI, AI_2],
        doi="10.1000/ai.1001",
        references=["W9001", "W9002"],
    ),
    # -- INTENDED DOI duplicate of W1002: different id, shared DOI ----------
    work(
        "W1014",
        "Instruction Tuning for Scholarly QA (preprint)",
        A_QA,
        AI,
        [AI],
        doi="10.1000/ai.1002",
        work_type="preprint",
    ),
    # -- validation failures, each with its own unique abstract -------------
    # abstract_inverted_index absent entirely
    work("W1020", "A Paper With No Abstract Available", None, AI, [AI], doi="10.1000/ai.1020"),
    # abstract below validation.min_abstract_chars
    work("W1021", "Short Abstract Paper", SHORT, CV, [CV], doi="10.1000/cv.1021"),
    # title below validation.min_title_chars
    work("W1022", "Brief", A_COMPILER, NET, [NET], doi="10.1000/net.1022"),
    # blank title
    work("W1023", "   ", A_TESTING, NET, [NET], doi="10.1000/net.1023"),
    # language outside validation.allowed_languages
    work(
        "W1024",
        "Réseaux de Neurones pour la Vision par Ordinateur",
        A_RADAR,
        CV,
        [CV],
        doi="10.1000/cv.1024",
        language="fr",
    ),
    # no primary_topic -> unlabelable at any taxonomy level
    work("W1025", "Paper Without Any Topic Assignment", A_FILTER, None, []),
    # -- rare class: a single Signal Processing record ----------------------
    # Trips labels.min_class_count; must be dropped rather than left to break
    # stratified splitting with a one-member class. Empty authorships too.
    work(
        "W1026",
        "Robust Adaptive Beamforming Under Steering Vector Mismatch",
        A_BEAMFORM,
        SIG,
        [SIG],
        doi="10.1000/sig.1026",
        authors=[],
    ),
    # -- structural robustness ---------------------------------------------
    # Nulls where lists are expected, and a score outside [0, 1].
    {
        **work(
            "W1027",
            "Robustness Probe Record",
            A_ACTIVE,
            AI,
            [AI],
            doi="10.1000/ai.1027",
        ),
        "authorships": None,
        "keywords": None,
        "referenced_works": None,
        "primary_topic": {**AI, "score": 1.4},
    },
    # No usable identifier -> parse_record must return None, not raise.
    work(None, "Record With No Identifier", A_NLP, AI, [AI]),
]

# --------------------------------------------------------------------------
# Property checks
# --------------------------------------------------------------------------

#: Pairs the **exact** key path catches, via identical source id or DOI. Their
#: text similarity is irrelevant to detection — exact keys are checked before
#: shingling — so no margin is required of them. They are listed only so the
#: unintended-collision checks do not flag them.
EXACT_DUPLICATE_PAIRS = {
    ("W1001", "W1001"),  # same work id, ingested twice
    ("W1002", "W1014"),  # shared DOI: preprint and published version
}

#: Pairs that **only** the near-duplicate path can catch: distinct ids, distinct
#: DOIs, lightly edited text. These must clear the threshold by ``MARGIN``,
#: because an edit that drops one below it turns a passing test into a vacuous
#: one — dedup would remove nothing and the assertion would still be satisfiable
#: by the exact path.
NEAR_DUPLICATE_PAIRS = {
    ("W1004", "W1013"),  # retitled resubmission, closing clause rewritten
}


def _matching_texts() -> list[tuple[str, str]]:
    """Return ``(short_id, matching_text)`` for every record with an abstract.

    The abstract is rebuilt from the inverted index *this script* just built, in
    the forward direction, so the round-trip check stays independent of the
    parser. The title is then prepended and the result normalised exactly as
    ``dedup`` does, because the title is part of what dedup compares.
    """
    texts: list[tuple[str, str]] = []
    for record in RECORDS:
        raw_id, index = record["id"], record.get("abstract_inverted_index")
        if raw_id is None or not index:
            continue
        positions: dict[int, str] = {}
        for term, spots in index.items():
            for spot in spots:
                positions[spot] = term
        abstract = " ".join(positions[i] for i in sorted(positions))
        title = record["title"] or ""
        texts.append(
            (raw_id.rsplit("/", 1)[-1], normalize_for_matching(f"{title} {abstract}"))
        )
    return texts


def verify() -> None:
    """Assert the properties the tests rely on.

    Similarity is measured with ``dedup``'s own functions rather than a local
    reimplementation. That is deliberate: the guarantee being asserted is "the
    pipeline *will* detect this pair", which a second, subtly different metric
    cannot establish. An earlier version scored the intended near-duplicate pair
    on abstracts alone and reported 0.94 for a pair the pipeline actually scores
    at 0.89 — the margin looked safe while being thinner than intended.

    Raises:
        AssertionError: If an unintended duplicate pair appears, or an intended
            near-duplicate pair falls too close to the decision threshold.
    """
    texts = _matching_texts()
    shingled = [(short_id, shingle_hashes(text, SHINGLE_SIZE)) for short_id, text in texts]

    problems: list[str] = []
    measured: dict[tuple[str, ...], float] = {}
    for i, (left_id, left) in enumerate(shingled):
        for right_id, right in shingled[i + 1 :]:
            score = jaccard(left, right)
            pair = tuple(sorted((left_id, right_id)))

            if pair in NEAR_DUPLICATE_PAIRS:
                measured[pair] = score
                if score < JACCARD_THRESHOLD + MARGIN:
                    problems.append(
                        f"near-duplicate pair {pair} too close to threshold: {score:.4f} "
                        f"(want >= {JACCARD_THRESHOLD + MARGIN:.2f})"
                    )
                continue

            if pair in EXACT_DUPLICATE_PAIRS or left_id == right_id:
                measured[pair] = score
                continue  # caught by id/DOI; similarity is not what detects it

            if score >= JACCARD_THRESHOLD:
                problems.append(f"unintended duplicate {pair}: {score:.4f}")
            elif score >= 0.60:
                problems.append(f"pair {pair} uncomfortably similar: {score:.4f}")

    missing = (NEAR_DUPLICATE_PAIRS | EXACT_DUPLICATE_PAIRS) - set(measured)
    if missing:
        problems.append(f"declared duplicate pair(s) absent from the corpus: {sorted(missing)}")

    if problems:
        raise AssertionError("Fixture property violations:\n  " + "\n  ".join(problems))

    print(f"Verified {len(texts)} abstracts: no unintended near-duplicates.")
    for pair, score in sorted(measured.items()):
        kind = "near" if pair in NEAR_DUPLICATE_PAIRS else "exact"
        print(f"  intended {kind} pair {pair[0]}~{pair[1]}: {score:.4f}")


def main() -> None:
    """Write the fixture corpus as UTF-8 JSONL with LF line endings."""
    verify()
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for record in RECORDS:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(RECORDS)} records to {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
