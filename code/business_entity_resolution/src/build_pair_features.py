from pathlib import Path
import argparse
import csv
import hashlib
import random
import re
import sys

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

NORMALIZED_DIR = PROJECT_ROOT / "output" / "normalized"

SOURCE1_PATH = NORMALIZED_DIR / "train_source1_normalized.parquet"
SOURCE2_PATH = NORMALIZED_DIR / "train_source2_normalized.parquet"
SOURCE3_PATH = NORMALIZED_DIR / "train_source3_normalized.parquet"

DEFAULT_CANDIDATE_PATH = (
    PROJECT_ROOT
    / "output"
    / "candidate_pairs_benchmark_sample.tsv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "benchmark_pair_features.parquet"
)

PARQUET_BATCH_SIZE = 100_000


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_text(value):
    """Return a clean string and convert missing values to ''."""

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_country(value):
    """Normalize country for exact comparison."""

    return safe_text(value).casefold()


def normalized_numbers(value):
    """
    Extract digit groups from an address.

    Leading zeroes are removed, matching the teammate feature code.
    """

    value = safe_text(value)

    if not value:
        return set()

    numbers = re.findall(r"\d+", value)

    result = set()

    for number in numbers:
        number = number.lstrip("0")

        if number == "":
            number = "0"

        result.add(number)

    return result


def compact_text_from_normalized_name(norm_name):
    """
    Build compact name from an already normalized name.

    This is equivalent to the teammate compact_text() definition:
    remove non-alphanumeric characters after name normalization.
    """

    normalized = safe_text(norm_name)

    return "".join(
        char
        for char in normalized
        if char.isalnum()
    )


def token_set(value):
    """Return unique tokens of length >= 2."""

    value = safe_text(value)

    if not value:
        return set()

    return {
        token
        for token in value.split()
        if len(token) >= 2
    }


def jaccard_similarity(tokens_a, tokens_b):
    """Jaccard similarity, matching the teammate feature code."""

    if not tokens_a and not tokens_b:
        return 1.0

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    if union == 0:
        return 0.0

    return intersection / union


def sequence_similarity(text_a, text_b):
    """
    Character-level similarity using difflib.SequenceMatcher.

    Kept identical to the teammate feature definition.
    """

    import difflib

    if not text_a or not text_b:
        return 0.0

    return difflib.SequenceMatcher(
        None,
        text_a,
        text_b,
    ).ratio()


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_record(row):
    """
    Convert one normalized Parquet row into a compact prepared record.

    The feature formulas below follow the teammate's feature definitions:

    - name_exact
    - name_compact_exact
    - name_token_jaccard
    - name_char_similarity
    - address_token_jaccard
    - address_char_similarity
    - number_overlap
    - number_jaccard
    - country_match
    - missing-value flags
    """

    entity_id = safe_text(
        row.get("entity_id")
    )

    norm_name = safe_text(
        row.get("norm_name")
    )

    compact_name = safe_text(
        row.get("compact_name")
    )

    if not compact_name and norm_name:
        compact_name = compact_text_from_normalized_name(
            norm_name
        )

    norm_address = safe_text(
        row.get("norm_address")
    )

    country = clean_country(
        row.get("country")
    )

    return {
        "entity_id": entity_id,
        "name": norm_name,
        "compact_name": compact_name,
        "name_tokens": token_set(norm_name),
        "address": norm_address,
        "address_tokens": token_set(norm_address),
        "numbers": normalized_numbers(norm_address),
        "country": country,
    }


def calculate_features(s1_record, candidate_record, label, split):
    """
    Calculate the pair features defined in the teammate's file.
    """

    s1_name = s1_record["name"]
    candidate_name = candidate_record["name"]

    s1_address = s1_record["address"]
    candidate_address = candidate_record["address"]

    s1_country = s1_record["country"]
    candidate_country = candidate_record["country"]

    s1_compact_name = s1_record["compact_name"]
    candidate_compact_name = candidate_record["compact_name"]

    s1_name_tokens = s1_record["name_tokens"]
    candidate_name_tokens = candidate_record["name_tokens"]

    s1_address_tokens = s1_record["address_tokens"]
    candidate_address_tokens = candidate_record["address_tokens"]

    s1_numbers = s1_record["numbers"]
    candidate_numbers = candidate_record["numbers"]

    # ========================================================
    # NAME FEATURES
    # ========================================================

    name_exact = int(
        bool(
            s1_name
            and candidate_name
            and s1_name == candidate_name
        )
    )

    name_compact_exact = int(
        bool(
            s1_compact_name
            and candidate_compact_name
            and s1_compact_name == candidate_compact_name
        )
    )

    name_token_jaccard = jaccard_similarity(
        s1_name_tokens,
        candidate_name_tokens,
    )

    name_char_similarity = sequence_similarity(
        s1_name,
        candidate_name,
    )

    # ========================================================
    # ADDRESS FEATURES
    # ========================================================

    address_token_jaccard = jaccard_similarity(
        s1_address_tokens,
        candidate_address_tokens,
    )

    address_char_similarity = sequence_similarity(
        s1_address,
        candidate_address,
    )

    # ========================================================
    # ADDRESS NUMBER FEATURES
    # ========================================================

    number_intersection = (
        s1_numbers
        & candidate_numbers
    )

    number_union = (
        s1_numbers
        | candidate_numbers
    )

    number_overlap = int(
        len(number_intersection) > 0
    )

    number_jaccard = (
        len(number_intersection)
        / len(number_union)
        if number_union
        else 0.0
    )

    # ========================================================
    # COUNTRY FEATURE
    # ========================================================

    country_match = int(
        bool(
            s1_country
            and candidate_country
            and s1_country == candidate_country
        )
    )

    # ========================================================
    # MISSING VALUE FEATURES
    # ========================================================

    s1_name_missing = int(
        not bool(s1_name)
    )

    candidate_name_missing = int(
        not bool(candidate_name)
    )

    s1_address_missing = int(
        not bool(s1_address)
    )

    candidate_address_missing = int(
        not bool(candidate_address)
    )

    return {
        "s1_id": s1_record["entity_id"],
        "candidate_id": candidate_record["entity_id"],
        "label": int(label),
        "split": split,
        "name_exact": name_exact,
        "name_compact_exact": name_compact_exact,
        "name_token_jaccard": name_token_jaccard,
        "name_char_similarity": name_char_similarity,
        "address_token_jaccard": address_token_jaccard,
        "address_char_similarity": address_char_similarity,
        "number_overlap": number_overlap,
        "number_jaccard": number_jaccard,
        "country_match": country_match,
        "s1_name_missing": s1_name_missing,
        "candidate_name_missing": candidate_name_missing,
        "s1_address_missing": s1_address_missing,
        "candidate_address_missing": candidate_address_missing,
    }


# ============================================================
# GROUND TRUTH DISCOVERY / LOADING
# ============================================================

def find_ground_truth_file(project_root):
    """
    Discover a ground-truth file from its required column names.

    This avoids hard-coding a filename that may differ in the local
    challenge dataset.
    """

    dataset_root = project_root / "dataset"

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{dataset_root}"
        )

    required_columns = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    matches = []

    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".tsv", ".csv"}:
            continue

        try:
            with open(
                path,
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                first_line = file.readline().strip()

            if not first_line:
                continue

            delimiter = "\t" if "\t" in first_line else ","
            header = {
                value.strip()
                for value in first_line.split(delimiter)
            }

            if required_columns.issubset(header):
                matches.append(path)

        except (
            OSError,
            UnicodeDecodeError,
        ):
            continue

    if not matches:
        raise FileNotFoundError(
            "Could not automatically find the ground-truth file. "
            "Use --ground-truth to specify its path."
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Multiple ground-truth files were found:\n"
            + "\n".join(str(path) for path in matches)
            + "\nUse --ground-truth to select the correct file."
        )

    return matches[0]


def load_ground_truth(ground_truth_path):
    """
    Load ground truth as:

        s1_id -> set(candidate entity IDs)
    """

    try:
        delimiter = "\t"

        with open(
            ground_truth_path,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            first_line = file.readline()

        if "\t" not in first_line and "," in first_line:
            delimiter = ","

        ground_truth = {}

        with open(
            ground_truth_path,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file,
                delimiter=delimiter,
            )

            if not reader.fieldnames:
                raise ValueError(
                    "Ground-truth file has no header."
                )

            required = {
                "source1_entity_id",
                "matched_entity_ids",
            }

            missing = required - set(reader.fieldnames)

            if missing:
                raise ValueError(
                    "Ground-truth file is missing columns: "
                    + ", ".join(sorted(missing))
                )

            for row in reader:

                s1_id = safe_text(
                    row.get("source1_entity_id")
                )

                if not s1_id:
                    continue

                matched_text = safe_text(
                    row.get("matched_entity_ids")
                )

                if matched_text:
                    matched_ids = {
                        item.strip()
                        for item in matched_text.split(",")
                        if item.strip()
                    }
                else:
                    matched_ids = set()

                ground_truth[s1_id] = matched_ids

    except OSError as exc:
        raise FileNotFoundError(
            f"Could not read ground truth:\n{ground_truth_path}"
        ) from exc

    return ground_truth


# ============================================================
# CANDIDATE SAMPLING
# ============================================================

def deterministic_sample(items, count, seed_text):
    """
    Deterministically sample strings without relying on Python's
    randomized hash().
    """

    if count <= 0 or not items:
        return []

    if len(items) <= count:
        return sorted(items)

    scored = []

    for item in items:
        digest = hashlib.sha256(
            f"{seed_text}|{item}".encode("utf-8")
        ).hexdigest()

        scored.append(
            (digest, item)
        )

    scored.sort()

    return [
        item
        for _, item in scored[:count]
    ]


def read_candidate_file(
    candidate_path,
    ground_truth,
    negatives_per_positive,
    min_negatives_per_s1,
    max_negatives_per_s1,
    seed,
):
    """
    Read the compact one-row-per-S1 candidate file.

    Returns:

        selected_pairs:
            list of (s1_id, candidate_id, label)

        candidate_s1_ids:
            exact S1 IDs in candidate file

        selected_candidate_ids:
            candidate IDs needed for feature computation

        summary:
            sampling statistics
    """

    selected_pairs = []
    candidate_s1_ids = []
    selected_candidate_ids = set()

    total_candidate_links = 0
    total_positive_candidates = 0
    total_missed_ground_truth = 0
    s1_with_true_matches = 0
    s1_with_no_true_matches = 0

    with open(
        candidate_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        header_line = file.readline().strip()

        if not header_line:
            raise ValueError(
                "Candidate file is empty."
            )

        header = header_line.split("\t")

        if len(header) < 2:
            raise ValueError(
                "Candidate file must contain two tab-separated columns."
            )

        if header[0] != "source1_entity_id":
            raise ValueError(
                "Expected first candidate column to be "
                "'source1_entity_id'."
            )

        if header[1] != "candidate_entity_ids":
            raise ValueError(
                "Expected second candidate column to be "
                "'candidate_entity_ids'."
            )

        for line_number, line in enumerate(file, start=2):

            line = line.rstrip("\r\n")

            if not line:
                continue

            parts = line.split("\t", 1)

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid candidate row at line {line_number}."
                )

            s1_id = safe_text(parts[0])
            candidate_text = safe_text(parts[1])

            if not s1_id:
                continue

            candidate_s1_ids.append(s1_id)

            if candidate_text:
                candidate_ids = {
                    item.strip()
                    for item in candidate_text.split(",")
                    if item.strip()
                }
            else:
                candidate_ids = set()

            total_candidate_links += len(candidate_ids)

            true_ids = ground_truth.get(
                s1_id,
                set()
            )

            positives = sorted(
                candidate_ids & true_ids
            )

            missed = true_ids - candidate_ids

            total_positive_candidates += len(positives)
            total_missed_ground_truth += len(missed)

            if positives:
                s1_with_true_matches += 1
            else:
                s1_with_no_true_matches += 1

            # Keep every positive candidate that the blocker captured.
            for candidate_id in positives:
                selected_pairs.append(
                    (
                        s1_id,
                        candidate_id,
                        1,
                    )
                )
                selected_candidate_ids.add(candidate_id)

            # Sample negatives so the feature dataset stays compact.
            if positives:
                desired_negatives = max(
                    min_negatives_per_s1,
                    len(positives) * negatives_per_positive,
                )
            else:
                # Unmatched S1 entities still provide useful negatives.
                desired_negatives = min_negatives_per_s1

            desired_negatives = min(
                desired_negatives,
                max_negatives_per_s1,
            )

            negatives = deterministic_sample(
                candidate_ids - set(positives),
                desired_negatives,
                f"{seed}|{s1_id}",
            )

            for candidate_id in negatives:
                selected_pairs.append(
                    (
                        s1_id,
                        candidate_id,
                        0,
                    )
                )
                selected_candidate_ids.add(candidate_id)

    summary = {
        "candidate_s1_count": len(candidate_s1_ids),
        "candidate_link_count": total_candidate_links,
        "candidate_positive_count": total_positive_candidates,
        "missed_ground_truth_count": total_missed_ground_truth,
        "s1_with_true_matches": s1_with_true_matches,
        "s1_with_no_true_matches": s1_with_no_true_matches,
        "selected_pair_count": len(selected_pairs),
        "selected_positive_count": sum(
            1
            for _, _, label in selected_pairs
            if label == 1
        ),
        "selected_negative_count": sum(
            1
            for _, _, label in selected_pairs
            if label == 0
        ),
        "selected_candidate_id_count": len(
            selected_candidate_ids
        ),
    }

    return (
        selected_pairs,
        set(candidate_s1_ids),
        selected_candidate_ids,
        summary,
    )


# ============================================================
# NORMALIZED PARQUET LOOKUPS
# ============================================================

def load_selected_s1_records(selected_s1_ids):
    """
    Read only the Source-1 records needed by the candidate sample.
    """

    records = {}

    parquet_file = pq.ParquetFile(
        SOURCE1_PATH
    )

    for batch in parquet_file.iter_batches(
        batch_size=PARQUET_BATCH_SIZE,
        columns=[
            "entity_id",
            "country",
            "norm_name",
            "compact_name",
            "norm_address",
        ],
    ):

        chunk = batch.to_pandas()

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = safe_text(
                row.entity_id
            )

            if entity_id not in selected_s1_ids:
                continue

            records[entity_id] = prepare_record(
                {
                    "entity_id": entity_id,
                    "country": row.country,
                    "norm_name": row.norm_name,
                    "compact_name": row.compact_name,
                    "norm_address": row.norm_address,
                }
            )

        if len(records) >= len(selected_s1_ids):
            break

    return records


def load_selected_candidate_records(selected_candidate_ids):
    """
    Read only the candidate records actually selected for features.

    Source 2 and Source 3 are scanned once each.
    """

    records = {}

    for source_path, source_label in [
        (
            SOURCE2_PATH,
            "Source 2",
        ),
        (
            SOURCE3_PATH,
            "Source 3",
        ),
    ]:

        parquet_file = pq.ParquetFile(
            source_path
        )

        print(
            f"Scanning {source_label} for selected candidates...",
            flush=True,
        )

        for batch in parquet_file.iter_batches(
            batch_size=PARQUET_BATCH_SIZE,
            columns=[
                "entity_id",
                "country",
                "norm_name",
                "compact_name",
                "norm_address",
            ],
        ):

            chunk = batch.to_pandas()

            for row in chunk.itertuples(
                index=False
            ):

                entity_id = safe_text(
                    row.entity_id
                )

                if entity_id not in selected_candidate_ids:
                    continue

                records[entity_id] = prepare_record(
                    {
                        "entity_id": entity_id,
                        "country": row.country,
                        "norm_name": row.norm_name,
                        "compact_name": row.compact_name,
                        "norm_address": row.norm_address,
                    }
                )

            del chunk

        print(
            f"{source_label} selected records found so far: "
            f"{len(records):,}",
            flush=True,
        )

    return records


# ============================================================
# BUILD FEATURES
# ============================================================

def build_feature_dataframe(
    selected_pairs,
    s1_records,
    candidate_records,
    validation_s1_ids,
):
    """Build the final feature DataFrame."""

    feature_rows = []
    missing_s1 = 0
    missing_candidates = 0

    for index, (s1_id, candidate_id, label) in enumerate(
        selected_pairs,
        start=1,
    ):

        s1_record = s1_records.get(
            s1_id
        )

        if s1_record is None:
            missing_s1 += 1
            continue

        candidate_record = candidate_records.get(
            candidate_id
        )

        if candidate_record is None:
            missing_candidates += 1
            continue

        split = (
            "validation"
            if s1_id in validation_s1_ids
            else "train"
        )

        feature_rows.append(
            calculate_features(
                s1_record,
                candidate_record,
                label,
                split,
            )
        )

        if index % 5_000 == 0:
            print(
                f"Feature rows built: {index:,}",
                flush=True,
            )

    print(
        f"Missing S1 records during feature build: {missing_s1:,}"
    )

    print(
        "Missing candidate records during feature build: "
        f"{missing_candidates:,}"
    )

    return pd.DataFrame(feature_rows)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build labeled pairwise ML features from the compact "
            "candidate-pair file."
        )
    )

    parser.add_argument(
        "--candidate-file",
        type=str,
        default=str(DEFAULT_CANDIDATE_PATH),
        help=(
            "Compact candidate TSV with one row per S1 entity."
        ),
    )

    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help=(
            "Ground-truth file. If omitted, it is discovered "
            "from its required column names."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=(
            "Output Parquet feature file."
        ),
    )

    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=5,
        help=(
            "Number of sampled negatives per captured positive. "
            "Default: 5."
        ),
    )

    parser.add_argument(
        "--min-negatives-per-s1",
        type=int,
        default=10,
        help=(
            "Minimum negative candidates kept for each S1. "
            "Default: 10."
        ),
    )

    parser.add_argument(
        "--max-negatives-per-s1",
        type=int,
        default=50,
        help=(
            "Maximum negative candidates kept for each S1. "
            "Default: 50."
        ),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction of S1 entities assigned to validation. "
            "Default: 0.20."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help=(
            "Random seed for the S1 train/validation split."
        ),
    )

    args = parser.parse_args()

    candidate_path = Path(
        args.candidate_file
    )

    output_path = Path(
        args.output
    )

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found:\n{candidate_path}"
        )

    for path in [
        SOURCE1_PATH,
        SOURCE2_PATH,
        SOURCE3_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Normalized Parquet file not found:\n{path}"
            )

    if args.negatives_per_positive < 0:
        raise ValueError(
            "--negatives-per-positive must be >= 0."
        )

    if args.min_negatives_per_s1 < 0:
        raise ValueError(
            "--min-negatives-per-s1 must be >= 0."
        )

    if args.max_negatives_per_s1 < args.min_negatives_per_s1:
        raise ValueError(
            "--max-negatives-per-s1 must be >= "
            "--min-negatives-per-s1."
        )

    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError(
            "--validation-fraction must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    if args.ground_truth:
        ground_truth_path = Path(
            args.ground_truth
        )
    else:
        ground_truth_path = find_ground_truth_file(
            PROJECT_ROOT
        )

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found:\n{ground_truth_path}"
        )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "PAIR FEATURE BUILDING"
    )

    print(
        "=" * 70
    )

    print(
        f"Candidate file          : {candidate_path}"
    )

    print(
        f"Ground truth            : {ground_truth_path}"
    )

    print(
        f"Output                  : {output_path}"
    )

    print(
        f"Negatives / positive    : {args.negatives_per_positive}"
    )

    print(
        f"Min negatives / S1      : {args.min_negatives_per_s1}"
    )

    print(
        f"Max negatives / S1      : {args.max_negatives_per_s1}"
    )

    print(
        f"Validation fraction     : {args.validation_fraction:.2f}"
    )

    print(
        f"Random state             : {args.random_state}"
    )

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    print(
        "\nLoading ground truth..."
    )

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    print(
        f"Ground-truth S1 rows    : {len(ground_truth):,}"
    )

    # --------------------------------------------------------
    # Read candidates and sample negatives
    # --------------------------------------------------------

    print(
        "\nReading candidate file and selecting training pairs..."
    )

    (
        selected_pairs,
        candidate_s1_ids,
        selected_candidate_ids,
        candidate_summary,
    ) = read_candidate_file(
        candidate_path,
        ground_truth,
        args.negatives_per_positive,
        args.min_negatives_per_s1,
        args.max_negatives_per_s1,
        args.random_state,
    )

    print(
        "\nCandidate/sampling summary:"
    )

    for key, value in candidate_summary.items():
        print(
            f"  {key:32s}: {value:,}"
        )

    # --------------------------------------------------------
    # Build grouped train/validation S1 split
    # --------------------------------------------------------

    candidate_s1_ids_list = sorted(
        candidate_s1_ids
    )

    validation_count = max(
        1,
        int(
            round(
                len(candidate_s1_ids_list)
                * args.validation_fraction
            )
        ),
    )

    if validation_count >= len(candidate_s1_ids_list):
        validation_count = max(
            1,
            len(candidate_s1_ids_list) - 1,
        )

    rng = random.Random(
        args.random_state
    )

    validation_s1_ids = set(
        rng.sample(
            candidate_s1_ids_list,
            validation_count,
        )
    )

    print(
        "\nGrouped split:"
    )

    print(
        f"  Train S1 entities     : "
        f"{len(candidate_s1_ids) - len(validation_s1_ids):,}"
    )

    print(
        f"  Validation S1 entities: "
        f"{len(validation_s1_ids):,}"
    )

    # --------------------------------------------------------
    # Load records required for selected pairs
    # --------------------------------------------------------

    print(
        "\nLoading selected Source-1 records..."
    )

    s1_records = load_selected_s1_records(
        candidate_s1_ids
    )

    print(
        f"Selected S1 records loaded: "
        f"{len(s1_records):,}"
    )

    missing_s1_ids = (
        candidate_s1_ids
        - set(s1_records)
    )

    if missing_s1_ids:
        print(
            f"WARNING: Missing S1 IDs in Parquet: "
            f"{len(missing_s1_ids):,}"
        )

    print(
        "\nLoading selected candidate records..."
    )

    candidate_records = load_selected_candidate_records(
        selected_candidate_ids
    )

    print(
        f"Selected candidate records loaded: "
        f"{len(candidate_records):,}"
    )

    missing_candidate_ids = (
        selected_candidate_ids
        - set(candidate_records)
    )

    if missing_candidate_ids:
        print(
            f"WARNING: Missing candidate IDs in Parquet: "
            f"{len(missing_candidate_ids):,}"
        )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    print(
        "\nBuilding feature rows..."
    )

    features_df = build_feature_dataframe(
        selected_pairs,
        s1_records,
        candidate_records,
        validation_s1_ids,
    )

    if features_df.empty:
        raise RuntimeError(
            "No feature rows were generated."
        )

    # --------------------------------------------------------
    # Reorder columns for stable output
    # --------------------------------------------------------

    ordered_columns = [
        "s1_id",
        "candidate_id",
        "label",
        "split",
        "name_exact",
        "name_compact_exact",
        "name_token_jaccard",
        "name_char_similarity",
        "address_token_jaccard",
        "address_char_similarity",
        "number_overlap",
        "number_jaccard",
        "country_match",
        "s1_name_missing",
        "candidate_name_missing",
        "s1_address_missing",
        "candidate_address_missing",
    ]

    features_df = features_df[
        ordered_columns
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_df.to_parquet(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    train_rows = int(
        (features_df["split"] == "train").sum()
    )

    validation_rows = int(
        (features_df["split"] == "validation").sum()
    )

    positive_rows = int(
        features_df["label"].sum()
    )

    negative_rows = int(
        len(features_df) - positive_rows
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PAIR FEATURE BUILD COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Feature rows             : {len(features_df):,}"
    )

    print(
        f"Positive rows            : {positive_rows:,}"
    )

    print(
        f"Negative rows            : {negative_rows:,}"
    )

    print(
        f"Train rows               : {train_rows:,}"
    )

    print(
        f"Validation rows          : {validation_rows:,}"
    )

    print(
        f"Feature columns          : "
        f"{len(ordered_columns) - 4:,}"
    )

    print(
        f"Output                   : {output_path}"
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()
