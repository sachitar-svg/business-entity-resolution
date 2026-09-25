import os
import json
import re
import pandas as pd
from collections import defaultdict
from preprocessing import normalize_name, normalize_address

# ============================================================
# CONFIGURATION
# ============================================================
S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"

# Paths to the token stats you generated earlier
NAME_STATS_PATH = "code/business_entity_resolution/data/token_stats/name_token_counts.json"
ADDRESS_STATS_PATH = "code/business_entity_resolution/data/token_stats/address_token_counts.json"

OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# The frequency threshold for a token to be considered "Rare"
# Lower = fewer candidates (more precision), Higher = more candidates (more recall)
SINGLE_TOKEN_MAX_FREQ = 5000 
CHUNK_SIZE = 200_000

# ============================================================
# HELPERS
# ============================================================

def safe_text(value):
    if value is None or pd.isna(value): return ""
    return str(value).strip()

def compact_text(value):
    normalized = normalize_name(value)
    return "".join(char for char in normalized if char.isalnum())

def get_tokens(value, normalizer):
    normalized = normalizer(value)
    if not normalized: return set()
    return {t for t in normalized.split() if len(t) >= 2}

def extract_numbers(value):
    """Extracts numbers with 3+ digits to avoid noise like '1st floor'"""
    if not value: return set()
    return set(re.findall(r'\d{3,}', str(value)))

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("Starting Final Candidate Generation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Token Stats
    with open(NAME_STATS_PATH, "r", encoding="utf-8") as f:
        name_counts = json.load(f)
    with open(ADDRESS_STATS_PATH, "r", encoding="utf-8") as f:
        address_counts = json.load(f)

    # 2. Build Indices for Source 2 and Source 3
    # Logic: token -> set of entity_ids
    name_index = defaultdict(set)
    compact_index = defaultdict(set)
    address_index = defaultdict(set)
    number_index = defaultdict(set)
    candidate_countries = {}

    source_files = [S2_PATH, S3_PATH]
    for path in source_files:
        print(f"Indexing {path}...")
        for chunk in pd.read_csv(path, sep="\t", chunksize=CHUNK_SIZE):
            for row in chunk.itertuples():
                cid = safe_text(row.entity_id)
                country = safe_text(row.country).casefold()
                candidate_countries[cid] = country

                # Index Exact Normalized Name
                norm_name = normalize_name(row.business_name)
                if norm_name: name_index[norm_name].add(cid)

                # Index Compact Name
                comp_name = compact_text(row.business_name)
                if comp_name: compact_index[comp_name].add(cid)

                # Index Rare Name Tokens
                name_toks = get_tokens(row.business_name, normalize_name)
                for t in name_toks:
                    if name_counts.get(t, 10**9) <= SINGLE_TOKEN_MAX_FREQ:
                        name_index[t].add(cid)

                # Index Rare Address Tokens
                addr_toks = get_tokens(row.business_address, normalize_address)
                for t in addr_toks:
                    if address_counts.get(t, 10**9) <= SINGLE_TOKEN_MAX_FREQ:
                        address_index[t].add(cid)

                # Index Numeric Anchors
                nums = extract_numbers(row.business_address)
                for n in nums:
                    number_index[n].add(cid)

    # 3. Process Source 1 and Generate Pairs
    print("\nGenerating pairs for Source 1...")
    
    # We open the file in write mode and write S1 entities one by one
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")

        s1_df = pd.read_csv(S1_PATH, sep="\t")
        for _, row in s1_df.iterrows():
            s1_id = safe_text(row.entity_id)
            s1_country = safe_text(row.country).casefold()
            
            candidates = set()

            # Channel 1: Exact Name
            norm_name = normalize_name(row.business_name)
            if norm_name: candidates.update(name_index.get(norm_name, set()))

            # Channel 2: Compact Name
            comp_name = compact_text(row.business_name)
            if comp_name: candidates.update(compact_index.get(comp_name, set()))

            # Channel 3: Rare Name Tokens
            name_toks = get_tokens(row.business_name, normalize_name)
            for t in name_toks:
                if name_counts.get(t, 10**9) <= SINGLE_TOKEN_MAX_FREQ:
                    candidates.update(name_index.get(t, set()))

            # Channel 4: Rare Address Tokens
            addr_toks = get_tokens(row.business_name, normalize_address) # Note: use address here
            # Correcting the variable:
            addr_toks = get_tokens(row.business_address, normalize_address)
            for t in addr_toks:
                if address_counts.get(t, 10**9) <= SINGLE_TOKEN_MAX_FREQ:
                    candidates.update(address_index.get(t, set()))

            # Channel 5: Numeric Anchors
            nums = extract_numbers(row.business_address)
            for n in nums:
                candidates.update(number_index.get(n, set()))

            # FINAL FILTER: Country Match
            # Only keep candidates in the same country
            final_candidates = [
                cid for cid in candidates 
                if candidate_countries.get(cid) == s1_country
            ]

            # Write to TSV: S1_ID [TAB] comma-separated candidate IDs
            cid_string = ",".join(final_candidates)
            f_out.write(f"{s1_id}\t{cid_string}\n")

    print(f"\nDone! Candidates saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
