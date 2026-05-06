import io
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import arabic_reshaper
from bidi.algorithm import get_display
import networkx as nx
from matplotlib.patches import Ellipse
import re
from collections import defaultdict
from itertools import combinations
from adjustText import adjust_text
from scipy.cluster import hierarchy
import six
import json


#@title assess
def collation_to_df(collation):
    """
    Converts a Collation object from the collatex library to a pandas DataFrame.

    The function first converts the Collation object to a CSV string,
    reads the CSV string into a pandas DataFrame, converts all empty cells
    to the number 0 transposes the DataFrame, and sets the first row as the
    column headers assuming it contains the names of the variants.

    Args:
        collation (Collation): A Collation object containing the collated text.

    Returns:
        pd.DataFrame: A pandas DataFrame representing the collated text,
                      with variant labels as column headers.
    """
    collation_csv = collate(collation, output="csv", layout="vertical")
    df = pd.read_csv(io.StringIO(collation_csv), header=None)
    df.fillna('GAP', inplace=True)
    df = df.T
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.T
    tokens = []
    for i in range(len(df.columns)):
        tokens.append("token " + str(i))
    df.columns = tokens
    return df


from collatex import *

#@title input_df_to_collation_df(input_df)
def input_df_to_collation_df(input_df):
    '''
        takes df with variant hadith in rows and one column
        returns a df with variant hadith in same rows, but columns are now aligned CollateX tokens
    '''
    #calculate labels for variants.
    labels = []

    for variant in range(len(input_df)):
        labels.append(f"variant {variant+1}")


    #format texts for collation
    full_texts = []
    for index, row in input_df.iterrows():
        concatenated_text = ""
        for col in reversed(input_df.columns):
            if pd.notna(row[col]):
                concatenated_text += str(row[col]) + " "
        full_texts.append(concatenated_text.strip())

    collation = Collation()


    for label, text in zip(labels, full_texts):
        collation.add_plain_witness(label, text)

    #Prints collation:
    # alignment_table = collate(collation, output="html", layout="vertical")

    return collation_to_df(collation)



def input_df_to_just_collation(input_df, format='html'):
    '''
        takes df with variant hadith in rows and one column
        returns an alignment table in specified format
    '''
    #calculate labels for variants.
    labels = []

    for variant in range(len(input_df)):
        labels.append(f"variant {variant+1}")


    #format texts for collation
    full_texts = []
    for index, row in input_df.iterrows():
        concatenated_text = ""
        for col in reversed(input_df.columns):
            if pd.notna(row[col]):
                concatenated_text += str(row[col]) + " "
        full_texts.append(concatenated_text.strip())


    collation = Collation()

    for label, text in zip(labels, full_texts):
        collation.add_plain_witness(label, text)

    #Prints collation:
    alignment_table = collate(collation, output=format, layout="vertical")

    return alignment_table



#@title MICRO-LEVEL (with rescaled similarities and witness tracking)

def micro_semantic_similarity(
    df,
    embedding_model,
    rescale_low=0.7,
    rescale_high=1.0,
    return_pairwise=False
):
    """
    Compute micro-level semantic similarity within each column.

    Optionally rescale similarities from [rescale_low, rescale_high] → [0, 1].

    Parameters
    ----------
    df : pd.DataFrame
        Aligned witness table (rows = witnesses, columns = alignment positions)
    embedding_model : sentence-transformers model or similar
    rescale_low, rescale_high : float
        Rescale raw cosine similarities from [rescale_low, rescale_high] to [0, 1]
    return_pairwise : bool
        If True, also returns pairwise witness similarities for dendrogram use.

    Returns
    -------
    if return_pairwise=False:
        pd.DataFrame with per-column stats (original behavior)
    else:
        tuple: (column_stats_df, pairwise_sim_df)
    """

    # Cache embeddings 
    unique_tokens = list({t for col in df.columns for t in df[col] if t != 'GAP' and pd.notna(t)})
    if unique_tokens:
        token_to_emb = dict(zip(unique_tokens, embedding_model.encode(unique_tokens)))
    else:
        token_to_emb = {}

    scale_range = rescale_high - rescale_low
    if scale_range <= 0:
        raise ValueError("rescale_high must be > rescale_low")

    # Prepare containers
    column_scores = []

    if return_pairwise:
        # For pairwise: accumulate similarities per witness pair across columns
        pair_sim_sum = defaultdict(float)   # sum of similarities
        pair_sim_count = defaultdict(int)   # number of shared columns

    witness_names = df.index.tolist()

    # Loop through columns 
    for col_idx, column_name in enumerate(df.columns):
        col_series = df[column_name]
        # Get non-GAP, non-NaN entries with witness names
        valid_mask = (col_series != 'GAP') & (col_series.notna())
        valid_tokens = col_series[valid_mask]
        valid_witnesses = valid_mask[valid_mask].index.tolist()

        if len(valid_tokens) < 2:
            column_scores.append({
                'column_index': col_idx,
                'column_name': column_name,
                'avg_similarity': np.nan,
                'min_similarity': np.nan,
                'max_similarity': np.nan,
                'comparison_count': 0,
                'tokens': valid_tokens.tolist()
            })
            continue

        # Get embeddings
        embeddings = np.array([token_to_emb[t] for t in valid_tokens])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / norms
        sims = np.dot(normed, normed.T)  # shape: (k, k)

        # Extract pairwise similarities (lower triangle)
        k = len(valid_tokens)
        tril_indices = np.tril_indices(k, k=-1)
        raw_similarities = sims[tril_indices]

        # Rescale
        clipped_sims = np.clip(raw_similarities, rescale_low, rescale_high)
        rescaled_similarities = (clipped_sims - rescale_low) / scale_range

        # Column stats
        avg_sim = np.mean(rescaled_similarities) if rescaled_similarities.size > 0 else 1.0
        min_sim = np.min(rescaled_similarities) if rescaled_similarities.size > 0 else 1.0
        max_sim = np.max(rescaled_similarities) if rescaled_similarities.size > 0 else 1.0

        column_scores.append({
            'column_index': col_idx,
            'column_name': column_name,
            'avg_similarity': avg_sim,
            'min_similarity': min_sim,
            'max_similarity': max_sim,
            'comparison_count': rescaled_similarities.size,
            'tokens': valid_tokens.tolist()
        })

        # if pairwise mode: accumulate per witness pair
        if return_pairwise:
            # Map local indices to global witness names
            for idx_i, idx_j in zip(tril_indices[0], tril_indices[1]):
                w_i = valid_witnesses[idx_i]
                w_j = valid_witnesses[idx_j]
                sim_val = rescaled_similarities[
                    np.where((tril_indices[0] == idx_i) & (tril_indices[1] == idx_j))[0][0]
                ]
                # Ensure consistent ordering
                if w_i > w_j:
                    w_i, w_j = w_j, w_i
                pair_key = (w_i, w_j)
                pair_sim_sum[pair_key] += sim_val
                pair_sim_count[pair_key] += 1

    column_stats_df = pd.DataFrame(column_scores)

    if not return_pairwise:
        return column_stats_df

    # === 4. Build pairwise DataFrame ===
    pairwise_records = []
    for (w_a, w_b), total_sim in pair_sim_sum.items():
        count = pair_sim_count[(w_a, w_b)]
        avg_sim = total_sim / count  # average over shared columns
        pairwise_records.append({
            'witness_a': w_a,
            'witness_b': w_b,
            'similarity': avg_sim
        })

    # Handle witness pairs that never co-occur in any column
    all_pairs = set()
    for i, wa in enumerate(witness_names):
        for wb in witness_names[i+1:]:
            all_pairs.add((wa, wb))

    missing_pairs = all_pairs - set(pair_sim_sum.keys())
    for wa, wb in missing_pairs:
        # Option 1: assign similarity = 0 (max distance)
        # Option 2: assign similarity = NaN (will be handled later)
        # We choose NaN here to signal "no data"
        pairwise_records.append({
            'witness_a': wa,
            'witness_b': wb,
            'similarity': np.nan
        })

    pairwise_df = pd.DataFrame(pairwise_records)

    return column_stats_df, pairwise_df




#@title MESO-LEVEL
def reconstruct_witness_phrases(df, window_size=3, min_valid_tokens=2):
    witness_phrases = defaultdict(list)
    data = df.values
    n_witnesses, n_columns = data.shape
    witness_names = df.index

    for witness_idx in range(n_witnesses):
        # Pre-process to find valid token positions and tokens
        # Combine token extraction and position mapping into one efficient step
        row_data = data[witness_idx]

        valid_data_and_pos = [
            (str(token), col_idx) # Store (token string, column index)
            for col_idx, token in enumerate(row_data)
            if token != 'GAP' and pd.notna(token)
        ]

        if len(valid_data_and_pos) < min_valid_tokens:
            continue

        # windowing logic
        valid_tokens = [item[0] for item in valid_data_and_pos]
        valid_positions = [item[1] for item in valid_data_and_pos]

        n_valid = len(valid_positions)

        # Create phrases using valid positions
        current_phrases = []
        for i in range(n_valid):
            # i is the index into valid_positions and valid_tokens (the start of the phrase)
            end_idx = i
            while (end_idx < n_valid and
                   valid_positions[end_idx] - valid_positions[i] < window_size):
                end_idx += 1

            # The start and end column indices for slicing the original row
            start_col = valid_positions[i]
            end_col = valid_positions[end_idx-1] + 1

            # Token Segment Extraction
            # Slice the original data array once and then list comprehension filtering
            segment_slice = row_data[start_col:end_col]

            segment_tokens = [
                str(t) for t in segment_slice
                if t != 'GAP' and pd.notna(t)
            ]

            if len(segment_tokens) >= min_valid_tokens:
                current_phrases.append(' '.join(segment_tokens))

        if current_phrases:
             witness_phrases[witness_names[witness_idx]] = current_phrases

    return witness_phrases

def get_or_encode_embeddings(phrases, embedding_model, cache):
    """Encodes phrases, using the cache if available."""
    to_encode = []
    new_indices = []

    for i, phrase in enumerate(phrases):
        if phrase not in cache:
            to_encode.append(phrase)
            new_indices.append(i)

    if to_encode:
        new_embeddings = embedding_model.encode(to_encode)
        for phrase, emb in zip(to_encode, new_embeddings):
            cache[phrase] = emb

    embeddings = np.array([cache[phrase] for phrase in phrases])
    return embeddings

def meso_semantic_similarity(
    df,
    embedding_model,
    window_size=3,
    rescale_low=0.7,
    rescale_high=1.0,
    min_valid_tokens=2,
    return_pairwise=False
):
    """
    Compute meso-level (phrase) semantic similarity.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned witness table
    embedding_model : sentence-transformers model
    window_size : int
        Max column span for phrase reconstruction
    rescale_low, rescale_high : float
        Rescale raw cosine similarities from [rescale_low, rescale_high] --> [0, 1]
    min_valid_tokens : int
        Min non-GAP tokens per phrase
    return_pairwise : bool
        If True, also returns flat pairwise witness similarities

    Returns
    -------
    if return_pairwise=False:
        (phrase_stats_df, witness_similarities_dict) 
    else:
        (phrase_stats_df, witness_similarities_dict, pairwise_df)
    """

    # Reconstruct phrases per witness
    witness_phrases = reconstruct_witness_phrases(
        df, window_size=window_size, min_valid_tokens=min_valid_tokens
    )

    if not witness_phrases:
        empty_df = pd.DataFrame(columns=[
            'phrase_index', 'avg_similarity', 'min_similarity',
            'max_similarity', 'comparison_count', 'sample_phrases'
        ])
        empty_dict = {}
        if return_pairwise:
            pairwise_empty = pd.DataFrame(columns=['witness_a', 'witness_b', 'similarity'])
            return empty_df, empty_dict, pairwise_empty
        return empty_df, empty_dict

    # Validate rescaling
    scale_range = rescale_high - rescale_low
    if scale_range <= 0:
        raise ValueError("rescale_high must be > rescale_low")

    phrase_embedding_cache = {}
    phrase_scores = []
    witness_similarities = defaultdict(list)

    # Get all unique phrase positions (indices) that exist across witnesses
    max_phrases = max(len(phrases) for phrases in witness_phrases.values())
    all_witnesses = list(witness_phrases.keys())

    if return_pairwise:
        pair_sim_sum = defaultdict(float)
        pair_sim_count = defaultdict(int)

    # Process each phrase index independently
    for phrase_idx in range(max_phrases):
        phrases_in_position = []
        witness_names_in_position = []

        for witness in all_witnesses:
            phrases = witness_phrases[witness]
            if phrase_idx < len(phrases):
                phrases_in_position.append(phrases[phrase_idx])
                witness_names_in_position.append(witness)

        if len(phrases_in_position) < 2:
            continue

        # Embed and compute similarities
        embeddings = get_or_encode_embeddings(
            phrases_in_position, embedding_model, phrase_embedding_cache
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.where(norms == 0, 1, norms)
        sims = np.dot(normed, normed.T)

        # Extract lower triangle
        tril_i, tril_j = np.tril_indices_from(sims, k=-1)
        raw_similarities = sims[tril_i, tril_j]

        # Rescale
        clipped = np.clip(raw_similarities, rescale_low, rescale_high)
        rescaled = (clipped - rescale_low) / scale_range

        phrase_scores.append({
            'phrase_index': phrase_idx,
            'avg_similarity': np.mean(rescaled) if rescaled.size > 0 else np.nan,
            'min_similarity': np.min(rescaled) if rescaled.size > 0 else np.nan,
            'max_similarity': np.max(rescaled) if rescaled.size > 0 else np.nan,
            'comparison_count': rescaled.size,
            'sample_phrases': phrases_in_position[:2]
        })

        # Store per-witness
        for idx, (i_local, j_local) in enumerate(zip(tril_i, tril_j)):
            w_i = witness_names_in_position[i_local]
            w_j = witness_names_in_position[j_local]
            sim_val = rescaled[idx]
            phrase_i = phrases_in_position[i_local]
            phrase_j = phrases_in_position[j_local]

            witness_similarities[w_i].append({
                'phrase_index': phrase_idx,
                'similarity': float(sim_val),
                'other_witness': w_j,
                'phrase': phrase_i,
                'other_phrase': phrase_j
            })
            witness_similarities[w_j].append({
                'phrase_index': phrase_idx,
                'similarity': float(sim_val),
                'other_witness': w_i,
                'phrase': phrase_j,
                'other_phrase': phrase_i
            })

        # Accumulate for pairwise mode
        if return_pairwise:
            for idx, (i_local, j_local) in enumerate(zip(tril_i, tril_j)):
                w_i = witness_names_in_position[i_local]
                w_j = witness_names_in_position[j_local]
                sim_val = rescaled[idx]
                # Canonical ordering
                if w_i > w_j:
                    w_i, w_j = w_j, w_i
                pair_key = (w_i, w_j)
                pair_sim_sum[pair_key] += sim_val
                pair_sim_count[pair_key] += 1

    phrase_stats_df = pd.DataFrame(phrase_scores)
    witness_sim_dict = dict(witness_similarities)

    if not return_pairwise:
        return phrase_stats_df, witness_sim_dict

    #  pairwise df
    pairwise_records = []
    all_pairs = set()
    n = len(all_witnesses)
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((all_witnesses[i], all_witnesses[j]))

    for (w_a, w_b) in all_pairs:
        if (w_a, w_b) in pair_sim_sum:
            avg_sim = pair_sim_sum[(w_a, w_b)] / pair_sim_count[(w_a, w_b)]
            pairwise_records.append({'witness_a': w_a, 'witness_b': w_b, 'similarity': avg_sim})
        else:
            pairwise_records.append({'witness_a': w_a, 'witness_b': w_b, 'similarity': np.nan})

    pairwise_df = pd.DataFrame(pairwise_records)

    return phrase_stats_df, witness_sim_dict, pairwise_df



#@title MACRO-LEVEL

def reconstruct_full_texts(df):
    data = df.values
    witness_names = df.index

    # Vectorized filtering
    mask = (data != 'GAP') & (~pd.isna(data))

    full_texts = {}
    for i, name in enumerate(witness_names):
        valid_tokens = data[i, mask[i]]
        text = ' '.join(str(token) for token in valid_tokens)
        full_texts[name] = text

    return full_texts


def macro_semantic_similarity(
    df,
    embedding_model,
    rescale_low=0.7,
    rescale_high=1.0
):
    """
    Computes macro-level (full-text) semantic similarity with optional rescaling.

    Similarities in [rescale_low, rescale_high] are linearly mapped to [0, 1].
    Values outside are clipped to the bounds before rescaling.
    """
    full_texts = reconstruct_full_texts(df)

    if len(full_texts) < 2:
        return pd.DataFrame()

    witnesses = list(full_texts.keys())
    texts = list(full_texts.values())

    # Batch encode all texts
    embeddings = np.array(embedding_model.encode(texts))

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.where(norms == 0, 1, norms)

    # Compute raw cosine similarity matrix
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # rescale entire similarity matrix
    scale_range = rescale_high - rescale_low
    if scale_range <= 0:
        raise ValueError("rescale_high must be > rescale_low")

    # Clip to [rescale_low, rescale_high], then rescale to [0, 1]
    clipped_sim_matrix = np.clip(similarity_matrix, rescale_low, rescale_high)
    rescaled_sim_matrix = (clipped_sim_matrix - rescale_low) / scale_range

    # Extract unique pairs using the rescaled matrix
    witness_pairs = []
    n = len(witnesses)

    for i in range(n):
        for j in range(i + 1, n):
            sim_value = float(rescaled_sim_matrix[i, j])  # or np.float32
            witness_pairs.append({
                'witness_a': witnesses[i],
                'witness_b': witnesses[j],
                'similarity': sim_value,
                'text_a_preview': texts[i][:50] + '...' if len(texts[i]) > 50 else texts[i],
                'text_b_preview': texts[j][:50] + '...' if len(texts[j]) > 50 else texts[j]
            })

    return pd.DataFrame(witness_pairs)



#@title STRUCTURAL
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from rapidfuzz.distance.JaroWinkler import normalized_similarity as jaro_winkler_similarity

def structural_similarity(df):
    """
    Compute structural similarity based on CollateX alignment patterns
    Returns: DataFrame with structural metrics
    """

    def col_similarity(tokens, ignore_case=True, strip_punct=True, ignore_empty=True):
        """
        Measures internal similarity of a CollateX collation slot.
        Returns float [0, 1]. 1.0 = perfect agreement, 0.0 = completely divergent.
        """
        # print(type(tokens))
        # print(tokens)
        # Filter empty/gap tokens (CollateX uses "", "-", or "∅" for missing witnesses)
        if ignore_empty:
            tokens = [t for t in tokens if (t and t.strip() not in {"", "-", "∅", "—", "GAP"} and pd.notna(t))]

        if len(tokens) <= 1:
            return 1.0

        # Deduplicate (collation slots often have many exact matches)
        unique = list(set(tokens))
        if len(unique) == 1:
            return 1.0

        # Average pairwise similarity (O(U²) where U = unique tokens)
        # Jaro-Winkler is a standard metric for short textual variants
        total_sim = 0.0
        pairs = 0
        n = len(tokens)
        for i in range(n):
            for j in range(i + 1, n):
                # jaro_winkler_similarity already normalizes to 0-1
                total_sim += jaro_winkler_similarity(tokens[i], tokens[j])
                pairs += 1

        # print(total_sim / pairs)
        return total_sim / pairs
    structural_metrics = []

    # Column level structural analysis
    for col_idx, column_name in enumerate(df.columns):
        column_data = df[column_name]

        # Count gaps and unique tokens
        gap_count = sum(1 for token in column_data if token == 'GAP')
        unique_tokens = set(token for token in column_data if token != 'GAP' and pd.notna(token))

        structural_metrics.append({
            'column_index': col_idx,
            'column_name': column_name,
            'gap_freq': 1 - (gap_count / len(column_data)),
            'struct_score': col_similarity(column_data),
            'unique_tokens_count': len(unique_tokens),
            'is_identical': len(unique_tokens) <= 1,  # All witnesses have same token (or gaps)
            'sample_tokens': list(unique_tokens)[:3] if unique_tokens else []
        })

    # Witness-pair structural analysis
    witness_pair_metrics = []
    for witness_a, witness_b in combinations(df.index, 2):
        agreement_count = 0
        total_comparable = 0

        for col in df.columns:
            # Access the scalar value from the DataFrame
            token_a = df.loc[witness_a, col]
            token_b = df.loc[witness_b, col]

            # Check if the value is a Series and extract the scalar if necessary
            if isinstance(token_a, pd.Series):
                token_a = token_a.iloc[0]
            if isinstance(token_b, pd.Series):
                token_b = token_b.iloc[0]

            # Only compare when both have non GAP tokens
            if token_a != 'GAP' and token_b != 'GAP' and pd.notna(token_a) and pd.notna(token_b):
                total_comparable += 1
                if token_a == token_b:  # Character level identity
                    agreement_count += 1

        structural_agreement = agreement_count / total_comparable if total_comparable > 0 else 0

        witness_pair_metrics.append({
            'witness_a': witness_a,
            'witness_b': witness_b,
            'structural_agreement': structural_agreement,
            'comparable_positions': total_comparable
        })

    return {
        'column_metrics': pd.DataFrame(structural_metrics),
        'witness_pair_metrics': pd.DataFrame(witness_pair_metrics)
    }



#@title combined_similarity_assessment(df, embedding_model)

def combined_similarity_assessment(df, embedding_model):
    """
    Run complete multi-level similarity analysis
    Returns: Comprehensive results dictionary
    """
    print("Computing Micro-level (column) similarities...")
    micro_df, mirco_pairs = micro_semantic_similarity(df, embedding_model, return_pairwise=True)

    print("Computing Meso-level (phrase) similarities...")
    meso_df, meso_witsim, meso_pairs = meso_semantic_similarity(df, embedding_model, return_pairwise=True)

    print("Computing Macro-level (full text) similarities...")
    macro_df = macro_semantic_similarity(df, embedding_model)

    print("Computing Structural similarities...")
    structural_results = structural_similarity(df)

    # Calculate overall scores
    overall_micro = micro_df['avg_similarity'].mean()
    overall_meso = meso_df['avg_similarity'].mean() if len(meso_df) > 0 else np.nan
    overall_macro = macro_df['similarity'].mean()
    overall_structural = structural_results['witness_pair_metrics']['structural_agreement'].mean()

    # Identify interesting discrepancies
    interesting_columns = micro_df[
        (micro_df['avg_similarity'] < 0.3) |  # Low semantic similarity
        (micro_df['comparison_count'] > 2)    # Multiple variants to compare
    ].sort_values('avg_similarity')

    return {
        'micro_level': micro_df,
        'meso_level':  meso_df,
        'macro_level': macro_df,
        'structural':  structural_results,
        'summary_metrics': {
            'overall_micro_similarity': overall_micro,
            'overall_meso_similarity':  overall_meso,
            'overall_macro_similarity': overall_macro,
            'overall_structural_similarity': overall_structural,
            'semantic_structural_discrepancy': overall_macro - overall_structural
        },
        'interesting_variants': interesting_columns,
        'full_texts': reconstruct_full_texts(df),
        'meso_witsim': meso_witsim,
        'meso_pairs': meso_pairs,
        'mirco_pairs': mirco_pairs
    }



#@title analyze_witness_similarity_optimized(phrase_scores_df, witness_similarities)
def analyze_witness_similarity_optimized(phrase_scores_df, witness_similarities):
    """
    Optimized witness similarity analysis using vectorized operations
    """
    witness_analysis = {}

    for witness, comparisons in witness_similarities.items():
        if not comparisons:
            continue

        comp_df = pd.DataFrame(comparisons)

        # precompute quantiles once
        similarity_series = comp_df['similarity']
        high_sim_threshold = similarity_series.quantile(0.75)
        low_sim_threshold = similarity_series.quantile(0.25)

        high_sim_mask = similarity_series >= high_sim_threshold
        low_sim_mask = similarity_series <= low_sim_threshold

        # groupby with named aggregation
        phrase_stats = comp_df.groupby('phrase_index').agg(
            avg_similarity=('similarity', 'mean'),
            min_similarity=('similarity', 'min'),
            max_similarity=('similarity', 'max'),
            comparison_count=('similarity', 'count'),
            compared_with=('other_witness', lambda x: x.tolist())
        ).round(3)

        witness_analysis[witness] = {
            'overall_avg_similarity': similarity_series.mean(),
            'phrase_stats': phrase_stats,
            'high_similarity_phrases': comp_df[high_sim_mask],
            'low_similarity_phrases': comp_df[low_sim_mask],
            'total_comparisons': len(comparisons),
            'high_sim_threshold': high_sim_threshold,  # useful for interpretation
            'low_sim_threshold': low_sim_threshold
        }

    return witness_analysis




#@title visualize

#@title 1. plot_variant_significance_scatter(results, figsize=(10, 8), label_all=False)

from scipy.stats import linregress

def plot_gap_freq_significance_scatter(results, figsize=(10, 8), label_all=False, label_top=True):
    """
    SCATTER PLOT: Which variant columns represent meaningful differences
    X-axis: Structural gap frequency | Y-axis: Semantic similarity
    """
    # merge micro-level semantic with structural column metrics
    micro_df = results['micro_level'].dropna()
    structural_df = results['structural']['column_metrics']

    # create combined dataframe
    combined = pd.merge(
        micro_df[['column_index', 'avg_similarity', 'tokens']],
        structural_df[['column_index', 'gap_freq', 'struct_score', 'unique_tokens_count']],
        on='column_index'
    )

    fig, ax = plt.subplots(figsize=figsize)

    # combined['gap_freq'] = 1 - combined['gap_freq'] 

    x = combined['gap_freq']
    y = combined['avg_similarity']

    # create scatter plot
    scatter = ax.scatter(
        x=x,
        y=y,
        c=combined['unique_tokens_count'],
        s=100,
        alpha=0.7,
        cmap='viridis'
    )

    #  Add linear trend line and R^2 
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    print(f"R^2 value is: {r_squared:.4f}")

    # generate trend line
    x_trend = np.linspace(x.min(), x.max(), 100)
    y_trend = slope * x_trend + intercept
    # ax.plot(x_trend, y_trend, color='red', linestyle='--', linewidth=2, label=f'$R^2 = {r_squared:.3f}$')

    # add R^2 as legend or text
    # ax.legend(loc='lower right')

    # Label each point
    for idx, row in combined.iterrows():
        if label_all:
            ax.annotate(f"col{row['column_index']}",
                    (row['gap_freq'], row['avg_similarity']),
                    xytext=(2, 10), textcoords='offset points', fontsize=13)
        elif label_top:
            if row['gap_freq'] > 0.9 or row['avg_similarity'] > 0.85 or (row['gap_freq'] < 0.3 and row['avg_similarity'] < 0.53):
                ax.annotate(f"col{row['column_index']}",
                        (row['gap_freq'], row['avg_similarity']),
                        xytext=(2, 10), textcoords='offset points', fontsize=13)

    ax.set_xlabel('Structural Gap Frequency\n(Low = major textual omissions)')
    ax.set_ylabel('Average Semantic Similarity Per Collated Group\n(Low = major meaning differences)')
    ax.set_title('Collation vs Micro-Semantic Similarity\nHow Well Does CollateX Group Tokens by Meaning?')

    # colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Unique Tokens', rotation=270, labelpad=15)

    # optional thresholds
    ax.axhline(y=0.7, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=0.9, color='red', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()

    return combined


def plot_jaro_winkler_significance_scatter(results, figsize=(10, 8), label_all=False, label_top=True):
    """
    SCATTER PLOT: Which variant columns represent meaningful differences
    X-axis: Structural completeness (struct_score)
    Y-axis: Semantic similarity
    Point size: Number of unique tokens (larger = more lexical diversity)
    All points use a uniform color for clarity.
    """
    micro_df = results['micro_level'].dropna()
    structural_df = results['structural']['column_metrics']

    combined = pd.merge(
        micro_df[['column_index', 'avg_similarity', 'tokens']],
        structural_df[['column_index', 'gap_freq', 'struct_score', 'unique_tokens_count']],
        on='column_index'
    )

    # combined['gap_freq'] = 1 - combined['gap_freq'] 
    fig, ax = plt.subplots(figsize=figsize)

    x = combined['struct_score']
    y = combined['avg_similarity']
    # sizes_raw = combined['unique_tokens_count']

    # Normalize size for visual clarity (avoid too small/too large)
    # Map to range [30, 400] — adjust as needed
    # size_min, size_max = mini, maxi
    # sizes = np.interp(sizes_raw, (sizes_raw.min(), sizes_raw.max()), (size_min, size_max))

    # Uniform color (e.g., steelblue, darkgray, or tab:blue)
    scatter = ax.scatter(
        x=x,
        y=y,
        s=150,
        alpha=0.7,
        color='steelblue',        # uniform color
        edgecolors='k',           # subtle outline for separation
        linewidth=0.3
    )

    # Trend line
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    print(f"R^2 value is: {r_squared:.4f}")

    x_trend = np.linspace(x.min(), x.max(), 100)
    y_trend = slope * x_trend + intercept
    # ax.plot(x_trend, y_trend, color='red', linestyle='--', linewidth=1.5, label=f'$R^2 = {r_squared:.3f}$')
    # ax.legend(loc='lower right')

    # Labels
    for idx, row in combined.iterrows():
        if label_all:
            ax.annotate(f"col{row['column_index']}",
                    (row['struct_score'], row['avg_similarity']),
                    xytext=(2, 10), textcoords='offset points', fontsize=13)
        elif label_top:
            if row['struct_score'] > 0.9 or row['avg_similarity'] > 0.85 or (row['struct_score'] < 0.3 and row['avg_similarity'] < 0.53):
                ax.annotate(f"col{row['column_index']}",
                        (row['struct_score'], row['avg_similarity']),
                        xytext=(2, 10), textcoords='offset points', fontsize=13)

    # Axes and thresholds
    ax.set_xlabel('Structural Similarity\n(Averaged Jaro Winkler Similarity of Collation Group)')
    ax.set_ylabel('Average Semantic Similarity Per Collated Group\n(Averaged Semantic Textual Similarity from LLM)')
    ax.set_title('CollateX Text Similarity vs Micro-Semantic Similarity\nHow Similar Are CollateX Groupings Across Letter and Meaning?')

    ax.axhline(y=0.9, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=0.9, color='red', linestyle=':', alpha=0.5)


    # ax.text(0.87, 0.02, f'Size ∝\nNumber of\nUnique Tokens', transform=ax.transAxes,
    #         fontsize=10, ha='left', va='bottom', color='gray')

    plt.tight_layout()
    plt.show()

    return combined
    

#@title 2. plot_focused_variant_significance(results, top_n=15)
def plot_focused_variant_significance(results, top_n=15):
    """
    Focused view: Show only the most significant variants with clear interpretation
    """
    #Print
    # print("\nVARIANT SIGNIFICANCE ANALYSIS")
    # print("-" * 40)
    # merge and prep data
    micro_df = results['micro_level'].dropna()
    structural_df = results['structural']['column_metrics']

    combined = pd.merge(
        micro_df[['column_index', 'avg_similarity', 'tokens']],
        structural_df[['column_index', 'struct_score', 'unique_tokens_count']],
        on='column_index'
    )

    # calculate variant significance score
    combined['significance_score'] = (1 - combined['avg_similarity']) * combined['struct_score']
    combined = combined.nlargest(top_n, 'significance_score')

    fig, ax = plt.subplots(figsize=(12, 8))

    combined['struct_score'] = combined['struct_score']
    combined['significance_score'] = 1 - combined['significance_score']  # Invert

    # Create bubbles: size = number of unique tokens, color = semantic similarity
    scatter = ax.scatter(
        x=combined['struct_score'],
        y=combined['avg_similarity'],
        s=combined['unique_tokens_count'] * 50,  # Size by complexity
        c=combined['significance_score'],
        cmap='RdYlGn',  # red for low similarity, Green for high
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    #LABLING ALL VARIANTS
    # Label only the most significant points
    top_labels = combined.nsmallest(5, 'significance_score')
    for idx, row in top_labels.iterrows():
    # for idx, row in combined.iterrows():
        # if row['significance_score'] > 0.1:  # Only label significant variants
        ax.annotate(f"Col{row['column_index']}",
                    (row['struct_score'], row['significance_score']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # # Collect annotations for adjustment
    # texts = []
    # for idx, row in combined.iterrows():
    #     txt = ax.annotate(f"Col{row['column_index']}",
    #                     (row['struct_score'], row['significance_score']),
    #                     xytext=(8, 8), textcoords='offset points',
    #                     fontsize=9, fontweight='bold',
    #                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    #     texts.append(txt)

    # # Adjust labels to avoid overlap
    # adjust_text(texts,
    #             arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5),
    #             ax=ax)

    ax.set_xlabel('Structural Complexity\nMinor gaps ——— Major gaps')
    ax.set_ylabel('Variant Significance Score\nMinor ——— Major')
    ax.set_title('Top 15 Most Significant Variants\n(Larger bubbles = more unique readings)')

    # Colorbar for semantic similarity
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Semantic Similarity\nDifferent ——— Similar', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.show()

    # # Print some insights
    # print("Insights - Variant Significance")
    # print("=" * 60)
    # print("Highest Scores: ")
    # top_5 = combined.nlargest(5, 'significance_score')
    # for i, (_, variant) in enumerate(top_5.iterrows(), 1):
    #     print(f"\n{i}. COLUMN {variant['column_index']} (Priority {i}):")
    #     print(f"   Significance Score: {variant['significance_score']:.3f}")
    #     print(f"   Semantic Similarity: {variant['avg_similarity']:.3f}")
    #     print(f"   Gap Frequency: {variant['struct_score']:.3f}")
    #     print(f"   Unique Readings: {variant['unique_tokens_count']}")
    #     print(f"   Sample Tokens: {variant['tokens'][:3]}...")  # Show first 3 tokens

    # print("Lowest Scores: ")
    # bot_5 = combined.nsmallest(5, 'significance_score')
    # for i, (_, variant) in enumerate(bot_5.iterrows(), 1):
    #     print(f"\n{i}. COLUMN {variant['column_index']} (Priority {i}):")
    #     print(f"   Significance Score: {variant['significance_score']:.3f}")
    #     print(f"   Semantic Similarity: {variant['avg_similarity']:.3f}")
    #     print(f"   Gap Frequency: {variant['struct_score']:.3f}")
    #     print(f"   Unique Readings: {variant['unique_tokens_count']}")
    #     print(f"   Sample Tokens: {variant['tokens'][:3]}...")  # Show first 3 tokens


    return combined


#@title 3. plot_witness_similarity_matrix(results)
def plot_witness_similarity_matrix(results, low=0.3, high=0.7):
    """
    Clean similarity matrix with clear clustering and interpretation
    """
    # print
    # print("\nWITNESS RELATIONSHIP ANALYSIS")
    # print("-" * 40)
    macro_df = results['macro_level']

    # Create similarity matrix
    witnesses = sorted(set(macro_df['witness_a']).union(set(macro_df['witness_b'])))
    sim_matrix = pd.DataFrame(1.0, index=witnesses, columns=witnesses)

    for _, row in macro_df.iterrows():
        sim_matrix.loc[row['witness_a'], row['witness_b']] = row['similarity']
        sim_matrix.loc[row['witness_b'], row['witness_a']] = row['similarity']

    # Reorder by similarity (clustering)
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    # Convert similarity --> distance
    distance_matrix = 1 - sim_matrix.values

    # Convert to condensed form
    condensed_distance = squareform(distance_matrix, checks=False)

    linkage = hierarchy.linkage(condensed_distance, method='average')
    order = hierarchy.leaves_list(linkage)

    # linkage = hierarchy.linkage(1 - sim_matrix.values, method='average')
    # order = hierarchy.leaves_list(linkage)
    sim_matrix = sim_matrix.iloc[order, order]

    fig, ax = plt.subplots(figsize=(10, 8))

    data_min = sim_matrix.values.min()
    data_max = sim_matrix.values.max()
    # Create heatmap
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1.1, aspect='auto')

    # Customize ticks
    witness_labels = [w.replace('variant ', 'V') for w in sim_matrix.index]
    ax.set_xticks(range(len(witnesses)))
    ax.set_yticks(range(len(witnesses)))

    ax.set_xticklabels(witness_labels, rotation=45, ha='right')
    ax.set_yticklabels(witness_labels)

    # Add values only for significant similarities/differences
    for i in range(len(witnesses)):
        for j in range(len(witnesses)):
            similarity = sim_matrix.iloc[i, j]
            if i != j and (similarity < low or similarity > high):  # Only label notable values
                color = 'white' if similarity < 0.5 else 'black'
                ax.text(j, i, f'{similarity:.2f}', ha='center', va='center',
                       fontsize=8, color=color, fontweight='bold')


    ax.set_title('Semantic Similarity Matrix\n(Full Text STS from LLM)', fontsize=14, pad=20)
    # ax.set_title('Witness Matrix\n(Full Text Semantic Similarity)', fontsize=14, pad=20)
    plt.colorbar(im, ax=ax, label='Semantic Similarity')

    # Add interpretation
    cluster_threshold = 0.8
    clusters = []
    visited = set()

    for i, witness in enumerate(sim_matrix.index):
        if witness not in visited:
            cluster = [witness]
            visited.add(witness)
            for j, other_witness in enumerate(sim_matrix.index[i+1:], i+1):
                if sim_matrix.iloc[i, j] > cluster_threshold:
                    cluster.append(other_witness)
                    visited.add(other_witness)
            if len(cluster) > 1:
                clusters.append(cluster)


    # print("\n WITNESS CLUSTERING ANALYSIS")
    # print("=" * 50)

    # if clusters:
    #     print(f"Detected {len(clusters)} natural witness groups (similarity > {cluster_threshold}):")
    #     for i, cluster in enumerate(clusters, 1):
    #         cluster_names = [w.replace('variant ', 'V') for w in cluster]
    #         avg_similarity = sim_matrix.loc[cluster, cluster].mean().mean()
    #         print(f"  Group {i}: {cluster_names} (avg similarity: {avg_similarity:.3f})")
    # else:
    #     print("No strong clustering detected - witnesses form a continuum")

    # Identify outliers
    avg_similarities = sim_matrix.mean(axis=1)
    outlier_threshold = avg_similarities.quantile(0.25)
    outliers = avg_similarities[avg_similarities < outlier_threshold]

    # if len(outliers) > 0:
    #     print(f"\nPOTENTIAL OUTLIERS (low average similarity):")
    #     for witness, similarity in outliers.items():
    #         print(f"  {witness.replace('variant ', 'V')}: {similarity:.3f}")

    plt.tight_layout()
    plt.show()

    return sim_matrix



#@title 3.2 plot_structural_similarity_heatmap(results)
from scipy.spatial.distance import squareform

def plot_structural_similarity_heatmap(results, low=0.3, high=0.7):
    """
    Plot a clustered heatmap of structural similarity between witnesses,
    based on token-level agreement in aligned columns (CollateX output).

    Parameters:
        structural_results: dict returned by structural_similarity()
        low, high: thresholds for annotating cells (only show values < low or > high)
    """
    # print("\nSTRUCTURAL SIMILARITY ANALYSIS")
    # print("-" * 40)

    structural_results = results['structural']
    pair_df = structural_results['witness_pair_metrics']

    # Get all unique witnesses
    witnesses = sorted(set(pair_df['witness_a']).union(set(pair_df['witness_b'])))
    n = len(witnesses)

    # Initialize similarity matrix with 1.0 on diagonal (self-similarity)
    sim_matrix = pd.DataFrame(1.0, index=witnesses, columns=witnesses)

    # Fill in pairwise structural agreement
    for _, row in pair_df.iterrows():
        a, b = row['witness_a'], row['witness_b']
        sim_matrix.loc[a, b] = row['structural_agreement']
        sim_matrix.loc[b, a] = row['structural_agreement']

    #Clustering
    # Convert to distance matrix
    distance_matrix = 1 - sim_matrix.values
    condensed_distance = squareform(distance_matrix, checks=False)
    linkage = hierarchy.linkage(condensed_distance, method='average')
    order = hierarchy.leaves_list(linkage)

    # Reorder matrix
    sim_matrix = sim_matrix.iloc[order, order]

    #Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1.0, aspect='auto')

    # Labels
    short_labels = [w.replace('variant ', 'V') for w in sim_matrix.index]
    tick_positions = np.arange(len(short_labels))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.set_yticklabels(short_labels)

    # Annotate notable cells
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # skip diagonal
            val = sim_matrix.iloc[i, j]
            if val < low or val > high:
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')

    ax.set_title('Structural Agreement Matrix\n(Token Identity from Collation)', fontsize=14, pad=20)
    # ax.set_title('Structural Agreement Matrix\n(Token Identity in Aligned Columns)', fontsize=14, pad=20)
    plt.colorbar(im, ax=ax, label='Structural Agreement')

    plt.tight_layout()
    plt.show()


    #@title 4. plot_witness_similarity_scatter(results)

def plot_witness_similarity_scatter(results):
    """
    Scatter plot: Structural vs Semantic similarity for each witness pair.
    Highlights cases where semantic and structural judgments diverge.

    Parameters:
    -----------
    results : dict
        Must contain:
        - 'macro_level': DataFrame with columns ['witness_a', 'witness_b', 'similarity']
        - 'structural': dict with key 'witness_pair_metrics' -> DataFrame with
          ['witness_a', 'witness_b', 'structural_agreement']

    Returns:
    --------
    pd.DataFrame
        Comparison DataFrame with columns: pair, semantic, structural, discrepancy
    """
    def _shorten_pair_label(label):
        label = str(label)
        # # Try to match common patterns: word + space + number
        # match = re.match(r'^(?:[Vv]ariant|[Cc]olumn|[Ww]itness)\s*(\d+)$', label)
        # if match:
        #     return f"v{match.group(1)}"  # or use first letter: label[0].lower() + match.group(1)
        if label.startswith('variant '):
            return label.replace('variant ', 'V')
        return label

    macro_df = results['macro_level'].copy()
    structural_df = results['structural']['witness_pair_metrics'].copy()

    # Helper to create order-invariant pair key
    def _canonical_pair(a, b):
        return '-'.join(sorted([str(a), str(b)]))

    # Apply canonical key to both DataFrames
    macro_df['pair_key'] = macro_df.apply(
        lambda row: _canonical_pair(row['witness_a'], row['witness_b']), axis=1
    )
    structural_df['pair_key'] = structural_df.apply(
        lambda row: _canonical_pair(row['witness_a'], row['witness_b']), axis=1
    )

    # Merge on canonical pair key
    merged = pd.merge(
        macro_df[['pair_key', 'witness_a', 'witness_b', 'similarity']],
        structural_df[['pair_key', 'structural_agreement']],
        on='pair_key',
        how='inner'
    )

    if merged.empty:
        raise ValueError("No matching witness pairs found between semantic and structural results.")

    # Build comparison DataFrame
    comp_df = pd.DataFrame({
        'pair': merged['pair_key'],
        'semantic': merged['similarity'],
        'structural': merged['structural_agreement'],
        'discrepancy': merged['similarity'] - merged['structural_agreement']
    })

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        x=comp_df['structural'],
        y=comp_df['semantic'],
        c=comp_df['discrepancy'],
        # cmap='RdYlBu_r', #reverse so red is high discrenensey
        cmap='RdYlBu',
        s=100,
        alpha=0.7,
        edgecolors='black'
    )

    # Identity line (perfect agreement)
    max_val = max(comp_df['structural'].max(), comp_df['semantic'].max())
    min_val = min(comp_df['structural'].min(), comp_df['semantic'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect agreement')

    # Optional: thresholds
    ax.axhline(y=0.7, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=0.7, color='red', linestyle=':', alpha=0.5)

    ax.set_xlabel('Structural Similarity')
    ax.set_ylabel('Semantic Similarity')
    # ax.set_title('\nText Relationships: Structural vs Semantic Similarity')
    ax.set_title('Collation vs Macro-Semantic Similarity\nHow Well Does CollateX Align Witnesses by Meaning?')
    ax.legend()

    plt.colorbar(scatter, ax=ax, label='Discrepancy\n(Semantic − Structural)')
    plt.tight_layout()


    # Decide how many top points to label in each direction
    n_top = 5

    # Get top N most positive (semantic >> structural)
    top_positive = comp_df.nlargest(n_top, 'discrepancy')

    # Get top N most negative (structural >> semantic)
    top_negative = comp_df.nsmallest(n_top, 'discrepancy')

    # Combine without duplicates
    high_extreme = pd.concat([top_positive, top_negative]).drop_duplicates()

    # Collect labels
    texts = []
    for _, row in high_extreme.iterrows():
        short_label = _shorten_pair_label(row['pair'])  # Apply shortening here
        txt = ax.annotate(
            short_label,
            xy=(row['structural'], row['semantic']),
            fontsize=9,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
        )
        texts.append(txt)

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5), ax=ax)

    plt.show()

    return comp_df


#@title 6. plot_similarity_levels_comparison(results)
def plot_similarity_levels_comparison(results):
    """
    Simple bar chart comparing similarity across analysis levels
    Shows how granularity affects similarity measurements
    """
    summary = results['summary_metrics']

    levels = ['Micro (Column)', 'Meso (Phrase)', 'Macro (Full Text)', 'Structural']
    similarities = [
        summary['overall_micro_similarity'],
        summary['overall_meso_similarity'] if not np.isnan(summary['overall_meso_similarity']) else 0,
        summary['overall_macro_similarity'],
        summary['overall_structural_similarity']
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(levels, similarities, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)

    # Add value labels on bars
    for bar, value in zip(bars, similarities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Average Similarity Score')
    ax.set_title('Similarity Comparison Across Analysis Levels\n' +
                'Shows how granularity affects textual relationships')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    return dict(zip(levels, similarities))


#@title 7. plot_semantic_clustering_dendrogram(results, language="en",threshold=0.5, shorten_labels=True, variant_prefix="variant ")

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_SUPPORT = True
except ImportError:
    HAS_ARABIC_SUPPORT = False


def plot_semantic_clustering_dendrogram(
    results,
    language="en",
    threshold=0.5,
    shorten_labels=True,
    variant_prefix="variant ",
    level="Macro"
):
    """
    Plot dendrogram of witnesses based on semantic similarity.

    Parameters
    ----------
    results : dict
        Must contain 'macro_level' DataFrame with columns:
        ['witness_a', 'witness_b', 'similarity']
    language : str, optional
        Language code (e.g., 'en', 'ar'). If 'ar', applies BiDi rendering.
    threshold : float, optional
        Similarity threshold for horizontal cutoff line (0.0–1.0).
    shorten_labels : bool, optional
        If True, replace 'variant X' with 'VX' in labels.
    variant_prefix : str, optional
        Prefix to replace when shortening labels.

    Returns
    -------
    linkage_matrix : ndarray
        The linkage matrix from hierarchical clustering.
    """
    sim_threshold = threshold
    
    macro_df = results['macro_level']

    if macro_df.empty:
        raise ValueError("macro_level DataFrame is empty.")

    # Get all unique witnesses
    witnesses = sorted(set(macro_df['witness_a']) | set(macro_df['witness_b']))
    n = len(witnesses)

    # Initialize similarity matrix
    sim_matrix = pd.DataFrame(np.nan, index=witnesses, columns=witnesses)

    # Fill known similarities
    for _, row in macro_df.iterrows():
        a, b, sim = row['witness_a'], row['witness_b'], row['similarity']
        sim_matrix.loc[a, b] = sim
        sim_matrix.loc[b, a] = sim

    # Set diagonal to 1.0 (self-similarity)
    # np.fill_diagonal(sim_matrix.values, 1.0) #read-only error
    temp_arr = sim_matrix.to_numpy().copy()
    np.fill_diagonal(temp_arr, 1.0)
    sim_matrix[:] = temp_arr

    # Check for missing pairs
    if sim_matrix.isna().any().any():
        missing_pairs = list(zip(*np.where(sim_matrix.isna())))
        missing_count = len(missing_pairs)
        raise ValueError(
            f"Missing semantic similarities for {missing_count} witness pairs. "
            "Ensure macro_level contains all unordered pairs."
        )

    # Convert to distance
    distance_matrix = 1.0 - sim_matrix.values

    # Compute linkage
    condensed = squareform(distance_matrix)
    linkage_matrix = hierarchy.linkage(condensed, method='average')
    # linkage_matrix = hierarchy.linkage(distance_matrix, method='average')

    # Prepare labels
    if shorten_labels:
        labels = [w.replace(variant_prefix, 'V') for w in witnesses]
    else:
        labels = witnesses[:]

    # Handle Arabic BiDi if needed
    if language == 'ar':
        if not HAS_ARABIC_SUPPORT:
            raise ImportError(
                "Arabic support requires 'arabic_reshaper' and 'python-bidi'."
            )
        bidi_labels = [
            get_display(arabic_reshaper.reshape(label)) for label in labels
        ]
        plot_labels = bidi_labels
    else:
        plot_labels = labels

    # Plot
    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 8))
    hierarchy.dendrogram(
        linkage_matrix,
        labels=plot_labels,
        orientation='top',
        leaf_rotation=45,
        ax=ax
    )

    # Threshold line
    distance_threshold = 1.0 - sim_threshold
    # ax.axhline(
    #     y=distance_threshold,
    #     color='red',
    #     linestyle='--',
    #     alpha=0.7,
    #     label=f'Similarity > {sim_threshold:.2f}'
    # )

    ax.set_ylabel('Semantic Distance\n(1 − Similarity)')

    ax.set_title(f'Witness Clustering Based on {level}-Semantic Similarity\n'
                 'Lower branches = closer semantically')

    # ax.set_title(f'Witness Clustering Based on {level}-Semantic Similarity\n'
    #              '*Processed V1 and V9 (bāraka to harama)')
    plt.tight_layout()
    plt.show()

    # # Optional: report cluster count
    # clusters = hierarchy.fcluster(linkage_matrix, distance_threshold, criterion='distance')
    # cluster_groups = {}
    # for i, cid in enumerate(clusters):
    #     name = labels[i]
    #     cluster_groups.setdefault(cid, []).append(name)

    # print("SEMANTIC CLUSTERING ANALYSIS")
    # print("=" * 50)
    # print(f"Clustering threshold: {distance_threshold:.1f} (max allowed disagreement)")
    # print(f"\nDetected {len(cluster_groups)} semantic groups:")
    # for cid, members in sorted(cluster_groups.items()):
    #     print(f"Group {cid}: {members}")

    # n_clusters = len(set(clusters))
    # print(f"Number of semantic clusters (similarity > {sim_threshold}): {n_clusters}")

    return linkage_matrix



#@title 8. plot_structural_dendrogram(results, original_df, threshold=0.3)
def plot_structural_dendrogram(results, original_df, threshold=0.5):
    """
    Plot dendrogram of witnesses based on structural alignment patterns.
    Also prints clustering analysis.
    Parameters:
        results : dict
        Must contain 'macro_level' DataFrame with columns:
        ['witness_a', 'witness_b', 'similarity']
        original_df: dataframe
        a dataframe of the original collation.
        threshold : float, optional
        Similarity threshold for horizontal cutoff line (0.0–1.0).
    Returns:
        linkage_matrix, distance_matrix, witnesses
    """
    sim_threshold = threshold

    def compute_structural_distance_matrix(original_df):
        """
        Compute pairwise structural distance matrix based on alignment disagreements.
        Returns:
            distance_matrix (np.ndarray): symmetric matrix of shape (n, n)
            witnesses (list): list of witness names in order
        """

        witnesses = original_df.index.tolist()
        n = len(witnesses)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    disagreements = 0
                    total_comparable = 0
                    for col_idx in range(original_df.shape[1]):
                        token_i = original_df.iloc[i, col_idx]
                        token_j = original_df.iloc[j, col_idx]
                        if (
                            token_i != 'GAP' and token_j != 'GAP' and
                            pd.notna(token_i) and pd.notna(token_j)
                        ):
                            total_comparable += 1
                            if token_i != token_j:
                                disagreements += 1
                    distance_matrix[i, j] = (
                        disagreements / total_comparable if total_comparable > 0 else 1.0
                    )
        return distance_matrix, witnesses


    distance_matrix, witnesses = compute_structural_distance_matrix(original_df)
    condensed = squareform(distance_matrix)
    linkage_matrix = hierarchy.linkage(condensed, method='average')
    # linkage_matrix = hierarchy.linkage(distance_matrix, method='average')

    # Create labels
    labels = [w.replace('variant ', 'V') for w in witnesses]

    fig, ax = plt.subplots(figsize=(10, max(6, len(witnesses) * 0.3)))
    hierarchy.dendrogram(
        linkage_matrix,
        labels=labels,
        orientation='top',
        ax=ax
    )
    distance_threshold = 1.0 - sim_threshold
    # ax.axhline(
    #     y=distance_threshold,
    #     color='red',
    #     linestyle='--',
    #     alpha=0.7,
    #     label=f'Similarity > {sim_threshold:.2f}'
    # )
    ax.set_xlabel('Structural Distance\n(Based on alignment disagreements)')
    ax.set_title('Witness Clustering by Structural Alignment\n'
                 'Lower branches = closer collation')
    # ax.set_title('Witness Clustering by Structural Alignment\n'
    #              '*Processed V1 and V9 (bāraka to harama)')
    plt.tight_layout()
    plt.show()

    # # Clustering analysis
    # clusters = hierarchy.fcluster(linkage_matrix, distance_threshold, criterion='distance')
    # cluster_groups = {}
    # for i, cid in enumerate(clusters):
    #     name = labels[i]
    #     cluster_groups.setdefault(cid, []).append(name)

    # print("STRUCTURAL CLUSTERING ANALYSIS")
    # print("=" * 50)
    # print(f"Clustering threshold: {distance_threshold:.1f} (max allowed disagreement)")
    # print(f"\nDetected {len(cluster_groups)} structural groups:")
    # for cid, members in sorted(cluster_groups.items()):
    #     print(f"Group {cid}: {members}")

    # avg_dist = distance_matrix[np.triu_indices_from(distance_matrix, k=1)].mean()
    # print(f"\nAverage structural distance: {avg_dist:.3f}")

    return linkage_matrix, distance_matrix, witnesses


