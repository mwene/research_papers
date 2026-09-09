import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# ====================================================================
# 1. CONFIGURATION & PARAMETERS
# ====================================================================
@dataclass
class FamilyParameters:
    """Family-specific parameters for divergence estimation."""
    retention_rate: float  # r = retention rate per millennium
    burst_correction: float  # b = burst correction factor
    sound_shifts: Dict[str, str]  # systematic consonant correspondences
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            "retention_rate": self.retention_rate,
            "burst_correction": self.burst_correction,
            "sound_shifts": self.sound_shifts,
            "description": self.description
        }

# Pre-configured parameter sets for major language families
PARAMETERS = {
    "bantu": FamilyParameters(
        retention_rate=0.75,
        burst_correction=0.67,
        sound_shifts={"r": "l", "l": "r", "g": "k", "k": "g", "b": "p", "p": "b"},
        description="Bantu and Niger-Congo (calibrated on Gikuyu-Kinyarwanda, ~1,500 yrs)"
    ),
    "niger_congo": FamilyParameters(
        retention_rate=0.75,
        burst_correction=0.67,
        sound_shifts={"r": "l", "l": "r", "g": "k", "k": "g"},
        description="Niger-Congo (calibrated on Swadesh-archaeological alignment)"
    ),
    "indo_european": FamilyParameters(
        retention_rate=0.805,
        burst_correction=1.0,
        sound_shifts={"p": "f", "t": "th", "k": "h", "d": "t"},
        description="Indo-European (standard rate from Lees, 1953)"
    ),
    "afro_asiatic": FamilyParameters(
        retention_rate=0.776,
        burst_correction=1.0,
        sound_shifts={"\u1e0f": "t", "q": "k", "\u1e2b": "h", "\u1e25": "h", "\u02bf": ""},
        description="Afro-Asiatic (calibrated on Egyptian-Coptic, ~2,510 yrs)"
    ),
    "ethio_semitic": FamilyParameters(
        retention_rate=0.743,
        burst_correction=1.0,
        sound_shifts={"\u1e25": "h", "\u02bf": "", "\u0121": "g"},
        description="Ethio-Semitic (in-situ diversification, Amharic-Tigrinya calibration)"
    ),
}

# ====================================================================
# 2. CONSONANT EXTRACTION
# ====================================================================
def extract_consonants(word: str) -> str:
    """
    Extract the consonantal skeleton from a word.
    Strips vowels, tones, diacritics, and retains only consonants.
    """
    if not isinstance(word, str):
        return ""
    # Normalize to lowercase
    word = word.lower().strip()
    # Remove vowels (a, e, i, o, u, y in some cases, but y can be consonant)
    # Standard approach: remove all vowels explicitly
    vowels = set('aeiou')
    consonants = ''.join([ch for ch in word if ch not in vowels and ch.isalpha()])

    # Remove diacritics (e.g., macrons, accents)
    # We just keep ASCII letters for simplicity
    consonants = re.sub(r'[^a-z]', '', consonants)
    return consonants

def normalize_consonants(cons: str, shifts: Dict[str, str]) -> str:
    """Apply systematic sound shifts to a consonant skeleton."""
    result = cons
    for from_c, to_c in shifts.items():
        result = result.replace(from_c, to_c)
    return result

def share_any_consonant(cons1: str, cons2: str) -> bool:
    """Check if two consonant skeletons share at least one consonant."""
    return bool(set(cons1) & set(cons2))

# ====================================================================
# 3. COMPARISON ENGINE
# ====================================================================
class ComparisonEngine:
    def __init__(self, parameters: FamilyParameters):
        self.params = parameters

    def compare_words(self, word1: str, word2: str) -> Tuple[str, float]:
        """
        Compare two words and return a match type and score.
        Returns: (match_type, score)
        match_type: 'full', 'partial', 'no'
        score: 1.0, 0.5, 0.0
        """
        cons1 = extract_consonants(word1)
        cons2 = extract_consonants(word2)
        if not cons1 or not cons2:
            return "no", 0.0
        # Apply sound shifts
        cons1_norm = normalize_consonants(cons1, self.params.sound_shifts)
        cons2_norm = normalize_consonants(cons2, self.params.sound_shifts)
        # Full match
        if cons1_norm == cons2_norm:
            return "full", 1.0
        # Partial match
        if share_any_consonant(cons1_norm, cons2_norm):
            return "partial", 0.5
        return "no", 0.0

    def compare_lists(self, list1: Dict[str, str], list2: Dict[str, str]) -> Dict:
        """
        Compare two Swadesh lists.
        Both are dicts mapping English gloss -> local word.
        Returns full results.
        """
        total = 0
        full = 0
        partial = 0
        no_match = 0
        total_score = 0.0
        results = []
        for gloss, word1 in list1.items():
            if gloss in list2 and list2[gloss]:
                word2 = list2[gloss]
                match_type, score = self.compare_words(word1, word2)
                total += 1
                total_score += score
                if match_type == "full":
                    full += 1
                elif match_type == "partial":
                    partial += 1
                else:
                    no_match += 1
                results.append({
                    "gloss": gloss,
                    "word1": word1,
                    "word2": word2,
                    "cons1": extract_consonants(word1),
                    "cons2": extract_consonants(word2),
                    "match_type": match_type,
                    "score": score
                })
        cognate_rate = total_score / total if total > 0 else 0.0
        return {
            "total": total,
            "full": full,
            "partial": partial,
            "no_match": no_match,
            "total_score": total_score,
            "cognate_rate": cognate_rate,
            "results": results
        }

    def estimate_divergence(self, cognate_rate: float) -> float:
        """
        Estimate divergence time in years from the cognate rate.
        t = (log(C) / log(r)) * b * 1000
        """
        if cognate_rate <= 0 or cognate_rate > 1:
            return float('nan')
        if self.params.retention_rate <= 0:
            return float('nan')
        log_c = np.log(cognate_rate)
        log_r = np.log(self.params.retention_rate)
        if log_r == 0:
            return float('nan')
        t_ky = (log_c / log_r) * self.params.burst_correction
        return t_ky * 1000  # convert to years

    def run_full(self, list1: Dict[str, str], list2: Dict[str, str]) -> Dict:
        """Run full comparison and divergence estimation."""
        comp = self.compare_lists(list1, list2)
        divergence = self.estimate_divergence(comp["cognate_rate"])
        comp["divergence_years"] = divergence
        comp["family"] = self.params.description
        return comp

# ====================================================================
# 4. DATA LOADING
# ====================================================================
def load_swadesh_from_csv(filepath: str) -> Dict[str, Dict[str, str]]:
    """Load Swadesh lists from a CSV file."""
    df = pd.read_csv(filepath)
    languages = [col for col in df.columns if col != 'word']
    data = {}
    for lang in languages:
        data[lang] = {}
        for _, row in df.iterrows():
            gloss = row['word'].strip()
            word = str(row[lang]) if pd.notna(row[lang]) else ""
            if word:
                data[lang][gloss] = word
    return data

def load_swadesh_from_json(filepath: str) -> Dict[str, Dict[str, str]]:
    """Load Swadesh lists from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_parameters_from_json(filepath: str) -> Dict[str, FamilyParameters]:
    """Load family parameters from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    params = {}
    for family, data in raw.items():
        params[family] = FamilyParameters(
            retention_rate=data['retention_rate'],
            burst_correction=data['burst_correction'],
            sound_shifts=data.get('sound_shifts', {}),
            description=data.get('description', '')
        )
    return params

# ====================================================================
# 5. OUTPUT
# ====================================================================
def format_result(result: Dict) -> str:
    """Format a comparison result for printing."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"FAMILY: {result.get('family', 'Unknown')}")
    lines.append(f"TOTAL COMPARISONS: {result['total']}")
    lines.append(f" Full matches: {result['full']} ({result['full']/result['total']*100:.1f}%)")
    lines.append(f" Partial matches: {result['partial']} ({result['partial']/result['total']*100:.1f}%)")
    lines.append(f" No matches: {result['no_match']} ({result['no_match']/result['total']*100:.1f}%)")
    lines.append(f"COGNATE RATE: {result['cognate_rate']*100:.1f}%")
    lines.append(f"DIVERGENCE: {result['divergence_years']:.0f} years ago")
    lines.append("=" * 70)
    return "\n".join(lines)


def format_detailed_result(result: Dict) -> str:
    """Format a detailed comparison result with word-by-word matches."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"FAMILY: {result.get('family', 'Unknown')}")
    lines.append(f"COGNATE RATE: {result['cognate_rate']*100:.1f}%")
    lines.append(f"DIVERGENCE: {result['divergence_years']:.0f} years ago")
    lines.append("-" * 80)
    lines.append(f"{'GLOSS':<20} {'WORD1':<20} {'CONS1':<15} {'WORD2':<20} {'CONS2':<15} {'MATCH':<10}")
    lines.append("-" * 80)
    for r in result['results']:
        lines.append(
            f"{r['gloss']:<20} {r['word1']:<20} {r['cons1']:<15} {r['word2']:<20} {r['cons2']:<15} {r['match_type']:<10}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)

# ====================================================================
# 6. MAIN APPLICATION
# ====================================================================
def main():
    """Main entry point for command-line usage."""
    import sys
    import os
    if len(sys.argv) < 2:
        print("Usage: python general_divergence.py <swadesh_file>")
        print(" Supported formats: .csv, .json")
        print("")
        print(" CSV format: word,lang1,lang2,...")
        print(" JSON format: { 'lang1': {'gloss': 'word', ...}, ...}")
        print("")
        print(" Available parameter families:", list(PARAMETERS.keys()))
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    # Load data
    if filepath.endswith('.csv'):
        data = load_swadesh_from_csv(filepath)
    elif filepath.endswith('.json'):
        data = load_swadesh_from_json(filepath)
    else:
        print("Unsupported format. Use .csv or .json")
        sys.exit(1)

    languages = list(data.keys())
    if len(languages) < 2:
        print("Need at least two languages to compare.")
        sys.exit(1)

    # Choose family (can be extended)
    print("\nAvailable parameter families:")
    for i, family in enumerate(PARAMETERS.keys()):
        print(f" {i+1}. {family}: {PARAMETERS[family].description}")
    print(f" {len(PARAMETERS)+1}. custom (enter parameters manually)")

    choice = input("\nSelect family (number): ").strip()
    try:
        idx = int(choice) - 1
        if idx < len(PARAMETERS):
            family_name = list(PARAMETERS.keys())[idx]
            params = PARAMETERS[family_name]
        else:
            # Custom parameters
            r = float(input("Enter retention rate per millennium: "))
            b = float(input("Enter burst correction factor: "))
            shifts = input("Enter sound shifts (e.g., 'r:l,g:k'): ")
            shift_dict = {}
            for pair in shifts.split(','):
                if ':' in pair:
                    k, v = pair.split(':')
                    shift_dict[k.strip()] = v.strip()
            params = FamilyParameters(
                retention_rate=r,
                burst_correction=b,
                sound_shifts=shift_dict,
                description="Custom family"
            )
    except (ValueError, IndexError):
        print("Invalid selection. Using Indo-European parameters.")
        params = PARAMETERS["indo_european"]

    # Run comparisons
    engine = ComparisonEngine(params)
    results = []
    for i, lang1 in enumerate(languages):
        for lang2 in languages[i+1:]:
            result = engine.run_full(data[lang1], data[lang2])
            result['lang1'] = lang1
            result['lang2'] = lang2
            results.append(result)

    # Output
    print("\n" + "=" * 80)
    print("DIVERGENCE ESTIMATES")
    print("=" * 80)
    df = pd.DataFrame([{
        'Language 1': r['lang1'],
        'Language 2': r['lang2'],
        'Cognate Rate': f"{r['cognate_rate']*100:.1f}%",
        'Divergence (years)': f"{r['divergence_years']:.0f}",
        'Family': r['family']
    } for r in results])
    print(df.to_string(index=False))
    print("\n")

    # Detailed output option
    if len(results) == 1:
        show_detail = input("Show detailed word-by-word comparison? (y/n): ").strip().lower()
        if show_detail == 'y':
            print("\n" + format_detailed_result(results[0]) + "\n")

if __name__ == "__main__":
    main()