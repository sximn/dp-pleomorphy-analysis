import json
import os
from pathlib import Path
from collections import Counter
from typing import List
from tqdm import tqdm

def normalize_label(label: str) -> str:
    """Basic text normalization."""
    return (
        label.strip()
        .lower()
        .replace("  ", " ")
    )

# mapping z nekonzistentných názvov na canonical
LABEL_MAPPING = {

    # veľké jadro
    "veľké jadro": "veľké jadro",
    "velké jadro": "veľké jadro",

    # nepravidelné jadro
    "nepravidelné jadro": "nepravidelné jadro",
    "nepravidelne jadro": "nepravidelné jadro",

    # veľké nepravidelné jadro
    "veľké nepravidelné jadro": "veľké nepravidelné jadro",
    "veľké, nepravidelné jadro": "veľké nepravidelné jadro",

    # hyperchrómne jadro
    "hyperchrómne jadro": "hyperchrómne jadro",
    "hyperchromne bunky": "hyperchrómne jadro",

    # hyperchrómne nepravidelné
    "hyperchrómne nepravidelné jadro": "hyperchrómne nepravidelné jadro",
    "yperchrómne nepravidelné jadro": "hyperchrómne nepravidelné jadro",

    # jadierka
    "veľké jadierko": "veľké jadierko",
    "veľké jadierko ": "veľké jadierko",

    "viacpočetné jadierka": "viacpočetné jadierka",
    "viacjadierkove": "viacpočetné jadierka",

    "viacpočetné jadierka,nepravidelné": "viacpočetné jadierka nepravidelné",

    # multinucleated
    "viacjadrová bunka": "viacjadrová bunka",

    # vesicular
    "vezikulárne jadro": "vezikulárne jadro",
    "veľké vezikulárne jadro": "vezikulárne jadro",

    # large hyperchromatic irregular
    "veľké nepravidelné hyperchrómne jadro": "veľké nepravidelné hyperchrómne jadro",

    # combo
    "veľké jadro s početnými jadierkami": "veľké jadro s početnými jadierkami",
    "velka bunka s velkym jadierkom": "veľké jadro s početnými jadierkami",

    # cytoplasm atypia
    "nepravidelná bunka": "nepravidelná bunka",

    # mitosis
    "mitóza": "mitóza",

    # reference cells
    "referenčná bunky - lymfocyt": "referenčná bunka - lymfocyt",
    "referenčná bunka (lymfocyt)": "referenčná bunka - lymfocyt",

    "referenčné bunky - erytrocyt": "referenčná bunka - erytrocyt",
    "referencna bunka (er)": "referenčná bunka - erytrocyt",
}

def standardize_labels(input_files: List[str], output_dir: str, state=None) -> List[str]:
    """
    Standardizes the labels in the provided geojson files and saves them to the output_dir.
    Populates statistics in the state object if provided.
    Returns a list of paths to the cleaned geojson files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    unknown_labels = Counter()
    final_counts = Counter()
    clean_files = []

    print("\n--- Stage 2: Label Standardization ---")
    for file_path in tqdm(input_files, desc="Standardizing Labels"):
        file = Path(file_path)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for feature in data.get("features", []):
            properties = feature.get("properties", {})
            classification = properties.get("classification", {})
            if "name" not in classification:
                continue

            cls = classification["name"]
            cls_norm = normalize_label(cls)

            if cls_norm not in LABEL_MAPPING:
                unknown_labels[cls_norm] += 1
                continue

            new_label = LABEL_MAPPING[cls_norm]
            classification["name"] = new_label
            final_counts[new_label] += 1

        output_path = Path(output_dir) / file.name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        clean_files.append(str(output_path.resolve()))

    if state:
        state.label_counts = dict(final_counts)
        state.unknown_labels = dict(unknown_labels)

    print("\n==== FINAL CLASS COUNTS ====")
    for k, v in sorted(final_counts.items()):
        print(f"{k:40} {v}")

    print("\n==== UNKNOWN LABELS ====")
    for k, v in unknown_labels.items():
        print(f"{k:40} {v}")
        
    return clean_files