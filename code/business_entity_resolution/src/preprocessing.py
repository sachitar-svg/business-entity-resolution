import re
import unicodedata
import pandas as pd


def normalize_text(value):
    """
    Unicode-safe normalization that also handles pandas NaN values.
    """

    # Handle None and pandas missing values
    if value is None or pd.isna(value):
        return ""

    value = str(value)

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Case normalization
    value = value.casefold()

    cleaned = []

    for char in value:
        category = unicodedata.category(char)

        # Replace punctuation and symbols with spaces
        if category.startswith("P") or category.startswith("S"):
            cleaned.append(" ")
        else:
            cleaned.append(char)

    value = "".join(cleaned)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value):
    """
    Normalize business names.
    """
    return normalize_text(value)


def normalize_address(value):
    """
    Normalize business addresses.
    """
    return normalize_text(value)


if __name__ == "__main__":

    examples = [
        "Payne Énterprises",
        "PAYNE-ENRTPRMISES",
        "Maure Williams Colombier Inc",
        "Dréxkor",
        "राज इन्वेस्टमेंट्स एलएलपी",
        "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
        "3315 Fremont Street, Peoria, IL",
        "6(29), C.I.T. Colony, 2ND Main Road"
    ]

    print("===== NORMALIZATION TEST =====")

    for text in examples:
        print()
        print("Original :", text)
        print("Normalized:", normalize_text(text))