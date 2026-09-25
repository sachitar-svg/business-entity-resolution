from rapidfuzz import fuzz

from preprocessing import normalize_name, normalize_address




def name_similarity(name1, name2):
    """
    Calculate similarity between two business names.

    Returns a score between 0 and 100.
    """

    name1 = normalize_name(name1)
    name2 = normalize_name(name2)

    if not name1 or not name2:
        return 0.0

    return fuzz.token_set_ratio(name1, name2)


def address_similarity(address1, address2):
    """
    Calculate similarity between two business addresses.

    Returns a score between 0 and 100.
    """

    address1 = normalize_address(address1)
    address2 = normalize_address(address2)

    if not address1 or not address2:
        return 0.0

    return fuzz.token_set_ratio(address1, address2)


def country_match(country1, country2):
    """
    Return 1 if countries match, otherwise 0.
    """

    if not country1 or not country2:
        return 0

    return int(
        str(country1).casefold().strip()
        == str(country2).casefold().strip()
    )


if __name__ == "__main__":

    s1_name = "Payne Enterprises"
    s2_name = "PAYNE-ENRTPRMISES"

    s1_address = "3315 Fremont Street, Peoria, IL"
    s2_address = "3315 FREMONTSAINT, PEORIA, IL"

    print("===== SIMILARITY TEST =====")

    print("\nName:")
    print(s1_name)
    print(s2_name)
    print(
        "Similarity:",
        name_similarity(s1_name, s2_name)
    )

    print("\nAddress:")
    print(s1_address)
    print(s2_address)
    print(
        "Similarity:",
        address_similarity(
            s1_address,
            s2_address
        )
    )

    print("\nCountry:")
    print("US vs US")
    print(
        "Match:",
        country_match("US", "US")
    )