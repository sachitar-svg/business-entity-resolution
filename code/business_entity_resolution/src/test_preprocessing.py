from preprocessing import normalize_name, normalize_address


examples = [
    (
        "Maure Williams Colombier Inc",
        "85 Wayne Avenue, Ticonderoga, NY"
    ),
    (
        "Maure Wilblims Colombier",
        None
    ),
    (
        "Dréxkor",
        "85 Wanye Avenue, Ticonderoga Townshiip, New York"
    ),
    (
        "Payne Enterprises",
        "3315 Fremont Street, Peoria, IL"
    ),
    (
        "PAYNE-ENRTPRMISES",
        "3315 FREMONTSAINT, PEORIA, IL"
    ),
    (
        "राज इन्वेस्टमेंट्स एलएलपी",
        "6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI"
    )
]


for name, address in examples:

    print("\n-------------------------------")

    print("Original name:")
    print(name)

    print("Normalized name:")
    print(normalize_name(name))

    print("\nOriginal address:")
    print(address)

    print("Normalized address:")
    print(normalize_address(address))