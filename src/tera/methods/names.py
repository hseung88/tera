method_aliases = {
    "standard gp": "exact gp",
    "standard dgp": "exact dgp",
    "standard dgp-deroos": "exact dgp-deroos",
    "exact_gp": "exact gp",
    "exact_dgp": "exact dgp",
    "exact_dgp_deroos": "exact dgp-deroos",
    "vecchia_gp": "vecchia gp",
    "vecchia_dgp": "vecchia dgp",
    "vecchia_dgp_deroos": "vecchia dgp-deroos",
    "tera_batched": "tera",
}


def normalize_method_name(name):
    key = name.strip().lower()
    return method_aliases.get(key, key)


def normalize_md22_method_name(name):
    key = normalize_method_name(name).replace(" ", "_")
    key = {"exact_gp": "standard_gp", "vecchia_gp": "vecchia"}.get(key, key)
    if key not in {"standard_gp", "dsoftki", "ddsvgp", "tera", "vecchia"}:
        raise ValueError(f"Unknown MD22 method: {name}")
    return key


def md22_method_label(name):
    return {"standard_gp": "exact gp", "dsoftki": "dsoftki", "ddsvgp": "ddsvgp",
            "tera": "tera", "vecchia": "vecchia gp"}[normalize_md22_method_name(name)]
