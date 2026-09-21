from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Any

LEXICAL_SUBSTITUTIONS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\bporque\b", re.IGNORECASE), ("ya que", "puesto que")),
    (re.compile(r"\bpero\b", re.IGNORECASE), ("sin embargo", "aunque")),
    (re.compile(r"\bsiempre\b", re.IGNORECASE), ("constantemente", "todo el tiempo")),
    (re.compile(r"\bnunca\b", re.IGNORECASE), ("jamas",)),
    (re.compile(r"\bmuy\b", re.IGNORECASE), ("sumamente", "bastante")),
    (re.compile(r"\bademas\b", re.IGNORECASE), ("aparte", "encima")),
    (re.compile(r"\bentonces\b", re.IGNORECASE), ("por eso", "asi que")),
    (re.compile(r"\bobliga\b", re.IGNORECASE), ("fuerza",)),
    (re.compile(r"\bobligo\b", re.IGNORECASE), ("forzo",)),
    (re.compile(r"\bobligar\b", re.IGNORECASE), ("forzar",)),
    (re.compile(r"\bamenaza\b", re.IGNORECASE), ("intimida",)),
    (re.compile(r"\bamenazo\b", re.IGNORECASE), ("intimido",)),
    (re.compile(r"\bamenazar\b", re.IGNORECASE), ("intimidar",)),
    (re.compile(r"\bcontrola\b", re.IGNORECASE), ("domina",)),
    (re.compile(r"\bcontrolaba\b", re.IGNORECASE), ("dominaba",)),
    (re.compile(r"\binsulta\b", re.IGNORECASE), ("humilla", "ofende")),
    (re.compile(r"\binsulto\b", re.IGNORECASE), ("humillo", "ofendio")),
    (re.compile(r"\binsultar\b", re.IGNORECASE), ("humillar", "ofender")),
    (re.compile(r"\bgolpea\b", re.IGNORECASE), ("agrede", "pega")),
    (re.compile(r"\bgolpeo\b", re.IGNORECASE), ("agredio", "pego")),
    (re.compile(r"\bgolpear\b", re.IGNORECASE), ("agredir", "pegar")),
    (re.compile(r"\bpega\b", re.IGNORECASE), ("golpea", "agrede")),
    (re.compile(r"\bpego\b", re.IGNORECASE), ("golpeo", "agredio")),
    (re.compile(r"\bpegar\b", re.IGNORECASE), ("golpear", "agredir")),
)

NEUTRAL_PREFIXES: tuple[str, ...] = (
    "Segun el relato, ",
    "En el testimonio, ",
    "De acuerdo con lo narrado, ",
)

REPORT_STYLE_OPENINGS: tuple[str, ...] = (
    "USUARIA REFIERE QUE AGRESOR ES {relation} Y QUE {time_clause}",
    "LA USUARIA MANIFIESTA QUE AGRESOR ES {relation} Y REFIERE QUE {time_clause}",
    "USUARIA MENCIONA QUE AGRESOR ES {relation} Y QUE {time_clause}",
)

REPORT_STYLE_TIME_CLAUSES: tuple[str, ...] = (
    "EL DIA DE HOY, EN EL DOMICILIO,",
    "EL DIA DE AYER POR LA NOCHE, AL INTERIOR DE LA CASA,",
    "EN LA MADRUGADA, DENTRO DEL DOMICILIO,",
    "APROXIMADAMENTE A LAS 2:00 AM, EN EL DOMICILIO,",
)

REPORT_STYLE_TIME_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "MADRUGADA",
        (
            "EN LA MADRUGADA, DENTRO DEL DOMICILIO,",
            "APROXIMADAMENTE A LAS 2:00 AM, EN EL DOMICILIO,",
        ),
    ),
    (
        "NOCHE",
        (
            "EL DIA DE AYER POR LA NOCHE, AL INTERIOR DE LA CASA,",
            "POR LA NOCHE, CUANDO USUARIA SE ENCONTRABA EN SU CASA,",
        ),
    ),
    (
        "VIA PUBLICA",
        (
            "EN LA VIA PUBLICA, CUANDO USUARIA SE ENCONTRABA SOLA,",
            "CUANDO USUARIA IBA CAMINANDO CERCA DE SU DOMICILIO,",
        ),
    ),
    (
        "TRABAJO",
        (
            "CUANDO USUARIA SE ENCONTRABA EN SU LUGAR DE TRABAJO,",
            "AL SALIR DEL TRABAJO Y DIRIGIRSE A SU DOMICILIO,",
        ),
    ),
)

REPORT_STYLE_RELATION_HINTS: tuple[tuple[str, str], ...] = (
    ("EX NOVIO", "SU EX NOVIO"),
    ("EX PAREJA", "SU EX PAREJA"),
    ("ESPOSO", "SU ESPOSO"),
    ("CONCUBINO", "SU CONCUBINO"),
    ("PAREJA", "SU PAREJA"),
    ("NOVIO", "SU NOVIO"),
    ("PADRASTRO", "SU PADRASTRO"),
    ("PADRE", "SU PADRE"),
    ("HIJO", "SU HIJO"),
    ("HERMANO", "SU HERMANO"),
    ("SUPERVISOR", "SU SUPERVISOR"),
    ("YERNO", "SU YERNO"),
)

REPORT_STYLE_WEAPON_HINTS: tuple[tuple[str, str], ...] = (
    ("ARMA DE FUEGO", "CON UN ARMA DE FUEGO"),
    ("PISTOLA", "CON UNA PISTOLA"),
    ("NAVAJA", "CON UNA NAVAJA"),
    ("CUCHILLO", "CON UN CUCHILLO"),
    ("MACHETE", "CON UN MACHETE"),
    ("DESARMADOR", "CON UN DESARMADOR"),
    ("ACIDO", "CON UNA SUSTANCIA CORROSIVA"),
    ("HACHA", "CON UN HACHA"),
)

REPORT_STYLE_SEXUAL_HINTS: tuple[str, ...] = (
    "VIOL",
    "PENETR",
    "RELACIONES SEXUALES",
    "ABUS",
    "TOCAM",
    "OBLIGO A TENER RELACIONES",
)

REPORT_STYLE_STRANGULATION_HINTS: tuple[str, ...] = (
    "ESTRANG",
    "AHORC",
    "ASFIX",
    "CUELLO",
    "NO LA DEJABA RESPIRAR",
)

REPORT_STYLE_CONFINEMENT_HINTS: tuple[str, ...] = (
    "ENCERR",
    "NO LA DEJABA SALIR",
    "RETUVO",
    "RETEN",
    "NO LE PERMITIA SALIR",
    "LE QUITO EL TELEFONO",
)

REPORT_STYLE_CHILD_HINTS: tuple[str, ...] = (
    " HIJO",
    " HIJA",
    " MENORES",
    " MENOR",
    " NINA",
    " NINO",
)

REPORT_STYLE_POLICE_HINTS: tuple[str, ...] = (
    "POLICIA",
    "PATRULLA",
    "911",
    "EMERGENCIAS",
)

REPORT_STYLE_THREAT_HINTS: tuple[str, ...] = (
    "TE VOY A MATAR",
    "MATAR",
    "MUERTE",
    "NO LA IBA A DEJAR EN PAZ",
    "NO SERAS DE NADIE",
)

REPORT_STYLE_GENERIC_ACTIONS: tuple[str, ...] = (
    "LA AMENAZO DE MUERTE Y LA AGREDIO FISICAMENTE EN REPETIDAS OCASIONES",
    "LA AGARRO DEL CABELLO, LA TIRO AL PISO Y LA SIGUIO GOLPEANDO",
    "LA SOMETIO POR LA FUERZA Y LA LASTIMO EN DISTINTAS PARTES DEL CUERPO",
    "LA GOLPEO DE MANERA REPETIDA EN EL ROSTRO, LA CABEZA Y EL CUERPO",
)

REPORT_STYLE_GENERIC_CONTEXT: tuple[str, ...] = (
    "USUARIA REFIERE QUE TEME POR SU INTEGRIDAD Y QUE EL AGRESOR YA LA HABIA VIOLENTADO ANTES",
    "MENCIONA QUE TUVO QUE PEDIR AUXILIO A FAMILIARES O VECINOS PARA PODER PONERSE A SALVO",
    "SEÑALA QUE EL AGRESOR CONTINUO AMENAZANDOLA AUN DESPUES DE LA AGRESION",
)

REPORT_STYLE_MILD_ACTIONS: tuple[str, ...] = (
    "EL AGRESOR LA INSULTO, LE GRITO Y LE DIJO PALABRAS OFENSIVAS",
    "SE NEGO A APORTAR DINERO PARA LOS GASTOS DEL HOGAR Y LA DEJO SOLA CON LAS RESPONSABILIDADES",
    "LE DIJO QUE NO CUMPLIRIA CON LA PENSION ALIMENTICIA Y QUE SE ARREGLARA COMO PUDIERA",
    "MANTUVO UNA ACTITUD INDIFERENTE Y DESPECTIVA MIENTRAS LA HUMILLABA VERBALMENTE",
)

REPORT_STYLE_MILD_CONTEXT: tuple[str, ...] = (
    "USUARIA ACUDE PRINCIPALMENTE A SOLICITAR ASESORIA JURIDICA Y ORIENTACION",
    "REFIERE QUE LA SITUACION LE GENERA MALESTAR EMOCIONAL Y PROBLEMAS FAMILIARES",
    "SEÑALA QUE LOS HECHOS HAN SIDO REPETITIVOS Y DESEA SABER COMO PROCEDER",
    "MENCIONA QUE ACUDIO PARA RECIBIR APOYO PSICOLOGICO Y ASESORIA LEGAL",
)

REPORT_STYLE_MILD_CLOSINGS: tuple[str, ...] = (
    "USUARIA SOLICITA ORIENTACION JURIDICA Y VALORACION PSICOLOGICA.",
    "ACUDE PARA RECIBIR ASESORIA SOBRE ALIMENTOS, CONVIVENCIA O SEPARACION.",
    "USUARIA DESEA QUEDAR INFORMADA SOBRE SUS DERECHOS Y POSIBLES MEDIDAS DE APOYO.",
)

REPORT_STYLE_CLOSINGS: tuple[str, ...] = (
    "USUARIA DESEA PRESENTAR DENUNCIA, SOLICITA MEDIDA DE PROTECCION Y ATENCION PSICOLOGICA.",
    "ACUDE PARA DENUNCIAR LOS HECHOS, SOLICITA MEDIDAS DE PROTECCION Y ASESORIA LEGAL.",
    "USUARIA TEME POR SU INTEGRIDAD Y LA DE SUS HIJOS, POR LO QUE SOLICITA APOYO JURIDICO Y PSICOLOGICO.",
)


def _stable_seed(base_seed: int, record_id: str, variant_index: int) -> int:
    payload = f"{base_seed}:{record_id}:{variant_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_one(pattern: re.Pattern[str], text: str, replacement: str, rng: random.Random) -> tuple[str, bool]:
    matches = list(pattern.finditer(text))
    if not matches:
        return text, False
    match = rng.choice(matches)
    resolved = _match_case(match.group(0), replacement)
    return text[: match.start()] + resolved + text[match.end() :], True


def _prefix_variant(text: str, prefix: str) -> str:
    stripped = text.lstrip()
    if stripped[:1].isalpha() and stripped[:1].isupper():
        stripped = stripped[:1].lower() + stripped[1:]
    return f"{prefix}{stripped}"


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _uppercase_source(text: str) -> str:
    return _normalize_text(text).upper()


def _contains_any(source: str, hints: tuple[str, ...]) -> bool:
    return any(hint in source for hint in hints)


def _first_hint_match(source: str, hints: tuple[tuple[str, str], ...], default: str) -> str:
    for hint, value in hints:
        if hint in source:
            return value
    return default


def _time_clause_pool(source: str) -> tuple[str, ...]:
    for hint, clauses in REPORT_STYLE_TIME_HINTS:
        if hint in source:
            return clauses
    return REPORT_STYLE_TIME_CLAUSES


def generate_synonym_variants(text: str, record_id: str, *, seed: int, max_variants: int) -> list[str]:
    base_text = _normalize_text(text)
    if not base_text or max_variants <= 0:
        return []

    variants: list[str] = []
    seen = {base_text}
    max_attempts = max(12, max_variants * 10)

    for attempt in range(max_attempts):
        if len(variants) >= max_variants:
            break
        rng = random.Random(_stable_seed(seed, record_id, attempt))
        candidate = base_text
        replacement_count = 0

        substitution_indices = list(range(len(LEXICAL_SUBSTITUTIONS)))
        rng.shuffle(substitution_indices)
        for substitution_index in substitution_indices:
            if replacement_count >= 3:
                break
            pattern, options = LEXICAL_SUBSTITUTIONS[substitution_index]
            candidate, changed = _replace_one(pattern, candidate, rng.choice(options), rng)
            if not changed:
                continue
            replacement_count += 1
            if replacement_count >= 1 and rng.random() < 0.35:
                break

        if replacement_count == 0 or rng.random() < 0.30:
            candidate = _prefix_variant(candidate, rng.choice(NEUTRAL_PREFIXES))

        candidate = _normalize_text(candidate)
        if candidate == base_text or candidate in seen:
            continue
        seen.add(candidate)
        variants.append(candidate)

    return variants


def generate_report_style_variants(
    text: str,
    record_id: str,
    *,
    seed: int,
    max_variants: int,
    target_label_id: int = 3,
) -> list[str]:
    source = _uppercase_source(text)
    if not source or max_variants <= 0:
        return []

    relation = _first_hint_match(source, REPORT_STYLE_RELATION_HINTS, "SU PAREJA")
    if int(target_label_id) == 0:
        aggression_pool = list(REPORT_STYLE_MILD_ACTIONS)
        context_pool = list(REPORT_STYLE_MILD_CONTEXT)
        closing_pool = list(REPORT_STYLE_MILD_CLOSINGS)

        if any(hint in source for hint in ("PENSION", "ALIMENT", "DINERO", "GASTOS", "NOMINA")):
            aggression_pool.extend(
                (
                    "DEJO DE DAR DINERO PARA LA MANUTENCION Y EVADIO SU RESPONSABILIDAD ECONOMICA",
                    "SE NEGO A CUMPLIR CON LOS ACUERDOS DE ALIMENTOS Y DEJO A USUARIA SIN APOYO",
                )
            )
        if _contains_any(source, REPORT_STYLE_CHILD_HINTS):
            aggression_pool.extend(
                (
                    "LE DIJO QUE NO LE PERMITIRIA VER A SUS HIJOS Y QUE TENDRIA QUE DEMANDARLO",
                    "UTILIZO EL TEMA DE SUS HIJOS PARA PRESIONARLA Y NEGARLE LA CONVIVENCIA",
                )
            )
            context_pool.extend(
                (
                    "USUARIA DESEA ORIENTACION SOBRE CONVIVENCIA, GUARDA Y CUSTODIA O PENSION",
                    "SEÑALA QUE EL CONFLICTO ESTA AFECTANDO LA CONVIVENCIA FAMILIAR Y EL BIENESTAR DE LOS MENORES",
                )
            )
        if any(hint in source for hint in ("INSULT", "GROSER", "GRIT", "OFEND")):
            aggression_pool.extend(
                (
                    "LA HUMILLO CON INSULTOS Y LEVANTO LA VOZ DURANTE LA DISCUSION",
                    "SE DIRIGIO A ELLA CON GROSERIAS Y DESCALIFICACIONES DE MANERA CONSTANTE",
                )
            )
        if any(hint in source for hint in ("ARROJ", "AVENT", "LANZ", "PALO", "OBJETO")):
            aggression_pool.extend(
                (
                    "LE ARROJO UN OBJETO SIN LOGRAR LESIONARLA Y CONTINUO DISCUTIENDO CON ELLA",
                    "DURANTE LA DISCUSION LE LANZO UN OBJETO Y DESPUES CONTINUO INSULTANDOLA",
                )
            )
        if any(hint in source for hint in ("CORRIO", "CASA", "DOMICILIO", "SALIR")):
            aggression_pool.extend(
                (
                    "LA AMENAZO CON SACARLA DE LA CASA Y DECIRLE QUE YA NO REGRESARA",
                    "LE EXIGIO QUE SE FUERA DEL DOMICILIO MIENTRAS CONTINUABA DISCUTIENDO CON ELLA",
                )
            )
    else:
        weapon_phrase = _first_hint_match(source, REPORT_STYLE_WEAPON_HINTS, "")
        aggression_pool = list(REPORT_STYLE_GENERIC_ACTIONS)
        context_pool = list(REPORT_STYLE_GENERIC_CONTEXT)
        closing_pool = list(REPORT_STYLE_CLOSINGS)

        if weapon_phrase:
            aggression_pool.extend(
                (
                    f"LA AMENAZO DE MUERTE {weapon_phrase} MIENTRAS LA TENIA SOMETIDA",
                    f"LA INTIMIDO {weapon_phrase} Y LE DIJO QUE SI PEDIA AYUDA LA IBA A MATAR",
                )
            )
        if _contains_any(source, REPORT_STYLE_SEXUAL_HINTS):
            aggression_pool.extend(
                (
                    "LA FORZO A TENER RELACIONES SEXUALES EN CONTRA DE SU VOLUNTAD",
                    "LA OBLIGO A REALIZAR ACTOS SEXUALES MIENTRAS LA AMENAZABA",
                )
            )
        if _contains_any(source, REPORT_STYLE_STRANGULATION_HINTS):
            aggression_pool.extend(
                (
                    "LA AGARRO DEL CUELLO Y LA ESTRANGULO HASTA DEJARLA SIN AIRE",
                    "LE TAPO LA BOCA Y LA NARIZ MIENTRAS LA AMENAZABA",
                )
            )
        if _contains_any(source, REPORT_STYLE_CONFINEMENT_HINTS):
            aggression_pool.extend(
                (
                    "LA MANTUVO ENCERRADA Y NO LE PERMITIA SALIR NI PEDIR AYUDA",
                    "LE QUITO EL TELEFONO Y LA RETUVO DENTRO DEL DOMICILIO",
                )
            )
        if _contains_any(source, REPORT_STYLE_THREAT_HINTS):
            context_pool.extend(
                (
                    "USUARIA REFIERE QUE EL AGRESOR LE DIJO QUE LA BUSCARIA Y LA MATARIA SI LO DENUNCIABA",
                    "MENCIONA QUE DESDE ENTONCES CONTINUA CON MIEDO POR LAS AMENAZAS DE MUERTE",
                )
            )
        if _contains_any(source, REPORT_STYLE_CHILD_HINTS):
            context_pool.extend(
                (
                    "LOS MENORES PRESENCIARON LOS HECHOS Y COMENZARON A PEDIR AUXILIO",
                    "USUARIA REFIERE QUE SUS HIJOS VIERON LA AGRESION Y QUEDARON ASUSTADOS",
                )
            )
        if _contains_any(source, REPORT_STYLE_POLICE_HINTS):
            context_pool.extend(
                (
                    "AL PEDIR AYUDA ACUDIERON ELEMENTOS DE POLICIA, PERO USUARIA CONTINUA CON TEMOR",
                    "REFIERE QUE SOLICITO APOYO POLICIAL DESPUES DE LOS HECHOS",
                )
            )

    variants: list[str] = []
    seen = {source, _normalize_text(text)}
    max_attempts = max(16, max_variants * 12)
    time_clauses = _time_clause_pool(source)

    for attempt in range(max_attempts):
        if len(variants) >= max_variants:
            break
        rng = random.Random(_stable_seed(seed, record_id, 1000 + attempt))
        opening = rng.choice(REPORT_STYLE_OPENINGS).format(relation=relation, time_clause=rng.choice(time_clauses))
        action_count = min(len(aggression_pool), 3 if rng.random() < 0.5 else 2)
        action_clauses = rng.sample(aggression_pool, k=action_count)
        context_clause = rng.choice(context_pool)
        trailing_context = rng.choice(context_pool) if len(context_pool) > 1 and rng.random() < 0.45 else ""
        closing = rng.choice(closing_pool)

        sentences = [f"{opening} {action_clauses[0]}." ]
        sentences.extend(f"{clause}." for clause in action_clauses[1:])
        sentences.append(f"{context_clause}.")
        if trailing_context and trailing_context != context_clause:
            sentences.append(f"{trailing_context}.")
        sentences.append(closing)
        candidate = _normalize_text(" ".join(sentences))
        if candidate in seen:
            continue
        seen.add(candidate)
        variants.append(candidate)

    return variants


def augment_rows_for_label(
    rows: list[dict[str, Any]],
    *,
    target_label_id: int,
    mode: str,
    target_ratio: float,
    max_copies_per_row: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows = list(rows)
    target_rows = [row for row in base_rows if int(row["labels"]) == int(target_label_id)]
    base_count = len(base_rows)
    base_target_count = len(target_rows)
    base_ratio = (base_target_count / base_count) if base_count else 0.0
    resolved_target_ratio = max(0.0, min(float(target_ratio), 0.49))

    summary: dict[str, Any] = {
        "mode": mode,
        "target_label_id": int(target_label_id),
        "target_ratio": resolved_target_ratio,
        "max_copies_per_row": int(max_copies_per_row),
        "base_rows": base_count,
        "base_target_rows": base_target_count,
        "base_target_ratio": base_ratio,
        "added_rows": 0,
        "final_rows": base_count,
        "final_target_rows": base_target_count,
        "final_target_ratio": base_ratio,
    }

    if mode == "none" or resolved_target_ratio <= 0.0 or max_copies_per_row <= 0 or not target_rows:
        return base_rows, summary
    if mode == "synonym":
        variant_builder = generate_synonym_variants
    elif mode == "report_style":
        def variant_builder(text: str, record_id: str, *, seed: int, max_variants: int) -> list[str]:
            return generate_report_style_variants(
                text,
                record_id,
                seed=seed,
                max_variants=max_variants,
                target_label_id=target_label_id,
            )
    else:
        raise ValueError(f"Modo de augmentacion no soportado: {mode}")

    desired_extra_rows = math.ceil(
        max(0.0, (resolved_target_ratio * base_count - base_target_count) / max(1e-9, 1.0 - resolved_target_ratio))
    )
    desired_extra_rows = min(desired_extra_rows, base_target_count * int(max_copies_per_row))
    if desired_extra_rows <= 0:
        return base_rows, summary

    per_row_variants: list[tuple[dict[str, Any], list[str]]] = []
    total_available_variants = 0
    for row in target_rows:
        variants = variant_builder(
            str(row["text"]),
            str(row["record_id"]),
            seed=seed,
            max_variants=int(max_copies_per_row),
        )
        per_row_variants.append((row, variants))
        total_available_variants += len(variants)

    desired_extra_rows = min(desired_extra_rows, total_available_variants)
    augmented_rows: list[dict[str, Any]] = []
    for copy_index in range(int(max_copies_per_row)):
        for row, variants in per_row_variants:
            if len(augmented_rows) >= desired_extra_rows:
                break
            if copy_index >= len(variants):
                continue
            augmented_row = dict(row)
            augmented_row["record_id"] = f"{row['record_id']}__aug_{mode}_{copy_index + 1}"
            augmented_row["text"] = variants[copy_index]
            augmented_rows.append(augmented_row)
        if len(augmented_rows) >= desired_extra_rows:
            break

    final_rows = base_rows + augmented_rows
    final_target_count = base_target_count + len(augmented_rows)
    summary.update(
        {
            "added_rows": len(augmented_rows),
            "final_rows": len(final_rows),
            "final_target_rows": final_target_count,
            "final_target_ratio": (final_target_count / len(final_rows)) if final_rows else 0.0,
        }
    )
    return final_rows, summary
