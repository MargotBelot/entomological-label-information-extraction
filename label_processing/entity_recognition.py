#!/usr/bin/env python3
"""
Entity Recognition Module

Extracts structured biodiversity entities from OCR text using Gemini LLM,
validates against GBIF, geocodes with OpenStreetMap, and converts to
Darwin Core / OpenDS standard formats.

Workflow per label:
    1. Send OCR text to Gemini → structured JSON (scientific names, collectors,
       dates, localities, traits, institutions)
    2. Validate scientific names via GBIF Species Match API
    3. Geocode localities via OSM Nominatim
    4. Score extraction quality
    5. Convert to DwC / OpenDS (optional)
"""

import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from google import genai
from google.genai import types


# --------------- External API helpers --------------- #


def validate_with_gbif(scientific_name: str) -> Optional[Dict[str, Any]]:
    """Validate a scientific name against the GBIF Species Match API.

    Returns a dict with matchType, usageKey, scientificName, status
    or None if no match.
    """
    if not scientific_name:
        return None
    try:
        res = requests.get(
            "https://api.gbif.org/v1/species/match",
            params={"name": scientific_name, "verbose": "true"},
            timeout=10,
        )
        if res.status_code == 200 and res.json().get("matchType") != "NONE":
            data = res.json()
            if data.get("usageKey"):
                return {
                    "matchType": data.get("matchType"),
                    "usageKey": data.get("usageKey"),
                    "scientificName": data.get("scientificName"),
                    "status": data.get("status"),
                }
    except Exception as e:
        print(f"GBIF Error: {e}")
    return None


def geocode_with_osm(modern_query: str) -> Optional[Dict[str, str]]:
    """Geocode a locality string using OpenStreetMap Nominatim.

    Returns {country, city} or None.
    """
    if not modern_query:
        return None
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "ELIE-MuseumLabelPipeline/1.0"}
    params = {"q": modern_query, "format": "json", "limit": 1, "addressdetails": 1}
    try:
        time.sleep(1.1)  # Respect OSM rate limits
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200 and response.json():
            address = response.json()[0].get("address", {})
            return {
                "country": address.get("country", ""),
                "city": address.get(
                    "city", address.get("town", address.get("state", ""))
                ),
            }
    except Exception as e:
        print(f"OSM Error: {e}")
    return None


# --------------- LLM Entity Extraction --------------- #

# JSON schema sent to Gemini for structured output.
SINGLE_LABEL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scientific_names": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "authority": {"type": "STRING"},
                },
            },
        },
        "traits_and_status": {
            "type": "OBJECT",
            "properties": {
                "type_status": {"type": "STRING"},
                "identified_by": {"type": "STRING"},
                "dateIdentified": {
                    "type": "STRING",
                    "description": "Date the identification was made",
                },
                "sex": {"type": "STRING"},
                "lifeStage": {
                    "type": "STRING",
                    "description": "e.g., adult, larva, pupa",
                },
                "preparations": {"type": "STRING"},
            },
        },
        "recordedBy": {
            "type": "STRING",
            "description": "Parsed and cleaned collector name",
        },
        "verbatimCollector": {
            "type": "STRING",
            "description": "The raw collector string exactly as written on the label",
        },
        "recordNumber": {
            "type": "STRING",
            "description": "Identifier given by the collector in the field",
        },
        "verbatimEventDate": {"type": "STRING"},
        "eventDate": {
            "type": "STRING",
            "description": "ISO 8601 format YYYY-MM-DD if possible",
        },
        "geographic_data": {
            "type": "OBJECT",
            "properties": {
                "verbatimLocality": {"type": "STRING"},
                "locality": {"type": "STRING"},
                "stateProvince": {
                    "type": "STRING",
                    "description": "State, province, or region deduced from the text",
                },
                "verbatimElevation": {
                    "type": "STRING",
                    "description": "Exact elevation text as written on label",
                },
                "minimumElevationInMeters": {"type": "STRING"},
                "habitat": {"type": "STRING"},
                "modern_location_query": {"type": "STRING"},
            },
        },
        "institutionCode": {"type": "STRING"},
        "catalogNumber": {"type": "STRING"},
        "occurrenceID": {"type": "STRING"},
    },
}

EXTRACTION_PROMPT = """
You are an expert biodiversity data scientist and taxonomist specializing in natural history museum labels (both historical and modern).
Your objective is to extract all available biological, geographic, and curatorial data from raw, often messy OCR text into a structured JSON format.

### EXTRACTION LOGIC & HEURISTICS:

1. STRICT NO HALLUCINATION: If a field is not present in the text, return null or an empty string. DO NOT guess, infer, or invent data.
2. HANDLING MESSY OCR: Museum OCR is often noisy. Mentally separate conjoined words, ignore stray characters (e.g., pipes '|', random underscores), and clean obvious optical artifacts before extracting.
3. THE "INFER AND SEPARATE" RULE: If a single string contains multiple data points (e.g., "Amasya 23/5/1880"), mentally separate them into Locality ("Amasya") and Date ("23/5/1880") before mapping.
4. CONTEXTUAL DEDUCTION: Differentiate carefully between Collectors (often marked by 'leg.', 'coll.') and Determiners/Identifiers (often marked by 'det.', 'vid.').
5. NO DATA LEFT BEHIND: Check very small strings, abbreviations, and numbers carefully before discarding them as noise.

### STRICT SCHEMA MAPPING RULES:

* **SCIENTIFIC NAME & AUTHORSHIP:** Extract the species name and authority. If you find an orphaned species epithet with an authority (e.g., "celticola Stgr.") without the Genus, extract it exactly as written. Do not invent the Genus.
* **TYPE STATUS:** Extract any type status (e.g., 'Holotype', 'Paratype', 'Lectotype') into `type_status`.
* **COLLECTOR:** Extract the raw, unedited collector text into `verbatimCollector`, and extract a standardized, parsed name into `recordedBy`.
* **DETERMINER:** Extract the person who identified the specimen into `identified_by`.
* **DATES:** If a date is next to "leg." or stands alone, map to `eventDate` / `verbatimEventDate`. If a date is associated with an identification/determiner, map to `dateIdentified`.
* **LOCALITY DATA:** Extract verbatim locality into `verbatimLocality`. Mentally parse the state/province into `stateProvince` if obvious from the locality string.
* **ELEVATION:** Map any mention of altitude/elevation exactly as written (e.g., "~2000ft", "500-600 m") into `verbatimElevation`.
* **SEX & LIFE STAGE:** Extract the biological sex (look for '♂'/Male or '♀'/Female) into `sex`. Look for developmental keywords (e.g., adult, pupa, larva) and extract to `lifeStage`.
* **GENITALIA & PREPS:** Extract identifiers for genitalia slides, DNA extractions, or prep numbers (e.g., "GU: 8963") into `preparations`.
* **CATALOG vs RECORD NUMBER:** Distinguish between institutional barcodes/accession numbers (`catalogNumber`) and a collector's personal field sequence numbers (`recordNumber`).
* **INSTITUTION:** Extract the institution or museum name (e.g., 'Zool. Mus. Berlin') into `institutionCode`.
* **URI:** Map any persistent web link (URI, LSID, URL) to `occurrenceID`.

Return ONLY a valid JSON object matching the provided schema exactly.

Label Text:
\"\"\"{text}\"\"\"
"""


def extract_entities_for_label(
    ocr_text: str,
    client: genai.Client,
    model: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """Send a single label's OCR text to Gemini and return structured entities."""
    prompt = EXTRACTION_PROMPT.format(text=ocr_text)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SINGLE_LABEL_SCHEMA,
            temperature=0.0,
        ),
    )
    return json.loads(response.text)


def extract_and_enrich(
    consolidated_labels: List[Dict[str, Any]],
    client: genai.Client,
    model: str = "gemini-2.0-flash",
) -> List[Dict[str, Any]]:
    """Run entity extraction + GBIF/OSM enrichment on every label.

    Args:
        consolidated_labels: List of label dicts from consolidated_results.json.
        client: Authenticated Gemini client.
        model: Gemini model ID.

    Returns:
        The same list with an ``entity_extraction`` key added to each label.
    """
    enriched = []

    for label in consolidated_labels:
        ocr_text = ""
        ocr_block = label.get("ocr", {})
        if isinstance(ocr_block, dict):
            ocr_text = ocr_block.get("text", "")
        elif isinstance(ocr_block, str):
            ocr_text = ocr_block

        if not ocr_text.strip():
            # Nothing to extract — keep label as-is
            enriched.append(label)
            continue

        start = time.time()
        try:
            entities = extract_entities_for_label(ocr_text, client, model)
        except Exception as e:
            print(f"  LLM extraction failed for {label.get('label_filename', '?')}: {e}")
            enriched.append(label)
            continue
        llm_time = round(time.time() - start, 2)

        # --- GBIF enrichment ---
        sci_names = entities.get("scientific_names", [])
        if sci_names and sci_names[0].get("name"):
            query = f"{sci_names[0].get('name')} {sci_names[0].get('authority', '')}".strip()
            gbif_val = validate_with_gbif(query)
            if gbif_val:
                sci_names[0]["gbif_validation"] = gbif_val

        # --- OSM enrichment ---
        geo = entities.get("geographic_data", {})
        if geo and geo.get("modern_location_query"):
            osm_val = geocode_with_osm(geo["modern_location_query"])
            if osm_val:
                geo["parsed"] = osm_val
            # Remove the query field from output
            geo.pop("modern_location_query", None)

        # --- Telemetry ---
        entities["telemetry"] = {
            "model": model,
            "execution_time_s": llm_time,
        }

        result = dict(label)
        result["entity_extraction"] = {k: v for k, v in entities.items() if v}
        enriched.append(result)

        lbl_id = label.get("label_filename", label.get("label_index", "?"))
        print(f"  Processed {lbl_id} in {llm_time}s")

    return enriched


# --------------- Quality Scoring --------------- #


def _normalize_name(name_str: Optional[str]) -> Optional[str]:
    """Title-case multi-word names, preserve all-caps acronyms."""
    if not name_str:
        return None
    if name_str.isupper() and "_" not in name_str:
        return name_str
    return name_str.title() if " " in name_str else name_str.capitalize()


def assess_extraction_success(
    entity_extraction: Dict[str, Any], ocr_text: str = ""
) -> Tuple[int, List[str], List[str], List[str]]:
    """Score extraction success based on what was extractable from visible text.

    Returns (score, extracted_fields, missing_visible_fields, notes).
    """
    extracted: List[str] = []
    missing_visible: List[str] = []
    notes: List[str] = []
    ocr_lower = ocr_text.lower()

    # Scientific Name
    sci_names = entity_extraction.get("scientific_names", [])
    if sci_names and sci_names[0].get("name"):
        extracted.append("scientific_name")
        if sci_names[0].get("gbif_validation"):
            extracted.append("gbif_validated")
        else:
            missing_visible.append("gbif_validation")
    else:
        if re.search(r"\b[A-Z][a-z]+\s[a-z]+\b", ocr_text) or any(
            kw in ocr_lower for kw in ("sp.", "var.", "subsp.", "lectotype", "type")
        ):
            missing_visible.append("scientific_name")

    # Traits
    traits = entity_extraction.get("traits_and_status", {})
    if traits.get("type_status"):
        extracted.append("type_status")
    elif "type" in ocr_lower:
        missing_visible.append("type_status")

    if traits.get("sex"):
        extracted.append("sex")
    elif "♂" in ocr_text or "♀" in ocr_text:
        missing_visible.append("sex")

    if traits.get("lifeStage"):
        extracted.append("lifeStage")
    elif any(kw in ocr_lower for kw in ("larva", "pupa", "adult", "nymph")):
        missing_visible.append("lifeStage")

    if traits.get("preparations"):
        extracted.append("preparations")
    elif "gu:" in ocr_lower or "prep" in ocr_lower:
        missing_visible.append("preparations")

    # People
    if (
        traits.get("identified_by")
        or entity_extraction.get("recordedBy")
        or entity_extraction.get("verbatimCollector")
    ):
        extracted.append("person_name")
    elif any(kw in ocr_lower for kw in ("leg.", "det.", "coll.")):
        missing_visible.append("person_name")

    # Geography
    geo = entity_extraction.get("geographic_data", {})
    if geo.get("locality") or geo.get("verbatimLocality") or geo.get("stateProvince"):
        extracted.append("geography")

    # Institution
    if entity_extraction.get("institutionCode"):
        extracted.append("institution")
    elif any(kw in ocr_lower for kw in ("mus.", "univ", "zool")):
        missing_visible.append("institution")

    # Dates
    if (
        entity_extraction.get("eventDate")
        or entity_extraction.get("verbatimEventDate")
        or traits.get("dateIdentified")
    ):
        extracted.append("date")
    elif re.search(r"\d{1,4}[/.-]\d{1,2}", ocr_text):
        missing_visible.append("date")

    # Identifiers
    if (
        entity_extraction.get("catalogNumber")
        or entity_extraction.get("occurrenceID")
        or entity_extraction.get("recordNumber")
    ):
        extracted.append("identifier")
    elif any(kw in ocr_lower for kw in ("http", "uri", "mfn", "no.")):
        missing_visible.append("identifier")

    # Score
    total = len(extracted) + len(missing_visible)
    if total == 0:
        score = 100
        notes.append("No extractable biological entities in this fragment")
    else:
        rate = len(extracted) / total
        score = int(30 + (rate * 70))
        if rate == 1.0:
            notes.append("Perfect extraction")
        elif rate >= 0.75:
            notes.append("Good extraction")
        elif rate >= 0.5:
            notes.append("Partial extraction")
        else:
            notes.append("Sparse extraction")

    return score, extracted, missing_visible, notes


def validate_and_normalize(
    labels: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Normalize names and build a quality report across all labels.

    Returns (validated_labels, quality_report).
    """
    report: Dict[str, Any] = {
        "total_records": len(labels),
        "extraction_quality": {"excellent": 0, "good": 0, "partial": 0, "sparse": 0},
        "issues": [],
        "summary": [],
    }
    validated = []
    all_extracted: List[str] = []
    all_missing: List[str] = []

    for label in labels:
        entities = label.get("entity_extraction", {})

        # Normalize names
        if entities.get("recordedBy"):
            entities["recordedBy"] = _normalize_name(entities["recordedBy"])
        traits = entities.get("traits_and_status", {})
        if traits.get("identified_by"):
            traits["identified_by"] = _normalize_name(traits["identified_by"])

        ocr_text = ""
        ocr_block = label.get("ocr", {})
        if isinstance(ocr_block, dict):
            ocr_text = ocr_block.get("text", "")

        score, extracted, missing, notes = assess_extraction_success(entities, ocr_text)
        all_extracted.extend(extracted)
        all_missing.extend(missing)

        if score >= 90:
            report["extraction_quality"]["excellent"] += 1
        elif score >= 70:
            report["extraction_quality"]["good"] += 1
        elif score >= 50:
            report["extraction_quality"]["partial"] += 1
        else:
            report["extraction_quality"]["sparse"] += 1

        if missing:
            report["issues"].append(
                {
                    "label_id": label.get("label_filename", "?"),
                    "score": score,
                    "extracted": extracted,
                    "missing_from_visible": missing,
                    "notes": notes,
                }
            )

        validated.append(label)

    report["summary"] = [
        f"Excellent (≥90%): {report['extraction_quality']['excellent']}",
        f"Good (70-89%): {report['extraction_quality']['good']}",
        f"Partial (50-69%): {report['extraction_quality']['partial']}",
        f"Sparse (<50%): {report['extraction_quality']['sparse']}",
    ]
    total = len(all_extracted) + len(all_missing)
    if total:
        report["overall_extraction_rate"] = f"{len(all_extracted) / total * 100:.1f}%"

    return validated, report


# --------------- DwC / OpenDS Conversion --------------- #


def _get_field(obj: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts."""
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return default
    return obj if obj else default


def _occurrence_id(label: Dict, entities: Dict) -> str:
    """Derive an occurrenceID from available identifiers."""
    if entities.get("occurrenceID"):
        return entities["occurrenceID"]
    cat = entities.get("catalogNumber", "")
    inst = entities.get("institutionCode", "")
    lbl = label.get("label_filename", label.get("label_index", ""))
    return f"{inst}:{cat}" if cat else str(lbl)


def _shared_fields(label: Dict, entities: Dict) -> Dict[str, Any]:
    """Build the set of fields common to DwC and OpenDS."""
    geo = entities.get("geographic_data", {})
    traits = entities.get("traits_and_status", {})
    return {
        "scientificName": _get_field(entities, "scientific_names")
        and entities["scientific_names"][0].get("name"),
        "scientificNameAuthorship": _get_field(entities, "scientific_names")
        and entities["scientific_names"][0].get("authority"),
        "recordedBy": entities.get("recordedBy"),
        "verbatimCollector": entities.get("verbatimCollector"),
        "recordNumber": entities.get("recordNumber"),
        "eventDate": entities.get("eventDate"),
        "verbatimEventDate": entities.get("verbatimEventDate"),
        "verbatimLocality": _get_field(geo, "verbatimLocality"),
        "locality": _get_field(geo, "locality"),
        "stateProvince": _get_field(geo, "stateProvince"),
        "country": _get_field(geo, "parsed", "country"),
        "institutionCode": entities.get("institutionCode"),
        "catalogNumber": entities.get("catalogNumber"),
        "typeStatus": _get_field(traits, "type_status"),
        "identifiedBy": _get_field(traits, "identified_by"),
        "dateIdentified": _get_field(traits, "dateIdentified"),
        "sex": _get_field(traits, "sex"),
        "lifeStage": _get_field(traits, "lifeStage"),
        "preparations": _get_field(traits, "preparations"),
    }


def _merge_labels_per_specimen(
    labels: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Group labels by source_image and merge entity fields into one record.

    For each field the first non-empty value wins; scientific names with
    GBIF validation take priority over unvalidated ones.
    """
    specimens: Dict[str, Dict[str, Any]] = {}

    for label in labels:
        entities = label.get("entity_extraction", {})
        if not entities:
            continue
        src = label.get("source_image", "unknown")
        shared = _shared_fields(label, entities)

        if src not in specimens:
            specimens[src] = {
                "fields": {},
                "gbif_usage_key": None,
                "occurrence_id": None,
                "label_ids": [],
            }

        rec = specimens[src]
        rec["label_ids"].append(label.get("label_filename", ""))

        # Merge: first non-empty value per field wins
        for key, val in shared.items():
            if val and not rec["fields"].get(key):
                rec["fields"][key] = val

        # Prefer GBIF-validated scientific name over plain ones
        sci = entities.get("scientific_names", [])
        if sci and sci[0].get("gbif_validation"):
            rec["fields"]["scientificName"] = sci[0]["name"]
            rec["fields"]["scientificNameAuthorship"] = sci[0].get("authority")
            rec["gbif_usage_key"] = sci[0]["gbif_validation"].get("usageKey")

        # Best occurrence ID: prefer URI > catalogNumber > label filename
        oid = _occurrence_id(label, entities)
        if entities.get("occurrenceID"):  # explicit URI always wins
            rec["occurrence_id"] = oid
        elif rec["occurrence_id"] is None:
            rec["occurrence_id"] = oid

    return specimens


def generate_dwc(labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert enriched labels to one Darwin Core record per specimen."""
    specimens = _merge_labels_per_specimen(labels)
    records = []
    for src_image, spec in specimens.items():
        record = {f"dwc:{k}": v for k, v in spec["fields"].items() if v}
        record["dwc:basisOfRecord"] = "PreservedSpecimen"
        record["dwc:occurrenceID"] = spec["occurrence_id"] or src_image
        record["source_image"] = src_image
        record["_labels_used"] = spec["label_ids"]
        if spec["gbif_usage_key"]:
            record["_gbif:usageKey"] = spec["gbif_usage_key"]
        records.append(record)
    return records


def generate_opends(labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert enriched labels to one OpenDS record per specimen."""
    specimens = _merge_labels_per_specimen(labels)
    records = []
    for src_image, spec in specimens.items():
        record = {f"ods:{k}": v for k, v in spec["fields"].items() if v}
        record["ods:ID"] = spec["occurrence_id"] or src_image
        record["ods:type"] = "PhysicalSpecimenDigitalRecord"
        record["source_image"] = src_image
        record["_labels_used"] = spec["label_ids"]
        records.append(record)
    return records


def export_to_csv(dwc_records: List[Dict[str, Any]], output_path: str) -> None:
    """Write Darwin Core records to a CSV file."""
    if not dwc_records:
        return
    all_headers = set()
    for row in dwc_records:
        all_headers.update(row.keys())
    tracking = [
        "source_image", "label_id", "classification",
        "dwc:occurrenceID", "dwc:basisOfRecord",
    ]
    headers = [c for c in tracking if c in all_headers] + sorted(
        c for c in all_headers if c not in tracking
    )
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(dwc_records)
    print(f"  Saved CSV → {output_path}")


# --------------- Master JSON Builder --------------- #


def build_master_json(
    enriched_labels: List[Dict[str, Any]],
    quality_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build the master output JSON grouping labels by source image."""
    images: Dict[str, List[Dict]] = {}
    for lbl in enriched_labels:
        src = lbl.get("source_image", "unknown")
        images.setdefault(src, []).append(lbl)

    master = []
    for src_image, labels in images.items():
        master.append(
            {
                "source_image": src_image,
                "labels": labels,
                "provenance": {
                    "pipeline": "ELIE Gemini Entity Recognition",
                },
            }
        )
    return master
