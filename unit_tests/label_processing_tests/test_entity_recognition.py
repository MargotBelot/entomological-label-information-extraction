"""Tests for label_processing.entity_recognition — pure-logic functions only (no API calls)."""

import unittest

from label_processing.entity_recognition import (
    _normalize_name,
    assess_extraction_success,
    generate_dwc,
    generate_opends,
    build_master_json,
)


# --------------- Helpers --------------- #


def _make_label(source_image="img001.jpg", label_filename="img001_label_1.jpg",
                entities=None):
    """Build a minimal label dict with entity_extraction."""
    label = {
        "source_image": source_image,
        "label_filename": label_filename,
        "ocr": {"text": "Apis mellifera L. leg. Smith 2020-05-01 Berlin"},
    }
    if entities is not None:
        label["entity_extraction"] = entities
    return label


def _rich_entities():
    """Entities dict with every major field populated."""
    return {
        "scientific_names": [
            {"name": "Apis mellifera", "authority": "L.",
             "gbif_validation": {"matchType": "EXACT", "usageKey": 1311477,
                                 "scientificName": "Apis mellifera", "status": "ACCEPTED"}}
        ],
        "traits_and_status": {
            "type_status": "Holotype",
            "identified_by": "Doe, J.",
            "sex": "Female",
            "lifeStage": "adult",
            "preparations": "GU: 1234",
        },
        "recordedBy": "smith, john",
        "verbatimCollector": "Smith leg.",
        "eventDate": "2020-05-01",
        "verbatimEventDate": "1.V.2020",
        "geographic_data": {
            "verbatimLocality": "Berlin, Germany",
            "locality": "Berlin",
            "stateProvince": "Berlin",
            "parsed": {"country": "Germany", "city": "Berlin"},
        },
        "institutionCode": "MFN",
        "catalogNumber": "MFN-001",
        "occurrenceID": "http://example.org/MFN-001",
    }


# --------------- _normalize_name --------------- #


class TestNormalizeName(unittest.TestCase):

    def test_none_returns_none(self):
        self.assertIsNone(_normalize_name(None))

    def test_empty_returns_none(self):
        self.assertIsNone(_normalize_name(""))

    def test_all_caps_acronym_preserved(self):
        self.assertEqual(_normalize_name("MFN"), "MFN")

    def test_multi_word_title_cased(self):
        self.assertEqual(_normalize_name("john smith"), "John Smith")

    def test_single_word_capitalized(self):
        self.assertEqual(_normalize_name("smith"), "Smith")

    def test_already_correct_unchanged(self):
        self.assertEqual(_normalize_name("John Smith"), "John Smith")


# --------------- assess_extraction_success --------------- #


class TestAssessExtractionSuccess(unittest.TestCase):

    def test_perfect_extraction(self):
        """All fields extracted → score = 100, note = 'Perfect extraction'."""
        score, extracted, missing, notes = assess_extraction_success(
            _rich_entities(),
            "Apis mellifera L. Holotype ♀ adult GU: 1234 leg. Smith 2020 Berlin Mus. http://x"
        )
        self.assertEqual(score, 100)
        self.assertEqual(missing, [])
        self.assertIn("Perfect extraction", notes)

    def test_empty_extraction_empty_text(self):
        """No entities and no visible text → score 100 (nothing to extract)."""
        score, extracted, missing, notes = assess_extraction_success({}, "")
        self.assertEqual(score, 100)
        self.assertIn("No extractable biological entities", notes[0])

    def test_partial_extraction(self):
        """Missing some fields that are visible in OCR → score < 100."""
        entities = {
            "scientific_names": [{"name": "Apis mellifera"}],
        }
        ocr = "Apis mellifera L. leg. Smith 2020-05-01 Berlin Mus."
        score, extracted, missing, notes = assess_extraction_success(entities, ocr)
        self.assertIn("scientific_name", extracted)
        self.assertGreater(len(missing), 0)
        self.assertLess(score, 100)

    def test_sex_detected_from_symbol(self):
        """♂ symbol in OCR triggers 'sex' as missing if not extracted."""
        entities = {"scientific_names": [{"name": "Bombus"}]}
        ocr = "Bombus ♂"
        _, _, missing, _ = assess_extraction_success(entities, ocr)
        self.assertIn("sex", missing)

    def test_gbif_validated_counts(self):
        """GBIF validation adds 'gbif_validated' to extracted list."""
        entities = {
            "scientific_names": [{
                "name": "Apis mellifera",
                "gbif_validation": {"matchType": "EXACT", "usageKey": 1}
            }]
        }
        _, extracted, _, _ = assess_extraction_success(entities, "Apis mellifera")
        self.assertIn("gbif_validated", extracted)


# --------------- generate_dwc --------------- #


class TestGenerateDwc(unittest.TestCase):

    def test_single_label_produces_one_record(self):
        labels = [_make_label(entities=_rich_entities())]
        records = generate_dwc(labels)
        self.assertEqual(len(records), 1)

    def test_record_has_dwc_prefix(self):
        labels = [_make_label(entities=_rich_entities())]
        records = generate_dwc(labels)
        record = records[0]
        self.assertIn("dwc:scientificName", record)
        self.assertIn("dwc:recordedBy", record)
        self.assertIn("dwc:basisOfRecord", record)
        self.assertEqual(record["dwc:basisOfRecord"], "PreservedSpecimen")

    def test_occurrence_id_from_entity(self):
        labels = [_make_label(entities=_rich_entities())]
        records = generate_dwc(labels)
        self.assertEqual(records[0]["dwc:occurrenceID"], "http://example.org/MFN-001")

    def test_multiple_labels_same_specimen_merge(self):
        """Two labels from the same source_image merge into one DwC record."""
        ent1 = {"scientific_names": [{"name": "Bombus"}], "recordedBy": "Smith"}
        ent2 = {"eventDate": "2020-01-01", "geographic_data": {"locality": "Berlin"}}
        labels = [
            _make_label(source_image="img.jpg", label_filename="l1.jpg", entities=ent1),
            _make_label(source_image="img.jpg", label_filename="l2.jpg", entities=ent2),
        ]
        records = generate_dwc(labels)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["dwc:scientificName"], "Bombus")
        self.assertEqual(records[0]["dwc:eventDate"], "2020-01-01")

    def test_empty_entities_skipped(self):
        """Labels without entity_extraction produce no DwC records."""
        labels = [_make_label(entities=None)]
        records = generate_dwc(labels)
        self.assertEqual(len(records), 0)

    def test_gbif_validated_name_takes_priority(self):
        """GBIF-validated scientific name overwrites an earlier unvalidated one."""
        ent1 = {"scientific_names": [{"name": "Bombus sp."}]}
        ent2 = {
            "scientific_names": [{
                "name": "Bombus terrestris",
                "authority": "L.",
                "gbif_validation": {"usageKey": 99},
            }]
        }
        labels = [
            _make_label(source_image="x.jpg", label_filename="a.jpg", entities=ent1),
            _make_label(source_image="x.jpg", label_filename="b.jpg", entities=ent2),
        ]
        records = generate_dwc(labels)
        self.assertEqual(records[0]["dwc:scientificName"], "Bombus terrestris")


# --------------- generate_opends --------------- #


class TestGenerateOpends(unittest.TestCase):

    def test_single_label_produces_one_record(self):
        labels = [_make_label(entities=_rich_entities())]
        records = generate_opends(labels)
        self.assertEqual(len(records), 1)

    def test_record_has_ods_prefix(self):
        labels = [_make_label(entities=_rich_entities())]
        records = generate_opends(labels)
        record = records[0]
        self.assertIn("ods:scientificName", record)
        self.assertIn("ods:type", record)
        self.assertEqual(record["ods:type"], "PhysicalSpecimenDigitalRecord")

    def test_ods_id_from_occurrence(self):
        labels = [_make_label(entities=_rich_entities())]
        records = generate_opends(labels)
        self.assertEqual(records[0]["ods:ID"], "http://example.org/MFN-001")


# --------------- build_master_json --------------- #


class TestBuildMasterJson(unittest.TestCase):

    def test_groups_by_source_image(self):
        labels = [
            _make_label(source_image="a.jpg", label_filename="a_l1.jpg", entities={}),
            _make_label(source_image="a.jpg", label_filename="a_l2.jpg", entities={}),
            _make_label(source_image="b.jpg", label_filename="b_l1.jpg", entities={}),
        ]
        master = build_master_json(labels, {})
        self.assertEqual(len(master), 2)
        src_images = {entry["source_image"] for entry in master}
        self.assertEqual(src_images, {"a.jpg", "b.jpg"})

    def test_provenance_included(self):
        labels = [_make_label(entities={})]
        master = build_master_json(labels, {})
        self.assertIn("provenance", master[0])
        self.assertIn("pipeline", master[0]["provenance"])

    def test_empty_input(self):
        master = build_master_json([], {})
        self.assertEqual(master, [])

    def test_labels_nested_correctly(self):
        labels = [_make_label(source_image="x.jpg", entities={})]
        master = build_master_json(labels, {})
        self.assertEqual(len(master[0]["labels"]), 1)


if __name__ == "__main__":
    unittest.main()
