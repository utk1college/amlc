"""Transliterated and English spellings of a name must share a skeleton."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from text_normalize import normalize_address, normalize_name, skeleton  # noqa: E402


def key(name):
    return skeleton(normalize_name(name))


class TextNormalizeTest(unittest.TestCase):
    def test_indic_scripts_meet_english_spelling(self):
        english = key("Lakshmi Consultancy Pvt. Ltd.")
        self.assertEqual(key("लक्ष्मी कंसल्टेंसी प्राइवेट लिमिटेड"), english)
        self.assertEqual(key("శ్రీ సాయి ట్రేడర్స్"), key("Sri Sai Traders"))

    def test_noise_wrappers_and_domains(self):
        self.assertEqual(normalize_name("-- Holloway Peak Inc Seafood"), "holloway peak incorporated seafood")
        self.assertEqual(normalize_name("porternall.com"), "porternall")
        self.assertEqual(normalize_name("<< Team École"), "team ecole")

    def test_address_abbreviations(self):
        self.assertEqual(normalize_address("3220. Gale Street, Indianapolis"),
                         normalize_address("3220 GALE SAINT, INDIANAPOLIS"))


if __name__ == "__main__":
    unittest.main()
