"""Transliterated and English spellings of a name must share a skeleton."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from text_normalize import (core_name, name_variants, normalize_address, normalize_name,  # noqa: E402
                            script_flags, skeleton)


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

    def test_dotted_abbreviations_become_one_token(self):
        self.assertEqual(normalize_name("Brightos Safety P.L.L.C."), "brightos safety pllc")
        self.assertEqual(normalize_name("NEUVILLE FILMS ECOLE S.C.I."), "neuville films ecole sci")
        self.assertEqual(normalize_name("Verte S.A.R.L"), normalize_name("Verte SARL"))
        self.assertEqual(normalize_name("Lakshmi Pvt. Ltd."), "lakshmi private limited")

    def test_core_name_drops_legal_forms_and_dotted_credentials_only(self):
        self.assertEqual(core_name("Howell Diamond L.L.C."), "howell diamond")
        self.assertEqual(core_name("Mary Smith D.O."), "mary smith")
        self.assertEqual(core_name("Do Good Foods LLC"), "do good foods")
        self.assertEqual(core_name("Entraineurs Club SARL"), "entraineurs club")

    def test_alias_names_are_split_only_between_two_names(self):
        self.assertEqual(name_variants("Pyrapyra aka Interstate Prudential PLLC"),
                         ["pyrapyra aka interstate prudential pllc", "pyrapyra", "interstate prudential pllc"])
        self.assertEqual(name_variants("Rizavera D.B.A. Montgomery Newbury")[2], "montgomery newbury")
        self.assertEqual(name_variants("Korsynio DBA: Club du Encres")[2], "club du encres")
        self.assertEqual(name_variants("Miranovi trading as Centre Hospitalier")[2], "centre hospitalier")
        self.assertEqual(name_variants("Fayeumbra Co F/K/A First Catholic Church")[1:],
                         ["fayeumbra company", "first catholic church"])
        self.assertEqual(name_variants("Aka Solution Private Limited"), ["aka solution private limited"])
        self.assertEqual(name_variants("DBA Exports Private Limited"), ["dba exports private limited"])

    def test_french_street_types_and_number_markers(self):
        self.assertEqual(normalize_address("N° 9 Place Vanhoenacker, Lille"), normalize_address("9 PL. VANHOENACKER, LILLE"))
        self.assertEqual(normalize_address("(27) R. CHALRES HOUSSET"), "27 rue chalres housset")
        self.assertEqual(normalize_address("12 Bd de la Liberte"), normalize_address("12 Boulevard de la Liberté"))
        self.assertEqual(normalize_address("Shop No. 5, MG Road"), "shop 5 mg rd")

    def test_script_flags(self):
        self.assertEqual(script_flags("लक्ष्मी कंसल्टेंसी"), (True, False))
        self.assertEqual(script_flags("Mérignac Services SARL"), (False, True))
        self.assertEqual(script_flags("Acme Inc"), (False, False))


if __name__ == "__main__":
    unittest.main()
