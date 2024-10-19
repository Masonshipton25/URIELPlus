import logging
import os
import sys

import numpy as np

class BaseURIEL:
    """
        Configuration options:
            cache (bool, optional): Whether to cache distance languages and changes to databases. 
            Defaults to False.

            aggregation (str, optional): Whether to perform a union ('U') or average ('A') operation on data for aggregation and distance
            calculations. 
            Defaults to 'U'.

            fill_with_base_lang (bool, optional): Whether to fill missing values during aggregation using parent language data. 
            Defaults to False.

            distance_metric (str, optional): The distance metric to use for distance calculations ("angular" or "cosine"). 
            Defaults to "angular".
    """
    cache = False
    aggregation = 'U'
    fill_with_base_lang = True
    distance_metric = "angular"

    def __init__(self, feats, langs, data, sources):
        self.files = ["family_features.npz", "features.npz", "geocoord_features.npz"]
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.logger = logging.getLogger(self.__class__.__name__)

        self.feats = feats
        self.langs = langs
        self.data = data
        self.sources = sources


    def get_cache(self): 
        """
            Returns whether to cache distance languages and changes to databases.

            Returns:
                bool: True if caching is enabled, False otherwise.
        """
        return self.cache

    def set_cache(self, cache):
        """
            Sets whether to cache distance languages and changes to databases.

            Args:
                cache (bool): True to enable caching, False otherwise.
                
            Logging:
                Error: Logs an error if the provided cache value is not a valid boolean value (True or False).
            
        """
        if isinstance(cache, bool):
            self.cache = cache
        else:
            logging.error(f"Invalid boolean value: {cache}. Valid boolean values are True and False.")
            sys.exit(1)

        
    def get_aggregation(self):
        """
            Returns whether to perform a union ('U') or average ('A') operation on data for aggregation and distance calculations.

            Returns:
                str: 'U' if aggregation is union, 'A' if aggregation is average.
        """
        return self.aggregation
    
    def set_aggregation(self, aggregation):
        """
            Sets whether to perform a union ('U') or average ('A') operation on data for aggregation and distance calculations.

            Args:
                aggregation (str): Whether to perform a union ('U') or average ('A') operation on data for aggregation and distance calculations..
                
            Logging:
                Error: Logs an error if the provided strategy value is invalid.
            
        """
        aggregations = ['U', 'A']
        if aggregation in aggregations:
            self.aggregation = aggregation
        else:
            logging.error(f"Invalid aggregation: {aggregation}. Valid aggregations are {aggregations}.")
            sys.exit(1)

        
    def get_fill_with_base_lang(self):
        """
            Returns whether to fill missing values during aggregation using parent language data.

            Returns:
                bool: True if filling missing values with parent language data is enabled, False otherwise.
        """
        return self.fill_with_base_lang

    def set_fill_with_base_lang(self, fill_with_base_lang):
        """
            Sets whether to fill missing values during aggregation using parent language data.

            Args:
                fill_with_base_lang (bool): True to enable filling with base language, False otherwise.
                
            Logging:
                Error: Logs an error if the provided fill_with_base_lang value is not a valid boolean value (True or False).
            
        """
        if isinstance(fill_with_base_lang, bool):
            self.fill_with_base_lang = fill_with_base_lang
            self.dialects = self.get_dialects()
        else: 
            logging.error(f"Invalid boolean value: {fill_with_base_lang}. Valid boolean values are True and False.") 
            sys.exit(1)   

    
    def get_distance_metric(self):
        """
            Returns the distance metric to use for distance calculations.

            Returns:
                str: The distance metric to use for distance calculations.
        """
        return self.distance_metric
    
    def set_distance_metric(self, distance_metric):
        """
            Sets the distance metric to use for distance calculations.

            Args:
                distance_metric (bool): The distance metric to use for distance calculations.
                
            Logging:
                Error: Logs an error if the provided distance metric value is invalid.
            
        """
        distance_metrics = ["angular", "cosine"]
        if distance_metric in distance_metrics:
            self.distance_metric = distance_metric
        else:
            logging.error(f"Invalid distance metric: {distance_metric}. Valid distance metrics are {distance_metrics}.")
            sys.exit(1)


    def is_iso_code(self, lang):
        """
            Checks if a provided language code is in ISO 639-3 code format.

            Args:
                lang (str): The language code to check.

            Returns:
                bool: True if the code is in ISO 639-3 code format (3 alphabetic characters); otherwise, False.
        """
        return (len(lang) == 3 and lang.isalpha())
    
    def is_iso_codes(self):
        """
            Checks if all the languages in URIEL+ are represented in ISO 639-3 code format.

            Returns:
                bool: True if all languages are in ISO 639-3 code format; otherwise, False.
        """
        return all(self.is_iso_code(lang) for langs in self.langs for lang in langs)
    

    def is_glottocode(self, lang):
        """
            Checks if a provided language code is in Glottocode format.

            Args:
                lang (str): The language code to check.

            Returns:
                bool: True if the code is in Glottocode format (4 alphabetic characters followed by 4 numeric characters); otherwise, False.
        """
        return (len(lang) == 8 and lang[:4].isalpha() and lang[4:].isnumeric())

    def is_glottocodes(self):
        """
            Checks if all the languages in URIEL+ are represented in Glottocode format.

            Returns:
                bool: True if all languages are in Glottocode format; otherwise, False.
        """
        return all(self.is_glottocode(lang) for langs in self.langs for lang in langs)
    


    def get_english_dialects(self):
        """
            Returns a list of English dialects.

            This function identifies English dialects based on the presence of Macro-English and the absence of Guinea Coast Croele English and 
            Pacific Creole English in a language's phylogeny vector language representation (ISO 639-3 or Glottocode).

            Returns:
                list: A list of code representations for English dialects.

            Logging:
                Error: If the languages in URIEL+ are not all in either ISO 639-3 or Glottocode representation.
        """
        if not self.is_glottocodes() and not self.is_iso_codes():
            logging.error("Cannot retrieve English dialects if languages in URIEL+ are not all of either ISO 639-3 or Glottocode language representation.")
            sys.exit(1)
        eng_dialects = []

        feat_indices = []
        for feat in ["F_Macro-English", "F_Guinea Coast Creole English", "F_Pacific Creole English"]:
            feat_indices.append(np.where(self.feats[0] == feat)[0][0])

        for lang in self.langs[0]:
            lang_index = np.where(self.langs[0] == lang)[0][0]
            if 1.0 in self.data[0][lang_index][feat_indices[0]] and 0.0 in self.data[0][lang_index][feat_indices[1]] and 0.0 in self.data[0][lang_index][feat_indices[2]]:
                eng_dialects.append(lang)

        if self.is_glottocodes():
            eng_dialects.remove("stan1293")
        elif self.is_iso_codes():
            eng_dialects.remove("eng")

        return eng_dialects
    
    def get_dialects(self):
        """
            Returns a dictionary of dialects, with keys being the base languages and values being a list of the dialects.

            This function identifies dialects for specific languages (e.g., Spanish, French, English) based on the current 
            language representation (ISO 639-3 or Glottocode).

            Returns:
                dict: A dictionary where keys are indices of base languages, and values are lists of dialect language codes.

            Logging:
                Error: If the languages in URIEL+ are not all in either ISO 639-3 or Glottocode representation.
        """
        if not self.is_glottocodes() and not self.is_iso_codes():
            logging.error("Cannot retrieve English dialects if languages in URIEL+ are not all of either ISO 639-3 or Glottocode language representation.")
            sys.exit(1)
        if self.is_glottocodes():
            SPANISH_DIALECTS = ["lore1243"]
            FRENCH_DIALECTS = ["caju1236", "gulf1242"]
            ENGLISH_DIALECTS = self.get_english_dialects()
            GERMAN_DIALECTS = ["colo1254", "hutt1235", "midd1318", "midd1343", "nort2627", "north2628", "penn1240", "uppe1400"]
            MALAY_DIALECTS = ["ambo1250", "baba1267", "baca1243", "bali1279", "band1353", "bera1262", "buki1247", "cent2053", "coco1260", "jamb1236", "keda1251", "kota1275", "kupa1239", "lara1260", "maka1305", "mala1479", "mala1480", "mala1481", "nege1240", "nort2828", "papu1250", "patt1249", "saba1263", "sril1245", "teng1267"]
            ARABIC_DIALECTS = ["alge1239", "alge1240", "anda1287", "baha1259", "chad1249", "cypr1248", "dhof1235", "east2690", "egyp1253", "gulf1241", "hadr1236", "hija1235", "jude1264", "jude1265", "jude1266", "jude1267", "khor1274", "liby1240", "meso1252", "moro1292", "najd1235", "nort3139", "nort3142", "oman1239", "said1239", "sana1295", "suda1236", "taiz1242", "taji1248", "tuni1259", "uzbe1248"]
            DIALECTS = {np.where(self.langs[1] == "stan1288")[0][0]: SPANISH_DIALECTS,
                        np.where(self.langs[1] == "stan1290")[0][0]: FRENCH_DIALECTS,
                        np.where(self.langs[1] == "stan1293")[0][0]: ENGLISH_DIALECTS,
                        np.where(self.langs[1] == "stan1295")[0][0]: GERMAN_DIALECTS,
                        np.where(self.langs[1] == "stan1306")[0][0]: MALAY_DIALECTS,
                        np.where(self.langs[1] == "stan1318")[0][0]: ARABIC_DIALECTS}
        elif self.is_iso_codes():
            SPANISH_DIALECTS = ["spq"]
            FRENCH_DIALECTS = ["frc"]
            ENGLISH_DIALECTS = self.get_english_dialects()
            GERMAN_DIALECTS = ["gct", "geh", "gml", "gmh", "nds", "frs", "pdc", "sxu"]
            MALAY_DIALECTS = ["abs", "mbf", "btj", "mhp", "bpq", "bve", "bvu", "pse", "coa", "jax", "meo", "mqg", "mkn", "lrt", "mfp", "zlm", "xdy", "xmm", "zmi", "max", "pmy", "mfa", "msi", "sci", "vkt"]
            ARABIC_DIALECTS = ["arq", "aao", "xaa", "abv", "shu", "acy", "adf", "avl", "arz", "afb", "ayh", "acw", "yud", "aju", "yhd", "jye", "ayl", "acm", "ary", "ars", "apc", "ayp", "acx", "aec", "ayn", "apd", "acq", "abh", "aeb", "auz"]
            DIALECTS = {np.where(self.langs[1] == "spa")[0][0]: SPANISH_DIALECTS,
                        np.where(self.langs[1] == "fra")[0][0]: FRENCH_DIALECTS,
                        np.where(self.langs[1] == "eng")[0][0]: ENGLISH_DIALECTS,
                        np.where(self.langs[1] == "deu")[0][0]: GERMAN_DIALECTS,
                        np.where(self.langs[1] == "zsm")[0][0]: MALAY_DIALECTS,
                        np.where(self.langs[1] == "arb")[0][0]: ARABIC_DIALECTS}

        return DIALECTS