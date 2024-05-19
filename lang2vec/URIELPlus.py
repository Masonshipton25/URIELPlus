import os, shutil
import csv

import math
import re

from .urielplus_csvs.duplicate_feature_sets import _u, _ug, _uai, _ue

import sys

import logging

import datetime
import contexttimer
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, \
    accuracy_score, precision_score, recall_score
from fancyimpute import SoftImpute

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from .urielplus_csvs.resource_level_langs import high_resource_languages_URIELPlus, medium_resource_languages_URIELPlus, low_resource_languages_URIELPlus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

''' 
URIEL+ library for integrating new and updated databases into URIEL and robust distance calculations.

Authors: Aditya Khan, Mason Shipton, David Anugraha, Kaiyao Duan, Phuong H. Hoang, Eric Khiu, A. Seza Doğruöz, En-Shiun Annie Lee
Last modified: September 27, 2024
'''

class URIELPlus:
    def __init__(self, cache=False, aggregation='U', fill_with_base_lang=False, distance_metric="angular"):
        """
            Initializes the URIEL+ class, setting up vector identifications of languages and configuration options.

            Args:
                cache (bool, optional): Whether to cache distance languages and changes to databases. Defaults to False.

                aggregation (str, optional): Whether to perform a union ('U') or average ('A') operation on data for aggregation and distance
                calculations. Defaults to 'U'.

                fill_with_base_lang (bool, optional): Whether to fill missing values during aggregation using parent language data. 
                Defaults to False.

                distance_metric (str, optional): The distance metric to use for distance calculations ("angular" or "cosine"). 
                Defaults to "angular".

            Logging:
                Info: Logs information when a file is missing in the data directory and copied from the old_data directory.

                Error: Logs an error if a file is not found in the old_data directory.
        """
        self.files = ['family_features.npz', 'features.npz', 'geocoord_features.npz']
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.logger = logging.getLogger(self.__class__.__name__)
        self.loaded_features  = []

        # Load each vector type, logging info and errors as necessary
        for file in self.files:
            file_path = os.path.join(self.cur_dir, 'data', file)
            if not os.path.isfile(file_path):
                logging.info(f"{file_path} is missing in 'data'. Copying from 'old_data'...")

                old_file_path = os.path.join(self.cur_dir, 'data', 'old_data', file)
                try:
                    shutil.copy(old_file_path, file_path)
                except FileNotFoundError:
                    logging.error(f"{file} not found in 'old_data'.")
                    return 
            # Load the vector data
            with np.load(file_path, allow_pickle=True) as l:
                self.loaded_features.append(dict(l)) 

        # Extract features, languages, data, and sources from loaded features
        self.feats = [l["feats"] for l in self.loaded_features]
        self.langs = [l["langs"] for l in self.loaded_features]
        self.data = [l["data"] for l in self.loaded_features]
        self.sources = [l["sources"] for l in self.loaded_features]

        # Initialize other attributes
        self.cache = cache
        self.aggregation = aggregation
        self.fill_with_base_lang = fill_with_base_lang
        self.distance_metric = distance_metric



    def get_loaded_features(self, l_name):
        """
            Returns the URIEL+ loaded features associated with the provided name, if the name is valid.

            Args:
                l_name (str): The name of the loaded features to return. Valid options are 'phylogeny', 'typological', or 'geography'.

            Returns:
                np.ndarray: The corresponding loaded features as a NumPy array.

            Logging:
                Error: Logs an error if the provided loaded features name is invalid.
            
        """
        l_map = {
            "phylogeny": self.loaded_features[0],
            "typological": self.loaded_features[1],
            "geography": self.loaded_features[2],
        }
        if l_name in l_map:
            return l_map[l_name]
        logging.error(f"Unknown loaded features: {l_name}. Valid loaded features are {list(l_map.keys())}.")
        return None

    """
        The following three functions return loaded features representing phylogeny, typological, and geography vectors, respectively.

        Returns:
            np.ndarray: The corresponding loaded features as a NumPy array.
    """
    def get_phylogeny_loaded_features(self):
        """Returns the phylogeny loaded features."""
        return self.loaded_features[0]
    
    def get_typological_loaded_features(self):
        """Returns the typological loaded features."""
        return self.loaded_features[1]
    
    def get_geography_loaded_features(self):
        """Returns the geography loaded features."""
        return self.loaded_features[2]
    

    def set_loaded_features(self, l_name, file):
        """
            Updates the loaded features associated with the provided name by loading data from the provided file.

            Args:
                l_name (str): The name of the loaded_features to update. Valid options are 'phylogeny', 'typological', or 'geography'.
                file (str): The file name to load the loaded features data from.

            Logging:
                Error: Logs an error if the provided loaded features name is invalid or if the file loading fails.

        """
        l_map = {
            "phylogeny": 0,
            "typological": 1,
            "geography": 2,
        }

        if l_name in l_map:
            file_path = os.path.join(self.cur_dir, 'data', file)
            
            try:
                with np.load(file_path, allow_pickle=True) as l:
                    l_idx = l_map[l_name]
                    self.loaded_features[l_idx] = dict(l)
                    self.feats[l_idx] = l["feats"]
                    self.langs[l_idx] = l["langs"]
                    self.data[l_idx] = l["data"]
                    self.sources[l_idx] = l["sources"]
                    self.files[l_idx] = file
                    logging.info(f"{l_name} loaded features updated successfully from {file}.")
            except FileNotFoundError:
                logging.error(f"File not found: {file_path}. Failed to update {l_name} loaded features.")
            except Exception as e:
                logging.error(f"An error occurred while loading the file {file}: {e}")
        else:
            logging.error(f"Unknown loaded features: {l_name}. Valid loaded features are {list(l_map.keys())}.")
    
    """
        The following three functions updates loaded features representing phylogeny, typological, and geography vectors, respectively.

        Args:
            file (str): The file name to load the loaded features data from.
    """    
    def set_phylogeny_loaded_features(self, file):
        """Updates the phylogeny loaded features."""
        self.set_loaded_features(self, "phylogeny", file)

    def set_typological_loaded_features(self, file):
        """Updates the typological loaded features."""
        self.set_loaded_features(self, "typological", file)

    def set_geography_loaded_features(self, file):
        """Updates the geography loaded features."""
        self.set_loaded_features(self, "geography", file)
    


    def get_arrays(self, l_name):
        """
            Returns the arrays within the URIEL+ loaded features associated with the provided name, if the name is valid.

            Args:
                l_name (str): The name of the loaded features to return. Valid options are 'phylogeny', 'typological', or 'geography'.

            Returns:
                tuple: The arrays within the corresponding loaded features as NumPy arrays.
        """
        loaded_features = self.get_loaded_features(l_name)
        return loaded_features["feats"], loaded_features["data"], loaded_features["langs"], loaded_features["sources"]
    
    """
        The following three functions return all the arrays within loaded features representing 
        phylogeny, typological, and geography vectors, respectively.

        Returns:
            tuple: The arrays within the corresponding loaded features as NumPy arrays.
    """
    def get_phylogeny_arrays(self):
        """Returns the phylogeny arrays."""
        return self.get_arrays("phylogeny")
    
    def get_typological_arrays(self):
        """Returns the typological arrays."""
        return self.get_arrays("typological")
    
    def get_geography_arrays(self):
        """Returns the geography arrays."""
        return self.get_arrays("geography")


    """
        The following four functions return all the arrays from all loaded features representing 
        features, languages, feature data, and sources, respectively.

        Returns:
            list: A list of NumPy arrays containing the corresponding arrays from each loaded features.
    """
    def get_features_arrays(self):
        """Returns the features arrays."""
        return self.feats
    
    def get_languages_arrays(self):
        """Returns the languages arrays."""
        return self.langs
    
    def get_data_arrays(self):
        """Returns the data arrays."""
        return self.data
    
    def get_sources_arrays(self):
        """Returns the sources arrays."""
        return self.sources


    """
        The following functions return the array corresponding with a specific loaded features
        and one of either features, languages, data, or sources arrays.

        Returns:
            np.ndarray: A NumPy array of either the features, languages, data, or sources of a specific loaded features.
    """
    def get_phylogeny_features_array(self):
        """Returns the features array of the phylogeny loaded features."""
        return self.feats[0]
    
    def get_typological_features_array(self):
        """Returns the features array of the typological loaded features."""
        return self.feats[1]
    
    def get_geography_features_array(self):
        """Returns the features array of the geography loaded features."""
        return self.feats[2]
    
    def get_phylogeny_languages_array(self):
        """Returns the languages array of the phylogeny loaded features."""
        return self.langs[0]
    
    def get_typological_languages_array(self):
        """Returns the languages array of the typological loaded features."""
        return self.langs[1]
    
    def get_geography_languages_array(self):
        """Returns the languages array of the geography loaded features."""
        return self.langs[2]
    
    def get_phylogeny_data_array(self):
        """Returns the data array of the phylogeny loaded features."""
        return self.data[0]
    
    def get_typological_data_array(self):
        """Returns the data array of the typological loaded features."""
        return self.data[1]
    
    def get_geography_data_array(self):
        """Returns the data array of the geography loaded features."""
        return self.data[2]
    
    def get_phylogeny_sources_array(self):
        """Returns the sources array of the phylogeny loaded features."""
        return self.sources[0]
    
    def get_typological_sources_array(self):
        """Returns the sources array of the typological loaded features."""
        return self.sources[1]
    
    def get_geography_sources_array(self):
        """Returns the sources array of the geography loaded features."""
        return self.sources[2]
    

    
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

        
    def get_aggregation(self):
        """
            Returns whether to perform a union ('U') or average ('A') operation on data for aggregation and distance calculations.

            Returns:
                str: 'U if aggregation is union, 'A' if aggregation is average.
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
        distance_metrics = ['angular', 'cosine']
        if distance_metric in distance_metrics:
            self.distance_metric = distance_metric
        else:
            raise logging.error(f"Invalid distance metric: {distance_metric}. Valid distance metrics are {distance_metrics}.")



    def query_yes_no(self, question, default="yes"):
        """
            Prompts the user with a yes/no question and returns their response.

            Args:
                question (str): The question to ask the user.
                default (str): The default answer if the user just hits Enter. It must be "yes", "no", or None.

            Returns:
                bool: True if the user answered "yes"; False if the user answered "no".

            Raises:
                ValueError: If the default answer is not "yes", "no", or None.
        """
        valid = {"yes": True, "y": True, "ye": True,
                "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise logging.error("invalid default answer: '%s'" % default)

        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' "
                                "(or 'y' or 'n').\n")
    

    def set_uriel(self):
        """
            Restores the URIEL knowledge base by copying necessary files to the main data directory.

            The function prompts if the user wants to revert to URIEL, and if yes, then moves all old data files back to the main directory.
        """
        files_to_copy = ['distances_languages.txt', 'distances.zip', 'distances2.zip', 
                         'family_features.npz', 'feature_averages.npz', 'feature_predictions.npz', 
                         'features.npz', 'geocoord_features.npz', 'learned.npy', 'letter_codes.json']
        cont = self.query_yes_no(f"Resetting to URIEL, integrating all databases in URIEL+, or integrating custom databases in URIEL+ involves copying the files {files_to_copy} into the data directory. Any files with the same name will be replaced. Continue?")
        if not cont:
            return
        for file in files_to_copy:
            from_file_path = os.path.join(self.cur_dir, "data", "old_data", file)
            to_file_path = os.path.join(self.cur_dir, "data", file)
            try:
                shutil.copy(from_file_path, to_file_path)
            except Exception as e:
                logging.error(f"Difficulty copying {from_file_path} to {to_file_path}: {e}")
        self.loaded_features = []
        for file in self.files:
            file_path = os.path.join(self.cur_dir, 'data', file)
            with np.load(file_path, allow_pickle=True) as l:
                self.loaded_features.append(dict(l)) 

        self.feats = [l["feats"] for l in self.loaded_features]
        self.langs = [l["langs"] for l in self.loaded_features]
        self.data = [l["data"] for l in self.loaded_features]
        self.sources = [l["sources"] for l in self.loaded_features]

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
        eng_dialects = []
        feats, feature_data, langs = self.feats[0], self.data[0], self.langs[0]

        feat_indices = []
        for feat in ['F_Macro-English', 'F_Guinea Coast Creole English', 'F_Pacific Creole English']:
            feat_indices.append(np.where(feats == feat)[0][0])

        for lang in langs:
            lang_index = np.where(langs == lang)[0][0]
            if 1.0 in feature_data[lang_index][feat_indices[0]] and 0.0 in feature_data[lang_index][feat_indices[1]] and 0.0 in feature_data[lang_index][feat_indices[2]]:
                eng_dialects.append(lang)

        if self.is_glottocodes():
            eng_dialects.remove("stan1293")
        elif self.is_iso_codes():
            eng_dialects.remove("eng")

        return eng_dialects
    
    def get_dialects(self):
        """
            Returns a dictionary of dialects, with keys being the base languages and values being a list of the dialects.

            This function identifies dialects for specific languages (e.g., Spanish, French, English) based on the current language representation (ISO 639-3 or Glottocode).

            Returns:
                dict: A dictionary where keys are indices of base languages, and values are lists of dialect language codes.

            Logging:
                Error: If the languages in URIEL+ are not all in either ISO 639-3 or Glottocode representation.
        """
        if not self.is_glottocodes() and not self.is_iso_codes():
            logging.error("Cannot retrieve English dialects if languages in URIEL+ are not all of either ISO 639-3 or Glottocode language representation.")
        langs = self.langs[1]
        if self.is_glottocodes():
            SPANISH_DIALECTS = ["lore1243"]
            FRENCH_DIALECTS = ["caju1236", "gulf1242"]
            ENGLISH_DIALECTS = self.get_english_dialects()
            GERMAN_DIALECTS = ["colo1254", "hutt1235", "midd1318", "midd1343", "nort2627", "north2628", "penn1240", "uppe1400"]
            MALAY_DIALECTS = ["ambo1250", "baba1267", "baca1243", "bali1279", "band1353", "bera1262", "buki1247", "cent2053", "coco1260", "jamb1236", "keda1251", "kota1275", "kupa1239", "lara1260", "maka1305", "mala1479", "mala1480", "mala1481", "nege1240", "nort2828", "papu1250", "patt1249", "saba1263", "sril1245", "teng1267"]
            ARABIC_DIALECTS = ["alge1239", "alge1240", "anda1287", "baha1259", "chad1249", "cypr1248", "dhof1235", "east2690", "egyp1253", "gulf1241", "hadr1236", "hija1235", "jude1264", "jude1265", "jude1266", "jude1267", "khor1274", "liby1240", "meso1252", "moro1292", "najd1235", "nort3139", "nort3142", "oman1239", "said1239", "sana1295", "suda1236", "taiz1242", "taji1248", "tuni1259", "uzbe1248"]
            DIALECTS = {np.where(langs == "stan1288")[0][0]: SPANISH_DIALECTS,
                        np.where(langs == "stan1290")[0][0]: FRENCH_DIALECTS,
                        np.where(langs == "stan1293")[0][0]: ENGLISH_DIALECTS,
                        np.where(langs == "stan1295")[0][0]: GERMAN_DIALECTS,
                        np.where(langs == "stan1306")[0][0]: MALAY_DIALECTS,
                        np.where(langs == "stan1318")[0][0]: ARABIC_DIALECTS}
        elif self.is_iso_codes():
            SPANISH_DIALECTS = ["spq"]
            FRENCH_DIALECTS = ["frc"]
            ENGLISH_DIALECTS = self.get_english_dialects()
            GERMAN_DIALECTS = ["gct", "geh", "gml", "gmh", "nds", "frs", "pdc", "sxu"]
            MALAY_DIALECTS = ["abs", "mbf", "btj", "mhp", "bpq", "bve", "bvu", "pse", "coa", "jax", "meo", "mqg", "mkn", "lrt", "mfp", "zlm", "xdy", "xmm", "zmi", "max", "pmy", "mfa", "msi", "sci", "vkt"]
            ARABIC_DIALECTS = ["arq", "aao", "xaa", "abv", "shu", "acy", "adf", "avl", "arz", "afb", "ayh", "acw", "yud", "aju", "yhd", "jye", "ayl", "acm", "ary", "ars", "apc", "ayp", "acx", "aec", "ayn", "apd", "acq", "abh", "aeb", "auz"]
            DIALECTS = {np.where(langs == "spa")[0][0]: SPANISH_DIALECTS,
                        np.where(langs == "fra")[0][0]: FRENCH_DIALECTS,
                        np.where(langs == "eng")[0][0]: ENGLISH_DIALECTS,
                        np.where(langs == "deu")[0][0]: GERMAN_DIALECTS,
                        np.where(langs == "zsm")[0][0]: MALAY_DIALECTS,
                        np.where(langs == "arb")[0][0]: ARABIC_DIALECTS}

        return DIALECTS

    def set_glottocodes(self):
        """
            Sets the language codes in URIEL+ to Glottocodes.

            This function reads a mapping CSV file and applies the mappings to all relevant data files,
            saving the updated data back to disk if caching is enabled.
        """
        if self.is_glottocodes():
            logging.error('Already using Glottocodes.')
        
        logging.info("Converting ISO 639-3 codes to Glottocodes....")

        csv_path = os.path.join(self.cur_dir, 'urielplus_csvs', 'uriel_glottocode_map.csv')
        map_df = pd.read_csv(csv_path)

        for i, file in enumerate(self.files):
            #I tried combining some of these lines but I ran into errors.
            langs_df = pd.DataFrame(self.langs[i], columns=['code'])
            merged_df = pd.merge(langs_df, map_df, on='code', how='outer')
            merged_df = merged_df.dropna()
            merged_df = merged_df.drop(columns=['X', 'code'])
            merged_np = merged_df.to_numpy()
            na_indices = langs_df.index.difference(merged_df.index)
            data_cleaned = np.delete(self.data[i], na_indices, axis=0)

            self.langs[i] = merged_np
            self.data[i] = data_cleaned
            self.langs[i] = np.array([l[0] for l in self.langs[i]])

            if self.cache:
                np.savez(os.path.join(self.cur_dir, "data", file), 
                         feats=self.feats[i], data=self.data[i], langs=self.langs[i], sources=self.sources[i])
                
        logging.info("Conversion to Glottocodes complete.")

    def _get_new_features(self, feats, columns):
        """
            Identifies and returns the new features to URIEL+.

            Args:
                feats (np.ndarray): The current features array.
                columns (list): The list of all features.

            Returns:
                list: A list of new features to URIEL+.
        """
        featlist = feats.tolist()
        return [feat for feat in columns if feat not in featlist]
    
    def _get_new_languages(self, langs, data, column):
        """
            Identifies and returns the new languages to URIEL+.

            Args:
                langs (np.ndarray): The current languages array.
                data (pd.DataFrame): The new dataset containing all languages.
                column (str): The column in the dataset that contains language codes.

            Returns:
                list: A list of new languages to URIEL+.
        """
        langlist = langs.tolist()
        return [lang for lang in data[column] if lang not in langlist]
    
    def _set_new_data_dimensions(self, data, new_feats, new_langs, new_sources):
        """
            Expands the URIEL+ data array to accommodate new features, languages, and sources, initializing new values to -1.0.

            Args:
                data (np.ndarray): The current data array.
                new_feats (list): List of new features to add.
                new_langs (list): List of new languages to add.
                new_sources (list): List of new sources to add.

            Returns:
                np.ndarray: The expanded data array with new dimensions.
        """
        new_data = np.full(
            (data.shape[0] + len(new_langs), data.shape[1] + len(new_feats), data.shape[2] + len(new_sources)), 
            -1.0
        )
        new_data[:data.shape[0], :data.shape[1], :data.shape[2]] = data
        return new_data
    
    def is_database_incorporated(self, database):
        """
            Checks if a specific database has already been integrated into URIEL+.

            Args:
                database (str): The name of the database to check.

            Logging:
                Error: Logs if the database is already integrated.
        """
        if any(database in source for source in self.sources):
            logging.error(f'{database} database already integrated.')

    def _calculate_phylogeny_vectors(self):
        """
            This function reads the relevant CSV file and updates the phylogeny arrays based on the phylogeny classifications of new languages.

            If caching is enabled, updates the `family_features.npz` file.
            
        """
        csv_path = os.path.join(self.cur_dir, "urielplus_csvs", 'lang_fam_geo.csv')
        fam_geo_feat_csv = pd.read_csv(csv_path)

        new_langs = np.setdiff1d(self.langs[1], self.langs[0])

        for l in new_langs:
            fam = fam_geo_feat_csv.loc[fam_geo_feat_csv['code'] == l, 'families']
            if fam.empty:
                self.data[0] = np.vstack([self.data[0], -1.0 * np.ones((1, self.data[0].shape[1], self.data[0].shape[2]))])
                self.langs[0] = np.append(self.langs[0], l)
                continue
            fam = fam.values[0]
            if not isinstance(fam, str):
                self.data[0] = np.vstack([self.data[0], -1.0 * np.ones((1, self.data[0].shape[1], self.data[0].shape[2]))])
                self.langs[0] = np.append(self.langs[0], l)
                continue
            self.data[0] = np.vstack([self.data[0], 0.0 * np.ones((1, self.data[0].shape[1], self.data[0].shape[2]))])
            self.langs[0] = np.append(self.langs[0], l)
            new_lang_idx = np.where(self.langs[0] == l)[0]
            fams = fam.split(',') if ',' in fam else [fam]
            for f in fams:
                f = f.lstrip()
                fam_string = 'F_' + f
                family_idx = np.where(self.feats[0] == fam_string)[0]
                if len(family_idx) == 0:
                    continue
                self.data[0][new_lang_idx, family_idx, -1] = 1.0

        if self.cache:
            np.savez(os.path.join(self.cur_dir, "data", self.files[0]), feats=self.feats[0], data=self.data[0], langs=self.langs[0], sources=self.sources[0])

    def _calculate_geocoord_vectors(self):
        """
            This function calculates the geographic distances between new languages and geocoordinates, creating geography vectors.

            If caching is enabled, updates the `geocoord_features.npz` file.
        """
        new_langs = np.setdiff1d(self.langs[1], self.langs[2])
        self.langs[2] = np.append(self.langs[2], new_langs)
        self.data[2] = self._set_new_data_dimensions(self.data[2], [], new_langs, [])

        coords = [list(map(int, re.findall(r'-?\d+', feat))) for feat in self.feats[2]]
        csv_path = os.path.join(self.cur_dir, "urielplus_csvs", 'lang_fam_geo.csv')
        fam_geo_feat_csv = pd.read_csv(csv_path)

        for i, l in enumerate(new_langs):
            lat = fam_geo_feat_csv.loc[fam_geo_feat_csv['code'] == l, 'lat']
            lon = fam_geo_feat_csv.loc[fam_geo_feat_csv['code'] == l, 'lon']
            lat, lon = lat.values[0], lon.values[0]

            if math.isnan(lat) or math.isnan(lon):
                self.data[2][self.data[2].shape[0] - len(new_langs) + i, :, -1] = -1.0
                continue

            distances = [math.dist([lat, lon], coord) for coord in coords]
            min_index = np.argmin(distances)
            self.data[2][self.data[2].shape[0] - len(new_langs) + i, min_index, -1] = 1.0

        if self.cache:
            np.savez(os.path.join(self.cur_dir, "data", self.files[2]), feats=self.feats[2], data=self.data[2], langs=self.langs[2], sources=self.sources[2])

    def combine_features(self, second_source, feat_sets):
        """
            Combines duplicate features and handles opposite features within a specified category of URIEL+ data.

            Args:
                second_source (str): The source of the secondary features to combine.
                feat_sets (list): A list of feature sets, each containing the primary feature and the secondary features to combine.
        """
        source_idx = np.where(self.sources[1] == second_source)[0][0]

        for feat_set in feat_sets:
                first_feat = feat_set[0]
                if isinstance(feat_set[1], float):
                    spec_value = feat_set[1]
                    secondary_feats = feat_set[2:]
                else:
                    secondary_feats = feat_set[1:]

                if first_feat not in self.feats[1]:
                        self.data[1] = self._set_new_data_dimensions(self.data[1], [first_feat], [], [])
                        self.feats[1] = self._get_new_features(self.feats[1], first_feat)
                        first_feat_idx = len(self.feats[1])-1
                else:
                    first_feat_idx = np.where(self.feats[1] == first_feat)[0][0]

                for feat in secondary_feats:
                        secondary_feat_idx = np.where(self.feats[1] == feat)[0][0]

                        for i in range(len(self.langs[1])):
                            secondary_data = self.data[1][i][secondary_feat_idx]
                            first_data = self.data[1][i][first_feat_idx]


                            for j in range(len(first_data)):
                                    if j == source_idx:
                                        if isinstance(feat_set[1], float):
                                            if (first_data[j] == -1.0 or first_data[j] == 0.0) and (secondary_data[j] == spec_value):
                                                    (self.data[1][i][first_feat_idx])[j] = spec_value
                                        else:
                                            if(secondary_data[j] > first_data[j]):
                                                (self.data[1][i][first_feat_idx])[j] = secondary_data[j]

        if self.cache:
            np.savez(os.path.join(self.cur_dir, self.files[1]), feats=self.feats[1], langs=self.langs[1], data=self.data[1], sources=self.sources[1])

    def inferred_features(self):
        """
            Combines duplicate features across all sources in URIEL+ by using the `combine_features` function.

            The function iterates through the available sources and combines features based on predefined sets of duplicate features.
        """
        logging.info("Updating feature data based on inferred features...")
        for source in self.sources[1]:
            self.combine_features(source, _u)

    def integrate_saphon(self, convert_glottocodes_param=False):
        """
            Updates URIEL+ with data from the updated SAPHON database.

            This function integrates the updated SAPHON data.

            Args: 
                convert_glottocodes_param (bool): If True, converts language codes to Glottocodes.
        """
        self.is_database_incorporated('UPDATED_SAPHON')

        if not self.is_glottocodes() and convert_glottocodes_param:
            self.set_glottocodes()
            logging.info("Importing updated SAPHON from 'saphon_data_glottocodes.csv'....")
        else:
            logging.info("Importing updated SAPHON from 'saphon_data.csv'....")

        saphon_csv = os.path.join(self.cur_dir, "urielplus_csvs", 'saphon_data_glottocodes.csv') if self.is_glottocodes() else os.path.join(self.cur_dir, "urielplus_csvs", 'saphon_data.csv')

        with open(saphon_csv, mode='r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)

            for row in csvreader:
                lang = row[0]
                if lang in self.langs[1]:
                    data_string = '['
                    feat_num = 1
                    for r in row[1:]:
                        if feat_num != 1:
                            data_string += ' '
                        data_string += r
                        if feat_num == 289:
                            data_string += ']'
                        else:
                            data_string += ',\n'
                        feat_num += 1
                    lang_index = np.where(self.langs[1] == lang)[0][0]
                    (self.data[1][lang_index])[:289, :10] = np.array(eval(data_string))

        index = np.where(self.sources[1] == 'PHOIBLE_SAPHON')
        self.sources[1][index] = "UPDATED_SAPHON"

        if self.cache:
            np.savez(os.path.join(self.cur_dir, "data", self.files[1]), 
                     feats=self.feats[1], data=self.data[1], langs=self.langs[1], sources=self.sources[1])

        logging.info("Updated SAPHON integration complete..")

    def integrate_bdproto(self):
        """
            Updates URIEL+ with data from the BDPROTO database.

            This function integrates the BDPROTO data, converting language codes to Glottocodes if necessary, 
            and updates the feature data in URIEL+.
        """
        self.is_database_incorporated('BDPROTO')

        logging.info("Importing BDPROTO from 'bdproto_data.csv'....")

        if not self.is_glottocodes():
            self.set_glottocodes()

        csv_path = os.path.join(self.cur_dir, "urielplus_csvs", 'bdproto_data.csv')
        df = pd.read_csv(csv_path)

        langlist = self.langs[1].tolist()
        new_langs = []
        modify_langs = []
        for lang in df['name']:
            if lang != "altaic" and lang != "australian" and lang != "finno-permic" and lang != "finno-ugric" and lang != "nostratic" and lang != "proto-baltic-finnic" and lang != "proto-eastern-oceanic" and lang != "proto-finno-permic" and lang != "proto-finno-saamic" and lang != "proto-mamore-guapore" and lang != "proto-nilo-saharan" and lang != "proto-tibeto-burman" and lang != "proto-totozoquean" and lang != "uralo-siberian":
                if ("UPDATE_LANG_URIEL" not in lang and lang not in langlist) or lang == "oldp1255_UPDATE_LANG_URIEL":
                    new_langs.append(lang.replace("_UPDATE_LANG_URIEL", ""))
                modify_langs.append(lang.replace("_UPDATE_LANG_URIEL", ""))

        self.data[1] = self._set_new_data_dimensions(self.data[1], [], new_langs, ["BDPROTO"])

        self.langs[1] = np.append(self.langs[1], np.array(new_langs).flatten())
        self.sources[1] = np.append(self.sources[1], 'BDPROTO')

        with open(csv_path, mode='r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)

            for row in csvreader:
                lang = row[0].replace("_UPDATE_LANG_URIEL", "")
                if lang in modify_langs:
                    row_list = row[1:]
                    float_row_list = []
                    for element in row_list:
                        float_row_list.append(float(element))
                    lang_index = np.where(self.langs[1] == lang)[0][0]
                    column_index = self.data[1].shape[2]-1
                    (self.data[1][lang_index])[:289, column_index] = float_row_list

        if self.cache:
            np.savez(os.path.join(self.cur_dir, "data", self.files[1]), 
                     feats=self.feats[1], data=self.data[1], langs=self.langs[1], sources=self.sources[1])

        self._calculate_phylogeny_vectors()
        self._calculate_geocoord_vectors()

        logging.info("BDPROTO integration complete.")

    def integrate_grambank(self):
        """
            Updates URIEL+ with data from the Grambank database.

            This function integrates the Grambank data, converting language codes to Glottocodes if necessary, 
            and updates the feature data in URIEL+.
        """
        self.is_database_incorporated('GRAMBANK')

        logging.info("Importing Grambank from 'grambank_data.csv'....")

        if not self.is_glottocodes():
            self.set_glottocodes()

        grambank_data = pd.read_csv(os.path.join(self.cur_dir, "urielplus_csvs", 'grambank_data.csv'))

        new_feats = self._get_new_features(self.feats[1], grambank_data.columns[1:])
        new_langs = self._get_new_languages(self.langs[1], grambank_data, 'code')
        new_source = "GRAMBANK"

        old_num_langs = self.data[1].shape[0]
        self.data[1] = self._set_new_data_dimensions(self.data[1], new_feats, new_langs, [new_source])

        self.feats[1] = np.append(self.feats[1], new_feats)

        new_langs_added = 0
        for i, lang in enumerate(grambank_data['code']):
            lang_index = None
            try:
                lang_index = np.where(self.langs[1] == lang)[0][0]
                if type(lang_index) not in [int, np.int64, float, np.float64]:
                    lang_index = old_num_langs + new_langs_added
                    new_langs_added += 1
            except:
                lang_index = old_num_langs + new_langs_added
                new_langs_added += 1
            for feat in grambank_data.columns[1:]:
                feat_index = np.where(self.feats[1] == feat)
                self.data[1][lang_index, feat_index, -1] = grambank_data[feat][i]
        self.langs[1] = np.append(self.langs[1], np.array(new_langs).flatten())
        self.sources[1] = np.append(self.sources[1], new_source)

        if self.cache:
            np.savez(os.path.join(self.cur_dir, "data", self.files[1]), 
                     feats=self.feats[1], data=self.data[1], langs=self.langs[1], sources=self.sources[1])

        self._calculate_phylogeny_vectors()
        self._calculate_geocoord_vectors()

        self.combine_features('GRAMBANK', _ug)

        logging.info("Grambank integration complete.")

    def integrate_apics(self):
        """
            Updates URIEL+ with data from the APiCS database.

            This function integrates the APiCS data, converting language codes to Glottocodes if necessary, 
            and updates the feature data in URIEL+.
        """
        self.is_database_incorporated('APICS')

        logging.info("Importing APiCS from 'apics_data.csv'....")

        if not self.is_glottocodes():
            self.set_glottocodes()

        apics_data = pd.read_csv(os.path.join(self.cur_dir, "urielplus_csvs", 'apics_data.csv'))

        new_langs = self._get_new_languages(self.langs[1], apics_data, 'Language_ID')

        apics_data = apics_data.drop(columns=['Name'])
        apics_data = apics_data[['Language_ID'] + [col for col in apics_data.columns if col != 'Language_ID']]

        new_feats = self._get_new_features(self.feats[1], apics_data.columns[1:])

        new_source = "APICS"

        old_num_langs = self.data[1].shape[0]
        self.data[1] = self._set_new_data_dimensions(self.data[1], new_feats, new_langs, [new_source])

        self.feats[1] = np.append(self.feats[1], new_feats)

        new_langs_added = 0
        for i, lang in enumerate(apics_data['Language_ID']):
            lang_index = None
            try:
                lang_index = np.where(self.langs[1] == lang)[0][0]
                if type(lang_index) not in [int, np.int64, float, np.float64]:
                    lang_index = old_num_langs + new_langs_added
                    new_langs_added += 1
            except:
                lang_index = old_num_langs + new_langs_added
                new_langs_added += 1
            for feat in apics_data.columns[1:]:
                feat_index = np.where(self.feats[1] == feat)

                self.data[1][lang_index, feat_index, -1] = apics_data[feat][i]
        self.langs[1] = np.append(self.langs[1], np.array(new_langs).flatten())
        self.sources[1] = np.append(self.sources[1], new_source)

        if self.cache:
            np.savez(os.path.join(self.cur_dir, "data", self.files[1]), 
                     feats=self.feats[1], data=self.data[1], langs=self.langs[1], sources=self.sources[1])

        self._calculate_phylogeny_vectors()
        self._calculate_geocoord_vectors()

        self.combine_features('APICS', _uai)

        logging.info("APiCS integration complete.")

    def integrate_ewave(self):
        """
            Updates URIEL+ with data from the EWAVE database.

            This function integrates the EWAVE data, converting language codes to Glottocodes if necessary,
            and updates the feature data in URIEL+.
        """
        self.is_database_incorporated('EWAVE')

        logging.info("Importing eWAVE from 'english_dialect_data.csv'....")

        if not self.is_glottocodes():
            self.set_glottocodes()

        df = pd.read_csv(os.path.join(os.path.join(self.cur_dir, "urielplus_csvs", 'english_dialect_data.csv')))

        new_langs = self._get_new_languages(self.langs[1], df, 'name')
        new_feats = self._get_new_features(self.feats[1], df.columns[1:])
        new_source = "EWAVE"

        old_num_langs = self.data[1].shape[0]
        self.data[1] = self._set_new_data_dimensions(self.data[1], df.columns[1:], new_langs, [new_source])

        self.feats[1] = np.append(self.feats[1], new_feats)

        new_langs_added = 0
        for i, lang in enumerate(df['name']):
            lang_index = None
            try:
                lang_index = np.where(self.langs[1] == lang)[0][0]
                if type(lang_index) not in [int, np.int64, float, np.float64]:
                    lang_index = old_num_langs + new_langs_added
                    new_langs_added += 1
            except:
                lang_index = old_num_langs + new_langs_added
                new_langs_added += 1
            for feat in df.columns[1:]:
                feat_index = np.where(self.feats[1] == feat)
                # feat_index = self.data[1].shape[1] - len(df.columns[1:]) + j
                self.data[1][lang_index, feat_index, -1] = df[feat][i]
        self.langs[1] = np.append(self.langs[1], np.array(new_langs).flatten())
        self.sources[1] = np.append(self.sources[1], new_source)

        if self.cache:
            np.savez(os.path.join(self.cur_dir, "data", self.files[1]), 
                     feats=self.feats[1], data=self.data[1], langs=self.langs[1], sources=self.sources[1])

        self._calculate_phylogeny_vectors()
        self._calculate_geocoord_vectors()

        self.combine_features('EWAVE', _ue)

        logging.info("eWAVE integration complete.")

    def integrate_databases(self):
        """
            Updates URIEL+ with data from all available databases (UPDATED_SAPHON, BDPROTO, GRAMBANK, APICS, EWAVE).
        """
        logging.info("Importing all databases....")

        self.set_uriel()
        self.integrate_saphon()
        self.integrate_bdproto()
        self.integrate_grambank()
        self.integrate_apics()
        self.integrate_ewave()
        self.inferred_features()

        logging.info("All databases integration complete.")

    def integrate_custom_databases(self, *args):
        """
            Updates URIEL+ based on provided databases.

            Args:
                *args: Databases to update URIEL+ with.

            Logging:
                Error: Logs an error if a provided databases is invalid.
        """
        logging.info("Importing custom databases....")

        self.set_uriel()

        if len(args) == 1 and isinstance(args[0], list):
            databases = args[0]
        else:
            databases = list(args)

        valid_databases = ["UPDATED_SAPHON", "BDPROTO", "GRAMBANK", "APICS", "EWAVE", "INFERRED"]
        for database in databases:
            if database == "UPDATED_SAPHON":
                self.integrate_saphon()
            elif database == "BDPROTO":
                self.integrate_bdproto()
            elif database == "GRAMBANK":
                self.integrate_grambank()
            elif database == "APICS":
                self.integrate_apics()
            elif database == "EWAVE":
                self.integrate_ewave()
            elif database == "INFERRED":
                self.inferred_features()
            else:
                logging.error(f"Unknown database: {database}. Valid databases are {valid_databases}.")
            
        logging.info("Custom databases integration complete.")

    def aggregate(self):
        """
            Computes the union or average of feature data across sources in URIEL+.

            The union operation takes the maximum value across sources for each feature and language combination.

            If caching is enabled, creates an npz file with the union or average of feature data across sources in URIEL+.
        """
        if self.aggregation == 'U':
            logging.info("Creating union of data across sources....")
            aggregated_data = np.max(self.data[1], axis=-1)
            if self.cache:
                self.sources[1] = ['UNION']
                file_name = "feature_union.npz"
        else:
            logging.info("Creating average of data across sources....")
            aggregated_data = np.where(self.data[1] == -1, np.nan, self.data[1])
            aggregated_data = np.nanmean(aggregated_data, axis=-1)
            aggregated_data = np.where(np.isnan(aggregated_data), -1, aggregated_data)
            if self.cache:
                self.sources[1] = ['AVERAGE']
                file_name = "feature_average.npz"

        if self.fill_with_base_lang:
            for i in range(len(self.langs[1])):
                for parent, child in self.dialects.items():
                    if self.langs[i] in child:
                        for j in range(len(self.feats[1])):
                            parent_data = aggregated_data[parent][j]
                            if aggregated_data[i][j] == -1.0 and parent_data > -1.0:
                                aggregated_data[i][j] = parent_data
        
        if self.aggregation == 'U':
            aggregated_data = np.expand_dims(aggregated_data, axis=-1)

        if self.cache:
            np.savez(os.path.join(self.cur_dir, "data", file_name), feats=self.feats[1], data=aggregated_data, langs=self.langs[1], sources=self.sources[1])

        if self.aggregation == 'U':
            logging.info("Union across sources creation complete.")
        else:
            logging.info("Average across sources creation complete.")

        return aggregated_data



    def _preprocess_data(self, df, feature_prefixes=('S_', 'P_', 'INV_', 'M_')):
        """
            Preprocesses the data by replacing missing values and categorizing features.

            Args:
                df (pd.DataFrame): The input data frame.
                feature_prefixes (tuple): A tuple of prefixes used to categorize features.

            Returns:
                tuple: A tuple containing the feature data as a NumPy array and a dictionary categorizing features by type.
        """
        combined_df_u = df.replace(-1, np.nan)
        feature_types = {prefix: [] for prefix in feature_prefixes}
        for col in combined_df_u.columns:
            for key in feature_types.keys():
                if col.startswith(key):
                    feature_types[key].append(combined_df_u.columns.get_loc(col))
        X = combined_df_u.values
        return X, feature_types

    def _create_missing_values(self, X, missing_rate=0.1):
        """
            Creates missing values in the dataset based on a specified missing rate.

            Args:
                X (np.ndarray): The original dataset.
                missing_rate (float): The rate of missing values to introduce.

            Returns:
                tuple: A tuple containing the dataset with missing values and the indices of the missing values.
        """
        X_missing = X.copy()
        non_nan_indices = np.argwhere(~np.isnan(X_missing))
        num_non_missing = len(non_nan_indices)
        n_missing = int(missing_rate * num_non_missing)
        np.random.seed(0)
        selected_indices = np.random.choice(len(non_nan_indices), n_missing,
                                            replace=False)
        missing_indices = non_nan_indices[selected_indices]
        X_missing[tuple(zip(*missing_indices))] = np.nan
        return X_missing, missing_indices

    def _evaluate_imputation(self, X_test_orig, X_test_imputed, missing_indices):
        """
            Evaluates the imputation results by comparing the imputed values to the original values.

            Args:
                X_test_orig (np.ndarray): The original test dataset.
                X_test_imputed (np.ndarray): The imputed test dataset.
                missing_indices (list): The indices of the missing values.

            Returns:
                dict: A dictionary containing evaluation metrics (e.g., accuracy, precision, recall, F1 score, RMSE, MAE).
        """
        orig = [X_test_orig[i][j] for i, j in missing_indices]
        imputed = [X_test_imputed[i][j] for i, j in missing_indices]
        # check if nans are present in the imputed data
        try:
            assert len(orig) == len(imputed)
        except AssertionError:
            logging.info(f"Length of orig is {len(orig)}")
            logging.info(f"Length of imputed is {len(imputed)}")

        overall_metrics = {}
        if self.aggregation == 'U':
            orig_rounded = [round(o) for o in orig]
            imputed_rounded = [round(imp) for imp in imputed]

            accuracy = accuracy_score(orig_rounded, imputed_rounded)
            classification_error = 1 - accuracy
            precision = precision_score(orig_rounded, imputed_rounded)
            recall = recall_score(orig_rounded, imputed_rounded)
            f1 = f1_score(orig_rounded, imputed_rounded)

            overall_metrics['accuracy'] = accuracy
            overall_metrics['classification_error'] = classification_error
            overall_metrics['precision'] = precision
            overall_metrics['recall'] = recall
            overall_metrics['f1'] = f1

            # Calculate differences in proportions of 1s for each feature
            feature_differences = {}
            for idx, (i, j) in enumerate(missing_indices):
                if j not in feature_differences:
                    feature_differences[j] = {'orig': [], 'imputed': []}
                feature_differences[j]['orig'].append(orig_rounded[idx])
                feature_differences[j]['imputed'].append(imputed_rounded[idx])

            proportion_diffs = []
            for feature, values in feature_differences.items():
                orig_prop = np.mean(values['orig'])
                imputed_prop = np.mean(values['imputed'])
                proportion_diff = abs(orig_prop - imputed_prop)
                proportion_diffs.append(proportion_diff)

            # Print the largest, average, and smallest differences in proportion
            if proportion_diffs:
                largest_diff = max(proportion_diffs)
                average_diff = np.mean(proportion_diffs)
                smallest_diff = min(proportion_diffs)

                logging.info(f"Largest difference in proportion: {largest_diff}")
                logging.info(f"Average difference in proportion: {average_diff}")
                logging.info(f"Smallest difference in proportion: {smallest_diff}")

                overall_metrics['largest_diff'] = largest_diff
                overall_metrics['average_diff'] = average_diff
                overall_metrics['smallest_diff'] = smallest_diff

                proportion_diffs_sorted = sorted(proportion_diffs)
                q1 = np.percentile(proportion_diffs_sorted, 25)
                q2 = np.percentile(proportion_diffs_sorted, 50)
                q3 = np.percentile(proportion_diffs_sorted, 75)

                quartile_counts = {
                    f'Q1 ({q1})': sum(diff <= q1 for diff in proportion_diffs),
                    f'Q2 ({q2})': sum(q1 < diff <= q2 for diff in proportion_diffs),
                    f'Q3 ({q3})': sum(q2 < diff <= q3 for diff in proportion_diffs),
                    f'Q4 ({np.max(proportion_diffs)})': sum(
                        diff > q3 for diff in proportion_diffs)
                }

                overall_metrics[f'Q1 ({q1})'] = quartile_counts[f'Q1 ({q1})']
                overall_metrics[f'Q2 ({q2})'] = quartile_counts[f'Q2 ({q2})']
                overall_metrics[f'Q3 ({q3})'] = quartile_counts[f'Q3 ({q3})']
                overall_metrics[f'Q4 ({np.max(proportion_diffs)})'] = \
                quartile_counts[f'Q4 ({np.max(proportion_diffs)})']

        elif self.aggregation == 'A':
            rmse = mean_squared_error(orig, imputed, squared=False)
            mae = mean_absolute_error(orig, imputed)

            overall_metrics['rmse'] = rmse
            overall_metrics['mae'] = mae

        return overall_metrics

    def _evaluate_imputation_by_feature(self, X_test_orig, X_test_imputed, missing_indices,
                                   feature_types):
        """
            Evaluates the imputation results by feature type.

            Args:
                X_test_orig (np.ndarray): The original test dataset.
                X_test_imputed (np.ndarray): The imputed test dataset.
                missing_indices (list): The indices of the missing values.
                feature_types (dict): A dictionary categorizing features by type.

            Returns:
                dict: A dictionary containing evaluation metrics by feature type.

            Logging:
                Error: Logs error if any metric calculations results in errors.
        """
        feat_metrics = {key: {} for key in feature_types.keys()}

        for feature_type, indices in feature_types.items():
            orig = [X_test_orig[i][j] for i, j in missing_indices if j in indices]
            imputed = [X_test_imputed[i][j] for i, j in missing_indices if
                    j in indices]
            if self.aggregation == 'U':
                try:
                    if orig and imputed:  # Check if there are values to evaluate
                        orig_rounded = [round(o) for o in orig]
                        imputed_rounded = [round(imp) for imp in imputed]

                        accuracy = accuracy_score(orig_rounded, imputed_rounded)
                        classification_error = 1 - accuracy
                        precision = precision_score(orig_rounded, imputed_rounded)
                        recall = recall_score(orig_rounded, imputed_rounded)
                        f1 = f1_score(orig_rounded, imputed_rounded)

                        feat_metrics[feature_type] = {
                            'accuracy': accuracy,
                            'classification_error': classification_error,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1
                        }
                except Exception as e:
                    logging.error(f"{e} for feature type {feature_type}")

            else:
                try:
                    if orig and imputed:  # Check if there are values to evaluate
                        rmse = mean_squared_error(orig, imputed, squared=False)
                        mae = mean_absolute_error(orig, imputed)

                        feat_metrics[feature_type] = {
                            'rmse': rmse,
                            'mae': mae
                        }
                except Exception as e:
                    logging.error(f"{e} for feature type {feature_type}")
        return feat_metrics

    def _impute_wrt_hyperparameter(self, X_train, X_test, X_test_missing, missing_indices,
                              strategy, feature_types, hyperparameter,
                              eval_metric):
        """
            Imputes missing values in the dataset using a specified hyperparameter.

            Args:
                X_train (np.ndarray): The training dataset.
                X_test (np.ndarray): The test dataset.
                X_test_missing (np.ndarray): The test dataset with missing values.
                missing_indices (list): The indices of the missing values.
                strategy (str): The imputation strategy to use ('knn', 'softimpute').
                feature_types (dict): A dictionary categorizing features by type.
                hyperparameter (int or float): The hyperparameter value for the imputer.
                eval_metric (str): The evaluation metric to use ('f1', 'rmse', etc.).

            Returns:
                tuple: A tuple containing the imputed dataset, the hyperparameter used, and the evaluation metric value.
        """
        logging.info("starting impute_wrt_hyperparameter")
        # SystemExit("Exiting")
        imputer = None
        with contexttimer.Timer() as t:
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=hyperparameter, keep_empty_features=True)
                _ = imputer.fit_transform(X_train)
                X_test_imputed = imputer.transform(X_test_missing)
            elif strategy == 'softimpute':
                imputer = SoftImpute(shrinkage_value=hyperparameter, max_iters=400,
                                    max_value=1, min_value=0,
                                    init_fill_method='mean')
                _ = imputer.fit_transform(X_train)
                X_test_imputed = imputer.fit_transform(X_test_missing)
        metrics = self._evaluate_imputation(X_test, X_test_imputed, missing_indices)
        # metrics = evaluate_imputation(orig=X_test_missing, imputed=X_test_imputed, missing_indices=missing_indices, union_or_average=union_or_average)
        feature_metrics = self._evaluate_imputation_by_feature(X_test, X_test_imputed,
                                                        missing_indices,
                                                        feature_types)

        # print(f"Time taken for {strategy} with hyperparameter {hyperparameter}: {t.elapsed}")
        logging.info(
            f"Metrics for {strategy} with hyperparameter {hyperparameter}: {metrics}")
        logging.info(
            f"Feature metrics for {strategy} with hyperparameter {hyperparameter}: {feature_metrics}")

        return X_test_imputed, hyperparameter, metrics[eval_metric]

    def _cross_val_impute_wrt_hyperparameter(self, X_train, strategy, feature_types, hyperparameter, eval_metric, n_splits=10, missing_rate=0.2):
        """
            Performs k-fold cross-validation to evaluate imputation quality using a specified hyperparameter.

            The function splits the training data into k folds, imputes missing values using the specified strategy (e.g., KNN or SoftImpute) with a given hyperparameter, and evaluates the imputation using the specified evaluation metric.

            Args:
                X_train (np.ndarray): Training data.
                strategy (str): Imputation strategy ('knn' or 'softimpute').
                feature_types (dict): Dictionary categorizing features by type.
                hyperparameter (int or float): Hyperparameter for the imputation strategy (e.g., number of neighbors for KNN).
                eval_metric (str): Metric to evaluate imputation quality ('accuracy', 'precision', 'recall', 'f1', 'rmse', or 'mae').
                n_splits (int, optional): Number of folds for cross-validation. Defaults to 10.
                missing_rate (float, optional): Proportion of data to artificially make missing for evaluation. Defaults to 0.2.

            Returns:
                tuple: (X_train, hyperparameter, avg_metric[eval_metric])
                    - X_train: The original training data.
                    - hyperparameter: The hyperparameter used for the imputation.
                    - avg_metric[eval_metric]: The average value of the specified evaluation metric across the folds.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        fold_metrics = []
        fold_feature_metrics = []
        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            X_val_missing, missing_indices = self._create_missing_values(X_val_fold, missing_rate=missing_rate)
            imputer = None
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=hyperparameter, keep_empty_features=True)
                _ = imputer.fit_transform(X_train_fold)
                X_val_imputed = imputer.transform(X_val_missing)
            elif strategy == 'softimpute':
                imputer = SoftImpute(shrinkage_value=hyperparameter, max_iters=400, max_value=1, min_value=0, init_fill_method='mean')
                _ = imputer.fit_transform(X_train_fold)
                X_val_imputed = imputer.fit_transform(X_val_missing)
            metrics = self._evaluate_imputation(X_val_fold, X_val_imputed, missing_indices)
            feature_metrics = self._evaluate_imputation_by_feature(X_val_fold, X_val_imputed, missing_indices, feature_types)
            fold_metrics.append(metrics)
            fold_feature_metrics.append(feature_metrics)
        # compute mean of metrics across folds, understanding metrics are dictionaries
        avg_metric = {}
        for key in fold_metrics[0].keys():
            avg_metric[key] = np.mean([fold_metrics[i][key] for i in range(n_splits)])
        avg_feature_metrics = {key: {} for key in feature_types.keys()}
        for key in feature_types.keys():
            for metric in fold_feature_metrics[0][key].keys():
                avg_feature_metrics[key][metric] = np.mean([fold_feature_metrics[i][key].get(metric, np.nan) for i in range(n_splits)])
        logging.info(f"Average metrics for {strategy} on {n_splits}-fold CV with hyperparameter {hyperparameter}: {avg_metric}")
        logging.info(f"Average feature metrics for {strategy} on {n_splits}-fold CV with hyperparameter {hyperparameter}: {avg_feature_metrics}")
        return X_train, hyperparameter, avg_metric[eval_metric]

    def _choose_hyperparameter_cv(self, X_train, X_test, strategy,
                          feature_types, missing_rate=0.2,
                          hyperparameter_range=range(3, 21, 3),
                          eval_metric='f1', n_splits=10):
        logging.info("Starting parallel processing for hyperparameter selection")
        results = Parallel(n_jobs=-1)(delayed(self._cross_val_impute_wrt_hyperparameter)(X_train, strategy, feature_types, hyperparameter, eval_metric, n_splits, missing_rate) for hyperparameter in hyperparameter_range)
        """
            Selects the best hyperparameter for an imputation strategy using k-fold cross-validation.

            The function evaluates different hyperparameters, using cross-validation, and selects the one that optimizes the specified evaluation metric.

            Args:
                X_train (np.ndarray): Training data.
                X_test (np.ndarray): Test data (not used directly for hyperparameter selection).
                strategy (str): Imputation strategy ('knn' or 'softimpute').
                feature_types (dict): Dictionary categorizing features by type.
                missing_rate (float, optional): Proportion of data to artificially make missing for evaluation. Defaults to 0.2.
                hyperparameter_range (range, optional): Range of hyperparameters to evaluate. Defaults to range(3, 21, 3).
                eval_metric (str, optional): Metric to evaluate imputation quality ('f1', 'accuracy', 'precision', 'recall', 'rmse', or 'mae'). Defaults to 'f1'.
                n_splits (int, optional): Number of folds for cross-validation. Defaults to 10.

            Returns:
                int or float: The best hyperparameter for the specified strategy based on the evaluation metric.
        """
        logging.info("Completed parallel processing for hyperparameter selection")
        if eval_metric in ['accuracy', 'precision', 'recall', 'f1']:
            best_hyperparameter = max(results, key=lambda x: x[2])[1]
        elif eval_metric in ['rmse', 'mae']:
            best_hyperparameter = min(results, key=lambda x: x[2])[1]

        logging.info(f"Best hyperparameter for {strategy} is {best_hyperparameter}")
        return best_hyperparameter

    def _choose_hyperparameter(self, X_train, X_test, strategy, average_or_union,
                          feature_types, missing_rate=0.2,
                          hyperparameter_range=range(3, 21, 3),
                          eval_metric='f1'):
        """
            Selects the best hyperparameter for a specified imputation strategy using the given evaluation metric.

            Args:
                X_train (np.ndarray): The training dataset.
                X_test (np.ndarray): The test dataset.
                strategy (str): The imputation strategy ('knn', 'softimpute', etc.).
                feature_types (dict): A dictionary categorizing features by type.
                missing_rate (float): The rate of missing values to simulate for testing.
                hyperparameter_range (range): The range of hyperparameters to test.
                eval_metric (str): The evaluation metric to use ('f1', 'rmse', etc.).

            Returns:
                int or float: The best hyperparameter value.
        """
        X_test_missing, missing_indices = self._create_missing_values(X_test, missing_rate=missing_rate)
        logging.info("Missing values created")
        logging.info(f"X_missing shape: {X_test_missing.shape}")
        logging.info(f"Number of missing indices: {len(missing_indices)}")

        results = Parallel(n_jobs=-1)(
            delayed(self._impute_wrt_hyperparameter)(X_train, X_test, X_test_missing,
                                            missing_indices, strategy,
                                            feature_types, hyperparameter,
                                            eval_metric) for
            hyperparameter in hyperparameter_range)

        logging.info("Completed parallel processing for hyperparameter selection")

        if eval_metric in ['accuracy', 'precision', 'recall', 'f1']:
            best_hyperparameter = max(results, key=lambda x: x[2])[1]
        elif eval_metric in ['rmse', 'mae']:
            best_hyperparameter = min(results, key=lambda x: x[2])[1]

        logging.info(f"Best hyperparameter for {strategy} is {best_hyperparameter}")

        return best_hyperparameter

    def _standard_impute(self, X, imputer_class, strategy, X_missing=None,
                    hyperparameter=None,
                    feature_types=None, missing_indices=None, midas_bin_vars=None,
                    on_test_set=False, file_path_to_save_npz=None):
        """
            Imputes missing data using a specified imputation strategy.

            Args:
                X (np.ndarray or pd.DataFrame): The dataset to impute.
                imputer_class (class): The imputation class to use (e.g., KNNImputer, SoftImpute).
                strategy (str): The imputation strategy ('knn', 'softimpute', 'mean', etc.).
                X_missing (np.ndarray or pd.DataFrame, optional): The dataset with missing values. Default is None.
                hyperparameter (int or float, optional): The hyperparameter for the imputer (e.g., number of neighbors for KNN).
                feature_types (dict, optional): Dictionary categorizing features by type. Default is None.
                missing_indices (list, optional): Indices of missing values in the dataset. Default is None.
                midas_bin_vars (list, optional): List of binary variables for MIDAS imputation. Default is None.
                on_test_set (bool, optional): Whether the imputation is being performed on a test set. Default is False.
                file_path_to_save_npz (str, optional): Path to save the imputed data in NPZ format. Default is None.

            Returns:
                np.ndarray: The imputed dataset.
        """
        if file_path_to_save_npz == None:
            file_path_to_save_npz == os.path.join(self.cur_dir, "data")
        if strategy == 'knn':
            imputer = imputer_class(n_neighbors=hyperparameter, keep_empty_features=True)
        elif strategy == 'softimpute':
            imputer = imputer_class(max_iters=400, max_value=1, min_value=0,
                                    shrinkage_value=hyperparameter)
        elif strategy in ['mean', 'most_frequent']:
            imputer = imputer_class(strategy=strategy, keep_empty_features=True)
        elif strategy == 'constant_0':
            imputer = imputer_class(strategy='constant', fill_value=0, keep_empty_features=True)
        elif strategy == 'constant_1':
            imputer = imputer_class(strategy='constant', fill_value=1, keep_empty_features=True)
        elif strategy == 'midas':
            import MIDASpy as md
            X_true = X.values
            curr_date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join('midas_outputs', f'run_{curr_date_time}')
            os.makedirs(file_path, exist_ok=True)
            if X_missing is None:
                with contexttimer.Timer() as t:
                    # 256
                    file_path = os.path.join('midas_outputs', f'{curr_date_time}', 'tmp')
                    imputer = md.Midas(layer_structure=[256, 256, 256], vae_layer=False, seed=0, input_drop=0.8, savepath=file_path)
                    imputer.build_model(X, binary_columns=midas_bin_vars)
                    imputer.train_model(training_epochs=50)
            else:
                with contexttimer.Timer() as t:
                    # 256
                    file_path = os.path.join('midas_outputs', f'{curr_date_time}', 'tmp')
                    imputer = md.Midas(layer_structure=[256, 256, 256], vae_layer=False, seed=0, input_drop=0.8, savepath=file_path)
                    imputer.build_model(X_missing, binary_columns=midas_bin_vars)
                    imputer.train_model(training_epochs=50)
            logging.info('Time taken to train the model: ', t.elapsed)
            num_samples = 5
            imputations = imputer.generate_samples(m=num_samples).output_list
            for i in range(num_samples):
                file_path = os.path.join("midas_outputs", f"run_{curr_date_time}", f"nl_256_256_256_imputation_{i}.csv")
                imputations[i].to_csv(file_path, index=False)
            if missing_indices is None:
                X_imputed = imputations[0].to_numpy()
                return X_imputed

            assert missing_indices is not None
            multiple_metrics = []
            multiple_feature_metrics = []
            for i in range(num_samples):
                X_imputed = imputations[i].to_numpy()
                metrics = self._evaluate_imputation(X_true, X_imputed, missing_indices)
                feature_metrics = self._evaluate_imputation_by_feature(X_true, X_imputed, missing_indices, feature_types)
                multiple_metrics.append(metrics)
                multiple_feature_metrics.append(feature_metrics)
                logging.info(f"Metrics for {strategy} with hyperparameter {hyperparameter} and imputed dataset sample {i}: {metrics}")
                logging.info(f"Feature metrics for {strategy} with hyperparameter {hyperparameter} and imputed dataset sample {i}: {feature_metrics}")

            avg_metrics = {}
            for key in multiple_metrics[0].keys():
                avg_metrics[key] = np.mean([multiple_metrics[i][key] for i in range(num_samples)])

            avg_feature_metrics = {key: {} for key in feature_types.keys()}
            for key in feature_types.keys():
                for metric in multiple_feature_metrics[0][key].keys():
                    avg_feature_metrics[key][metric] = np.mean([multiple_feature_metrics[i][key].get(metric, np.nan) for i in range(num_samples)])

            logging.info(f"Average metrics for {strategy} with hyperparameter {hyperparameter} across each imputed dataset: {avg_metrics}")
            logging.info(f"Average feature metrics for {strategy} with hyperparameter {hyperparameter} across each imputed dataset: {avg_feature_metrics}")
            metrics_df = pd.DataFrame(avg_metrics, index=[0])
            file_path = os.path.join(file_path_to_save_npz, "imputation_metrics.csv")
            metrics_df.to_csv(file_path, index=False)
            X_imputed = imputations[0].to_numpy()
            return X_imputed

        if X_missing is not None:
            with contexttimer.Timer() as t:
                X_imputed = imputer.fit_transform(X_missing)
            logging.info(f"Time taken for {strategy} (initial imputation): {t.elapsed}")
        else:
            with contexttimer.Timer() as t:
                X_imputed = imputer.fit_transform(X)
            logging.info(f"Time taken for {strategy} (final imputation): {t.elapsed}")
        if missing_indices is not None:
            metrics = self._evaluate_imputation(X, X_imputed, missing_indices)
            feature_metrics = self._evaluate_imputation_by_feature(X, X_imputed,
                                                            missing_indices,
                                                            feature_types)
            if not on_test_set:
                logging.info(f"Overall metrics for {strategy} with hyperparameter {hyperparameter}: {metrics}")
                logging.info(f"Overall feature metrics for {strategy} with hyperparameter {hyperparameter}: {feature_metrics}")
                # save metrics as a csv file
                metrics_df = pd.DataFrame(metrics, index=[0])
                file_path = os.path.join(file_path_to_save_npz, "imputation_metrics.csv")
                metrics_df.to_csv(file_path, index=False)
            else:
                logging.info(f"On test set, metrics for {strategy} with hyperparameter {hyperparameter}: {metrics}")
                logging.info(f"On test set, feature metrics for {strategy} with hyperparameter {hyperparameter}: {feature_metrics}")
        return X_imputed

    def _hyperparameter_imputation(self, X, strategy, feature_types,
                              hyperparameter_range, eval_metric,
                              test_quality=True, file_path_to_save_npz=None):
        """
            Performs imputation using the best hyperparameter selected through cross-validation.

            Args:
                X (np.ndarray): The dataset to impute.
                strategy (str): The imputation strategy ('knn', 'softimpute', etc.).
                feature_types (dict): A dictionary categorizing features by type.
                hyperparameter_range (range): The range of hyperparameters to test.
                eval_metric (str): The evaluation metric to use ('f1', 'rmse', etc.).
                test_quality (bool, optional): Whether to evaluate the quality of imputation. Default is True.
                file_path_to_save_npz (str, optional): Path to save the imputed data in NPZ format. Default is None.

            Returns:
                np.ndarray: The imputed dataset.
        """
        if file_path_to_save_npz == None:
            file_path_to_save_npz = os.path.join(self.cur_dir, "data")
        X_train, X_test = train_test_split(X, test_size=0.25, random_state=0)
        logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        best_hyperparameter = self._choose_hyperparameter_cv(X_train, X_test,
                                                    strategy=strategy,
                                                    feature_types=feature_types,
                                                    missing_rate=0.2,
                                                    hyperparameter_range=hyperparameter_range,
                                                    eval_metric=eval_metric)

        imputer_class = KNNImputer if strategy == 'knn' else SoftImpute
        if test_quality:
            X_missing, missing_indices = self._create_missing_values(X_test, missing_rate=0.2)
            _ = self._standard_impute(X=X_test, imputer_class=imputer_class,
                                        strategy=strategy, X_missing=X_missing,
                                        hyperparameter=best_hyperparameter,
                                        feature_types=feature_types,
                                        missing_indices=missing_indices,
                                        on_test_set=True)
            X_missing, missing_indices = self._create_missing_values(X, missing_rate=0.2)
            _ = self._standard_impute(X=X, imputer_class=imputer_class,
                                strategy=strategy, X_missing=X_missing,
                                hyperparameter=best_hyperparameter,
                                feature_types=feature_types,
                                missing_indices=missing_indices)

        X_imputed = self._standard_impute(X=X, imputer_class=imputer_class,
                                    strategy=strategy,
                                    hyperparameter=best_hyperparameter,
                                    feature_types=feature_types, file_path_to_save_npz=file_path_to_save_npz)
        return X_imputed

    def _preprocess_midas(self, combined_df_u):
        """
            Preprocesses data for MIDAS imputation by converting the data into the appropriate format.

            Args:
                combined_df_u (pd.DataFrame): The input data frame.

            Returns:
                tuple: A tuple containing the preprocessed data and the list of binary variables.
        """
        combined_df_u.columns.str.strip()
        data_in = combined_df_u
        data_in = data_in.replace(-1, np.nan)
        vals = data_in.nunique()
        bin_vars = data_in.columns
        constructor_list = [data_in]
        data_in = pd.concat(constructor_list, axis=1)
        return data_in, bin_vars

    def _make_csv(self, file_path_to_save_npz):
        """
            Generates a CSV file from the URIEL+ features dataset.

            Args:
                file_path_to_save_npz (str): The path to save the NPZ file.

            Returns:
                pd.DataFrame: The generated data frame.
        """
        if ['imputed'] not in self.sources[1]:
            updated_path = os.path.join(file_path_to_save_npz, 'non_imputed_data')
            # check if the directory exists
            if not os.path.exists(updated_path):
                os.makedirs(updated_path)
            np.savez(os.path.join(updated_path, self.files[1]), data=self.data[1], feats=self.feats[1], langs=self.langs[1], sources=self.sources[1])

        aggregate_data = self.aggregate()

        if self.aggregation == 'U':
            combined_df_u_no_ffg = pd.DataFrame(aggregate_data.squeeze(), columns=self.feats[1])
            combined_df_u_no_ffg.insert(0, 'language', self.langs[1])
            return combined_df_u_no_ffg
        else:
            combined_df_a_no_ffg = pd.DataFrame(aggregate_data.squeeze(), columns=self.feats[1])
            combined_df_a_no_ffg.insert(0, 'language', self.langs[1])
            return combined_df_a_no_ffg

    def imputation_interface(self, csv_path=None, strategy='softimpute', file_path_to_save_npz=None,
                         feature_prefixes=('S_', 'P_', 'INV_', 'M_'),
                         eval_metric='f1', hyperparameter_range=None,
                         test_quality=True, return_csv=True, save_as_npz=True):
        """
            Interface for imputing missing values in URIEL datasets using different imputation strategies.

            Args:
                csv_path (str, optional): Path to the CSV file to load. Default is None.
                strategy (str, optional): The imputation strategy to use ('mean', 'knn', 'softimpute', 'midas'). Default is 'softimpute'.
                file_path_to_save_npz (str, optional): Path to save the imputed data in NPZ format. Default is None.
                feature_prefixes (tuple, optional): Prefixes to categorize features. Default is ('S_', 'P_', 'INV_', 'M_').
                eval_metric (str, optional): The evaluation metric to use ('f1', 'rmse', etc.). Default is 'f1'.
                hyperparameter_range (range, optional): The range of hyperparameters to test for strategies like 'knn' and 'softimpute'. Default is None.
                test_quality (bool, optional): Whether to evaluate the quality of imputation. Default is True.
                return_csv (bool, optional): Whether to return the imputed data as a CSV. Default is True.
                save_as_npz (bool, optional): Whether to save the imputed data as an NPZ file. Default is True.

            Returns:
                pd.DataFrame: The imputed data frame if return_csv is True; otherwise, None.
        """
        logging.info("Starting imputation_interface")

        if file_path_to_save_npz == None:
            file_path_to_save_npz = os.path.join(self.cur_dir, "data")

        if csv_path is None:
            combined_df_u = self._make_csv(file_path_to_save_npz)
        else:
            combined_df_u = pd.read_csv(csv_path)
            f = np.load(file_path_to_save_npz + 'features.npz', allow_pickle=True)
            f = dict(f)
            f_sources = f['sources']
            if ['imputed'] not in f_sources:
                updated_path = os.path.join(file_path_to_save_npz, 'non_imputed_data')
                np.savez(os.path.join(updated_path, 'features.npz'), data=f['data'], feats=f['feats'], langs=f['langs'], sources=f_sources)

        old_combined_df_u = combined_df_u.copy()
        combined_df_u = combined_df_u.drop(combined_df_u.columns[0], axis=1)

        X, feature_types = self._preprocess_data(combined_df_u, feature_prefixes)
        if strategy in ['knn', 'softimpute'] and hyperparameter_range is not None:
            imputed = self._hyperparameter_imputation(X=X, strategy=strategy,
                                                feature_types=feature_types,
                                                hyperparameter_range=hyperparameter_range,
                                                eval_metric=eval_metric,
                                                test_quality=test_quality,
                                                file_path_to_save_npz=file_path_to_save_npz)
        elif strategy == 'midas':
            data_in, bin_vars = self._preprocess_midas(combined_df_u)
            X, feature_types = self._preprocess_data(data_in, feature_prefixes)
            if test_quality:
                X_missing, missing_indices = self._create_missing_values(X,
                                                                missing_rate=0.2)
                # turn X_missing back into dataframe
                X_missing_df = pd.DataFrame(X_missing, columns=data_in.columns)
                X_missing_df, bin_vars = self._preprocess_midas(X_missing_df)
                imputed = self._standard_impute(X=data_in, imputer_class=None,
                                        strategy=strategy, X_missing=X_missing_df,
                                        feature_types=feature_types,
                                        missing_indices=missing_indices,
                                        midas_bin_vars=bin_vars,
                                        file_path_to_save_npz=file_path_to_save_npz)
            else:
                imputed = self._standard_impute(X=data_in, imputer_class=None,
                                        strategy=strategy,
                                        feature_types=feature_types,
                                        midas_bin_vars=bin_vars,
                                        file_path_to_save_npz=file_path_to_save_npz)
        else:
            if test_quality:
                X_missing, missing_indices = self._create_missing_values(X,
                                                                missing_rate=0.2)
                imputer_class = SoftImpute if strategy == 'softimpute' else SimpleImputer
                imputed = self._standard_impute(X=X, imputer_class=imputer_class,
                                        strategy=strategy, X_missing=X_missing,
                                        feature_types=feature_types,
                                        missing_indices=missing_indices,
                                        file_path_to_save_npz=file_path_to_save_npz)
            else:
                imputer_class = SoftImpute if strategy == 'softimpute' else SimpleImputer
                imputed = self._standard_impute(X=X, imputer_class=imputer_class,
                                        strategy=strategy,
                                        feature_types=feature_types,
                                        file_path_to_save_npz=file_path_to_save_npz)

        if self.aggregation == 'U':
            imputed = np.round(imputed)

        if save_as_npz:
            print("Hi...")
            df = pd.DataFrame(imputed, columns=combined_df_u.columns)
            df = df.fillna(-1)
            data = df.to_numpy()
            feats = df.columns
            reshaped_data = data.reshape(data.shape[0], data.shape[1], 1)
            f_sources_new = np.array(['imputed'])
            #IMPORTANT
            file_path = os.path.join(file_path_to_save_npz, self.files[1])
            np.savez(file_path,
                    data=reshaped_data, feats=feats,
                    langs=old_combined_df_u['language'], sources=f_sources_new)

        if return_csv:
            completed_df = pd.DataFrame(imputed, columns=combined_df_u.columns)
            completed_df.insert(0, 'language', old_combined_df_u['language'])

            return completed_df

    def midaspy_imputation(self):
        """Imputes missing values in URIEL+ data using MIDASpy imputation."""
        if self.aggregation == 'U':
            eval_metric = 'f1'
        else:
            eval_metric = 'rmse'
        _ = self.imputation_interface(strategy='midas', save_as_npz=False, test_quality=True, eval_metric=eval_metric)
        _ = self.imputation_interface(strategy='midas', save_as_npz=True, test_quality=False, eval_metric=eval_metric)

    def knn_imputation(self):
        """Imputes missing values in URIEL+ data using k-nearest-neighbour imputation."""
        if self.aggregation == 'U':
            eval_metric = 'f1'
        else:
            eval_metric = 'rmse'
        _ = self.imputation_interface(strategy='knn', save_as_npz=True, test_quality=True, eval_metric=eval_metric, hyperparameter_range=(3, 6, 9, 12, 15))


    def softimpute_imputation(self):
        """Imputes missing values in URIEL datasets using softImpute."""
        if self.aggregation == 'U':
            eval_metric = 'f1'
        else:
            eval_metric = 'rmse'
        _ = self.imputation_interface(strategy='softimpute', save_as_npz=False, test_quality=True, eval_metric=eval_metric)
        _ = self.imputation_interface(strategy='softimpute', save_as_npz=True, test_quality=False, eval_metric=eval_metric)

    def mean_imputation(self):
        """Imputes missing values in URIEL datasets using mean imputation."""
        if self.aggregation == 'U':
            eval_metric = 'f1'
        else:
            eval_metric = 'rmse'
        _ = self.imputation_interface(strategy='mean', save_as_npz=False, test_quality=True, eval_metric=eval_metric)
        _ = self.imputation_interface(strategy='mean', save_as_npz=True, test_quality=False, eval_metric=eval_metric)




    def map_new_distance_to_database(self, distance):
        """
            Maps a distance type to the corresponding filename for URIEL+.

            Args:
                distance (str): The type of distance to map.

            Returns:
                str: The filename corresponding to the distance type.

            Logging:
                Error: Logs error if the distance type is not available in URIEL+.
        """
        d = {"genetic": 0,
        "geographic": 2, 
        "syntactic": 1,
        "phonological": 1,
        "inventory": 1,
        "featural": 1,
        "morphological": 1}
        if distance in d.keys():
            return d[distance]
        logging.error(f"{distance} is not an available feature category in URIEL+. Feature categories are genetic, geographic, featural (syntactic, phonological, inventory, morphological).")

    def get_available_distance_languages(self, distance_type):
        """
            Retrieves a list of languages that contain at least one non-empty feature of the specified distance type.

            Args:
                distance_type (str): The type of distance to check.

            Returns:
                list: A list of languages.
        """

        db_index = self.map_new_distance_to_database(distance_type)

        features, data_matrix, languages = self.feats[db_index], self.data[db_index], self.langs[db_index]
        
        available_languages = []
        for lang_index in range(len(languages)):
            flattened_data = data_matrix[lang_index].flatten()
            if not np.all(np.isclose(flattened_data, -1.0)):
                available_languages.append(languages[lang_index])

        if distance_type in ["syntactic", "phonological", "inventory", "morphological"]:
            subdomain_languages = []
            languages_to_remove = available_languages.copy()

            for feature_index in range(len(features)):
                feature_prefix = features[feature_index][:2]
                is_matching_feature = (
                    (distance_type == "syntactic" and feature_prefix == "S_") or
                    (distance_type == "phonological" and feature_prefix == "P_") or
                    (distance_type == "inventory" and features[feature_index][:4] == "INV_") or
                    (distance_type == "morphological" and feature_prefix == "M_")
                )

                if is_matching_feature:
                    for lang_index in range(len(languages)):
                        if languages[lang_index] in languages_to_remove:
                            if any(data_matrix[lang_index][feature_index] > -1.0):
                                subdomain_languages.append(languages[lang_index])
                                languages_to_remove.remove(languages[lang_index])
            
            if self.cache:
                cache_file_path = os.path.join(self.cur_dir, "data", distance_type + "_distances_languages.txt")
                with open(cache_file_path, 'w', encoding='utf-8') as cache_file:
                    cache_file.write(','.join(subdomain_languages))

            return subdomain_languages

        if self.cache:
            cache_file_path = os.path.join(self.cur_dir, "data", distance_type + "_distances_languages.txt")
            with open(cache_file_path, 'w', encoding='utf-8') as cache_file:
                cache_file.write(','.join(available_languages))

        return available_languages

    def get_vector(self, category, *args):
        """
            Retrieves the vectors of the specified category for each language.

            Args:
                category (str): The category of features to retrieve.
                *args: Language codes for which to retrieve vectors.

            Returns:
                dict: A dictionary with language codes as keys and vectors as values.

            Logging:
                Error: Logs error if only one language is provided or if the language is unknown.
        """
        if len(args) == 1 and not isinstance(args[0],list):
            logging.error("You only provided one language argument.\nProvide multiple language arguments, or a single list of languages as arguments.")
        if len(args) == 1 and isinstance(args[0],list):
            langs = args[0]
        else:
            langs = [l for l in args]

        database = self.map_new_distance_to_database(category)
        languages = self.langs[database]
        for lang in langs:
            if lang not in languages:
                logging.error(f"Unknown language: {lang}.")

        vectors = {}
        for i in range(len(langs)):
            lang_index = np.where(languages == langs[i])[0][0]
            feats = self.feats[database]
            feature_data = self.data[database]
            vector = []
            for j in range(len(feats)):
                if category in ['genetic', 'featural', 'geographic'] or (category == "morphological" and (feats[j])[:2] == 'M_') or (category == 'inventory' and (feats[j])[:4] == 'INV_') or (category == 'phonological' and (feats[j])[:2] == 'P_') or (category == 'syntactic' and (feats[j])[:2] == 'S_'):
                    i_data = feature_data[lang_index][j]
                    if len(i_data) == 1:
                        vector.extend(i_data)
                    elif self.aggregation == 'U':
                        vector.append(np.max(i_data))
                    elif self.aggregation == 'A':
                        if np.all(i_data == -1.0):
                            vector.append(-1.0)
                        else:
                            valid_data = i_data[i_data != -1.0]
                            vector.append(np.mean(valid_data) if valid_data.size > 0 else 0.0)
            vectors[langs[i]] = vector
        return vectors

    def _process_language(self, i, langs, feats, feature_data, languages, list_shared_indices, vec_num):
        """
            Processes a single language to compute feature vectors based on shared indices between languages.

            Args:
                i (int): Index of the language in the langs list.
                langs (list): List of language codes.
                feats (np.ndarray): Array of feature names.
                feature_data (np.ndarray): Array of feature values for the languages.
                languages (np.ndarray): Array of language codes.
                list_shared_indices (list): List of indices that indicate shared features between languages.
                vec_num (int): Index of the vector in the shared indices list.

            Returns:
                list: Computed feature vector for the language.
        """
        if len(langs) > 2 and -1 in list_shared_indices[vec_num]:
            return []

        lang_index = np.where(languages == langs[i])[0][0]
        vec = []

        for j in range(len(feats)):
            i_data = feature_data[lang_index][j]
            is_shared_index = (len(langs) > 2 and j in list_shared_indices[vec_num]) or (len(langs) == 2 and j in list_shared_indices)

            if is_shared_index:
                if len(i_data) == 1:
                    vec.extend(i_data)
                elif self.aggregation == 'U':
                    vec.append(1.0 if 1.0 in i_data else 0.0)
                elif self.aggregation == 'A':
                    valid_data = i_data[i_data != -1.0]
                    vec.append(np.mean(valid_data) if valid_data.size > 0 else 0.0)
        return vec


    def _create_vectors(self, langs, database, list_shared_indices, vec_num=-1):
        """
            Creates feature vectors for a list of languages using shared feature indices.

            Args:
                langs (list): List of language codes.
                database (str): Database of the feature data.
                list_shared_indices (list): List of indices that indicate shared features between languages.
                vec_num (int): Index of the vector in the shared indices list.

            Returns:
                tuple: A tuple containing the list of language vectors and the updated vec_num.
        """
        feats, feature_data, languages = self.feats[database], self.data[database], self.langs[database]
        lang_vectors = []

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._process_language,
                    i, langs, feats, feature_data, languages, list_shared_indices, vec_num + i
                )
                for i in range(len(langs))
            ]

            for future in futures:
                lang_vectors.append(future.result())

        if len(langs) > 2:
            vec_num += len(langs)

        return (lang_vectors, vec_num) if len(langs) > 2 else lang_vectors
    
    def _angular_distance(self, u, v):
        """
            Computes the angular or cosine distance between two vectors.

            Args:
                u (np.ndarray): First vector.
                v (np.ndarray): Second vector.

            Returns:
                float: The computed distance between the vectors.

            Logging:
                Error: Logs error if the vectors are not of the same length.
        """
        if len(u) != len(v):
            logging.error("Vectors must be of the same length")
        u = np.array(u)
        v = np.array(v)
        if np.linalg.norm(u) == 0:
            u = np.ones(u.shape[0]) * 1e-10
        if np.linalg.norm(v) == 0:
            v = np.ones(u.shape[0]) * 1e-10
        norm_u = u / np.linalg.norm(u)
        norm_v = v / np.linalg.norm(v)
        cos_theta = np.dot(norm_u, norm_v)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        if self.distance_metric == 'cosine':
            return round(1 - cos_theta, 4)
        angular_distance = 2 * np.arccos(cos_theta)
        angular_distance /= np.pi
        return round(angular_distance, 4)

    def new_distance(self, distance, *args):
        """
            Computes the distance between languages based on a specified feature category.

            Args:
                distance (str or list): The distance category or a list of distance categories.
                *args: Language codes for which to calculate the distance.

            Returns:
                list: A list of computed distances between languages or a distance matrix.

            Logging:
                Error: Logs error if only one language is provided or if the language is unknown.
        """
        start_time = time.time()
        # print(f"In new_distance, loaded dialects: {time.time() - start_time} seconds")

        if isinstance(distance, str):
            distance_list = [distance]
        elif isinstance(distance, list):
            distance_list = distance
        else:
            logging.error(f"Unknown distance type: {distance}. Provide a name (str) or a list of str.")

        if len(args) == 1 and not isinstance(args[0], list):
            logging.error("You only provided one language argument.\nProvide multiple language arguments, or a single list of languages as arguments.")
        if len(args) == 1 and isinstance(args[0], list):
            langs = args[0]
        else:
            langs = list(args)

        if any(lang not in self.langs[0] for lang in langs):
            logging.error(f"Unknown language: {lang}.")
        # print(f"In new_distance, getting languages array: {time.time() - start_time} seconds")

        angular_distances_list = []
        for dist in distance_list:
            dist_start_time = time.time()

            database = self.map_new_distance_to_database(dist)

            list_indices_with_data = []
            langs_no_info = []

            if os.path.exists(os.path.join(self.cur_dir, "data", distance + "_distances_languages.txt")):
                with open(os.path.join(self.cur_dir, "data", distance + "_distances_languages.txt")) as file:
                    available_distance_languages = file.read().split(',')
            else:
                available_distance_languages = self.get_available_distance_languages(dist)

            # print(f"In new_distance, checking available distance languages: {time.time() - dist_start_time:.2f} seconds")

            feats, feature_data, languages = self.feats[database], self.data[database], self.langs[database]

            for lang in langs:
                # print(f"In new_distance, loading feature data for language {lang}: {time.time() - dist_start_time:.2f} seconds")

                if lang not in available_distance_languages:
                    if len(langs) == 2:
                        logging.error(f"No {dist} information for language: {lang}. {lang}'s {dist} distance to other languages cannot be calculated.")
                    else:
                        langs_no_info.append(lang)
                        list_indices_with_data.append([-1])
                        continue

                lang_index = np.where(languages == lang)[0][0]

                indices_with_values = [
                    j for j, feat in enumerate(feats) if (dist in ['genetic', 'featural', 'geographic'] or
                                                                (dist == "morphological" and feat[:2] == 'M_') or
                                                                (dist == 'inventory' and feat[:4] == 'INV_') or
                                                                (dist == 'phonological' and feat[:2] == 'P_') or
                                                                (dist == 'syntactic' and feat[:2] == 'S_')) and not all(value == -1.0 for value in feature_data[lang_index][j])
                ]

                list_indices_with_data.append(indices_with_values)

            list_shared_indices = []

            if len(langs) == 2:
                shared_indices = list(set(list_indices_with_data[0]).intersection(list_indices_with_data[1]))

                if len(shared_indices) == 0:
                    logging.error(f"No shared {dist} features between {langs[0]} and {langs[1]} for which the two languages have information.\nUnable to calculate {dist} distance.")
                list_shared_indices = shared_indices
                lang_vectors = self._create_vectors(langs, database, list_shared_indices)
                angular_distances_list.append(self._angular_distance(lang_vectors[0], lang_vectors[1]))
                logging.info(f"In new_distance, calculated angular distance for {dist} with 2 languages: {time.time() - dist_start_time} seconds")

            else:
                langs_no_shared_info = []
                pairings = {}
                for l_a, list_a in enumerate(list_indices_with_data):
                    for l_b, list_b in enumerate(list_indices_with_data):
                        shared_indices = list(set(list_a).intersection(list_b))
                        if not shared_indices:
                            list_shared_indices.append([-1])
                            langs_no_shared_info.append((langs[l_a], langs[l_b]))
                        else: list_shared_indices.append(shared_indices)
                        pairings[f"{langs[l_a]}-{langs[l_b]}"] = shared_indices

                lang_vectors = []
                vec_num = 0
                for i in range(len(langs)):
                    vec, vec_num = self._create_vectors(langs, database, list_shared_indices, vec_num)
                    lang_vectors.append(vec)

                flattened_lang_vectors = [sublist for group in lang_vectors for sublist in group]



                indices_to_calculate_1 = []
                indices_to_calculate_2 = []

                pairings_keys_list = list(pairings.keys())
                pair_index_map = {pair: idx for idx, pair in enumerate(pairings_keys_list)}
                angular_distances = []

                for pair in pairings_keys_list:
                    index = pair_index_map[pair]
                    reversed_pair = '-'.join(pair.split('-')[::-1])
                    index_2 = pair_index_map[reversed_pair] if pair != reversed_pair else index

                    indices_to_calculate_1.append(index)
                    indices_to_calculate_2.append(index_2)

                for i, (index1, index2) in enumerate(zip(indices_to_calculate_1, indices_to_calculate_2)):
                    if len(flattened_lang_vectors[i]) == 0:
                        angular_distances.append(0 if index1 == index2 else -1)
                    else:
                        dis = self._angular_distance(flattened_lang_vectors[index1], flattened_lang_vectors[index2])
                        angular_distances.append(dis)

                array = np.array(angular_distances)
                matrix = array.reshape((len(lang_vectors[0]), len(lang_vectors[0])))

                if len(langs_no_info) > 0:
                    logging.error(f"No {dist} information for language(s) {str(langs_no_info)}. Their {dist} distance to other languages cannot be calculated.")
                if len(langs_no_shared_info) > 0:
                    logging.error(f"No shared {dist} features between pair(s) of languages {str(langs_no_shared_info)} for which pair(s) of languages have information. Unable to calculate their {dist} distance to each other.")
                if len(langs_no_info) > 0 or len(langs_no_shared_info) > 0:
                    logging.error("Distances which could not be calculated are marked with a -1.")
                angular_distances_list.append(matrix)
                logging.info(f"In new_distance, calculating distances for {len(langs)} languages: {time.time() - dist_start_time} seconds")

        logging.info(f"Total time for new_distance: {time.time() - start_time} seconds")
        return angular_distances_list
    
    """
        The next seven functions are used to compute specific distances between languages.

        Args:
            *args: Language codes for which to calculate the geographic distance.

        Returns:
            list: A list of computed distances between languages or a distance matrix.
    """
    def new_geographic_distance(self, *args):
        return self.new_distance("geographic", *args)

    def new_genetic_distance(self, *args):
        return self.new_distance("genetic", *args)

    def new_featural_distance(self, *args):
        return self.new_distance("featural", *args)

    def new_morphological_distance(self, *args):
        return self.new_distance("morphological", *args)

    def new_inventory_distance(self, *args):
        return self.new_distance("inventory", *args)

    def new_phonological_distance(self, *args):
        return self.new_distance("phonological", *args)

    def new_syntactic_distance(self, *args):
        return self.new_distance("syntactic", *args)
    
    def _process_custom_language(self, i, langs, flattened_feats, gen_feature_data, geo_feature_data, feat_feature_data, gen_languages, geo_languages, feat_languages, list_shared_indices, vec_num, source_num):
        """
            Processes a single language to compute feature vectors based on shared indices between languages.

            Args:
                i (int): Index of the language in the langs list.
                langs (list): List of language codes.
                flattened_feats (list): List of all features.
                gen_feature_data (np.ndarray): Array of genetic feature values for the languages.
                geo_feature_data (np.ndarray): Array of geographic feature values for the languages.
                feat_feature_data (np.ndarray): Array of typological feature values for the languages.
                gen_languages (list): Array of language codes for genetic data.
                geo_languages (list): Array of language codes for geographic data.
                feat_languages (list): Array of language codes for typological data.
                list_shared_indices (list): List of indices that indicate shared features between languages.
                vec_num (int): Index of the vector in the shared indices list.
                source_num (int): Column of The source to use features from.

            Returns:
                list: Computed feature vector for the language.
        """
        if len(langs) > 2 and -1 in list_shared_indices[vec_num]:
            return []

        vec = []
        for j in range(len(flattened_feats)):
            is_shared_index = (len(langs) > 2 and j in list_shared_indices[vec_num]) or (len(langs) == 2 and j in list_shared_indices)

            if is_shared_index:
                if(j >= 0 and j <= 3717):
                    feature_data = gen_feature_data
                    lang_index = np.where(langs[i] == gen_languages)[0][0]
                    i_data = feature_data[lang_index][j]
                    vec.extend(i_data)
                    continue
                elif(j >= 3718 and j <= 4016):
                    feature_data = geo_feature_data
                    lang_index = np.where(langs[i] == geo_languages)[0][0]
                    i_data = feature_data[lang_index][j - 3718]
                    vec.extend(i_data)
                    continue
                else:
                    feature_data = feat_feature_data
                    lang_index = np.where(langs[i] == feat_languages)[0][0]
                    if source_num == None:
                        i_data = feature_data[lang_index][j - 4017]
                    else:
                        i_data = feature_data[lang_index][j - 4017][source_num]
                if len(i_data) == 1:
                    vec.extend(i_data)
                if self.aggregation == 'U':
                    vec.append(1.0 if 1.0 in i_data else 0.0)
                elif self.aggregation == 'A':
                    valid_data = i_data[i_data != -1.0]
                    vec.append(np.mean(valid_data) if valid_data.size > 0 else 0.0)
        return vec

    def _create_custom_vectors(self, langs, flattened_feats, list_shared_indices, source_num=None, vec_num=-1):
        """
            Creates feature vectors for a list of languages using shared feature indices.

            Args:
                langs (list): List of language codes.
                flattened_feats (list): List of all features.
                list_shared_indices (list): List of indices that indicate shared features between languages.
                source_num (int): Column of The source to use features from.
                vec_num (int): Index of the vector in the shared indices list.

            Returns:
                tuple: A tuple containing the list of language vectors and the updated vec_num.
        """
        gen_feature_data = self.data[0] #0-3717
        geo_feature_data = self.data[2] #3718-4016
        feat_feature_data = self.data[1] #4017-4305
        gen_languages = self.langs[0] #0-3717
        geo_languages = self.langs[2] #3718-4016
        feat_languages = self.langs[1] #4017-4305
        lang_vectors = []

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._process_custom_language,
                    i, langs, flattened_feats, gen_feature_data, geo_feature_data, feat_feature_data, gen_languages, geo_languages, feat_languages, list_shared_indices, vec_num + i, source_num
                )
                for i in range(len(langs))
            ]

            for future in futures:
                lang_vectors.append(future.result())

        if len(langs) > 2:
            vec_num += len(langs)

        return (lang_vectors, vec_num) if len(langs) > 2 else lang_vectors
    
    def new_custom_distance(self, features, *args, source = 'A'):
        """
            Computes the distance between languages based on provided features.

            Args:
                features (list): A list of features to use data from.
                *args: Language codes for which to calculate the distance.
                source (str): The source to use features from ('A' for all sources, 'ETHNO', 'WALS', 'PHOIBLE_UPSID', 'PHOIBLE_SAPHON',
                'PHOIBLE_GM', 'PHOIBLE_PH', 'PHOIBLE_AA', 'SSWL', 'PHOIBLE_RA', or 'PHOIBLE_SPA').

            Returns:
                list: A list of computed distances between languages or a distance matrix.

            Logging:
                Error: Logs error if a source is unknown or the features inputted are not supported by the source,
                If the features entered are not a list or are unknown,
                If only one language is provided or if the language is unknown,
                If only two languages are provided and one does not have any information for all the provided features.
        """
        start_time = time.time()
        # print(f"In new_distance, loaded dialects: {time.time() - start_time} seconds")

        if source != 'A':
            if source not in self.sources().flatten():
                logging.error(f"Unknown source: {source}. Valid sources are {self.sources().flatten()}.")

        source_dict = {
            'ETHNO': ['S_', 'P_', 'INV_'],
            'WALS': ['S_', 'P_'],
            "PHOIBLE_UPSID": ['P_', 'INV_'],
            'PHOIBLE_SAPHON': ['P_', 'INV_'],
            'PHOIBLE_GM': ['P_', 'INV_'],
            'PHOIBLE_PH': ['P_', 'INV_'],
            'PHOIBLE_AA': ['P_', 'INV_'],
            'SSWL': ['S_'],
            'PHOIBLE_RA': ['P_', 'INV_'],
            'PHOIBLE_SPA': ['P_', 'INV_'],
            "BDPROTO": ['P_', 'INV_'],
            "GRAMBANK": ['S_', 'M_'],
            "APICS": ['S_', 'P_', 'INV_', 'M_'],
            "EWAVE": ['S_', 'M_']
        }

        #If the UPDATED_SAPHON database has been integrated, replaces the PHOIBLE_SAPHON source with UPDATED_SAPHON.
        new_source_dict = {}
        if "UPDATED_SAPHON" in self.sources[1]:
            for key, value in source_dict.items():
                if key == 'PHOIBLE_SAPHON':
                    new_source_dict['UPDATED_SAPHON'] = ['P_', 'INV_']
                else:
                    new_source_dict[key] = value
        else:
            new_source_dict = source_dict

        if not isinstance(features, list):
            logging.error(f"Unknown features type {features}. Provide a list of str.")
        
        #Creates a list of all features from all files.
        flattened_feats = np.concatenate([self.feats[0], self.feats[2], self.feats[1]])

        #Finds the indices of the provided features within the list of all the features.
        source_num = None
        feat_indices = []
        for feature in features:
            if (feature not in flattened_feats):
                logging.error(f"Unknown feature: {feature}.")
            #If a source is provided, checks if the provided features are all supported by the source.
            #Finds the column of the provided source.
            classifier = feature[:feature.index('_')+1]
            for src, classifiers in new_source_dict.items():
                if source == src and (classifier not in classifiers and classifier not in ['F_', 'GC_']):
                    logging.error(f"{feature} is not in a category of features {classifiers} that {src} supports.")
                elif source == src:
                    sources = list(new_source_dict.keys())
                    source_num = sources.index(source)
            feat_indices.append(np.where(flattened_feats == feature)[0][0])

        if len(args) == 1 and not isinstance(args[0], list):
            logging.error("You only provided one language argument.\nProvide multiple language arguments, or a single list of languages as arguments.")
        if len(args) == 1 and isinstance(args[0], list):
            langs = args[0]
        else:
            langs = list(args)

        languages = self.langs[0]
        # print(f"In new_distance, getting languages array: {time.time() - start_time} seconds")

        if any(lang not in languages for lang in langs):
            logging.error(f"Unknown language: {lang}.")
        
        dist_start_time = time.time()

        list_indices_with_data = []
        langs_no_info = []

        #Gets feature data for genetic, geographic, and featural features. Needed as languages are ordered differently in each file.
        gen_feature_data = self.data[0] #0-3717
        geo_feature_data = self.data[2] #3718-4016
        feat_feature_data = self.data[1] #4017-4305

        for lang in langs:
            canCalculate = False
            indices_with_values = []
            for idx in feat_indices:
                if(idx >= 0 and idx <= 3717):
                    languages = self.langs[0]
                    lang_index = np.where(lang == languages)[0][0]
                    if(np.any(gen_feature_data[lang_index][idx] != -1.0)):
                        canCalculate = True
                        indices_with_values.append(idx)
                elif(idx >= 3718 and idx <= 4016):
                    languages = self.langs[2]
                    lang_index = np.where(lang == languages)[0][0]
                    if(np.any(geo_feature_data[lang_index][idx - 3718] != -1.0)):
                        canCalculate = True
                        indices_with_values.append(idx)
                else:
                    languages = self.langs[1]
                    lang_index = np.where(lang == languages)[0][0]
                    if source == 'A':
                        if(np.any(feat_feature_data[lang_index][idx - 4017] != -1.0)):
                            canCalculate = True
                            indices_with_values.append(idx)
                    else:
                        if(feat_feature_data[lang_index][idx - 4017][source_num] != -1.0):
                            canCalculate = True
                            indices_with_values.append(idx)
            if not canCalculate:
                if len(langs) == 2:
                        logging.error(f"No information for language: {lang} for any provided features. {lang}'s distance to other languages cannot be calculated.")
                else:
                    langs_no_info.append(lang)
                    list_indices_with_data.append([-1])
                    continue
            list_indices_with_data.append(indices_with_values)

        list_shared_indices = []

        if len(langs) == 2:
            shared_indices = list(set(list_indices_with_data[0]).intersection(list_indices_with_data[1]))

            if len(shared_indices) == 0:
                logging.error(f"No shared inputted features between {langs[0]} and {langs[1]} for which the two languages have information.\nUnable to calculate customized distance.")
            list_shared_indices = shared_indices
            lang_vectors = self._create_custom_vectors(langs, flattened_feats, list_shared_indices, source_num)
            dist = self._angular_distance(lang_vectors[0], lang_vectors[1])   
            logging.info(f"In new_distance, calculated angular distance for {dist} with 2 languages: {time.time() - dist_start_time} seconds")  
            return dist  
        else:
            langs_no_shared_info = []
            pairings = {}
            for l_a, list_a in enumerate(list_indices_with_data):
                for l_b, list_b in enumerate(list_indices_with_data):
                    shared_indices = list(set(list_a).intersection(list_b))
                    if not shared_indices:
                        list_shared_indices.append([-1])
                        langs_no_shared_info.append((langs[l_a], langs[l_b]))
                    else: list_shared_indices.append(shared_indices)
                    pairings[f"{langs[l_a]}-{langs[l_b]}"] = shared_indices

            lang_vectors = []
            vec_num = 0
            for i in range(len(langs)):
                vec, vec_num = self._create_custom_vectors(langs, flattened_feats, list_shared_indices, source_num, vec_num)
                lang_vectors.append(vec)

            flattened_lang_vectors = [sublist for group in lang_vectors for sublist in group]
    
            indices_to_calculate_1 = []
            indices_to_calculate_2 = []

            pairings_keys_list = list(pairings.keys())
            pair_index_map = {pair: idx for idx, pair in enumerate(pairings_keys_list)}
            angular_distances = []

            for pair in pairings_keys_list:
                index = pair_index_map[pair]
                reversed_pair = '-'.join(pair.split('-')[::-1])
                index_2 = pair_index_map[reversed_pair] if pair != reversed_pair else index

                indices_to_calculate_1.append(index)
                indices_to_calculate_2.append(index_2)

            for i, (index1, index2) in enumerate(zip(indices_to_calculate_1, indices_to_calculate_2)):
                if len(flattened_lang_vectors[i]) == 0:
                    angular_distances.append(0 if index1 == index2 else -1)
                else:
                    dis = self._angular_distance(flattened_lang_vectors[index1], flattened_lang_vectors[index2])
                    angular_distances.append(dis)

            array = np.array(angular_distances)
            matrix = array.reshape((len(lang_vectors[0]), len(lang_vectors[0])))

            if len(langs_no_info) > 0:
                logging.error(f"No inputted feature information for language(s) {str(langs_no_info)}. Their customized distance to other languages cannot be calculated.")
            if len(langs_no_shared_info) > 0:
                logging.error(f"No shared inputted features between pair(s) of languages {str(langs_no_shared_info)} for which pair(s) of languages have information. Unable to calculate their customized distance to each other.")
            if len(langs_no_info) > 0 or len(langs_no_shared_info) > 0:
                logging.error("Distances which could not be calculated are marked with a -1.")
            logging.info(f"In new_distance, calculating distances for {len(langs)} languages: {time.time() - dist_start_time} seconds")
            return matrix

    def feature_coverage(self):
        """
            Prints the number of languages with available data in URIEL+, separating by resoure-level.
        """
        if not self.is_glottocodes():
            self.set_glottocodes()
        for distance_type in ['genetic', 'geographic', 'featural', 'syntactic', 'phonological', 'inventory', 'morphological']:
            distance_languages = self.get_available_distance_languages(distance_type)
            languages_with_data = 0
            for lang in high_resource_languages_URIELPlus:
                if lang in distance_languages:
                    languages_with_data += 1
            logging.info(f'Number of high-resource languages with available {distance_type} data: {languages_with_data}.')

            languages_with_data = 0
            for lang in medium_resource_languages_URIELPlus:
                if lang in distance_languages:
                    languages_with_data += 1
            logging.info(f'Number of medium-resource languages with available {distance_type} data: {languages_with_data}.')

            languages_with_data = 0
            for lang in low_resource_languages_URIELPlus:
                if lang in distance_languages:
                    languages_with_data += 1
            logging.info(f'Number of low-resource languages with available {distance_type} data: {languages_with_data}.')
    
    def featural_confidence_score(self, lang1, lang2, distance_type="featural"):
        """
            Computes the confidence score for the featural distance between two languages.

            Args:
                lang1 (str): The first language code.
                lang2 (str): The second language code.
                distance_type (str, optional): The type of distance (featural, syntactic, phonological, inventory, or morphological). 
                Default is "featural".

            Returns:
                float: The computed confidence scores.
        """
        # check if imputation_metrics.csv exists
        assert distance_type in ['featural', 'syntactic', 'phonological', 'inventory', 'morphological']
        imputed = False
        if os.path.exists(os.path.join(self.cur_dir, "data", "imputation_metrics.csv")):
            imputed = True

        def check_agreement(lang, distance_type="featural"):
            database = self.map_new_distance_to_database(distance_type)
            feats, feature_data, languages = self.feats[database], self.data[database], self.langs[database]
            lang_index = np.where(languages == lang)[0][0]

            if distance_type == "syntactic":
                feat_mask = [feat[:2] == "S_" for feat in feats]
            elif distance_type == "phonological":
                feat_mask = [feat[:2] == "P_" for feat in feats]
            elif distance_type == "inventory":
                feat_mask = [feat[:4] == "INV_" for feat in feats]
            elif distance_type == "morphological":
                feat_mask = [feat[:2] == "M_" for feat in feats]
            else:
                feat_mask = [True] * len(feats)

                feats = [feat for i, feat in enumerate(feats) if feat_mask[i]]
                feature_data = feature_data[:, feat_mask, :]


            agreement = []
            for i in range(len(feats)):
                data = feature_data[lang_index][i]
                # Ignore all -1 values
                data = [x for x in data if x != -1]
                if len(data) == 0:
                    agreement.append(1)
                else:
                    # Count the most frequent value to check for consensus
                    most_common_value = max(set(data), key=data.count)
                    consensus = data.count(most_common_value) / len(data)
                    agreement.append(consensus)

            average_agreement = sum(agreement) / len(agreement) if len(
                agreement) > 0 else 0
            return average_agreement

        agreement1 = check_agreement(lang1, distance_type)
        agreement2 = check_agreement(lang2, distance_type)
        agreement_score = (agreement2 + agreement1) / 2

        def check_missing_values(lang, distance_type="featural"):
            database = self.map_new_distance_to_database(distance_type)
            feats, feature_data, languages = self.feats[database], self.data[database], self.langs[database]
            lang_index = np.where(languages == lang)[0][0]

            if distance_type == "syntactic":
                feat_mask = [feat[:2] == "S_" for feat in feats]
            elif distance_type == "phonological":
                feat_mask = [feat[:2] == "P_" for feat in feats]
            elif distance_type == "inventory":
                feat_mask = [feat[:4] == "INV_" for feat in feats]
            elif distance_type == "morphological":
                feat_mask = [feat[:2] == "M_" for feat in feats]
            else:
                feat_mask = [True] * len(feats)

            # Apply the mask to filter features
            feats = [feat for i, feat in enumerate(feats) if feat_mask[i]]
            feature_data = feature_data[:, feat_mask, :]

            # Compute the proportion of missing values (-1) for each feature
            missing_value_proportions = []
            for i in range(len(feats)):
                data = feature_data[lang_index][i].tolist()
                missing_value_proportions.append(data.count(-1) / len(data))

            # Average the missing value proportions across all features
            average_missing_value_proportion = (
                sum(missing_value_proportions) / len(missing_value_proportions)
                if len(missing_value_proportions) > 0
                else 0
            )

            return average_missing_value_proportion

        missing_values1 = check_missing_values(lang1, distance_type)
        missing_values2 = check_missing_values(lang2, distance_type)
        missing_values_score = 1 - ((missing_values1 + missing_values2) / 2)

        if imputed:
            file_path = os.path.join(self.cur_dir, "data", "imputation_metrics.csv")
            imputation_metrics = pd.read_csv(file_path)
            imputation_accuracy = float(imputation_metrics['accuracy'][0])
            imputation_score = imputation_accuracy

        if imputed:
            return agreement_score, imputation_score 
        return agreement_score, missing_values_score

    def non_featural_confidence_score(self, lang1, lang2, distance_type):
        """
            Computes the confidence score for non-featural distances between two languages (genetic or geographic).

            Args:
                lang1 (str): The first language code.
                lang2 (str): The second language code.
                distance_type (str): The type of distance (genetic or geographic).

            Returns:
                float: The computed confidence scores.
        """
        assert distance_type in ['genetic', 'geographic']
        def check_agreement(lang, distance_type):
            database = self.map_new_distance_to_database(distance_type)
            feats, feature_data, languages = self.feats[database], self.data[database], self.langs[database]
            lang_index = np.where(languages == lang)[0][0]
            agreement = []
            for i in range(len(feats)):
                data = feature_data[lang_index][i]
                # Ignore all -1 values
                data = [x for x in data if x != -1]
                if len(data) == 0:
                    agreement.append(1)
                else:
                    # Count the most frequent value to check for consensus
                    most_common_value = max(set(data), key=data.count)
                    consensus = data.count(most_common_value) / len(data)
                    agreement.append(consensus)

            average_agreement = sum(agreement) / len(agreement) if len(agreement) > 0 else 0
            return average_agreement

        agreement1 = check_agreement(lang1, distance_type)
        agreement2 = check_agreement(lang2, distance_type)
        agreement_score = (agreement2 + agreement1) / 2

        def check_missing_values(lang, distance_type):
            database = self.map_new_distance_to_database(distance_type)
            feats, feature_data, languages = self.feats[database], self.data[database], self.langs[database]
            lang_index = np.where(languages == lang)[0][0]

            # Compute the proportion of missing values (-1) for each feature
            missing_value_proportions = []
            for i in range(len(feats)):
                data = feature_data[lang_index][i].tolist()
                missing_value_proportions.append(data.count(-1) / len(data))

            # Average the missing value proportions across all features
            average_missing_value_proportion = (
                sum(missing_value_proportions) / len(missing_value_proportions)
                if len(missing_value_proportions) > 0
                else 0
            )

            return average_missing_value_proportion

        missing_values1 = check_missing_values(lang1, distance_type)
        missing_values2 = check_missing_values(lang2, distance_type)
        missing_values_score = 1 - ((missing_values1 + missing_values2) / 2)

        return agreement_score, missing_values_score

    def confidence_score(self, lang1, lang2, distance_type):
        """
            Computes the confidence score for the distance between two languages based on the specified distance type.

            Args:
                lang1 (str): The first language code.
                lang2 (str): The second language code.
                distance_type (str): The type of distance (featural, syntactic, phonological, inventory, morphological, genetic, or geographic).

            Returns:
                float: The computed confidence scores.
        """
        distance_types = ['genetic', 'geographic', 'featural', 'syntactic', 'phonological', 'inventory', 'morphological']
        if distance_type in ['featural', 'syntactic', 'phonological', 'inventory', 'morphological']:
            return self.featural_confidence_score(lang1, lang2, distance_type)
        elif distance_type in ['genetic', 'geographic']:
            return self.non_featural_confidence_score(lang1, lang2, distance_type)
        else:
            logging.error(f"{distance_type} is not an available distance type in URIEL+. Distance types are {distance_types}.")