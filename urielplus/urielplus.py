from .urielplus_databases import URIELPlusDatabases
from .urielplus_imputation import URIELPlusImputation
from .urielplus_querying import URIELPlusQuerying


import logging
import os
import shutil
import sys


import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


'''
URIEL+ library for integrating new and updated databases into URIEL and robust distance calculations.


Authors: Aditya Khan, Mason Shipton, David Anugraha, Kaiyao Duan, Phuong H. Hoang, Eric Khiu, A. Seza Doğruöz,
En-Shiun Annie Lee


Last modified: December 21, 2024
'''


class URIELPlus(URIELPlusDatabases, URIELPlusImputation, URIELPlusQuerying):
    def __init__(self):
        """
            Initializes the URIEL+ class, setting up vector identifications of languages, and instantiating the classes
            needed for integrating databases, imputing missing values, and querying the knowledge base.


            Logging:
                Info: Logs information when a file is missing in the `database` directory and copied from the original_uriel
                directory.


                Error: Logs an error if a file is not found in the original_uriel directory.
        """
        self.files = ["family_features.npz", "features.npz", "geocoord_features.npz"]
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.loaded_features  = []


        for file in self.files:
            file_path = os.path.join(self.cur_dir, "database", file)
            if not os.path.isfile(file_path):
                logging.info(f"{file_path} is missing in \"database\". Copying from \"original_uriel\"...")


                old_file_path = os.path.join(self.cur_dir, "database", "original_uriel", file)
                try:
                    shutil.copy(old_file_path, file_path)
                except FileNotFoundError:
                    logging.error(f"{file} not found in \"original_uriel\".")
                    sys.exit(1)
            with np.load(file_path, allow_pickle=True) as l:
                self.loaded_features.append(dict(l))


        self.feats = [l["feats"] for l in self.loaded_features]
        self.langs = [l["langs"] for l in self.loaded_features]
        self.data = [l["data"] for l in self.loaded_features]
        self.sources = [l["sources"] for l in self.loaded_features]


        self.databases = URIELPlusDatabases(self.feats, self.langs, self.data, self.sources)
        self.imputation = URIELPlusImputation(self.feats, self.langs, self.data, self.sources)
        self.querying = URIELPlusQuerying(self.feats, self.langs, self.data, self.sources)

        self.codes = self.get_codes()






    def get_loaded_features(self, l_name):
        """
            Returns the URIEL+ loaded features associated with the provided name, if the name is valid.


            Args:
                l_name (str): The name of the loaded features to return. Valid options are "phylogeny", "typological",
                or "geography".


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
        sys.exit(1)


    """
        The following three functions return loaded features representing phylogeny, typological, and geography
        vectors, respectively.


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
                l_name (str): The name of the loaded_features to update. Valid options are "phylogeny", "typological",
                or "geography".
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
            file_path = os.path.join(self.cur_dir, "database", file)
           
            try:
                with np.load(file_path, allow_pickle=True) as l:
                    l_idx = l_map[l_name]
                    self.loaded_features[l_idx] = dict(l)
                    self.feats[l_idx] = l["feats"]
                    self.langs[l_idx] = l["langs"]
                    self.data[l_idx] = l["data"]
                    self.sources[l_idx] = l["sources"]
                    self.files[l_idx] = file
                    self.databases = URIELPlusDatabases(self.feats, self.langs, self.data, self.sources)
                    self.imputation = URIELPlusImputation(self.feats, self.langs, self.data, self.sources)
                    self.querying = URIELPlusQuerying(self.feats, self.langs, self.data, self.sources)
                    logging.info(f"{l_name} loaded features updated successfully from {file}.")
            except FileNotFoundError:
                logging.error(f"File not found: {file_path}. Failed to update {l_name} loaded features.")
                sys.exit(1)
            except Exception as e:
                logging.error(f"An error occurred while loading the file {file}: {e}")
                sys.exit(1)
        else:
            logging.error(f"Unknown loaded features: {l_name}. Valid loaded features are {list(l_map.keys())}.")
            sys.exit(1)
   
    """
        The following three functions updates loaded features representing phylogeny, typological, and geography
        vectors, respectively.


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
            Returns the arrays within the URIEL+ loaded features associated with the provided name, if the name is
            valid.


            Args:
                l_name (str): The name of the loaded features to return. Valid options are "phylogeny", "typological",
                or "geography".


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
            np.ndarray: A NumPy array of either the features, languages, data, or sources of a specific loaded
            features.
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
            logging.error("invalid default answer: '%s'" % default)
            sys.exit(1)


        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with \"yes\" or \"no\" "
                                "(or 'y' or 'n').\n")
   


    def reset(self):
        """
            Restores the URIEL knowledge base by copying necessary files to the main data directory.


            The function prompts if the user wants to revert to URIEL, and if yes, then moves all old data files back
            to the main directory.
        """
        files_to_copy = ["family_features.npz",
                         "features.npz", "geocoord_features.npz"]
        cont = self.query_yes_no(f"Resetting to URIEL involves copying the files {files_to_copy} into the data directory. Any files with the same name will be replaced. Continue?")
        if not cont:
            return
        for file in files_to_copy:
            from_file_path = os.path.join(self.cur_dir, "database", "original_uriel", file)
            to_file_path = os.path.join(self.cur_dir, "database", file)
            try:
                shutil.copy(from_file_path, to_file_path)
            except Exception as e:
                logging.error(f"Difficulty copying {from_file_path} to {to_file_path}: {e}")
                sys.exit(1)
        self.loaded_features = []
        for file in self.files:
            file_path = os.path.join(self.cur_dir, "database", file)
            with np.load(file_path, allow_pickle=True) as l:
                self.loaded_features.append(dict(l))


        self.feats = [l["feats"] for l in self.loaded_features]
        self.langs = [l["langs"] for l in self.loaded_features]
        self.data = [l["data"] for l in self.loaded_features]
        self.sources = [l["sources"] for l in self.loaded_features]


        self.databases = URIELPlusDatabases(self.feats, self.langs, self.data, self.sources)
        self.imputation = URIELPlusImputation(self.feats, self.langs, self.data, self.sources)
        self.querying = URIELPlusQuerying(self.feats, self.langs, self.data, self.sources)

        self.codes = "Iso"
