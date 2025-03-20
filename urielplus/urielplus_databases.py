import logging
import math
import os
import re
import sys


import numpy as np
import pandas as pd


from .base_uriel import BaseURIEL
from .database.urielplus_csvs.duplicate_feature_sets import _u, _ug, _uai, _ue


class URIELPlusDatabases(BaseURIEL):
    def __init__(self, feats, langs, data, sources):
        """
            Initializes the Databases class, setting up vector identifications of languages with the constructor of the
            BaseURIEL class.


            Args:
                feats (np.ndarray): The features of the three loaded features.
                langs (np.ndarray): The languages of the three loaded features.
                data (np.ndarray): The data of the three loaded features.
                sources (np.ndarray): The sources of the three loaded features.
        """
        super().__init__(feats, langs, data, sources)




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
            Expands the URIEL+ data array to accommodate new features, languages, and sources, initializing new values
            to -1.0.


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
        if any(database in source for source in self.sources[1]):
            logging.error(f"{database} database already integrated.")
            sys.exit(1)


    def _calculate_phylogeny_vectors(self):
        """
            This function reads the relevant CSV file and updates the phylogeny arrays based on the phylogeny
            classifications of new languages.


            If caching is enabled, updates the "family_features.npz" file.
           
        """
        csv_path = os.path.join(self.cur_dir, "database", "urielplus_csvs", "lang_fam_geo.csv")
        fam_geo_feat_csv = pd.read_csv(csv_path)


        new_langs = np.setdiff1d(self.langs[1], self.langs[0])


        for l in new_langs:
            fam = fam_geo_feat_csv.loc[fam_geo_feat_csv["code"] == l, "families"]
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
                fam_string = "F_" + f
                family_idx = np.where(self.feats[0] == fam_string)[0]
                if len(family_idx) == 0:
                    continue
                self.data[0][new_lang_idx, family_idx, -1] = 1.0


        if self.cache:
            np.savez(os.path.join(self.cur_dir, "database", self.files[0]), feats=self.feats[0], data=self.data[0], langs=self.langs[0], sources=self.sources[0])


    def _calculate_geocoord_vectors(self):
        """
            This function calculates the geographic distances between new languages and geocoordinates, creating
            geography vectors.


            If caching is enabled, updates the `geocoord_features.npz` file.
        """
        new_langs = np.setdiff1d(self.langs[1], self.langs[2])
        self.langs[2] = np.append(self.langs[2], new_langs)
        self.data[2] = self._set_new_data_dimensions(self.data[2], [], new_langs, [])


        coords = [list(map(int, re.findall(r"-?\d+", feat))) for feat in self.feats[2]]
        csv_path = os.path.join(self.cur_dir, "database", "urielplus_csvs", "lang_fam_geo.csv")
        fam_geo_feat_csv = pd.read_csv(csv_path)


        for i, l in enumerate(new_langs):
            lat = fam_geo_feat_csv.loc[fam_geo_feat_csv["code"] == l, "lat"]
            lon = fam_geo_feat_csv.loc[fam_geo_feat_csv["code"] == l, "lon"]
            lat, lon = lat.values[0], lon.values[0]


            #Language does not have a known location spoken.
            if math.isnan(lat) or math.isnan(lon):
                self.data[2][self.data[2].shape[0] - len(new_langs) + i, :, -1] = -1.0
                continue


            distances = [math.dist([lat, lon], coord) for coord in coords]
            min_index = np.argmin(distances)
            self.data[2][self.data[2].shape[0] - len(new_langs) + i, min_index, -1] = 1.0


        if self.cache:
            np.savez(os.path.join(self.cur_dir, "database", self.files[2]), feats=self.feats[2], data=self.data[2], langs=self.langs[2], sources=self.sources[2])


    def combine_features(self, secondary_source, feature_sets):
        """
            Combines duplicate features and handles opposite features within a specified category of URIEL+ data.


            Args:
                second_source (str): The source of the secondary features to combine.
                feat_sets (list): A list of feature sets, each containing the primary feature and the secondary
                features to combine.
        """
        secondary_source_index = np.where(self.sources[1] == secondary_source)[0][0]


        for feature_set in feature_sets:
            primary_feature = feature_set[0]


            if isinstance(feature_set[1], float):
                specific_value = feature_set[1]
                secondary_features = feature_set[2:]
            else:
                secondary_features = feature_set[1:]


            # Add the primary feature if it doesn't already exist
            if primary_feature not in self.feats[1]:
                self.data[1] = self._set_new_data_dimensions(self.data[1], [primary_feature], [], [])
                self.feats[1] = np.append(self.feats[1], primary_feature)
                primary_feature_index = len(self.feats[1]) - 1
            else:
                primary_feature_index = np.where(self.feats[1] == primary_feature)[0][0]


            for secondary_feature in secondary_features:
                secondary_feature_index = np.where(self.feats[1] == secondary_feature)[0][0]


                for lang_idx in range(len(self.langs[1])):
                    secondary_data = self.data[1][lang_idx][secondary_feature_index]
                    primary_data = self.data[1][lang_idx][primary_feature_index]


                    for src_idx in range(len(primary_data)):
                        if src_idx == secondary_source_index:
                            if isinstance(feature_set[1], float):
                                is_primary_unknown = (primary_data[src_idx] == -1.0)
                                is_primary_absent = (primary_data[src_idx] == 0.0)
                                is_secondary_present = (secondary_data[src_idx] == specific_value)


                                if (is_primary_unknown or is_primary_absent) and is_secondary_present:
                                    self.data[1][lang_idx][primary_feature_index][src_idx] = specific_value
                            else:
                                if secondary_data[src_idx] > primary_data[src_idx]:
                                    self.data[1][lang_idx][primary_feature_index][src_idx] = secondary_data[src_idx]


        if self.cache:
            np.savez(os.path.join(self.cur_dir, "database", self.files[1]), feats=self.feats[1], langs=self.langs[1], data=self.data[1], sources=self.sources[1])


    def inferred_features(self):
        """
            Combines duplicate features across all sources in URIEL+ by using the `combine_features` function.


            The function iterates through the available sources and combines features based on predefined sets of
            duplicate features.
        """
        logging.info("Inferring feature data based on similar features.....")
        for source in self.sources[1]:
            self.combine_features(source, _u)


        logging.info("Inferred feature data based on similar features.")


    def integrate_saphon(self, convert_glottocodes_param=False):
        """
            Updates URIEL+ with data from the updated SAPHON database.


            This function integrates the updated SAPHON data.


            Args:
                convert_glottocodes_param (bool): If True, converts language codes to Glottocodes.
        """
        self.is_database_incorporated("UPDATED_SAPHON")


        logging.info("Importing updated SAPHON from \"saphon_data.csv\"....")


        saphon_data = pd.read_csv(os.path.join(self.cur_dir, "database", "urielplus_csvs", "saphon_data.csv"))


        code_col = "code" if (self.codes == "Iso" and not convert_glottocodes_param) else "glottocode"


        source_index = np.where(self.sources[1] == "PHOIBLE_SAPHON")


        for i, lang in enumerate(saphon_data[code_col]):
            if not pd.isna(lang):
                lang_index = np.where(self.langs[1] == lang)[0][0]
                for feat in saphon_data.columns[2:]:
                    feat_index = np.where(self.feats[1] == feat)
                    self.data[1][lang_index, feat_index, source_index] = saphon_data[feat][i]


        self.sources[1][source_index] = "UPDATED_SAPHON"


        if self.cache:
            np.savez(os.path.join(self.cur_dir, "database", self.files[1]),
                     feats=self.feats[1], data=self.data[1], langs=self.langs[1], sources=self.sources[1])


        logging.info("Updated SAPHON integration complete..")


    def integrate_bdproto(self):
        """
            Updates URIEL+ with data from the BDPROTO database.


            This function integrates the BDPROTO data, converting language codes to Glottocodes if necessary,
            and updates the feature data in URIEL+.
        """
        self.is_database_incorporated("BDPROTO")


        logging.info("Importing BDPROTO from \"bdproto_data.csv\"....")


        if self.codes == "Iso":
            self.set_glottocodes()


        bdproto_data = pd.read_csv(os.path.join(self.cur_dir, "database", "urielplus_csvs", "bdproto_data.csv"))


        new_langs = self._get_new_languages(self.langs[1], bdproto_data, "name")
        new_source = "BDPROTO"


        old_num_langs = self.data[1].shape[0]
        self.data[1] = self._set_new_data_dimensions(self.data[1], [], new_langs, [new_source])


        new_langs_added = 0
        for i, lang in enumerate(bdproto_data["name"]):
            lang_index = None
            try:
                lang_index = np.where(self.langs[1] == lang)[0][0]
                if type(lang_index) not in [int, np.int64, float, np.float64]:
                    lang_index = old_num_langs + new_langs_added
                    new_langs_added += 1
            except:
                lang_index = old_num_langs + new_langs_added
                new_langs_added += 1
            for feat in bdproto_data.columns[1:]:
                feat_index = np.where(self.feats[1] == feat)
                self.data[1][lang_index, feat_index, -1] = bdproto_data[feat][i]
        self.langs[1] = np.append(self.langs[1], np.array(new_langs).flatten())
        self.sources[1] = np.append(self.sources[1], new_source)


        if self.cache:
            np.savez(os.path.join(self.cur_dir, "database", self.files[1]),
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
        self.is_database_incorporated("GRAMBANK")


        logging.info("Importing Grambank from \"grambank_data.csv\"....")

  

        if self.codes == "Iso":
            self.set_glottocodes()


        grambank_data = pd.read_csv(os.path.join(self.cur_dir, "database", "urielplus_csvs", "grambank_data.csv"))


        new_feats = self._get_new_features(self.feats[1], grambank_data.columns[1:])
        new_langs = self._get_new_languages(self.langs[1], grambank_data, "code")
        new_source = "GRAMBANK"


        old_num_langs = self.data[1].shape[0]
        self.data[1] = self._set_new_data_dimensions(self.data[1], new_feats, new_langs, [new_source])


        self.feats[1] = np.append(self.feats[1], new_feats)


        new_langs_added = 0
        for i, lang in enumerate(grambank_data["code"]):
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
            np.savez(os.path.join(self.cur_dir, "database", self.files[1]),
                     feats=self.feats[1], data=self.data[1], langs=self.langs[1], sources=self.sources[1])


        self._calculate_phylogeny_vectors()
        self._calculate_geocoord_vectors()


        self.combine_features("GRAMBANK", _ug)


        logging.info("Grambank integration complete.")


    def integrate_apics(self):
        """
            Updates URIEL+ with data from the APiCS database.


            This function integrates the APiCS data, converting language codes to Glottocodes if necessary,
            and updates the feature data in URIEL+.
        """
        self.is_database_incorporated("APICS")


        logging.info("Importing APiCS from \"apics_data.csv\"....")


        if self.codes == "Iso":
            self.set_glottocodes()


        apics_data = pd.read_csv(os.path.join(self.cur_dir, "database", "urielplus_csvs", "apics_data.csv"))


        new_langs = self._get_new_languages(self.langs[1], apics_data, "Language_ID")


        apics_data = apics_data.drop(columns=["Name"])
        apics_data = apics_data[["Language_ID"] + [col for col in apics_data.columns if col != "Language_ID"]]


        new_feats = self._get_new_features(self.feats[1], apics_data.columns[1:])


        new_source = "APICS"


        old_num_langs = self.data[1].shape[0]
        self.data[1] = self._set_new_data_dimensions(self.data[1], new_feats, new_langs, [new_source])


        self.feats[1] = np.append(self.feats[1], new_feats)


        new_langs_added = 0
        for i, lang in enumerate(apics_data["Language_ID"]):
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
            np.savez(os.path.join(self.cur_dir, "database", self.files[1]),
                     feats=self.feats[1], data=self.data[1], langs=self.langs[1], sources=self.sources[1])


        self._calculate_phylogeny_vectors()
        self._calculate_geocoord_vectors()


        self.combine_features("APICS", _uai)


        logging.info("APiCS integration complete.")


    def integrate_ewave(self):
        """
            Updates URIEL+ with data from the EWAVE database.


            This function integrates the EWAVE data, converting language codes to Glottocodes if necessary,
            and updates the feature data in URIEL+.
        """
        self.is_database_incorporated("EWAVE")


        logging.info("Importing eWAVE from \"english_dialect_data.csv\"....")


        if self.codes == "Iso":
            self.set_glottocodes()


        df = pd.read_csv(os.path.join(os.path.join(self.cur_dir, "database", "urielplus_csvs", "english_dialect_data.csv")))


        new_langs = self._get_new_languages(self.langs[1], df, "name")
        new_feats = self._get_new_features(self.feats[1], df.columns[1:])
        new_source = "EWAVE"


        old_num_langs = self.data[1].shape[0]
        self.data[1] = self._set_new_data_dimensions(self.data[1], df.columns[1:], new_langs, [new_source])


        self.feats[1] = np.append(self.feats[1], new_feats)


        new_langs_added = 0
        for i, lang in enumerate(df["name"]):
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
                self.data[1][lang_index, feat_index, -1] = df[feat][i]
        self.langs[1] = np.append(self.langs[1], np.array(new_langs).flatten())
        self.sources[1] = np.append(self.sources[1], new_source)


        if self.cache:
            np.savez(os.path.join(self.cur_dir, "database", self.files[1]),
                     feats=self.feats[1], data=self.data[1], langs=self.langs[1], sources=self.sources[1])


        self._calculate_phylogeny_vectors()
        self._calculate_geocoord_vectors()

        self.combine_features("EWAVE", _ue)


        logging.info("eWAVE integration complete.")


    def integrate_databases(self):
        """
            Updates URIEL+ with data from all available databases (UPDATED_SAPHON, BDPROTO, GRAMBANK, APICS, EWAVE).
        """
        logging.info("Importing all databases....")


        databases = {
            "UPDATED_SAPHON": self.integrate_saphon,
            "BDPROTO": self.integrate_bdproto,
            "GRAMBANK": self.integrate_grambank,
            "APICS": self.integrate_apics,
            "EWAVE": self.integrate_ewave
        }
       
        for db, integrate_method in databases.items():
            if db not in self.sources[1]:
                integrate_method()


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


        if len(args) == 1 and isinstance(args[0], list):
            databases = args[0]
        else:
            databases = list(args)


        valid_databases = {
            "UPDATED_SAPHON": self.integrate_saphon,
            "BDPROTO": self.integrate_bdproto,
            "GRAMBANK": self.integrate_grambank,
            "APICS": self.integrate_apics,
            "EWAVE": self.integrate_ewave,
            "INFERRED": self.inferred_features
        }


        for db in databases:
            if db in valid_databases and db not in self.sources[1]:
                valid_databases[db]()
            elif db in self.sources[1]:
                pass
            else:
                logging.error(f"Unknown database: {db}. Valid databases are {list(valid_databases.keys())}.")
                sys.exit(1)
           
        logging.info("Custom databases integration complete.")