import csv
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
        super().__init__(feats, langs, data, sources)

    def set_glottocodes(self):
        """
            Sets the language codes in URIEL+ to Glottocodes.

            This function reads a mapping CSV file and applies the mappings to all relevant data files,
            saving the updated data back to disk if caching is enabled.
        """
        if self.is_glottocodes():
            logging.error("Already using Glottocodes.")
            sys.exit(1)
        
        logging.info("Converting ISO 639-3 codes to Glottocodes....")

        csv_path = os.path.join(self.cur_dir, "database", "urielplus_csvs", "uriel_glottocode_map.csv")
        map_df = pd.read_csv(csv_path)

        for i, file in enumerate(self.files):
            langs_df = pd.DataFrame(self.langs[i], columns=["code"])
            merged_df = pd.merge(langs_df, map_df, on="code", how="inner")
            merged_df = merged_df.drop(columns=['X', "code"])
            merged_np = merged_df.to_numpy()
            na_indices = langs_df.index.difference(merged_df.index)
            data_cleaned = np.delete(self.data[i], na_indices, axis=0)

            self.langs[i] = merged_np
            self.data[i] = data_cleaned
            self.langs[i] = np.array([l[0] for l in self.langs[i]])

            if self.cache:
                np.savez(os.path.join(self.cur_dir, "database", file), 
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

            if math.isnan(lat) or math.isnan(lon):
                self.data[2][self.data[2].shape[0] - len(new_langs) + i, :, -1] = -1.0
                continue

            distances = [math.dist([lat, lon], coord) for coord in coords]
            min_index = np.argmin(distances)
            self.data[2][self.data[2].shape[0] - len(new_langs) + i, min_index, -1] = 1.0

        if self.cache:
            np.savez(os.path.join(self.cur_dir, "database", self.files[2]), feats=self.feats[2], data=self.data[2], langs=self.langs[2], sources=self.sources[2])

    def combine_features(self, second_source, feat_sets):
        """
            Combines duplicate features and handles opposite features within a specified category of URIEL+ data.

            Args:
                second_source (str): The source of the secondary features to combine.
                feat_sets (list): A list of feature sets, each containing the primary feature and the secondary 
                features to combine.
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
            np.savez(os.path.join(self.cur_dir, "database", self.files[1]), feats=self.feats[1], langs=self.langs[1], data=self.data[1], sources=self.sources[1])

    def inferred_features(self):
        """
            Combines duplicate features across all sources in URIEL+ by using the `combine_features` function.

            The function iterates through the available sources and combines features based on predefined sets of 
            duplicate features.
        """
        logging.info("Updating feature data based on inferred features...")
        for source in self.sources[1]:
            self.combine_features(source, _u)

        logging.info("Updated feature data based on inferred features complete.")

    def integrate_saphon(self, convert_glottocodes_param=False):
        """
            Updates URIEL+ with data from the updated SAPHON database.

            This function integrates the updated SAPHON data.

            Args: 
                convert_glottocodes_param (bool): If True, converts language codes to Glottocodes.
        """
        self.is_database_incorporated("UPDATED_SAPHON")

        if not self.is_glottocodes() and convert_glottocodes_param:
            self.set_glottocodes()
            logging.info("Importing updated SAPHON from \"saphon_data_glottocodes.csv\"....")
        else:
            logging.info("Importing updated SAPHON from \"saphon_data.csv\"....")

        saphon_csv = os.path.join(self.cur_dir, "database", "urielplus_csvs", "saphon_data_glottocodes.csv") if self.is_glottocodes() else os.path.join(self.cur_dir, "database", "urielplus_csvs", "saphon_data.csv")

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
                            data_string += ",\n"
                        feat_num += 1
                    lang_index = np.where(self.langs[1] == lang)[0][0]
                    (self.data[1][lang_index])[:289, :10] = np.array(eval(data_string))

        index = np.where(self.sources[1] == "PHOIBLE_SAPHON")
        self.sources[1][index] = "UPDATED_SAPHON"

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

        if not self.is_glottocodes():
            self.set_glottocodes()

        csv_path = os.path.join(self.cur_dir, "database", "urielplus_csvs", "bdproto_data.csv")
        df = pd.read_csv(csv_path)

        langlist = self.langs[1].tolist()
        new_langs = []
        modify_langs = []
        for lang in df["name"]:
            if lang != "altaic" and lang != "australian" and lang != "finno-permic" and lang != "finno-ugric" and lang != "nostratic" and lang != "proto-baltic-finnic" and lang != "proto-eastern-oceanic" and lang != "proto-finno-permic" and lang != "proto-finno-saamic" and lang != "proto-mamore-guapore" and lang != "proto-nilo-saharan" and lang != "proto-tibeto-burman" and lang != "proto-totozoquean" and lang != "uralo-siberian":
                if ("UPDATE_LANG_URIEL" not in lang and lang not in langlist) or lang == "oldp1255_UPDATE_LANG_URIEL":
                    new_langs.append(lang.replace("_UPDATE_LANG_URIEL", ""))
                modify_langs.append(lang.replace("_UPDATE_LANG_URIEL", ""))

        self.data[1] = self._set_new_data_dimensions(self.data[1], [], new_langs, ["BDPROTO"])

        self.langs[1] = np.append(self.langs[1], np.array(new_langs).flatten())
        self.sources[1] = np.append(self.sources[1], "BDPROTO")

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

        if not self.is_glottocodes():
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

        if not self.is_glottocodes():
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

        if not self.is_glottocodes():
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
                # feat_index = self.data[1].shape[1] - len(df.columns[1:]) + j
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
                sys.exit(1)
            
        logging.info("Custom databases integration complete.")
