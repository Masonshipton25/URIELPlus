import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

from .base_uriel import BaseURIEL
from .database.urielplus_csvs.resource_level_langs import (
    high_resource_languages_URIELPlus,
    medium_resource_languages_URIELPlus,
    low_resource_languages_URIELPlus,
)

class URIELPlusQuerying(BaseURIEL):
    def __init__(self, feats, langs, data, sources):
        """
            Initializes the Querying class, setting up vector identifications of languages with the constructor of the
            BaseURIEL class.

            Args:
                feats (np.ndarray): The features of the three loaded features.
                langs (np.ndarray): The languages of the three loaded features.
                data (np.ndarray): The data of the three loaded features.
                sources (np.ndarray): The sources of the three loaded features.
        """
        super().__init__(feats, langs, data, sources)

    def map_new_distance_to_loaded_features(self, distance):
        """
            Maps a distance type to the corresponding loaded features for URIEL+.

            Args:
                distance (str): The type of distance to map.

            Returns:
                str: The loaded features corresponding to the distance type.

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
        logging.error(f"{distance} is not an available feature category in URIEL+. Feature categories are {list(d.keys())}.")
        sys.exit(1)

    def is_available_distance_language(self, distance_type, lang):
        """
            Checks if a language contains at least one non-empty feature of the specified distance type.

            Args:
                distance_type (str): The type of distance to check.
                lang (str): The language to check.

            Returns:
                bool: True if language has one non-empty feature of the specified distance type, False otherwise.
        """
        loaded_features_idx = self.map_new_distance_to_loaded_features(distance_type)

        lang_index = np.where(self.langs[loaded_features_idx] == lang)[0][0]

        if loaded_features_idx == 0 or loaded_features_idx == 2:
            if not np.all(self.data[loaded_features_idx][lang_index] == -1.0):
                return True   
            return False

        aggregated_data = np.max(self.data[1], axis=-1)

        if self.fill_with_base_lang:
            self.dialects = self.get_dialects()
            for i in range(len(self.langs[1])):
                for parent, child in self.dialects.items():
                    if self.langs[1][i] in child:
                        for j in range(len(self.feats[1])):
                            parent_data = aggregated_data[parent][j]
                            if aggregated_data[i][j] == -1.0 and parent_data > -1.0:
                                aggregated_data[i][j] = parent_data
        
        aggregated_data = np.expand_dims(aggregated_data, axis=-1)

        if np.all(aggregated_data[lang_index] == -1.0):
            return False
        if distance_type == "featural":
            return True
        
        for feat_idx in range(len(self.feats[1])):
            feature_prefix = self.feats[1][feat_idx][:2]
            is_matching_feature = (
                (distance_type == "syntactic" and feature_prefix == "S_") or
                (distance_type == "phonological" and feature_prefix == "P_") or
                (distance_type == "inventory" and self.feats[1][feat_idx][:4] == "INV_") or
                (distance_type == "morphological" and feature_prefix == "M_")
            )

            if is_matching_feature:
                if (aggregated_data[lang_index][feat_idx] > -1):
                    return True
                
        return False
    
    def get_available_distance_languages(self, distance_type):
        """
            Retrieves a list of languages that contain at least one non-empty feature of the specified distance type.

            Args:
                distance_type (str): The type of distance to check.

            Returns:
                list: A list of languages.
        """
        loaded_features_idx = self.map_new_distance_to_loaded_features(distance_type)
        
        available_languages = []
        for lang_index in range(len(self.langs[loaded_features_idx])):
            flattened_data = self.data[loaded_features_idx][lang_index].flatten()
            if not np.all(np.isclose(flattened_data, -1.0)):
                available_languages.append(self.langs[loaded_features_idx][lang_index])

        if distance_type in ["syntactic", "phonological", "inventory", "morphological"]:
            subdomain_languages = []
            languages_to_remove = available_languages.copy()

            for feature_index in range(len(self.feats[loaded_features_idx])):
                feature_prefix = self.feats[loaded_features_idx][feature_index][:2]
                is_matching_feature = (
                    (distance_type == "syntactic" and feature_prefix == "S_") or
                    (distance_type == "phonological" and feature_prefix == "P_") or
                    (distance_type == "inventory" and self.feats[loaded_features_idx][feature_index][:4] == "INV_") or
                    (distance_type == "morphological" and feature_prefix == "M_")
                )

                if is_matching_feature:
                    for lang_index in range(len(self.langs[loaded_features_idx])):
                        if self.langs[loaded_features_idx][lang_index] in languages_to_remove:
                            if any(self.data[loaded_features_idx][lang_index][feature_index] > -1.0):
                                subdomain_languages.append(self.langs[loaded_features_idx][lang_index])
                                languages_to_remove.remove(self.langs[loaded_features_idx][lang_index])

            return subdomain_languages

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
            sys.exit(1)
        if len(args) == 1 and isinstance(args[0],list):
            langs = args[0]
        else:
            langs = [l for l in args]

        loaded_features_idx = self.map_new_distance_to_loaded_features(category)

        for lang in langs:
            if lang not in self.langs[loaded_features_idx]:
                logging.error(f"Unknown language: {lang}.")
                sys.exit(1)

        vectors = {}
        for i in range(len(langs)):
            # lang_index = np.where(self.langs[loaded_features_idx] == langs[i])[0][0]
            vector = []
            for j in range(len(self.feats[loaded_features_idx])):
                if category in ["genetic", "featural", "geographic"] or (category == "morphological" and (self.feats[loaded_features_idx][j])[:2] == "M_") or (category == "inventory" and (self.feats[loaded_features_idx][j])[:4] == "INV_") or (category == "phonological" and (self.feats[loaded_features_idx][j])[:2] == "P_") or (category == "syntactic" and (self.feats[loaded_features_idx][j])[:2] == "S_"):
                    i_data = self.data[loaded_features_idx][j]
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


    def _create_vectors(self, langs, loaded_features, list_shared_indices, vec_num=-1):
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
        lang_vectors = []

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._process_language,
                    i, langs, self.feats[loaded_features], self.data[loaded_features], self.langs[loaded_features], list_shared_indices, vec_num + i
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
            sys.exit(1)
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
        if self.distance_metric == "cosine":
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
            sys.exit(1)

        if len(args) == 1 and not isinstance(args[0], list):
            logging.error("You only provided one language argument.\nProvide multiple language arguments, or a single list of languages as arguments.")
            sys.exit(1)
        if len(args) == 1 and isinstance(args[0], list):
            langs = args[0]
        else:
            langs = list(args)

        unknown_langs = [lang for lang in langs if lang not in self.langs[0]]
        if unknown_langs:
            logging.error(f"Unknown languages: {', '.join(unknown_langs)}.")
            sys.exit(1)
        # print(f"In new_distance, getting languages array: {time.time() - start_time} seconds")

        angular_distances_list = []
        for dist in distance_list:
            dist_start_time = time.time()

            loaded_features_idx = self.map_new_distance_to_loaded_features(dist)

            list_indices_with_data = []
            langs_no_info = []

            # print(f"In new_distance, checking available distance languages: {time.time() - dist_start_time:.2f} seconds")

            for lang in langs:
                # print(f"In new_distance, loading feature data for language {lang}: {time.time() - dist_start_time:.2f} seconds")

                if not self.is_available_distance_language(dist, lang):
                    if len(langs) == 2:
                        logging.error(f"No {dist} information for language: {lang}. {lang}'s {dist} distance to other languages cannot be calculated.")
                        sys.exit(1)
                    else:
                        langs_no_info.append(lang)
                        list_indices_with_data.append([-1])
                        continue

                lang_index = np.where(self.langs[loaded_features_idx] == lang)[0][0]

                indices_with_values = [
                    j for j, feat in enumerate(self.feats[loaded_features_idx]) if (dist in ["genetic", "featural", "geographic"] or
                                                                (dist == "morphological" and feat[:2] == "M_") or
                                                                (dist == "inventory" and feat[:4] == "INV_") or
                                                                (dist == "phonological" and feat[:2] == "P_") or
                                                                (dist == "syntactic" and feat[:2] == "S_")) and not all(value == -1.0 for value in self.data[loaded_features_idx][lang_index][j])
                ]

                list_indices_with_data.append(indices_with_values)

            list_shared_indices = []

            if len(langs) == 2:
                shared_indices = list(set(list_indices_with_data[0]).intersection(list_indices_with_data[1]))

                if len(shared_indices) == 0:
                    logging.error(f"No shared {dist} features between {langs[0]} and {langs[1]} for which the two languages have information.\nUnable to calculate {dist} distance.")
                    sys.exit(1)
                list_shared_indices = shared_indices
                lang_vectors = self._create_vectors(langs, loaded_features_idx, list_shared_indices)
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
                    vec, vec_num = self._create_vectors(langs, loaded_features_idx, list_shared_indices, vec_num)
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
                    logging.info(f"No {dist} information for language(s) {str(langs_no_info)}. Their {dist} distance to other languages cannot be calculated.")
                if len(langs_no_shared_info) > 0:
                    logging.info(f"No shared {dist} features between pair(s) of languages {str(langs_no_shared_info)} for which pair(s) of languages have information. Unable to calculate their {dist} distance to each other.")
                if len(langs_no_info) > 0 or len(langs_no_shared_info) > 0:
                    logging.info("Distances which could not be calculated are marked with a -1.")
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
                source (str): The source to use features from ('A' for all sources, "ETHNO", "WALS", "PHOIBLE_UPSID", 
                "PHOIBLE_SAPHON", "PHOIBLE_GM", "PHOIBLE_PH", "PHOIBLE_AA", "SSWL", "PHOIBLE_RA", "PHOIBLE_SPA").

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
                sys.exit(1)

        source_dict = {
            "ETHNO": ["S_", "P_", "INV_"],
            "WALS": ["S_", "P_"],
            "PHOIBLE_UPSID": ["P_", "INV_"],
            "PHOIBLE_SAPHON": ["P_", "INV_"],
            "PHOIBLE_GM": ["P_", "INV_"],
            "PHOIBLE_PH": ["P_", "INV_"],
            "PHOIBLE_AA": ["P_", "INV_"],
            "SSWL": ["S_"],
            "PHOIBLE_RA": ["P_", "INV_"],
            "PHOIBLE_SPA": ["P_", "INV_"],
            "BDPROTO": ["P_", "INV_"],
            "GRAMBANK": ["S_", "M_"],
            "APICS": ["S_", "P_", "INV_", "M_"],
            "EWAVE": ["S_", "M_"]
        }

        #If the UPDATED_SAPHON database has been integrated, replaces the PHOIBLE_SAPHON source with UPDATED_SAPHON.
        new_source_dict = {}
        if "UPDATED_SAPHON" in self.sources[1]:
            for key, value in source_dict.items():
                if key == "PHOIBLE_SAPHON":
                    new_source_dict["UPDATED_SAPHON"] = ["P_", "INV_"]
                else:
                    new_source_dict[key] = value
        else:
            new_source_dict = source_dict

        if not isinstance(features, list):
            logging.error(f"Unknown features type {features}. Provide a list of str.")
            sys.exit(1)
        
        #Creates a list of all features from all files.
        flattened_feats = np.concatenate([self.feats[0], self.feats[2], self.feats[1]])

        #Finds the indices of the provided features within the list of all the features.
        source_num = None
        feat_indices = []
        for feature in features:
            if (feature not in flattened_feats):
                logging.error(f"Unknown feature: {feature}.")
                sys.exit(1)
            #If a source is provided, checks if the provided features are all supported by the source.
            #Finds the column of the provided source.
            classifier = feature[:feature.index('_')+1]
            for src, classifiers in new_source_dict.items():
                if source == src and (classifier not in classifiers and classifier not in ["F_", "GC_"]):
                    logging.error(f"{feature} is not in a category of features {classifiers} that {src} supports.")
                    sys.exit(1)
                elif source == src:
                    sources = list(new_source_dict.keys())
                    source_num = sources.index(source)
            feat_indices.append(np.where(flattened_feats == feature)[0][0])

        if len(args) == 1 and not isinstance(args[0], list):
            logging.error("You only provided one language argument.\nProvide multiple language arguments, or a single list of languages as arguments.")
            sys.exit(1)
        if len(args) == 1 and isinstance(args[0], list):
            langs = args[0]
        else:
            langs = list(args)

        # print(f"In new_distance, getting languages array: {time.time() - start_time} seconds")

        if any(lang not in self.langs[0] for lang in langs):
            logging.error(f"Unknown language: {lang}.")
            sys.exit(1)
        
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
                    lang_index = np.where(lang == self.langs[0])[0][0]
                    if(np.any(gen_feature_data[lang_index][idx] != -1.0)):
                        canCalculate = True
                        indices_with_values.append(idx)
                elif(idx >= 3718 and idx <= 4016):
                    lang_index = np.where(lang == self.langs[2])[0][0]
                    if(np.any(geo_feature_data[lang_index][idx - 3718] != -1.0)):
                        canCalculate = True
                        indices_with_values.append(idx)
                else:
                    lang_index = np.where(lang == self.langs[1])[0][0]
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
                        sys.exit(1)
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
                sys.exit(1)
            list_shared_indices = shared_indices
            lang_vectors = self._create_custom_vectors(langs, flattened_feats, list_shared_indices, source_num)
            dist = self._angular_distance(lang_vectors[0], lang_vectors[1])   
            logging.info(f"In new_distance, calculated angular distance for provided features with 2 languages: {time.time() - dist_start_time} seconds")  
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
                logging.info(f"No inputted feature information for language(s) {str(langs_no_info)}. Their customized distance to other languages cannot be calculated.")
            if len(langs_no_shared_info) > 0:
                logging.info(f"No shared inputted features between pair(s) of languages {str(langs_no_shared_info)} for which pair(s) of languages have information. Unable to calculate their customized distance to each other.")
            if len(langs_no_info) > 0 or len(langs_no_shared_info) > 0:
                logging.info("Distances which could not be calculated are marked with a -1.")
            logging.info(f"In new_distance, calculating distances for {len(langs)} languages: {time.time() - dist_start_time} seconds")
            return matrix

        
    def feature_coverage(self, resource_level, distance_type):
        """
            Prints the number of languages with available data in URIEL+ for the provided resoure-level and distance type.

            Args:
                resource_level (str): The resource level of the languages to check feature coverage of. Options are 
                high-resource, medium-resource, and low-resource.

                distance_type (str): The type of distance (featural, syntactic, phonological, inventory, or morphological). 

            Returns:
                list: List of languages of the provided resource-level with available distance type data.
        """
        if not self.is_glottocodes():
            logging.error("Cannot retrieve feature coverage if languages in URIEL+ are not all of Glottocode language representation.")
            sys.exit(1)
        r_map = {
            "high-resource": high_resource_languages_URIELPlus,
            "medium-resource": medium_resource_languages_URIELPlus,
            "low-resource": low_resource_languages_URIELPlus,
        }
        if resource_level in r_map:
            resource_level_languages = r_map[resource_level]
        
        distance_languages = self.get_available_distance_languages(distance_type)

        langs_with_data = 0

        for lang in resource_level_languages:
            if lang in distance_languages:
                langs_with_data += 1
        return langs_with_data

    def all_feature_coverage(self):
        """
            Prints the number of languages with available data in URIEL+ for all resoure-levels and distance types.
        """
        if not self.is_glottocodes():
            self.set_glottocodes()
        for distance_type in ["genetic", "geographic", "featural", "syntactic", "phonological", "inventory", "morphological"]:
            distance_languages = self.get_available_distance_languages(distance_type)
            languages_with_data = 0
            for lang in high_resource_languages_URIELPlus:
                if lang in distance_languages:
                    languages_with_data += 1
            logging.info(f"Number of high-resource languages with available {distance_type} data: {languages_with_data}.")

            languages_with_data = 0
            for lang in medium_resource_languages_URIELPlus:
                if lang in distance_languages:
                    languages_with_data += 1
            logging.info(f"Number of medium-resource languages with available {distance_type} data: {languages_with_data}.")

            languages_with_data = 0
            for lang in low_resource_languages_URIELPlus:
                if lang in distance_languages:
                    languages_with_data += 1
            logging.info(f"Number of low-resource languages with available {distance_type} data: {languages_with_data}.")

    
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
        assert distance_type in ["featural", "syntactic", "phonological", "inventory", "morphological"]
        imputed = False
        # if os.path.exists(os.path.join(self.cur_dir, "database", "imputation_metrics.csv")):
        #     imputed = True
        if "imputed" in self.sources[1]:
            imputed = True
            
        def check_agreement(lang, distance_type="featural"):
            loaded_features = self.map_new_distance_to_loaded_features(distance_type)
            feats = self.feats[loaded_features]
            feature_data = self.data[loaded_features]
            lang_index = np.where(self.langs[loaded_features] == lang)[0][0]

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
            loaded_features_idx = self.map_new_distance_to_loaded_features(distance_type)
            feats = self.feats[loaded_features_idx]
            feature_data = self.data[loaded_features_idx]
            lang_index = np.where(self.langs[loaded_features_idx] == lang)[0][0]

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
            file_path = os.path.join(self.cur_dir, "database", "imputation_metrics.csv")
            imputation_metrics = pd.read_csv(file_path)
            imputation_accuracy = float(imputation_metrics["accuracy"][0])
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
        assert distance_type in ["genetic", "geographic"]
        def check_agreement(lang, distance_type):
            loaded_features_idx = self.map_new_distance_to_loaded_features(distance_type)
            lang_index = np.where(self.langs[loaded_features_idx] == lang)[0][0]
            agreement = []
            for i in range(len(self.feats[loaded_features_idx])):
                data = self.data[loaded_features_idx][lang_index][i]
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
            loaded_features_idx = self.map_new_distance_to_loaded_features(distance_type)
            lang_index = np.where(self.langs[loaded_features_idx] == lang)[0][0]

            # Compute the proportion of missing values (-1) for each feature
            missing_value_proportions = []
            for i in range(len(self.feats[loaded_features_idx])):
                data = self.data[loaded_features_idx][lang_index][i].tolist()
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
        distance_types = ["genetic", "geographic", "featural", "syntactic", "phonological", "inventory", "morphological"]
        if distance_type in ["featural", "syntactic", "phonological", "inventory", "morphological"]:
            return self.featural_confidence_score(lang1, lang2, distance_type)
        elif distance_type in ["genetic", "geographic"]:
            return self.non_featural_confidence_score(lang1, lang2, distance_type)
        else:
            logging.error(f"{distance_type} is not an available distance type in URIEL+. Distance types are {distance_types}.")
            sys.exit(1)