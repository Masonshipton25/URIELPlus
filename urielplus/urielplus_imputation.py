import datetime
import logging
import os
import sys


import contexttimer
import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
from joblib import Parallel, delayed
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score)
from sklearn.model_selection import KFold, train_test_split


from .base_uriel import BaseURIEL


class URIELPlusImputation(BaseURIEL):
    def __init__(self, feats, langs, data, sources):
        """
            Initializes the Imputation class, setting up vector identifications of languages with the constructor of the
            BaseURIEL class.


            Args:
                feats (np.ndarray): The features of the three loaded features.
                langs (np.ndarray): The languages of the three loaded features.
                data (np.ndarray): The data of the three loaded features.
                sources (np.ndarray): The sources of the three loaded features.
        """
        super().__init__(feats, langs, data, sources)




    def aggregate(self):
        """
            Computes the union or average of feature data across sources in URIEL+.


            The union operation takes the maximum value across sources for each feature and language combination.
            The average operation takes the average across sources for each feature and language combination.


            If caching is enabled, creates an npz file with the union or average of feature data across sources in URIEL+.
        """
        if self.aggregation == 'U':
            logging.info("Creating union of data across sources....")
            aggregated_data = np.max(self.data[1], axis=-1)
            if self.cache:
                self.sources[1] = ["UNION"]
                file_name = "feature_union.npz"
        else:
            logging.info("Creating average of data across sources....")
            aggregated_data = np.where(self.data[1] == -1, np.nan, self.data[1])
            aggregated_data = np.nanmean(aggregated_data, axis=-1)
            aggregated_data = np.where(np.isnan(aggregated_data), -1, aggregated_data)
            if self.cache:
                self.sources[1] = ["AVERAGE"]
                file_name = "feature_average.npz"


        if self.fill_with_base_lang:
            self.dialects = self.get_dialects()
            for lang_idx in range(len(self.langs[1])):
                for parent, child in self.dialects.items():
                    if self.langs[1][lang_idx] in child:
                        for feat_idx in range(len(self.feats[1])):
                            parent_data = aggregated_data[parent][feat_idx]
                            dataUnknown = (aggregated_data[lang_idx][feat_idx] == -1.0)
                            parentDataKnown = parent_data > -1.0
                            if dataUnknown and parentDataKnown:
                                aggregated_data[lang_idx][feat_idx] = parent_data
       
        aggregated_data = np.expand_dims(aggregated_data, axis=-1)
       
        self.data[1] = aggregated_data


        if self.cache:
            np.savez(os.path.join(self.cur_dir, "database", file_name), feats=self.feats[1], data=self.data[1], langs=self.langs[1], sources=self.sources[1])


        if self.aggregation == 'U':
            logging.info("Union across sources creation complete.")
        else:
            logging.info("Average across sources creation complete.")


        return aggregated_data


    def _preprocess_data(self, df, feature_prefixes=("S_", "P_", "INV_", "M_")):
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


            overall_metrics["accuracy"] = accuracy
            overall_metrics["classification_error"] = classification_error
            overall_metrics["precision"] = precision
            overall_metrics["recall"] = recall
            overall_metrics["f"] = f1


            # Calculate differences in proportions of 1s for each feature
            feature_differences = {}
            for idx, (i, j) in enumerate(missing_indices):
                if j not in feature_differences:
                    feature_differences[j] = {"orig": [], "imputed": []}
                feature_differences[j]["orig"].append(orig_rounded[idx])
                feature_differences[j]["imputed"].append(imputed_rounded[idx])


            proportion_diffs = []
            for feature, values in feature_differences.items():
                orig_prop = np.mean(values["orig"])
                imputed_prop = np.mean(values["imputed"])
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


                overall_metrics["largest_diff"] = largest_diff
                overall_metrics["average_diff"] = average_diff
                overall_metrics["smallest_diff"] = smallest_diff


                proportion_diffs_sorted = sorted(proportion_diffs)
                q1 = np.percentile(proportion_diffs_sorted, 25)
                q2 = np.percentile(proportion_diffs_sorted, 50)
                q3 = np.percentile(proportion_diffs_sorted, 75)


                quartile_counts = {
                    f"Q1 ({q1})": sum(diff <= q1 for diff in proportion_diffs),
                    f"Q2 ({q2})": sum(q1 < diff <= q2 for diff in proportion_diffs),
                    f"Q3 ({q3})": sum(q2 < diff <= q3 for diff in proportion_diffs),
                    f"Q4 ({np.max(proportion_diffs)})": sum(
                        diff > q3 for diff in proportion_diffs)
                }


                overall_metrics[f"Q1 ({q1})"] = quartile_counts[f"Q1 ({q1})"]
                overall_metrics[f"Q2 ({q2})"] = quartile_counts[f"Q2 ({q2})"]
                overall_metrics[f"Q3 ({q3})"] = quartile_counts[f"Q3 ({q3})"]
                overall_metrics[f"Q4 ({np.max(proportion_diffs)})"] = \
                quartile_counts[f"Q4 ({np.max(proportion_diffs)})"]


        elif self.aggregation == 'A':
            rmse = mean_squared_error(orig, imputed, squared=False)
            mae = mean_absolute_error(orig, imputed)


            overall_metrics["rmse"] = rmse
            overall_metrics["mae"] = mae


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
                            "accuracy": accuracy,
                            "classification_error": classification_error,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1
                        }
                except Exception as e:
                    logging.error(f"{e} for feature type {feature_type}")
                    sys.exit(1)


            else:
                try:
                    if orig and imputed:  # Check if there are values to evaluate
                        rmse = mean_squared_error(orig, imputed, squared=False)
                        mae = mean_absolute_error(orig, imputed)


                        feat_metrics[feature_type] = {
                            "rmse": rmse,
                            "mae": mae
                        }
                except Exception as e:
                    logging.error(f"{e} for feature type {feature_type}")
                    sys.exit(1)
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
                strategy (str): The imputation strategy to use ("knn", "softimpute").
                feature_types (dict): A dictionary categorizing features by type.
                hyperparameter (int or float): The hyperparameter value for the imputer.
                eval_metric (str): The evaluation metric to use ("f1", "rmse", etc.).


            Returns:
                tuple: A tuple containing the imputed dataset, the hyperparameter used, and the evaluation metric value.
        """
        logging.info("starting impute_wrt_hyperparameter")
        # SystemExit("Exiting")
        imputer = None
        with contexttimer.Timer() as t:
            if strategy == "knn":
                imputer = KNNImputer(n_neighbors=hyperparameter, keep_empty_features=True)
                _ = imputer.fit_transform(X_train)
                X_test_imputed = imputer.transform(X_test_missing)
            elif strategy == "softimpute":
                imputer = SoftImpute(shrinkage_value=hyperparameter, max_iters=400,
                                    max_value=1, min_value=0,
                                    init_fill_method="mean")
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


            The function splits the training data into k folds, imputes missing values using the specified strategy
            (e.g., KNN or SoftImpute) with a given hyperparameter, and evaluates the imputation using the specified evaluation metric.


            Args:
                X_train (np.ndarray): Training data.
                strategy (str): Imputation strategy ("knn" or "softimpute").
                feature_types (dict): Dictionary categorizing features by type.
                hyperparameter (int or float): Hyperparameter for the imputation strategy (e.g., number of neighbors for KNN).
                eval_metric (str): Metric to evaluate imputation quality ("accuracy", "precision", "recall", "f1", "rmse", "mae").
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
            if strategy == "knn":
                imputer = KNNImputer(n_neighbors=hyperparameter, keep_empty_features=True)
                _ = imputer.fit_transform(X_train_fold)
                X_val_imputed = imputer.transform(X_val_missing)
            elif strategy == "softimpute":
                imputer = SoftImpute(shrinkage_value=hyperparameter, max_iters=400, max_value=1, min_value=0, init_fill_method="mean")
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
                          eval_metric="f1", n_splits=10):
        logging.info("Starting parallel processing for hyperparameter selection")
        results = Parallel(n_jobs=-1)(delayed(self._cross_val_impute_wrt_hyperparameter)(X_train, strategy, feature_types, hyperparameter, eval_metric, n_splits, missing_rate) for hyperparameter in hyperparameter_range)
        """
            Selects the best hyperparameter for an imputation strategy using k-fold cross-validation.


            The function evaluates different hyperparameters, using cross-validation, and selects the one that optimizes the specified evaluation metric.


            Args:
                X_train (np.ndarray): Training data.
                X_test (np.ndarray): Test data (not used directly for hyperparameter selection).
                strategy (str): Imputation strategy ("knn" or "softimpute").
                feature_types (dict): Dictionary categorizing features by type.
                missing_rate (float, optional): Proportion of data to artificially make missing for evaluation. Defaults to 0.2.
                hyperparameter_range (range, optional): Range of hyperparameters to evaluate. Defaults to range(3, 21, 3).
                eval_metric (str, optional): Metric to evaluate imputation quality ("f1", "accuracy", "precision", "recall", "rmse", or "mae").
                Defaults to "f1".
                n_splits (int, optional): Number of folds for cross-validation. Defaults to 10.


            Returns:
                int or float: The best hyperparameter for the specified strategy based on the evaluation metric.
        """
        logging.info("Completed parallel processing for hyperparameter selection")
        if eval_metric in ["accuracy", "precision", "recall", "f1"]:
            best_hyperparameter = max(results, key=lambda x: x[2])[1]
        elif eval_metric in ["rmse", "mae"]:
            best_hyperparameter = min(results, key=lambda x: x[2])[1]


        logging.info(f"Best hyperparameter for {strategy} is {best_hyperparameter}")
        return best_hyperparameter


    def _choose_hyperparameter(self, X_train, X_test, strategy,
                          feature_types, missing_rate=0.2,
                          hyperparameter_range=range(3, 21, 3),
                          eval_metric="f1"):
        """
            Selects the best hyperparameter for a specified imputation strategy using the given evaluation metric.


            Args:
                X_train (np.ndarray): The training dataset.
                X_test (np.ndarray): The test dataset.
                strategy (str): The imputation strategy ("knn", "softimpute", etc.).
                feature_types (dict): A dictionary categorizing features by type.
                missing_rate (float): The rate of missing values to simulate for testing.
                hyperparameter_range (range): The range of hyperparameters to test.
                eval_metric (str): The evaluation metric to use ("f1", "rmse", etc.).


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


        if eval_metric in ["accuracy", "precision", "recall", "f1"]:
            best_hyperparameter = max(results, key=lambda x: x[2])[1]
        elif eval_metric in ["rmse", "mae"]:
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
                strategy (str): The imputation strategy ("knn", "softimpute", "mean", etc.).
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
            file_path_to_save_npz == os.path.join(self.cur_dir, "database")
        if strategy == "knn":
            imputer = imputer_class(n_neighbors=hyperparameter, keep_empty_features=True)
        elif strategy == "softimpute":
            imputer = imputer_class(max_iters=400, max_value=1, min_value=0,
                                    shrinkage_value=hyperparameter)
        elif strategy in ["mean", "most_frequent"]:
            imputer = imputer_class(strategy=strategy, keep_empty_features=True)
        elif strategy == "constant_0":
            imputer = imputer_class(strategy="constant", fill_value=0, keep_empty_features=True)
        elif strategy == "constant_1":
            imputer = imputer_class(strategy="constant", fill_value=1, keep_empty_features=True)
        elif strategy == "midas":
            import MIDASpy as md
            X_true = X.values
            curr_date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join("midas_outputs", f"run_{curr_date_time}")
            os.makedirs(file_path, exist_ok=True)
            if X_missing is None:
                with contexttimer.Timer() as t:
                    # 256
                    file_path = os.path.join("midas_outputs", f"{curr_date_time}", "tmp")
                    imputer = md.Midas(layer_structure=[256, 256, 256], vae_layer=False, seed=0, input_drop=0.8, savepath=file_path)
                    imputer.build_model(X, binary_columns=midas_bin_vars)
                    imputer.train_model(training_epochs=50)
            else:
                with contexttimer.Timer() as t:
                    # 256
                    file_path = os.path.join("midas_outputs", f"{curr_date_time}", "tmp")
                    imputer = md.Midas(layer_structure=[256, 256, 256], vae_layer=False, seed=0, input_drop=0.8, savepath=file_path)
                    imputer.build_model(X_missing, binary_columns=midas_bin_vars)
                    imputer.train_model(training_epochs=50)
            logging.info("Time taken to train the model: ", t.elapsed)
            num_samples = 5
            imputations = imputer.generate_samples(m=num_samples).output_list
            for i in range(num_samples):
                if self.cache:
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
            if self.cache:
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
                if self.cache:
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
                strategy (str): The imputation strategy ("knn", "softimpute", etc.).
                feature_types (dict): A dictionary categorizing features by type.
                hyperparameter_range (range): The range of hyperparameters to test.
                eval_metric (str): The evaluation metric to use ("f1", "rmse", etc.).
                test_quality (bool, optional): Whether to evaluate the quality of imputation. Default is True.
                file_path_to_save_npz (str, optional): Path to save the imputed data in NPZ format. Default is None.


            Returns:
                np.ndarray: The imputed dataset.
        """
        if file_path_to_save_npz == None:
            file_path_to_save_npz = os.path.join(self.cur_dir, "database")
        X_train, X_test = train_test_split(X, test_size=0.25, random_state=0)
        logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


        best_hyperparameter = self._choose_hyperparameter_cv(X_train, X_test,
                                                    strategy=strategy,
                                                    feature_types=feature_types,
                                                    missing_rate=0.2,
                                                    hyperparameter_range=hyperparameter_range,
                                                    eval_metric=eval_metric)


        imputer_class = KNNImputer if strategy == "knn" else SoftImpute
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
        if ["imputed"] not in self.sources[1] and self.cache:
            updated_path = os.path.join(file_path_to_save_npz, "non_imputed_data")
            # check if the directory exists
            if not os.path.exists(updated_path):
                os.makedirs(updated_path)
            np.savez(os.path.join(updated_path, self.files[1]), data=self.data[1], feats=self.feats[1], langs=self.langs[1], sources=self.sources[1])


        aggregate_data = self.aggregate()


        if self.aggregation == 'U':
            combined_df_u_no_ffg = pd.DataFrame(aggregate_data.squeeze(), columns=self.feats[1])
            combined_df_u_no_ffg.insert(0, "language", self.langs[1])
            return combined_df_u_no_ffg
        else:
            combined_df_a_no_ffg = pd.DataFrame(aggregate_data.squeeze(), columns=self.feats[1])
            combined_df_a_no_ffg.insert(0, "language", self.langs[1])
            return combined_df_a_no_ffg


    def imputation_interface(self, csv_path=None, strategy="softimpute", file_path_to_save_npz=None,
                         feature_prefixes=("S_", "P_", "INV_", "M_"),
                         eval_metric="f1", hyperparameter_range=None,
                         test_quality=True, return_csv=True, save_as_npz=True):
        """
            Interface for imputing missing values in URIEL+ using different imputation strategies.


            Args:
                csv_path (str, optional): Path to the CSV file to load.
                Default is None.
                strategy (str, optional): The imputation strategy to use ("mean", "knn", "softimpute", "midas".
                Default is "softimpute".
                file_path_to_save_npz (str, optional): Path to save the imputed data in NPZ format.
                Default is None.
                feature_prefixes (tuple, optional): Prefixes to categorize features.
                Default is ("S_", "P_", "INV_", "M_").
                eval_metric (str, optional): The evaluation metric to use ("f1", "rmse", etc.).
                Default is "f1".
                hyperparameter_range (range, optional): The range of hyperparameters to test for strategies like "knn" and "softimpute".
                Default is None.
                test_quality (bool, optional): Whether to evaluate the quality of imputation.
                Default is True.
                return_csv (bool, optional): Whether to return the imputed data as a CSV.
                Default is True.
                save_as_npz (bool, optional): Whether to save the imputed data as an NPZ file.
                Default is True.


            Returns:
                pd.DataFrame: The imputed data frame if return_csv is True; otherwise, None.
        """
        logging.info("Starting imputation_interface")


        if file_path_to_save_npz == None:
            file_path_to_save_npz = os.path.join(self.cur_dir, "database")


        if csv_path is None:
            combined_df_u = self._make_csv(file_path_to_save_npz)
        else:
            combined_df_u = pd.read_csv(csv_path)
            updated_path = os.path.join(file_path_to_save_npz, "features.npz")
            f = np.load(updated_path, allow_pickle=True)
            f = dict(f)
            f_sources = f["sources"]
            if ["imputed"] not in f_sources and self.cache:
                updated_path = os.path.join(file_path_to_save_npz, "non_imputed_data")
                np.savez(os.path.join(updated_path, "features.npz"), data=f["data"], feats=f["feats"], langs=f["langs"], sources=f_sources)


        old_combined_df_u = combined_df_u.copy()
        combined_df_u = combined_df_u.drop(combined_df_u.columns[0], axis=1)


        X, feature_types = self._preprocess_data(combined_df_u, feature_prefixes)
        if strategy in ["knn", "softimpute"] and hyperparameter_range is not None:
            imputed = self._hyperparameter_imputation(X=X, strategy=strategy,
                                                feature_types=feature_types,
                                                hyperparameter_range=hyperparameter_range,
                                                eval_metric=eval_metric,
                                                test_quality=test_quality,
                                                file_path_to_save_npz=file_path_to_save_npz)
        elif strategy == "midas":
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
                imputer_class = SoftImpute if strategy == "softimpute" else SimpleImputer
                imputed = self._standard_impute(X=X, imputer_class=imputer_class,
                                        strategy=strategy, X_missing=X_missing,
                                        feature_types=feature_types,
                                        missing_indices=missing_indices,
                                        file_path_to_save_npz=file_path_to_save_npz)
            else:
                imputer_class = SoftImpute if strategy == "softimpute" else SimpleImputer
                imputed = self._standard_impute(X=X, imputer_class=imputer_class,
                                        strategy=strategy,
                                        feature_types=feature_types,
                                        file_path_to_save_npz=file_path_to_save_npz)


        if self.aggregation == 'U':
            imputed = np.round(imputed)


        if save_as_npz:
            df = pd.DataFrame(imputed, columns=combined_df_u.columns)
            df = df.fillna(-1)
            data = df.to_numpy()
            feats = df.columns
            reshaped_data = data.reshape(data.shape[0], data.shape[1], 1)
            f_sources_new = np.array(["imputed"])
            #IMPORTANT
            file_path = os.path.join(file_path_to_save_npz, self.files[1])


            self.data[1] = reshaped_data


            if self.cache:
                np.savez(file_path,
                        data=reshaped_data, feats=feats,
                        langs=old_combined_df_u["language"], sources=f_sources_new)


        if return_csv:
            completed_df = pd.DataFrame(imputed, columns=combined_df_u.columns)
            completed_df.insert(0, "language", old_combined_df_u["language"])


            return completed_df


    """
        Imputes missing values in URIEL+ with a specific imputation strategy.
    """
    def midaspy_imputation(self):
        """Imputes missing values in URIEL+ data using MIDASpy imputation."""
        if self.aggregation == 'U':
            eval_metric = "f1"
        else:
            eval_metric = "rmse"
        _ = self.imputation_interface(strategy="midas", save_as_npz=False, test_quality=True, eval_metric=eval_metric)
        _ = self.imputation_interface(strategy="midas", save_as_npz=True, test_quality=False, eval_metric=eval_metric)


    def knn_imputation(self):
        """Imputes missing values in URIEL+ data using k-nearest-neighbour imputation."""
        if self.aggregation == 'U':
            eval_metric = "f1"
        else:
            eval_metric = "rmse"
        _ = self.imputation_interface(strategy="knn", save_as_npz=True, test_quality=True, eval_metric=eval_metric, hyperparameter_range=(3, 6, 9, 12, 15))




    def softimpute_imputation(self):
        """Imputes missing values in URIEL+ data using softImpute imputation."""
        if self.aggregation == 'U':
            eval_metric = "f1"
        else:
            eval_metric = "rmse"
        _ = self.imputation_interface(strategy="softimpute", save_as_npz=False, test_quality=True, eval_metric=eval_metric)
        _ = self.imputation_interface(strategy="softimpute", save_as_npz=True, test_quality=False, eval_metric=eval_metric)


    def mean_imputation(self):
        """Imputes missing values in URIEL+ data using mean imputation."""
        if self.aggregation == 'U':
            eval_metric = "f1"
        else:
            eval_metric = "rmse"
        _ = self.imputation_interface(strategy="mean", save_as_npz=False, test_quality=True, eval_metric=eval_metric)
        _ = self.imputation_interface(strategy="mean", save_as_npz=True, test_quality=False, eval_metric=eval_metric)
