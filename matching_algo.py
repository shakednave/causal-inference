import logging
import multiprocessing
import os
import platform
import sys
import time
import timeit
from functools import partial

import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from tqdm import tqdm


"""
Uplift Matching Algorithm
-------------------------
The code performs fast Mahalanobis distance-based matching (see below) between treatment and control groups.
It selects relevant features using the MRMR algo, calculates the covariance matrix, 
and parallelizes the matching process. The hyperparameters, such as batch size and number of features, are pre-defined. 

Fast Mahalanobis Distance Calculation using Cholesky Decomposition
------------------------------------------------------------------------
The Mahalanobis Distance can be efficiently computed by leveraging Cholesky decomposition.
This approach reduces the computation to squared Euclidean distances after a transformation.
This is particularly advantageous when the dimensionality (p) is much smaller than the number 
of samples in X (m) and Y (n).

1. Cholesky Decomposition:
   - Given the covariance matrix C, perform Cholesky decomposition to obtain R^t * R,
     where R is an upper-triangular matrix.
   - This operation takes O(p^3) time.

2. Inverse of R:
   - Invert the triangular matrix R.
   - This inversion also takes O(p^3) time.

3. Transformation:
   - Multiply both the data matrices X and Y by the inverse of R on the right.
   - This step takes O(p^2 * (m+n)) time.
     TX_i = X_i * inv(R)
     TY_j = Y_j * inv(R)

4. Euclidean Distances:
   - Compute the squared Euclidean distances between pairs of transformed points TX_i and TY_j, i.e. ||TX_i - TY_j||^2.
   - This step takes O(m * n * p) time.
   
5. Proof:
    - Let X be an m x p matrix and Y be an n x p matrix.
    - Let C be the p x p covariance matrix.
    - Since C is a PSD matrix, it can be decomposed into C = L*L^t, where L is an upper-triangular matrix. 
        - Proof: https://statproofbook.github.io/P/covmat-psd.html#:~:text=Theorem%3A%20Each%20covariance%20matrix%20is,alla%E2%88%88Rn.
    - Define Z = L^-1 * X, where x is a vector in R^p. Then L*Z = X.
    - Define W = L^-1 * Y, where y is a vector in R^p. Then L*W = Y.
    - The squared Mahalanobis distance between x and y is defined as:
        d^2(x, y) = (x - y)^t * C^-1 * (x - y)
                    = (x - y)^t * L^-t * L^-1 * (x - y)
                    = (L^-1 * x - L^-1 * y)^t * (L^-1 * x - L^-1 * y)
                    = ||L^-1 * x - L^-1 * y||^2
                    = ||Z - W||^2

Total time complexity is O(m * n * p + (m + n) * p^2 + p^3), which is more efficient than
the original O(m * n * p^2) computation where 1 << p << m, n (which is the case in our problem).
"""

def select_features(treatment_df, label_col, n_features):
    logging.info("Selecting features")
    numeric_columns = [clm for clm in treatment_df.select_dtypes(include=[np.number]).columns if clm != label_col]
    treatment_df[numeric_columns] = treatment_df[numeric_columns].astype(float)
    selected_features = mrmr_classif(treatment_df[numeric_columns],
                                     treatment_df[label_col],
                                     K=n_features, n_jobs=-1)
    assert len(selected_features) > 0, "No features were selected, check for positive labels in the data"
    assert len(selected_features) == n_features, "Number of features is not equal to K, check the input data"
    logging.info(f"Selected features: {selected_features}")
    return selected_features


def calculate_covariance_matrix(treatment_df, features):
    logging.info("Calculating covariance matrix")
    cov_matrix = np.cov(treatment_df[features].values.T)
    chol_cov_matrix = np.linalg.cholesky(cov_matrix)
    inv_chol_cov_matrix = np.linalg.inv(chol_cov_matrix)
    return inv_chol_cov_matrix


def mahalanobis_distance(x, y, inv_cov):
    x, y = x @ inv_cov, y @ inv_cov
    return np.linalg.norm(x[:, np.newaxis, :] - y, axis=2)


def find_closest_control(batch_start, batch_end, treatment_df, control_df, inv_cov, n_controls):
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    treatment_batch = treatment_df.iloc[batch_start:batch_end, :]
    sample_idxs = [np.random.randint(0, len(control_df)) for _ in range(n_controls*len(treatment_batch))]
    distances = mahalanobis_distance(treatment_batch.values.astype(np.float).reshape(-1, len(treatment_df.columns)),
                                     control_df.iloc[sample_idxs, :].values.astype(np.float).reshape(-1, len(treatment_df.columns)),
                                     inv_cov)
    closest_indices = [sample_idxs[i] for i in np.argmin(distances, axis=1)]
    return closest_indices


def run_matching(treatment_df, control_df, label_col, n_controls, n_features, n_members_batch):
    # Subset the data to only the features we need
    features = select_features(treatment_df, label_col, n_features)
    treatment_subset = treatment_df[features]
    control_subset = control_df[features]

    # Calculate covariance matrix
    inv_cov = calculate_covariance_matrix(treatment_df, features)

    # handle batches
    n_batches = int(np.ceil(len(treatment_subset) / n_members_batch))
    batch_start_idxs = [i * n_members_batch for i in range(n_batches)]
    batch_end_idxs = [(i + 1) * n_members_batch if (i + 1) * n_members_batch < len(treatment_subset) else len(treatment_subset) for i in range(n_batches)]
    logging.info(f"Number of batches: {n_batches}")

    # Use multiprocessing.Pool for parallel processing
    logging.info("Running matching..")
    with multiprocessing.Pool() as pool:
        # Use partial to fix the extra arguments for each call
        worker_partial = partial(find_closest_control,
                                 treatment_df=treatment_subset,
                                 control_df=control_subset,
                                 inv_cov=inv_cov,
                                 n_controls=n_controls)

        # Use starmap to pass multiple arguments
        results = list(tqdm(pool.starmap(worker_partial, zip(batch_start_idxs, batch_end_idxs)),
                            total=len(batch_start_idxs)))

    # Unpack the results
    closest_controls = []
    for closest_indices in results:
        closest_controls.extend(closest_indices)

    # Return the closest control members
    closest_control_members = control_df.iloc[closest_controls]
    return closest_control_members



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s     %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout,
                        level='INFO')

    # Get the data
    treatment_df = .. # TODO: load the treatment group data here
    control_df = .. # TODO: load the control group data here
    logging.info("Fetched data successfully")
    logging.info(f"Treatment Size: {treatment_df.shape}")
    logging.info(f"Potential Control Size: {control_df.shape}")

    # Hyperparameters
    label_column = "label"  # Name of the label column
    n_members_batch = 100  # Number of members to process in each batch
    num_controls = 1000  # Number of potential controls nominated to match to each treatment
    num_features = 8  # Number of features to select
    logging.info(f"The algo will select the top {num_features} features, "
                 f"and will match each treatment (there are {len(treatment_df)} treatments) "
                 f"to its closest control out of {num_controls} potential control sample. "
                 f"Each batch will contain {n_members_batch} treatment members.")
    logging.info(f"Be aware of the memory consumption, which is approximately: "
                 f"{8 * n_members_batch * num_controls**2 / 1024**3} GB")

    # Run the matching
    start = timeit.default_timer()
    closest_controls_result = run_matching(treatment_df, control_df, label_column, num_controls, num_features, n_members_batch)
    end = timeit.default_timer()
    logging.info(f"Matching took {end - start} seconds")

    # Add treatment column
    closest_controls_result['treatment'] = 0
    treatment_df['treatment'] = 1

    # Concatenate the results
    matched_df = pd.concat([treatment_df, closest_controls_result], axis=0)

    # Write the results to the filesystem
    matched_df.to_parquet(...) # TODO: replace with the a writing statement

    logging.info(f"{' Great Success! ':-^30}")

