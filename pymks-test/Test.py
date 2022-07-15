
import sys
print(sys.version_info)

import warnings
warnings.filterwarnings('ignore')

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from dask_ml.decomposition import PCA
from dask_ml.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from pymks import (
    generate_multiphase,
    generate_delta,
    plot_microstructures,
    PrimitiveTransformer,
    FlattenTransformer,
    solve_fe,
    TwoPointCorrelation,
    FlattenTransformer,
    LocalizationRegressor
)



da.random.seed(10)
np.random.seed(10)

x_data = da.concatenate([
    generate_multiphase(shape=(1, 101, 101), grain_size=(25, 25), volume_fraction=(0.5, 0.5)),
    generate_multiphase(shape=(1, 101, 101), grain_size=(95, 15), volume_fraction=(0.5, 0.5))
]).persist()



x_data


plot_microstructures(*x_data, cmap='gray', colorbar=False)
plt.show()


correlation_pipeline = Pipeline([
    ("discritize",PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0)),
    ("correlations",TwoPointCorrelation())
])

x_corr = correlation_pipeline.fit(x_data).transform(x_data)

x_corr
#plt.show()

plot_microstructures(
    x_corr[0, :, :, 0],
    x_corr[0, :, :, 1],
    titles=['Correlation [0, 0]', 'Correlation [0, 1]']
)
plt.show()

plot_microstructures(
    x_corr[1, :, :, 0],
    x_corr[1, :, :, 1],
    titles=['Correlation [0, 0]', 'Correlation [0, 1]']
)
plt.show()


def generate_data(n_samples, chunks):
    tmp = [
        generate_multiphase(shape=(n_samples, 21, 21), grain_size=x, volume_fraction=(0.5, 0.5), chunks=chunks, percent_variance=0.15)
        for x in [(20, 2), (2, 20), (8, 8)]
    ]

    x_data = da.concatenate(tmp).persist()

    y_stress = solve_fe(x_data,
                         elastic_modulus=(100, 150),
                         poissons_ratio=(0.3, 0.3),
                         macro_strain=0.001)['stress'][..., 0]

    y_data = da.average(y_stress.reshape(y_stress.shape[0], -1), axis=1).persist()

    return x_data, y_data

np.random.seed(100000)
da.random.seed(100000)

x_train, y_train = generate_data(3, 50)

print("Finished!")










