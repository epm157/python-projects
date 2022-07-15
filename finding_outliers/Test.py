import numpy as np


dataset= [11,10,12,14,12,15,14,13,15,102,12,14,17,19,107, 10,13,12,14,12,108,12,11,14,13,15,10,15,12,10,14,13,15,10]



def detect_outliers_z_score(data):
    outliers = []
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)

    for i in data:
        z_score = (i - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)

    return outliers

outls = detect_outliers_z_score(dataset)
print(outls)





def detect_outliers_z_percentile(data):
    outliers = []
    #print(sorted(dataset))

    quantile1, quantile3 = np.percentile(dataset, [25, 75])
    #print(quantile1, quantile3)

    iqr_value = quantile3 - quantile1
    #print(iqr_value)

    lower_bound_value = quantile1 - (1.5 * iqr_value)
    upper_bound_value = quantile3 + (1.5 * iqr_value)
    #print(lower_bound_value)
    #print(upper_bound_value)
    for i in data:
        if i < lower_bound_value or i > upper_bound_value:
            outliers.append(i)

    return outliers

outls = detect_outliers_z_percentile(dataset)
print(outls)














