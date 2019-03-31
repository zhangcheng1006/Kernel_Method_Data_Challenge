# ensembled model by majority vote of 7 spectrum kernel model results
import pandas as pd

files = [5, 6, 7, 8, 9, 10]

models_loc = ["output/klr_dna_pca.csv"]
models_loc = models_loc + ["output/svm_spectrum"+str(i)+".csv" for i in files]

models_combined = pd.DataFrame()
for i in range(len(models_loc)):
    loc = models_loc[i]
    f = pd.read_csv(loc)
    models_combined["Id"] = f["Id"].values
    models_combined["Bound"+str(i)] = f["Bound"].values

cols = list(models_combined.columns)
cols = cols[1:]

models_combined["Bound"] = models_combined[cols].mode(axis=1)[0].astype(int)
models_combined[["Id", "Bound"]].to_csv("output/spectrum_ensemble.csv", index=False)
