from tqdm.auto import tqdm
from dataset import load_data
from dataset import all_datasets
from dataset import ds_dimensions

import time

total_iterations = 0
num_iters = 5

for ds in all_datasets:
    total_iterations += (ds_dimensions[ds][1] - 1) * num_iters

print("Varying Dim", all_datasets)
pbar = tqdm(total=total_iterations, desc="Overall Progress", position=0)
for ds in all_datasets:
    x, y = load_data(ds)
    pbar_inner = tqdm(total=(x.shape[1] - 1) * num_iters, desc=ds, leave=False)
    for i in range(num_iters):
        for n in range(2, x.shape[1] + 1):
            time.sleep(0.05)
            pbar.update()
            pbar_inner.update()
    pbar_inner.close()
pbar.close()
print("done")
