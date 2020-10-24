import pandas as pd
import glob
import joblib
from tqdm import tqdm


if __name__ == "__main__":
    files  = glob.glob('../input/train*.parquet')
    for f in files:
        df = pd.read_parquet(f)
        image_ids = df.loc[:, 'image_id'].values
        df = df.drop('image_id', axis=1)
        image_arrays= df.values
        for j, img_id in tqdm(enumerate(image_ids), total = len(image_ids)):
            joblib.dump(image_arrays[j, :], '/root/input/image_pickles/{}.pkl'.format(img_id))
