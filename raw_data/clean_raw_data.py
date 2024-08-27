import pathlib as pl
import shutil as sh

# Used for reducing the space taken up by files that don't have matching csvs or tiffs

# gets all the csvs and tiffs
csvs = [x.split()[0] for x in (pl.Path("./raw_data") / "csvs").glob("*.csv")]
tiffs = [x.split()[0] for x in (pl.Path("./raw_data") / "tiffs").glob("*.tiff")]

# gets the names of the csvs and tiffs without the extensions
clean_csv_names = (str(x.resolve()).split("\\")[-1].replace(".csv", "") for x in csvs)
clean_img_names = (str(x.resolve()).split("\\")[-1].replace(".tiff", "") for x in tiffs)
# gets all the common names between the csvs and tiffs
common = set(clean_csv_names).intersection(set(clean_img_names))

# gets the names of the csvs and tiffs that are not present in the other
not_present_csvs = set(csvs) - common
not_present_tiffs = set(tiffs) - common

# deletes the csvs and tiffs that don't have a matching file
for csv in not_present_csvs:
    csv_path = pl.Path("./raw_data/csvs") / (csv + ".csv")
    csv_path.unlink()

for tiff in not_present_tiffs:
    tiff_path = pl.Path("./raw_data/tiffs") / (tiff + ".tiff")
    tiff_path.unlink()
