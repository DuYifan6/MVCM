# Dataset placement

The dataset is intentionally not committed because its redistribution status
and privacy constraints must be checked by the project owner.

Place the input CSV at `data/data_all.csv`, or set `MVCT_DATA_PATH` to another
location. The required columns are:

| Column | Meaning |
| --- | --- |
| `ID` | Unique sample identifier |
| `title` | News title |
| `text` | Main article text |
| `desc` | Description or summary |
| `label_fake` | Binary fake/real target |
| `label_ai` | Binary AI/human target |

An optional `type` column is used by some reporting scripts. Do not commit
private, licensed, or personally identifying records without permission.
