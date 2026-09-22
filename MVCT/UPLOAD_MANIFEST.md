# GitHub upload manifest

Upload the complete contents of this `MVCT` directory, including hidden files
such as `.gitignore`. The ZIP beside this directory is only a transport copy;
do not commit the ZIP inside the repository.

## Included

- Maintained CAT/MVCT implementation and all directly related experiment code
- Automated tests
- Reproduction and experiment documentation
- Figure-generation source code and small source-data CSV
- Dependency and project configuration
- Placeholder documentation for external data and model snapshots

## Excluded from the source repository

| Local content | Reason | Recommended publication route |
| --- | --- | --- |
| `data_all.csv` | Data rights/privacy must be verified | Zenodo/Hugging Face Dataset or documented request process |
| `models/` weights | About 1.2 GB and available upstream | Official model links; Git LFS only if redistribution is allowed |
| `outputs/` checkpoints | About 1.7 GB, generated artifacts | Versioned GitHub Release, Zenodo, or model hub |
| `work/`, audit folders | Manuscript drafts and internal QA state | Keep private or publish a curated supplement separately |
| `.venv-figure/`, caches | Machine-specific/generated | Recreate from dependency files |
| Legacy prototypes | Superseded by `src/*_cat.py` pipeline | Preserve in local archive or a separate history branch |

## Before making the repository public

1. Choose and add a compatible open-source `LICENSE`.
2. Confirm permission to publish the dataset, examples, and trained weights.
3. Replace any author/institution placeholders and add `CITATION.cff`.
4. Create the GitHub repository, then run the commands shown below from this
   directory.

```bash
git init
git add .
git status
git commit -m "Initial public MVCT research release"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```
