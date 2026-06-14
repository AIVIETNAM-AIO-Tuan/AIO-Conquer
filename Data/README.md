# Data

Raw, processed, and intermediate datasets for the project.

## Suggested layout

| Subfolder | Purpose |
| --- | --- |
| `raw/` | Original, immutable source data (never edit in place) |
| `processed/` | Cleaned and transformed data ready for modeling |
| `interim/` | Intermediate files produced during processing |
| `external/` | Third-party or reference datasets |

## Notes

- Keep large or sensitive files out of version control; track them with a data versioning tool or external storage.
- Document the source, license, and date for each dataset.
