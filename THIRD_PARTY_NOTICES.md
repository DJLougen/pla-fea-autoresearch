# Third-party data and notices

The MIT License covers this repository's original software and documentation.
It does not replace the licenses of imported or derived research data.

## Included curve data

- Huang and Zhang, “Experimental and numerical data from comparative analysis
  of candidate structural models for active camber morphing wings,” Mendeley
  Data v2, DOI
  [10.17632/8jxw5533py.2](https://doi.org/10.17632/8jxw5533py.2), licensed
  CC BY 4.0. The repository includes literal extracted curve CSVs, a manifest,
  and a data-quality note.
- Al Amri and Al Azhar, “Brick Layering Experiment,” Zenodo, DOI
  [10.5281/zenodo.16935914](https://doi.org/10.5281/zenodo.16935914), licensed
  CC BY 4.0. The repository includes literal extracted curve CSVs, a manifest,
  and a limitations note.

The CSVs retain source identifiers and are not relicensed as MIT. Cite the
original datasets when using them.

## Reported-property tables

Files under `data/reported/` transcribe, digitize, or derive bounded quantities
from sources identified in their adjacent provenance files. Most source
articles state CC BY 4.0; the viscoelastic-wave source states CC BY 3.0.
Individual provenance files and DOI links are authoritative.

## Excluded source material

Downloaded papers, source workbooks, figures, CAD, and author solver decks are
deliberately excluded from Git. Checksum-pinned code and provenance records
identify their sources. Users must obtain those materials under the terms set
by their original publishers or repositories.

CalculiX, Gmsh, MeshIO, NumPy, pandas, SciPy, scikit-image, PyYAML, Pillow,
OpenPyXL, pytest, and other dependencies are separate projects distributed
under their own licenses.
