from astropy.table import QTable
from astropy import units as u
import numpy as np
import pandas as pd
import pandasql

if __name__ == '__main__':
    tbl = QTable.read("oeberg2011.txt", format='ascii.no_header')
    tbl = QTable(tbl, masked=True, copy=False)
    colnames = ["V-CO", "F-CO", "V-HCO", "F-HCO", "V-DCO", "F-DCO", "V-N2H", "F-N2H", "V-H2CO43", "F-H2CO43",
                "V-H2CO32", "F-H2CO32",
                "V-HCN", "F-HCN", "V-DCN", "F-DCN", "V-CN23", "F-CN23", "V-CN22", "F-CN22", ]
    tbl["Source"] = [entry["col1"] + " " + entry["col2"] for entry in tbl]
    del tbl["col1"]
    del tbl["col2"]
    for col, colname in zip(
            ["col%d" % i for i in range(3, 23)],
            colnames
    ):
        tbl[col][tbl[col] == -99.99] = np.nan
        tbl.rename_column(col, colname)
        if colname.startswith("F-"):
            tbl[colname].unit = u.Jy
        elif colname.startswith("V-"):
            tbl[colname].unit = u.km / u.s

    print(tbl)

    sources = sorted(set(tbl["Source"]))
    lines = sorted({name.split("-")[1] for name in colnames})
    print(lines)
    print(sources)
    fulldf = tbl.to_pandas()
    for source in sources:
        for line in lines:
            df = pandasql.sqldf(f"SELECT `V-{line}` as vel, `F-{line}` as flux FROM fulldf WHERE Source='{source}';")
            finite = df["flux"].notna() & df["vel"].notna()
            flux = np.trapz(df["flux"][finite] << u.Jy, df["vel"][finite] << (u.km/u.s))
            print(source, line, flux)

