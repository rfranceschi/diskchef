"""Class CTable(astropy.table.QTable) with additional features for CheF"""

from astropy.table import QTable


class CTable(QTable):
    """Subclass of astropy.table.Qtable that puts the name of a column to the __getitem__ output"""

    def __getitem__(self, item):
        column_quantity = super().__getitem__(item)
        column_quantity.name = item
        return column_quantity
