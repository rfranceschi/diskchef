class CHEFError(Exception):
    pass


class CHEFNotImplemented(CHEFError, NotImplementedError):
    pass
