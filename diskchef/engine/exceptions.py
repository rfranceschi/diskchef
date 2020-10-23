class CHEFError(Exception):
    pass


class CHEFNotImplementedError(CHEFError, NotImplementedError):
    pass


class CHEFSlowDownWarning(UserWarning):
    pass