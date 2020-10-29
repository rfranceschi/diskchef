class CHEFError(Exception):
    """Base class for diskchef exceptions"""
    pass


class CHEFNotImplementedError(CHEFError, NotImplementedError):
    """Exception raised by base classes methods which only define interface for child classes"""
    pass


class CHEFSlowDownWarning(UserWarning):
    """Warning issued by pieces of code which are known to be slow, if a possible alternative exists"""
    pass
