class CHEFError(Exception):
    """Base class for diskchef exceptions"""
    pass


class CHEFNotImplementedError(CHEFError, NotImplementedError):
    """Exception raised by base classes methods which only define interface for child classes"""
    pass


class CHEFRuntimeError(CHEFError, RuntimeError):
    """Exception raised in other cases"""


class CHEFWarning(UserWarning):
    """Base class for diskchef warningns"""
    pass


class CHEFValueWarning(CHEFWarning):
    """Warning issued when the values are not properly set, but it is not critical"""
    pass


class CHEFSlowDownWarning(CHEFWarning, ResourceWarning):
    """Warning issued by pieces of code which are known to be slow, if a possible alternative exists"""
    pass
