class CHEFError(Exception):
    """Base class for diskchef exceptions"""


class CHEFNotImplementedError(CHEFError, NotImplementedError):
    """Exception raised by base classes methods which only define interface for child classes"""


class CHEFValueError(CHEFError, ValueError):
    """Exception raised when the value of argument to the function is wrong"""


class CHEFTypeError(CHEFError, TypeError):
    """Exception raised when the type of argument is wrong"""


class CHEFRuntimeError(CHEFError, RuntimeError):
    """Exception raised in other cases"""


class CHEFWarning(UserWarning):
    """Base class for diskchef warningns"""


class RADMCWarning(CHEFWarning):
    """Base class for radmc warningns"""


class CHEFValueWarning(CHEFWarning):
    """Warning issued when the values are not properly set, but it is not critical"""


class CHEFSlowDownWarning(CHEFWarning, ResourceWarning):
    """Warning issued by pieces of code which are known to be slow, if a possible alternative exists"""
