import warnings
from functools import wraps


def ignore_warnings(msgs):
    def dec(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                for msg in msgs:
                    warnings.filterwarnings('ignore', message=msg)
                return f(*args, **kwargs)
        return wrapper
    return dec


class CachedProps:
    def free_cache(self, props):
        if isinstance(props, str):
            props = [props]
        for prop in props:
            if prop in self.__dict__:
                del self.__dict__[prop]
