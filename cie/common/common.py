
def inherit_doc(cls):
    """
    A decorator that makes a class inherit documentation from its parents.
    """
    for name, func in vars(cls).items():
        if name.startswith("_"):
            continue
        if not func.__doc__:
            for parent in cls.__bases__:
                parent_func = getattr(parent, name, None)
                if parent_func and getattr(parent_func, "__doc__", None):
                    func.__doc__ = parent_func.__doc__
                    break
    return cls


def since(version):
    import re
    indent_p = re.compile(r'\n( +)')

    def deco(f):
        indents = indent_p.findall(f.__doc__)
        indent = ' ' * (min(len(m) for m in indents) if indents else 0)
        f.__doc__ = f.__doc__.rstrip() + "\n\n%s.. versionadded:: %s" % (indent, version)
        return f
    return deco