def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import pathlib
    import sys, pkg_resources, importlib.util, os
    topdir = pathlib.Path(__file__).parent.parent

    __file__ = os.path.join(str(topdir.absolute()),
                            'TLRMVMpy.cpython-39-x86_64-linux-gnu.so')
    print("loading shared lib from ",__file__)
    __loader__ = None; del __bootstrap__, __loader__
    spec = importlib.util.spec_from_file_location("TLRMVMpy", __file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
__bootstrap__()

from ._wrapper import *
# from .tlrmvmtools import *
# from .tlrmat import *

