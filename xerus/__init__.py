import os
DIR = os.path.dirname(os.path.realpath(__file__))
import ctypes; ctypes.cdll.LoadLibrary(os.path.join(DIR, "libxerus_misc.so"))
import ctypes; ctypes.cdll.LoadLibrary(os.path.join(DIR, "libxerus.so"))
from xerus.xerus import *
