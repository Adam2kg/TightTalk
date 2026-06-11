# runtime_hook_tcl.py
# Fixes Tcl/Tk library paths when running inside a PyInstaller .app bundle
# built from the python.org Python 3.11 installer on macOS.
#
# Without this, tkinter raises:
#   TclError: Can't find a usable init.tcl in the following directories: ...
# because _tkinter.so expects TCL_LIBRARY to point to the bundled tcl8.6 data.

import os
import sys

if hasattr(sys, "_MEIPASS"):
    tcl_dir = os.path.join(sys._MEIPASS, "lib", "tcl8.6")
    tk_dir  = os.path.join(sys._MEIPASS, "lib", "tk8.6")
    if os.path.isdir(tcl_dir):
        os.environ["TCL_LIBRARY"] = tcl_dir
    if os.path.isdir(tk_dir):
        os.environ["TK_LIBRARY"] = tk_dir
