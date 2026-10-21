"""Launch a vendored (third-party, unmodified) MCP server file with `from __future__ import annotations`
forced on, so it also runs on Python 3.9. Usage: python run_vendored.py <path/to/server.py>"""
import __future__
import os
import sys

path = os.path.abspath(sys.argv[1])
sys.argv = [path]
sys.path.insert(0, os.path.dirname(path))
code = compile(open(path).read(), path, "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
exec(code, {"__name__": "__main__", "__file__": path})
