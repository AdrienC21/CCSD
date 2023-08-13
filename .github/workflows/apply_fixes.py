#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""apply_fixes.py: apply fixes to the installed packages.
Used in the GitHub Actions workflow.
"""

import os
import shutil
import site

packages = site.getsitepackages()
site_packages = None
for p in packages:
    if "site-packages" in p:
        site_packages = p
        break

# Path of the fixes (to be copied)
molsets_filepath = os.path.join("./", "fixes", "utils.py")
toponetx_filepath = os.path.join("./", "fixes", "combinatorial_complex.py")
# Path of the destination (where to copy the fixes)
molsets_path = os.path.join(site_packages, "moses", "metrics", "utils.py")
toponetx_path = os.path.join(
    site_packages, "toponetx", "classes", "combinatorial_complex.py"
)

# Make the copies
print("Fixing MOSES ... (utils.py)")
shutil.copyfile(molsets_filepath, molsets_path)
print("Fixing TopoNetX ... (combinatorial_complex.py)")
shutil.copyfile(toponetx_filepath, toponetx_path)
