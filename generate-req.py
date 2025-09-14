import pkg_resources

# Get all installed packages and their versions
installed_packages = pkg_resources.working_set
packages_list = sorted([f"{i.key}=={i.version}" for i in installed_packages])

# Write to requirements.txt
with open("requirements.txt", "w") as f:
    for pkg in packages_list:
        f.write(pkg + "\n")

print("requirements.txt has been generated with all installed packages.")
