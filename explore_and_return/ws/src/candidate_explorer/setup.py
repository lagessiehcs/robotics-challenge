from glob import glob

from setuptools import find_packages, setup

package_name = "candidate_explorer"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.xml") + glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="MESA Team",
    maintainer_email="raul.dominguez@aici.de",
    description="Candidate's solution for the explore-and-return challenge.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "explorer_node = candidate_explorer.explorer_node:main",
        ],
    },
)
