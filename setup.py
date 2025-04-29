from setuptools import find_packages, setup


def read_requirements():
    with open("requirements.txt", "r") as f:
        lines = [l.strip() for l in f.readlines()]
    install_requires = list(filter(None, lines))
    return install_requires


setup(
    name="gap_rl",
    version="0.1.0",
    packages=find_packages(include=["gap_rl", "gap_rl.*"]),
    description="GAP-RL: Grasps As Points for RL Towards Dynamic Object Grasping",
    author="THU-RoboLab",
    python_requires=">=3.8",
    setup_requires=["setuptools>=62.3.0"],
    # install_requires=read_requirements(),
    package_data={
        "gap_rl": [
            "assets/**",
        ],
    },
    exclude_package_data={"": ["*.convex.stl"]},
    extras_require={
        "tests": ["pytest", "black", "isort"],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
            # Copy button for code snippets
            "sphinx_copybutton",
            # Markdown parser
            "myst-parser",
            "sphinx-subfigure",
        ],
    },
)
