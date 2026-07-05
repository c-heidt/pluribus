import glob
import os
import re
import setuptools
from typing import List


def get_version() -> str:
    """Read ``__version__`` from ``poker_ai/__init__.py`` without importing it.

    setup.py must **not** ``import poker_ai``: under PEP 517 build isolation the
    build environment holds only the build requirements (setuptools, Cython,
    numpy), so importing the package — which pulls in runtime deps like
    ``rich`` — fails and breaks ``pip install``.  Parsing the version string
    out of the source keeps the build self-contained.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "poker_ai", "__init__.py")) as stream:
        source = stream.read()
    match = re.search(
        r"""^__version__\s*=\s*['"]([^'"]+)['"]""", source, re.MULTILINE
    )
    if not match:
        raise RuntimeError("Unable to find __version__ in poker_ai/__init__.py")
    return match.group(1)


def get_scripts_from_bin() -> List[str]:
    """Get all local scripts from bin so they are included in the package."""
    return glob.glob("bin/*")


def get_ext_modules() -> list:
    """Cythonize the optional compiled core (``poker_ai/_core/*.pyx``).

    Returns an empty list — a pure-Python install with no native extension —
    when the core should be skipped: ``POKER_AI_NO_EXT`` is set, Cython is not
    importable, or there are no ``.pyx`` sources.  Every path the core
    accelerates has a Python fallback (see ``poker_ai/_core/__init__.py``), so a
    build without the extension is fully functional.

    Kept deliberately simple: one ``Extension`` per ``.pyx`` with numpy's
    headers on the include path (harmless for the memoryview-only probe, needed
    once Phase-1 kernels use the numpy C-API).  Per-extension link settings
    (e.g. libxxhash for the info-set hash in Phase 1) are added at that point.
    """
    if os.environ.get("POKER_AI_NO_EXT"):
        return []
    try:
        from Cython.Build import cythonize
    except ImportError:
        return []
    import numpy as np

    # The core package dir carries vendored headers (``xxhash.h`` +
    # ``_xxh3.h``) that a kernel includes with quotes; add it to the include
    # path so the include resolves regardless of the build's working directory
    # (quote-include-relative-to-source already covers the in-tree build, this
    # makes it robust for out-of-tree / isolated builds too).
    core_dir = os.path.join("poker_ai", "_core")

    extensions = []
    for path in sorted(glob.glob("poker_ai/_core/*.pyx")):
        module = path[: -len(".pyx")].replace(os.sep, ".")
        extensions.append(
            setuptools.Extension(
                name=module,
                sources=[path],
                include_dirs=[np.get_include(), core_dir],
            )
        )
    if not extensions:
        return []
    return cythonize(extensions, compiler_directives={"language_level": "3"})


def get_package_description() -> str:
    """Returns a description of this package from the markdown files."""
    with open("README.md", "r") as stream:
        return stream.read()


def get_requirements() -> List[str]:
    """Returns all requirements for this package."""
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements


setuptools.setup(
    name="poker_ai",
    version=get_version(),
    author="Leon Fedden, Colin Manko",
    author_email="leonfedden@gmail.com",
    description="Open source implementation of a CFR based poker AI player.",
    long_description=get_package_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/fedden/poker_ai",
    packages=setuptools.find_packages(),
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    scripts=get_scripts_from_bin(),
    ext_modules=get_ext_modules(),
    python_requires=">=3.7",
)
