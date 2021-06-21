"""Setup file for the package."""
import os
import sys
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Helper function for c extensions."""

    def __init__(self, name, sourcedir=""):
        """Init."""
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Build extension using cmake."""

    def run(self):
        """Helper cmake function."""
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        """Build ext."""
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]
        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        build_args += ['--', '-j2']

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )
        print()  # Add an empty line for cleaner output


# Declare your non-python data files:
# Files underneath configuration/ will be copied into the build preserving the
# subdirectory structure if they exist.
data_files = []
for root, dirs, files in os.walk("configuration"):
    data_files.append(
        (os.path.relpath(root, "configuration"), [os.path.join(root, f) for f in files])
    )

setup(
    name="eXtremeContextualBandits",
    version="1.0",
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    data_files=data_files,
    root_script_source_version="default-only",
    test_command="pytest",
    check_format=False,
    test_mypy=False,
    test_flake8=False,
    ext_modules=[CMakeExtension("xcb/core")],
    # add custom build_ext command
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
