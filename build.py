from pybuilder.core import use_plugin, init, task

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.install_dependencies")
use_plugin("python.distutils")


name = "MachineIntel-HW1"
default_task = "publish"


@init
def set_properties(project):
    pass

@init
def initialize(project):
    project.build_depends_on('pybrain')
    project.build_depends_on('numpy')
    project.build_depends_on('matplotlib')
