# flake8: noqa

from catalyst.registry import Registry

#from .runner import CustomSupervisedConfigRunner
#from .model import SimpleNet
import runner
import model

Registry(runner.CustomSupervisedConfigRunner)
Registry(model.SimpleNet)
