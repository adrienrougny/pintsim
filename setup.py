from distutils.core import setup

setup(name='pintsim',
      version='0.1',
      description='Simulation of pint models',
      author='Adrien Rougny',
      author_email='adrienrougny@gmail.com',
      packages=['pintsim'],
      install_requires = [
        "pypint",
        "pathos"
      ],
     )
