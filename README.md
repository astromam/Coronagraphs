# Coronagraphs
In the context of exoplanet imaging, this object-oriented toolkit in Python 
enables the model and optimization of Lyot-style coronagraphs and Shaped pupil 
devices. The code is developed with two object classes:
- design for coronagraph model
- optim_1d and optim_2d for coronagraph design optimization.

In terms of design optimization, the package defines an optimization problem 
with a linear program (LP) model and calls for a solver to find the best 
solution. The tool currently works with the following solvers:
- [scipy.linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)
- [gurobi](http://www.gurobi.com)

### Prerequisites
- Gurobi solver for Python. For installation and license, see instructions at 
[gurobipy](http://www.gurobi.com/documentation/7.5/quickstart_mac/the_gurobi_python_interfac.html).
- [stdgrb](https://github.com/rflamary/stdgrb) gurobi cython wrapper for dense 
optimization problem.

### Installing


## Deployment

## Built With
- [spyder](https://pythonhosted.org/spyder/installation.html) the used python editor

## Versioning
We use [SemVer](http://semver.org/) for versioning. For the versions available,
see the [tags on this repository](https://github.com/astromam/Coronagraphs/tags).

## Authors

* **Mamadou N'Diaye** - *Initial work* - [astromam](https://github.com/astromam)

See also the list of [contributors](https://github.com/astromam/Coronagraphs/graphs/contributors) 
who participated in this project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for 
details.

## Acknowledgments

* [Rémy Flamary](https://github.com/rflamary) for providing countless guidance in Python 
coding and optimization.
* [Space Telescope Science Institute](http://www.stsci.edu/) collaborators, 
in particular, the Segmented Design and Analysis (SCDA) team.
* the [Lorentz Center](http://www.lorentzcenter.nl/) for hosting and to a large extent 
funding the [Optimal Optical Coronagraph workshop](https://www.lorentzcenter.nl/lc/web/2017/924/info.php3?wsid=924&venue=Snellius) 
held September 25-29, 2017 at the Lorentz Center in Leiden, the Netherlands. Also, all the
 participants for the very engaging discussions during the workshop.
