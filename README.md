# Coronagraphs

This package allows the user to design and simulate Lyot-style coronagraphs. 
For design optimization, the code defines a linear program model and uses a 
solver to find the best solution of the problem. Two different solvers can be used to 
solve the problem:
- [scipy.linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)
- [gurobi](http://www.gurobi.com)


### Prerequisites
- Gurobi solver for python: see instructions for installation and license at 
[gurobipy](http://www.gurobi.com/documentation/7.5/quickstart_mac/the_gurobi_python_interfac.html)

## Authors

* **Mamadou N'Diaye** - *Initial work* - [Astromam](https://github.com/astromam)

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
funding the Optimal Optical Coronagraph workshop held September 25-29, 2017 at the Lorentz
 Center in Leiden, the Netherlands. Also, all the participants of the workshop for very
 engaging discussions.
