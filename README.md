# Coronagraphs

This package allows you to design and simulate the propagation in Lyot-style coronagraphs.
For design optimization, the code defines a linear program model and uses a solver to find
the best solution of the problem. Two different solvers can be used to solve the problem:
- [scipy.linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)
- [gurobi](http://www.gurobi.com)


### Prerequisites
- Gurobi solver for python: see instructions for installation and license at 
[gurobipy](http://www.gurobi.com/documentation/7.5/quickstart_mac/the_gurobi_python_interfac.html)



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for 
details.


