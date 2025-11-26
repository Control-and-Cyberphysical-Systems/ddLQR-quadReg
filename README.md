# ddLQR-quadReg
Code for generating figures in the paper 'On the effect of quadratic regularization in direct data-driven LQR' Demonstrates effects of the three regularization terms identified in the paper.

*Dependencies:* [CVX](https://cvxr.com/cvx/doc/intro.html) is required for solving SDPs.

## License
The source code in this repository is shared under the MIT license. See the [LICENSE](./LICENSE) file.

## Citing
Citing information will be added once the paper is published.

## How to use
Run `GenerateFigures.m`. Doing so will use the same data samples as in the paper without re-computing the results for Figure 1 and 2.
To re-compute these results, set 'compute_figure1' and 'compute_figure2' to 'true'. Warning: This might take 15+ minutes, depending on your system.
To generate similar figures with new datasets, set 'randomize_experiment' to 'true'. The flags 'compute_figure1' and 'compute_figure2' are automatically overwritten as 'true', in this case.
