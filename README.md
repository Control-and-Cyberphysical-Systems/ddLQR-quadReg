# ddLQR-quadReg
Code for generating figures in the paper 'On the effect of quadratic regularization in direct data-driven LQR', to be presented at the 23rd IFAC World Congress 2026. Demonstrates effects of the three regularization terms identified in the paper.

*Dependencies:* [CVX](https://cvxr.com/cvx/doc/intro.html) is required for solving SDPs.

## License
The source code in this repository is shared under the MIT license. See the [LICENSE](./LICENSE) file.

## Citing
Until the conference proceeding of IFAC WC 2026 are published, please refer to the [arxiv preprint](https://arxiv.org/abs/2604.18453) if you would like to cite this project in scientific publications:
```bibtex
@misc{Klaedtke2026ddLQR_quadReg,
      title={On the Effect of Quadratic Regularization in Direct Data-Driven LQR}, 
      author={Kl\"{a}dtke, Manuel and Zhao, Feiran and D\"{o}rfler, Florian and Schulze Darup, Moritz},
      year={2026},
      eprint={2604.18453},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2604.18453}, 
}
```



## How to use
Run `GenerateFigures.m`. Doing so will use the same data samples as in the paper without re-computing the results for Figure 1 and 2.

To re-compute these results, set `compute_figure1` and `compute_figure2` to `true`. Warning: This might take 15+ minutes, depending on your system.

To generate similar figures with new datasets, set `randomize_experiment` to `true`. The flags `compute_figure1` and `compute_figure2` are automatically overwritten as `true`, in this case.

Note that randomized results may look drastically different to the ones in the paper, if $K_{\mathrm{LS}}$ does not stabilize $(A_{\mathrm{LS}}, B_{\mathrm{LS}})$. For instance, $K$ cannot converge to $K_{\mathrm{LS}}$ for the covariance parametrization, in this case.

To compare computation times of the proposed parameterizations, run `compuationTimes.m`. Warning: This might take 30+ minutes, depending on your system. For a less time-consuming numerical experiment, you can reduce the number of runs `nRuns` (default: 10) over which the computation times are averaged.
