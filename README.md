# cogsci-2024-likely
Companion repository for the 2024 article "Necessity, Possibility and Likelihood in Syllogistic Reasoning" published in the proceedings of the 46rd Annual Meeting of the Cognitive Science Society.

## Overview

- `data`: Contains the datasets with *possible* and *likely* task types as well as the dataset from Brand & Ragni (2023) providing multiple-choice responses for *Necessary*.
- `helper`: Contains helper files for the analysis.
- `helper/metrics.py`: Implementation of the metrics used in the analysis.
- `helper/syl_solver.py`: Implementation of a solver for syllogisms (supports possible and necessary task types).
- `mreasoner`: Contains the cache files with predictions of mReasoner for all Syllogisms and conclusions (necessary and possible). To obtain the file, the [implementation](https://github.com/nriesterer/cogsci-individualization) provided with the 2020 paper "Do Models Capture Individuals? Evaluating Parameterized Models for Syllogistic Reasoning" by Riesterer, Brand & Ragni was used.
- `mreasoner/necessary_full.npy`: Predictions of mReasoner for all Syllogisms and conclusions (necessary).
- `mreasoner/possible_full.npy`: Predictions of mReasoner for all Syllogisms and conclusions (possible).
- `analysis2024.html`: Static HTML file showing the analysis exported from marimo.
- `analysis2024.py`: Python file containing the marimo notebook with the analysis.

## Open the notebook

The analysis is provided as a [marimo notebook](https://marimo.io/) and needs Python 3 as well as marimo installed to run.

It is recommended to use a Python virtual environment before installing marimo, since it allows to automatically install further requirements.

After installing marimo using a virtual environment, it can be useful to open the notebook in edit mode first. This ensures that marimo can install the missing requirements:
 ```
cd /path/to/repository/
$> marimo edit analysis2024.py
```

Afterwards, the notebook can also be opened by using the following command:

```
cd /path/to/repository/
$> marimo run analysis2024.py
```

## References
Brand, D., Todorovikj, S., & Ragni, M. (2024). Necessity, Possibility and Likelihood in Syllogistic Reasoning. In *Proceedings of the 46rd Annual Meeting of the Cognitive Science Society*.
Brand, D., & Ragni, M. (2023). Effect of response format on syllogistic reasoning. In M. Goldwater, F. K. Anggoro, B. K. Hayes, & D. C. Ong (Eds.), *Proceedings of the 45th Annual Conference of the Cognitive Science Society*

Evans, J., Handley, S., Harper, C., & Johnson-Laird, P. (1999). Reasoning about necessity and possibility: A test of the mental model theory of deduction. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 25, 1495-1513.

Riesterer, N., Brand, D., & Ragni, M. (2020). Do models capture individuals? Evaluating parameterized models for syllogistic reasoning. In S. Denison, M. Mack, Y. Xu, & B. C. Armstrong (Eds.), *Proceedings of the 42nd Annual Conference of the Cognitive Science Society* (pp. 3377–3383). Toronto, ON: Cognitive Science Society
