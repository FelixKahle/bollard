# Data - BAP Benchmark Instances

## 📊 Overview

This directory contains benchmark instances for the Berth Allocation Problem (BAP) from the **Kramer–Lalla–Ruiz–Iori–Voss** benchmark suite.

**Source**: [https://github.com/elalla/DBAP](https://github.com/elalla/DBAP)

---

## Instance Sets

The benchmark includes two sets of problem instances:

### f200x15 Series
- **Vessels**: 200
- **Berths**: 15
- **Instances**: 10 (f200x15-01.txt through f200x15-10.txt)

### f250x20 Series
- **Vessels**: 250
- **Berths**: 20
- **Instances**: 10 (f250x20-01.txt through f250x20-10.txt)

---

## File Format

Each instance file follows a structured text format:

```raw
N // number of ships
M // number of berths
ta_1 ... ta_|N| (expected arrival of the vessels)
s_1 ... s_[M] (expected opening time of berths)
h_1_1 ... h_1_|M| (handling time of p_vessel_1_berth_1, p_vessel_1_berth_2, ...)
...
h_|N|_1 ... h_|N|_|M| (processing time of p_vessel_|N|_berth_1, p_vessel_|N|_berth_2, ...)
e_1 ... e_|M| (expected ending time of berths)
t'_1 ... t'_|N| (maximum departure time of vessels)
```

**Note**: A value of `99999` in the processing time matrix indicates that the berth is forbidden for that particular vessel.

---

## Citation

If you use these instances in your research, please cite:

```bibtex
@article{KRAMER2019170,
    title = {Novel formulations and modeling enhancements for the dynamic berth allocation problem},
    journal = {European Journal of Operational Research},
    volume = {278},
    number = {1},
    pages = {170-185},
    year = {2019},
    issn = {0377-2217},
    doi = {https://doi.org/10.1016/j.ejor.2019.03.036},
    url = {https://www.sciencedirect.com/science/article/pii/S0377221719302942},
    author = {Arthur Kramer and Eduardo Lalla-Ruiz and Manuel Iori and Stefan Voß},
    keywords = {OR in maritime industry, Dynamic berth allocation problem, Novel formulations, Modeling enhancements},
    abstract = {This paper addresses the well-known dynamic berth allocation problem (DBAP), which finds numerous applications at container terminals aiming to allocate and schedule incoming container vessels into berthing positions along the quay. Due to its impact on ports’ performance, having efficient DBAP formulations is of great importance, especially for determining optimal schedules in quick time as well as aiding managers and developers in the assessment of solution strategies and approximate approaches. In this work, we propose two novel formulations, a time-indexed formulation and an arc-flow one, to efficiently tackle the DBAP. Additionally, to improve computational performance, we propose problem-based modeling enhancements and a variable-fixing procedure that allows to discard some variables by considering their reduced costs. By means of these contributions, we improve the models’ performance for those instances where the optimal solutions were already known, and we solve to optimality for the first time other instances from the literature.}
}
```

## License

The files in the `bollard/data` are not licensed by the included MIT license and remain property of their original authors.
