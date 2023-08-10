# Random Schedule Generation and Constraint Violation Reduction for the TTP

This is the repository for my Master's project/thesis **"Random Schedule Generation and Constraint Violation Reduction for the TTP"**.

The project was carried out at **VU Amsterdam** and **UvA**, supervised by **Daan van den Berg**.

## Code

The code related to my project can be found in the main directory. 

`violations.py` contains the code for the violation experiment (**block 1/chapter 3 - Random Schedule Generation**).

`HC.py`, `PPA.py`, `SAGG.py`, `SALR.py`, and `SASC.py` contain the code for the algorithms used (**block 2/chapter 4 - Constraint Violation Reduction**).

Run the scripts using `python [file].py [nr_teams]`

## Data

All the data related to the project can be found in the **Data** folder in the main directory. 

Each folder corresponds to its algorithm/method, while each folder within contains its respective number of teams (6 = 6 number of teams).

The number behind each file within the folders signifies the run it originates from (_1.txt signifies it is from run 1).

The violations (`violations_.txt` and `violationsBest_.txt`) are stored in the format:

[maxSteakViolatoins] \
[noRepeatViolations]\
[doubleRoundRobinViolatoins]\
[gamesAgainstSelf] (always 0, present from a previous method)\
[unmatchedPairings] (always 0, present from a previous method)

PPA contains two additional data points:

[numberOfEvaluations]\
[generations]

## Figures

All figures (including figures not present in the thesis) can be found in the **Figures** folder in the main directory.

Block1 contains the figures related to **block 1/chapter 3 - Random Schedule Generation**

Block2 contains the figures related to **block 2/chapter 4 - Constraint Violation Reduction**

## Author

Kristian Verduin (kristianverduin@gmail.com)
