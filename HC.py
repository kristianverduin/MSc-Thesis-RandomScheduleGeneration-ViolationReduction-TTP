import numpy as np
import random
import math
from datetime import datetime
import sys

U = 3

def checkScheduleConstraints(schedule, nrTeams):
    """Calculates the number of violations present in the schedule
    Arguments:
        schedule ([int, int]) : Schedule 
        nrTeams (int) : The number of teams present in the schedule
    Returns:
        violations ([int]) : The number of violations present in the schedule, in the format [Home/Away streak, No-repeat, Double round-robin, mismatched games, games against itself]
    """

    nrRounds = (2*nrTeams)-2
    violations = [0, 0, 0, 0, 0]
    for team in range(nrTeams):
        homeStreak = 0
        awayStreak = 0
        gamesPlayed = np.zeros(nrTeams)
        homePlayed = np.zeros(nrTeams)
        awayPlayed = np.zeros(nrTeams)

        for round in range(nrRounds):
            #Check maxStreak
            if schedule[round, team] > 0:
                awayStreak = 0
                homeStreak += 1
                homePlayed[abs(schedule[round, team])-1] += 1
            else:
                awayStreak += 1
                homeStreak = 0
                awayPlayed[abs(schedule[round, team])-1] += 1
            if homeStreak > U or awayStreak > U:
                violations[0] += 1

            gamesPlayed[abs(schedule[round, team])-1] += 1

            #Check noRepeat
            if round > 0:
                if abs(schedule[round, team]) == abs(schedule[round-1, team]):
                    violations[1] += 1

            #Check if the opponent also has the current team as opponent (matches are paires)
            if team != abs(schedule[round, abs(schedule[round, team])-1])-1:
                violations[3] += 1

            #Check if the current team is playing against itself
            if abs(schedule[round, team])-1 == team:
                violations[4] += 1

        #Check for double round-robin violations
        for i in range(len(gamesPlayed)):
            if i != team:
                if gamesPlayed[i] == 0:
                    violations[2] += 2
                elif gamesPlayed[i] == 1:
                    violations[2] += 1
                elif gamesPlayed[i] == 2:
                    if homePlayed[i] != 1 and awayPlayed[i] != 1:
                        violations[2] += 1
                else:
                    violations[2] += gamesPlayed[i]-2
                    if homePlayed[i] == 0 or awayPlayed[i] == 0:
                        violations[2] += 1

    return  violations

def createRandomSchedulePairs(nrTeams):
    """Generates a randomly paired schedule

    Arguments:
        nrTeams (int) : The number of teams present in the schedule

    Returns:
        Schedule ([int, int]) : The randomly generated schedule
    """

    nrRounds = (2*nrTeams)-2
    schedule = np.full((nrRounds, nrTeams), None)
    choices = list(range(-nrTeams, nrTeams+1))
    choices.remove(0)

    for round in range(nrRounds):
        teamsToPick = choices.copy()
        for team in range(nrTeams):
            if schedule[round, team] == None:
                team += 1
                teamsToPick.remove(team)
                teamsToPick.remove(-team)
                choice = random.choice(teamsToPick)
                teamsToPick.remove(choice)
                teamsToPick.remove(-choice)
                if choice > 0:
                    schedule[round, team-1] = choice
                    schedule[round, choice-1] = -team
                else:
                    schedule[round, team-1] = choice
                    schedule[round, abs(choice)-1] = team

    return schedule

def swapSigns(originalSchedule):
    """Swaps the signs of a random match

    Arguments:
        Schedule ([int, int]) : The original schedule

    Returns:
        Schedule ([int, int]) : The new schedule
    """
    schedule = originalSchedule.copy()
    row = random.choice(range(len(schedule)))
    column = random.choice(range(len(schedule[row])))
    schedule[row][column] *= -1
    schedule[row][abs(schedule[row][column])-1] *= -1
    return schedule

def swapRounds(originalSchedule):
    """Swaps two random rounds

    Arguments:
        Schedule ([int, int]) : The original schedule

    Returns:
        Schedule ([int, int]) : The mutated schedule
    """
    schedule = originalSchedule.copy()
    row1, row2 = random.sample(range(len(schedule)), 2)
    firstRow = originalSchedule[row1]
    secondRow = originalSchedule[row2]
    schedule[row1] = secondRow
    schedule[row2] = firstRow
    return schedule

def swapPartialTeams(originalSchedule):
    """Swaps a random game in a random round

    Arguments:
        Schedule ([int, int]) : The original schedule

    Returns:
        schedule ([int, int]) : The mutated schedule
    """
    schedule = originalSchedule.copy()
    col1, col2 = random.sample(range(len(schedule[0])), 2)
    row = random.choice(range(len(schedule)))
    current = schedule[row][col1]

    while(True):
        if abs(current)-1 != col2:
            opponent = schedule[row][col2]
            curOpponent = schedule[row][abs(schedule[row][col1])-1]
            oppOpponent = schedule[row][abs(schedule[row][col2])-1]
            schedule[row][abs(schedule[row][col1])-1] = oppOpponent
            schedule[row][abs(schedule[row][col2])-1] = curOpponent
            schedule[row][col1] = opponent
            schedule[row][col2] = current
            break
        else:
            col1, col2 = random.sample(range(len(schedule[0])), 2)
            current = schedule[row][col1]

    return schedule


def swapTeams(originalSchedule):
    """Swaps a all games of two random teams

    Arguments:
        Schedule ([int, int]) : The original schedule

    Returns:
        schedule ([int, int]) : The mutated schedule
    """
    schedule = originalSchedule.copy()
    col1, col2 = random.sample(range(len(schedule[0])),2)

    for i in range(len(schedule)):
        current = schedule[i][col1]
        opponent = schedule[i][col2]
        if abs(current)-1 != col2:
            curOpponent = schedule[i][abs(schedule[i][col1])-1]
            oppOpponent = schedule[i][abs(schedule[i][col2])-1]
            schedule[i][abs(schedule[i][col1])-1] = oppOpponent
            schedule[i][abs(schedule[i][col2])-1] = curOpponent
            schedule[i][col1] = opponent
            schedule[i][col2] = current

    return schedule

def update(row, col):
    """ Updates the sign of the team 

    Arguments:
        row (int) : The row of the game
        col (int) : The column of the game
    """
    opponent = int(abs(row[col])-1)
    if row[col] < 0:
        row[opponent] = col+1
    else:
        row[opponent] = (col+1)*-1

def find(row, team):
    """ Finds the index of the team in the row

    Arguments:
        row ([int]) : The row containing the team
        team (int) : The team to find the index of

    Returns:
        idx : The index of the team
    """
    for idx, i in enumerate(row):
        if i is not None:
            if abs(i) == abs(team):
                return idx

def updateRow(newRow, oldRow):
    """ Updates the new row based on the old row, if impossible, otherwise finds new matches

    Arguments:
        newRow ([int]) : The row to update
        oldRow ([int]) : The old row to update from
    """
    for idx, i in enumerate(newRow):
        if i == None:
            old = oldRow[idx]
            if old not in newRow and old*-1 not in newRow:
                newRow[idx] = old
                update(newRow, idx)
            else:
                newIdx = find(newRow, old)
                newRow[idx] = oldRow[newIdx]
                update(newRow, idx)

def swapPartialRounds(originalSchedule):
    """Swaps two games at random rounds for a random team

    Arguments:
        originalSchedule ([int, int]) : The original schedule

    Returns:
        schedule ([int, int]) : The mutated schedule
    """
    schedule = originalSchedule.copy()
    newRow1 = np.full(len(originalSchedule[0]), None)
    newRow2 = np.full(len(originalSchedule[0]), None)
    col = random.choice(range(len(schedule[0])))
    row1, row2 = random.sample(range(len(schedule)), 2)
    newRow1[col] = originalSchedule[row2, col]
    newRow2[col] = originalSchedule[row1, col]

    update(newRow1, col)
    update(newRow2, col)

    updateRow(newRow1, originalSchedule[row1])
    updateRow(newRow2, originalSchedule[row2])

    schedule[row1] = newRow1
    schedule[row2] = newRow2

    return schedule

def hillClimber(nrTeams):
    schedule = createRandomSchedulePairs(nrTeams)

    violations = checkScheduleConstraints(schedule, nrTeams)
    totalViolations = np.sum(violations)

    start = datetime.now()
    counter = 0

    homeAway = []
    repeat = []
    robin1 = []
    robin2 = []
    robin3 = []

    homeAway.append(violations[0])
    repeat.append(violations[1])
    robin1.append(violations[2])
    robin2.append(violations[3])
    robin3.append(violations[4])

    while totalViolations != 0 and counter != 1000000:

        mutation = random.choice([swapRounds, swapSigns, swapTeams, swapPartialRounds, swapPartialTeams])
        secondSchedule = mutation(schedule)
        violations2 = checkScheduleConstraints(secondSchedule, nrTeams)
        totalViolationsSecond = np.sum(violations2)

        if totalViolationsSecond <= totalViolations:

            violations = violations2
            totalViolations = totalViolationsSecond
            schedule = secondSchedule

        counter += 1
        homeAway.append(violations[0])
        repeat.append(violations[1])
        robin1.append(violations[2])
        robin2.append(violations[3])
        robin3.append(violations[4])

    time = datetime.now()-start
    time = time.total_seconds()

if int(sys.argv[1]) % 2 == 0:
    for i in range(11):
        hillClimber(int(sys.argv[1]))
    else:
        print("n must be even")
        