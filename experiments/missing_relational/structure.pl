0.5::fire(X):-person(X).
0.5::burglary(X):-person(X).
0.5::alarm(X):-fire(X).
0.5::alarm(X):-burglary(X).
0.5::cares(X,Y):-person(X),person(Y).
0.5::calls(X,Y):-cares(X,Y),alarm(Y),\+same_person(X,Y).
