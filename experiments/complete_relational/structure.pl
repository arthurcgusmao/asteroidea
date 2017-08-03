0.5::fire(X).
0.5::burglary(X).
0.5::alarm(X):-fire(X).
0.5::alarm(X):-burglary(X).
0.5::neighbor(X,Y).
0.5::calls(X,Y):-alarm(X),neighbor(X,Y).
