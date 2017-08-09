0.3::fire(X):-person(X).
0.4::burglary(X):-person(X).
0.7::alarm(X):-fire(X).
0.9::alarm(X):-burglary(X).
0.8::cares(X,Y):-person(X),person(Y).
0.8::calls(X,Y):-cares(X,Y),alarm(Y),\+same_person(X,Y).
