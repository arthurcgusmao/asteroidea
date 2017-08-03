0.3::fire(X).
0.4::burglary(X).
0.7::alarm(X):-fire(X).
0.9::alarm(X):-burglary(X).
0.2::neighbor(X,Y).
neighbor(X,Y):-neighbor(Y,X).
0.8::calls(X,Y):-alarm(X),neighbor(X,Y).
