person(p_1).
same_person(p_1,p_1).
person(p_2).
same_person(p_2,p_2).
person(p_3).
same_person(p_3,p_3).
person(p_4).
same_person(p_4,p_4).
person(p_5).
same_person(p_5,p_5).
evidence(fire(p_1),true).
evidence(alarm(p_1),true).
evidence(burglary(p_1),none).
evidence(calls(p_1,p_1),none).
evidence(cares(p_1,p_1),true).
evidence(calls(p_1,p_2),false).
evidence(cares(p_1,p_2),true).
evidence(calls(p_1,p_3),false).
evidence(cares(p_1,p_3),none).
evidence(calls(p_1,p_4),true).
evidence(cares(p_1,p_4),true).
evidence(calls(p_1,p_5),false).
evidence(cares(p_1,p_5),false).
evidence(fire(p_2),false).
evidence(alarm(p_2),false).
evidence(burglary(p_2),none).
evidence(calls(p_2,p_1),true).
evidence(cares(p_2,p_1),none).
evidence(calls(p_2,p_2),false).
evidence(cares(p_2,p_2),true).
evidence(calls(p_2,p_3),false).
evidence(cares(p_2,p_3),false).
evidence(calls(p_2,p_4),false).
evidence(cares(p_2,p_4),none).
evidence(calls(p_2,p_5),false).
evidence(cares(p_2,p_5),true).
evidence(fire(p_3),none).
evidence(alarm(p_3),false).
evidence(burglary(p_3),false).
evidence(calls(p_3,p_1),none).
evidence(cares(p_3,p_1),true).
evidence(calls(p_3,p_2),false).
evidence(cares(p_3,p_2),true).
evidence(calls(p_3,p_3),false).
evidence(cares(p_3,p_3),true).
evidence(calls(p_3,p_4),true).
evidence(cares(p_3,p_4),true).
evidence(calls(p_3,p_5),false).
evidence(cares(p_3,p_5),true).
evidence(fire(p_4),none).
evidence(alarm(p_4),true).
evidence(burglary(p_4),true).
evidence(calls(p_4,p_1),false).
evidence(cares(p_4,p_1),false).
evidence(calls(p_4,p_2),false).
evidence(cares(p_4,p_2),true).
evidence(calls(p_4,p_3),none).
evidence(cares(p_4,p_3),true).
evidence(calls(p_4,p_4),false).
evidence(cares(p_4,p_4),true).
evidence(calls(p_4,p_5),none).
evidence(cares(p_4,p_5),false).
evidence(fire(p_5),false).
evidence(alarm(p_5),false).
evidence(burglary(p_5),false).
evidence(calls(p_5,p_1),true).
evidence(cares(p_5,p_1),true).
evidence(calls(p_5,p_2),false).
evidence(cares(p_5,p_2),true).
evidence(calls(p_5,p_3),false).
evidence(cares(p_5,p_3),none).
evidence(calls(p_5,p_4),true).
evidence(cares(p_5,p_4),true).
evidence(calls(p_5,p_5),false).
evidence(cares(p_5,p_5),false).
