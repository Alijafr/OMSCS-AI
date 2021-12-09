#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:17:00 2021

@author: labuser
"""

def implies(a, b):
  if a == True:
    return b
  return True

def Biconditional(a,b):
  A = implies(a, b)
  B = implies(b, a)
  if A == B:
    return True
  else:
    return False

#the logic gates

A = [True,False]
B = [True,False]
C = [True,False]
D = [True,False]
i =0
for a in A:
    for b in B:
        for c in C:
            for d in D: 
                #print(" {} {} {} {} ".format(a,b,c,d))
                board_1 = a or b
                board_2 = b and c 
                board_3 = a and (not c or d)
                board_4 = (a and b) or d  
                
                board_5 = implies(board_1,board_2)
                board_6 = Biconditional(board_3,board_4)
                
                out = not implies((board_5 or board_6),board_5)
                
                print("row {}: {}".format(i,out))
                i +=1
                
#last statement only 

for a in A:
    for b in B:
        out1 = not implies((a or b),a)
        out2 = not implies(b,a)
        if out1 == out2:
            print(True)
