#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:25:55 2021

@author: labuser
"""
import copy 
grid = [[0,0,0,0],
        [0,0,1,0],
        [1,0,0,0],
        [0,0,0,0]]
goal = [3,3]
discount_factor = 0.9
delta = [
        [-1, 0],  # go up
        [-1, -1],  # up left (diag)
        [0, -1],  # go left
        [1, -1],  # dn left (diag)
        [1, 0],  # go down
        [1, 1],  # dn right (diag)
        [0, 1],  # go right
        [-1, 1],  # up right (diag)]
        ]
success_prob = 0.72
delta_directions = ["n", "nw", "w", "sw", "s", "se", "e", "ne"]
value = [[0,0.0,0.0,0.0],[0.0,0.0,-200,0.0],[-200,0.0,0.0,0.0],[0.0,0.0,0.0,100]]
policy = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]

reward =[[-4,-4,-4,-4],
        [-4,-4,0,100],
        [0,-4,-4,-4],
        [-4,100,-4,0]]
change = True

for m in range(2):
    values_copy = copy.deepcopy(value)
    for x in range(len(grid)):
        for y in range(len(grid[0])):

            if goal[0] == x and goal[1] == y:
                if value[x][y] != 100:
                    value[x][y] = 100
                    policy[x][y] = '*'

            elif grid[x][y] == 0:
                legal_moves_count = 0
                v_max = -1000
                #count the legal moves
                for a in range(len(delta)):
                    x2 = x + delta[a][0]
                    y2 = y + delta[a][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):
                        legal_moves_count +=1
                
                for a in range(len(delta)):
                    if m==1 and x==1 and y==0:
                        print()
                    x2 = x + delta[a][0]
                    y2 = y + delta[a][1]
                    if not (x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0])):
                        continue
                    v2 = reward[x][y]
                    for a2 in range(len(delta)):
                        x2 = x + delta[a2][0]
                        y2 = y + delta[a2][1]
                        
                        if a2 == a:
                            p2 = success_prob
                        else:
                            p2 = (1.0 - success_prob)/(legal_moves_count-1)

                        if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):
                            v2 += discount_factor*p2* value[x2][y2]
                            if m==1 and x==1 and y==0:
                                print(value[x2][y2])
                                #pass
                            #if x==0 and y==0:
                            #    print(v2)
                            #     print(value[x2][y2] )
                            #     #pass
                            #     print(p2)
                            #     #print(v2)
                    
                    if v2 > v_max:
                            change = True
                            v_max = v2
                            values_copy[x][y] = v2
                            policy[x][y] = delta_directions[a]
    
    value =copy.deepcopy(values_copy)
    
    
    
    

#print(value)
    
                            
                            
                            
                            
                            