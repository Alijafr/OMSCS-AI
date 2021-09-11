# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math
from typing import Counter, Union



class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        # counter to keep track of which element inserted first so the queue become FIFO
        self.counter = 0 
    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        return heapq.heappop(self.queue)
        # TODO: finish this function!
        raise NotImplementedError

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """
        #self.queue[0] = self.queue[node] # move it to the top
        #heapq.heappop(self.queue) # remove it 
        #heapq.heapify(self.queue) #sort it again 
        #return self.queue
        return heapq.heappop(node)
        raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """
        self.counter += 1
        node = (node[0],self.counter,node[1])
        
        # TODO: finish this function!
        return heapq.heappush(self.queue,node)
        raise NotImplementedError
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []
    frontier = PriorityQueue()
    explored = set(start)
    current_node = None
    priority  = 0
    frontier.append((priority,start))
    found_path = False
    branch = {}
    
    while frontier:
        current_node = frontier.pop() 
        priority +=1
        # print(graph[current_node])
        for neighbour in sorted(graph.neighbors(current_node[2])): #the queue is structured as (priority, counter,node)
            if neighbour not in frontier and neighbour not in explored:
                explored.add(neighbour)
                if neighbour == goal:
                    found_path = True 
            
                else:
                    frontier.append((priority, neighbour))
                
                branch [neighbour] = current_node[2] #add the parent branch
        
        if found_path:
            n = goal
            path = []
            path.append(goal)
            while branch[n] != start:
                path.append(branch[n])
                n = branch[n]
            path.append(branch[n]) #append the start
            path.reverse()
            return path

    return "no path found"


    raise NotImplementedError


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []
    frontier = PriorityQueue()
    explored = set(start)
    current_node = None
    frontier.append((0,start))
    branch = {}
    
    while frontier:
        _, _ , current_node = frontier.pop() 
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]
        if current_node == goal:   
            n = goal
            path = []
            path.append(goal)
            while branch[n][1] != start:
                path.append(branch[n][1])
                n = branch[n][1]
            path.append(branch[n][1]) #append the start
            path.reverse()
            return path

        
        explored.add(current_node)
        # print(graph[current_node])
        for neighbour in sorted(graph.neighbors(current_node)): #the queue is structured as (priority, counter,node)
            neighbour_cost = graph.get_edge_weight(current_node,neighbour)
            cost_total = current_cost + neighbour_cost
            if neighbour not in frontier and neighbour not in explored:
                frontier.append((cost_total, neighbour))
                branch [neighbour] = (cost_total, current_node) #add the parent branch
                
            elif neighbour in frontier and cost_total < branch[neighbour][0]:
                #how to remove while not knowing the counter number?
                frontier.append((cost_total,neighbour))#is it okay to add without removing
                branch [neighbour] = (cost_total, current_node) #add the parent branch 
        

    return "no path found"
    raise NotImplementedError


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    goal_pos = graph.nodes[goal]['pos']
    v_pos = graph.nodes[v]['pos']
    h = ((goal_pos[0]-v_pos[0])**2 + (goal_pos[1]-v_pos[1])**2)**0.5
    # print("h for {} is {}".format(v,h))
    return h
    raise NotImplementedError
def euclidean_dist_heuristic1(graph, v, goal1,goal2):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    goal_pos1 = graph.nodes[goal1]['pos']
    goal_pos2 = graph.nodes[goal2]['pos']
    v_pos = graph.nodes[v]['pos']
    hs = []
    h1 = ((goal_pos1[0]-v_pos[0])**2 + (goal_pos1[1]-v_pos[1])**2)**0.5
    h2 = ((goal_pos2[0]-v_pos[0])**2 + (goal_pos2[1]-v_pos[1])**2)**0.5
    hs.append(h1)
    hs.append(h2)
    return min(hs)

def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []
    frontier = PriorityQueue()
    explored = set(start)
    current_node = None
    frontier.append((0.0,start))
    found_path = False
    branch = {}
    
    while frontier.size():
        _, _ , current_node = frontier.pop() 
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]
        # print(current_node)
        if current_node == goal:   
            found_path = True
            break
        
        explored.add(current_node)
        # print(graph[current_node])
        for neighbour in sorted(graph.neighbors(current_node)): #the queue is structured as (priority, counter,node)
            neighbour_cost = graph.get_edge_weight(current_node,neighbour)
            cost_total = current_cost + neighbour_cost
            # print("cost for {} is: {}".format(neighbour,cost_total))
            h = heuristic(graph,neighbour,goal)
            f = cost_total + h 
            # print(f)
            if  neighbour not in frontier and neighbour not in explored:
                frontier.append((f, neighbour))
                branch [neighbour] = (cost_total, current_node) #add the parent branch
                
            elif neighbour in frontier and cost_total < branch[neighbour][0]:
                #how to remove while not knowing the counter number?
                frontier.append((f,neighbour))#is it okay to add without removing
                branch [neighbour] = (cost_total, current_node) #add the parent branch 
            
    if found_path:
        # print(len(explored))
        n = goal
        path = []
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1]) #append the start
        path.reverse()
        return path
                

        

    return "no path found"
    raise NotImplementedError


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # print("start: {}, goal: {}".format(start,goal))
    if start == goal:
        return []
    #search from the start
    frontier_forward = PriorityQueue()
    explored_forward = set(start)
    current_node_forward = None
    frontier_forward.append((0,start))
    branch_forward = {}
    # branch_forward [start] = (0.0,start)
    #searh from the goal 
    frontier_backward = PriorityQueue()
    explored_backward = set(goal)
    current_node_backward = None
    frontier_backward.append((0,goal))
    branch_backward = {}
    # branch_backward [goal] = (0.0,goal)
    #Flag to to alternate between search 
    forward_search = True
    #flag to check if path is found
    found_path = False
    #intersection nodes
    intersection_nodes = None
    intersection_node = None
    while frontier_forward or frontier_backward:
        if forward_search:
            _, _ , current_node_forward = frontier_forward.pop() 
            explored_forward.add(current_node_forward)
            if explored_backward.intersection(explored_forward):   
                found_path = True
                frontier_backward_set = set([x[-1] for x in frontier_backward])
                intersection_nodes = list(explored_forward.intersection(explored_backward.union(frontier_backward_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    # if node not in branch_backward:
                    #     print("node {} not in backward branch".format(node))
                    # if node not in branch_forward:
                    #     print("node {} node in forward branch".format(node))
                    if node is start:
                        #the cost to the from start to start is 0
                        cost = branch_backward[node][0]
                    elif node is goal:
                        cost = branch_forward[node][0]
                    else:
                        cost = branch_forward[node][0] + branch_backward[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node =intersection_nodes[intersection_node_index] 
                
                break
            
            if current_node_forward == start:
                current_cost_forward = 0.0
            else:
                current_cost_forward = branch_forward[current_node_forward][0]
            

            
            
            # print(graph[current_node])
            for neighbour in sorted(graph.neighbors(current_node_forward)): #the queue is structured as (priority, counter,node)
                neighbour_cost = graph.get_edge_weight(current_node_forward,neighbour)
                cost_total_forward = current_cost_forward + neighbour_cost
                if neighbour not in frontier_forward and neighbour not in explored_forward:
                    frontier_forward.append((cost_total_forward, neighbour))
                    branch_forward [neighbour] = (cost_total_forward, current_node_forward) #add the parent branch
                    
                elif neighbour in frontier_forward and cost_total_forward < branch_forward[neighbour][0]:
                    #how to remove while not knowing the counter number?
                    frontier_forward.append((cost_total_forward,neighbour))#is it okay to add without removing
                    branch_forward [neighbour] = (cost_total_forward, current_node_forward) #add the parent branch 
            #alternate to the backward search 
            forward_search = False 
        else:
            _, _ , current_node_backward = frontier_backward.pop() 
            explored_backward.add(current_node_backward)
            if explored_backward.intersection(explored_forward):   
                found_path = True
                frontier_forward_set = set([x[-1] for x in frontier_forward])
                intersection_nodes = list(explored_backward.intersection(explored_forward.union(frontier_forward_set)))
                intersections_cost = []
                # print(intersection_nodes)
                for node in intersection_nodes:
                    # print(node)
                    # if node not in branch_backward:
                    #     print("node {} not in backward branch".format(node))
                    # if node not in branch_forward:
                    #     print("node {} node in forward branch".format(node))
                    if node is start:
                        #the cost to the from start to start is 0
                        cost = branch_backward[node][0]
                    elif node is goal:
                        cost = branch_forward[node][0]
                    else:
                        cost = branch_forward[node][0] + branch_backward[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node =intersection_nodes[intersection_node_index] 
                break

            if current_node_backward == goal:
                current_cost_backward = 0.0
            else:
                current_cost_backward = branch_backward[current_node_backward][0]

            # print(graph[current_node])
            for neighbour in sorted(graph.neighbors(current_node_backward)): #the queue is structured as (priority, counter,node)
                neighbour_cost = graph.get_edge_weight(current_node_backward,neighbour)
                cost_total_backward = current_cost_backward + neighbour_cost
                if neighbour not in frontier_backward and neighbour not in explored_backward:
                    frontier_backward.append((cost_total_backward, neighbour))
                    branch_backward [neighbour] = (cost_total_backward, current_node_backward) #add the parent branch
                    
                elif neighbour in frontier_backward and cost_total_backward < branch_backward[neighbour][0]:
                    #how to remove while not knowing the counter number?
                    frontier_backward.append((cost_total_backward,neighbour))#is it okay to add without removing
                    branch_backward [neighbour] = (cost_total_backward, current_node_backward) #add the parent branch 
            #alternate to the backward search 
            forward_search = True 
            
    if found_path:
        #back-propogate the path
        path = []
        n = intersection_node
        path.append(n)
        #to avoid duplicate path, more check statement is added  (if the intersection is the start or end, it may result in duplicate nodes in the path)
        if n != start:
            while branch_forward [n][1] != start:
                path.append(branch_forward[n][1])
                n = branch_forward[n][1]
            path.append(start)
            path.reverse()
        n = intersection_node
        if n != goal:
            while branch_backward[n][1] != goal:
                path.append(branch_backward[n][1])
                n = branch_backward[n][1]
            path.append(goal) #now path should contain path from intersection to goal 
        # print("start: {}, goal: {}".format(start,goal))
        # print("bi-path",path)
        return path

        

    raise NotImplementedError

def bidirectional_ucs1(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # print("start: {}, goal: {}".format(start,goal))
    if start == goal:
        return []
    #search from the start
    frontier_forward = PriorityQueue()
    explored_forward = set(start)
    current_node_forward = None
    frontier_forward.append((0,start))
    branch_forward = {}
    # branch_forward [start] = (0.0,start)
    #searh from the goal 
    frontier_backward = PriorityQueue()
    explored_backward = set(goal)
    current_node_backward = None
    frontier_backward.append((0,goal))
    branch_backward = {}
    # branch_backward [goal] = (0.0,goal)
    #Flag to to alternate between search 
    forward_search = True
    #flag to check if path is found
    found_path = False
    #intersection nodes
    intersection_nodes = None
    intersection_node = None
    while frontier_forward or frontier_backward:
        if forward_search:
            _, _ , current_node_forward = frontier_forward.pop() 
            explored_forward.add(current_node_forward)
            if explored_backward.intersection(explored_forward):   
                found_path = True
                frontier_backward_set = set([x[-1] for x in frontier_backward])
                intersection_nodes = list(explored_forward.intersection(explored_backward.union(frontier_backward_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    # if node not in branch_backward:
                    #     print("node {} not in backward branch".format(node))
                    # if node not in branch_forward:
                    #     print("node {} node in forward branch".format(node))
                    if node is start:
                        #the cost to the from start to start is 0
                        cost = branch_backward[node][0]
                    elif node is goal:
                        cost = branch_forward[node][0]
                    else:
                        cost = branch_forward[node][0] + branch_backward[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node =intersection_nodes[intersection_node_index] 
                
                break
            
            if current_node_forward == start:
                current_cost_forward = 0.0
            else:
                current_cost_forward = branch_forward[current_node_forward][0]
            

            
            
            # print(graph[current_node])
            for neighbour in sorted(graph.neighbors(current_node_forward)): #the queue is structured as (priority, counter,node)
                neighbour_cost = graph.get_edge_weight(current_node_forward,neighbour)
                cost_total_forward = current_cost_forward + neighbour_cost
                if neighbour not in frontier_forward and neighbour not in explored_forward:
                    frontier_forward.append((cost_total_forward, neighbour))
                    branch_forward [neighbour] = (cost_total_forward, current_node_forward) #add the parent branch
                    
                elif neighbour in frontier_forward and cost_total_forward < branch_forward[neighbour][0]:
                    #how to remove while not knowing the counter number?
                    frontier_forward.append((cost_total_forward,neighbour))#is it okay to add without removing
                    branch_forward [neighbour] = (cost_total_forward, current_node_forward) #add the parent branch 
            #alternate to the backward search 
            forward_search = False 
        else:
            _, _ , current_node_backward = frontier_backward.pop() 
            explored_backward.add(current_node_backward)
            if explored_backward.intersection(explored_forward):   
                found_path = True
                frontier_forward_set = set([x[-1] for x in frontier_forward])
                intersection_nodes = list(explored_backward.intersection(explored_forward.union(frontier_forward_set)))
                intersections_cost = []
                # print(intersection_nodes)
                for node in intersection_nodes:
                    # print(node)
                    # if node not in branch_backward:
                    #     print("node {} not in backward branch".format(node))
                    # if node not in branch_forward:
                    #     print("node {} node in forward branch".format(node))
                    if node is start:
                        #the cost to the from start to start is 0
                        cost = branch_backward[node][0]
                    elif node is goal:
                        cost = branch_forward[node][0]
                    else:
                        cost = branch_forward[node][0] + branch_backward[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node =intersection_nodes[intersection_node_index] 
                break

            if current_node_backward == goal:
                current_cost_backward = 0.0
            else:
                current_cost_backward = branch_backward[current_node_backward][0]

            # print(graph[current_node])
            for neighbour in sorted(graph.neighbors(current_node_backward)): #the queue is structured as (priority, counter,node)
                neighbour_cost = graph.get_edge_weight(current_node_backward,neighbour)
                cost_total_backward = current_cost_backward + neighbour_cost
                if neighbour not in frontier_backward and neighbour not in explored_backward:
                    frontier_backward.append((cost_total_backward, neighbour))
                    branch_backward [neighbour] = (cost_total_backward, current_node_backward) #add the parent branch
                    
                elif neighbour in frontier_backward and cost_total_backward < branch_backward[neighbour][0]:
                    #how to remove while not knowing the counter number?
                    frontier_backward.append((cost_total_backward,neighbour))#is it okay to add without removing
                    branch_backward [neighbour] = (cost_total_backward, current_node_backward) #add the parent branch 
            #alternate to the backward search 
            forward_search = True 
            
    if found_path:
        #back-propogate the path
        path = []
        n = intersection_node
        path.append(n)
        #to avoid duplicate path, more check statement is added  (if the intersection is the start or end, it may result in duplicate nodes in the path)
        if n != start:
            while branch_forward [n][1] != start:
                path.append(branch_forward[n][1])
                n = branch_forward[n][1]
            path.append(start)
            path.reverse()
        n = intersection_node
        if n != goal:
            while branch_backward[n][1] != goal:
                path.append(branch_backward[n][1])
                n = branch_backward[n][1]
            path.append(goal) #now path should contain path from intersection to goal 
        # print("start: {}, goal: {}".format(start,goal))
        # print(path)
        return path

        

    raise NotImplementedError

def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    #search from the start
    frontier_forward = PriorityQueue()
    explored_forward = set(start)
    current_node_forward = None
    frontier_forward.append((0,start))
    branch_forward = {}
    #searh from the goal 
    frontier_backward = PriorityQueue()
    explored_backward = set(goal)
    current_node_backward = None
    frontier_backward.append((0,goal))
    branch_backward = {}

    #Flag to to alternate between search 
    forward_search = True
    #flag to check if path is found
    found_path = False
    #intersection nodes
    intersection_nodes = None
    intersection_node = None
    while frontier_forward or frontier_backward:
        if forward_search:
            _, _ , current_node_forward = frontier_forward.pop() 
            explored_forward.add(current_node_forward)
            if explored_backward.intersection(explored_forward):   
                found_path = True
                frontier_backward_set = set([x[-1] for x in frontier_backward])
                intersection_nodes = list(explored_forward.intersection(explored_backward.union(frontier_backward_set)))
                intersections_cost = []
                for node in intersection_nodes:
                    if node is start:
                        #the cost to the from start to start is 0
                        cost = branch_backward[node][0]
                    elif node is goal:
                        cost = branch_forward[node][0]
                    else:
                        cost = branch_forward[node][0] + branch_backward[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node =intersection_nodes[intersection_node_index] 
                
                break
            
            if current_node_forward == start:
                current_cost_forward = 0.0
            else:
                current_cost_forward = branch_forward[current_node_forward][0]
            

            
            
            # print(graph[current_node])
            for neighbour in sorted(graph.neighbors(current_node_forward)): #the queue is structured as (priority, counter,node)
                neighbour_cost = graph.get_edge_weight(current_node_forward,neighbour)
        
                cost_total_forward = current_cost_forward + neighbour_cost
                # print("cost for {} is: {}".format(neighbour,cost_total))
                h = heuristic(graph,neighbour,goal)
                f = cost_total_forward + h 
                if neighbour not in frontier_forward and neighbour not in explored_forward:
                    frontier_forward.append((f, neighbour))
                    branch_forward [neighbour] = (cost_total_forward, current_node_forward) #add the parent branch
                    
                elif neighbour in frontier_forward and cost_total_forward < branch_forward[neighbour][0]:
                    #how to remove while not knowing the counter number?
                    frontier_forward.append((f,neighbour))#is it okay to add without removing
                    branch_forward [neighbour] = (cost_total_forward, current_node_forward) #add the parent branch 
            #alternate to the backward search 
            forward_search = False 
        else:
            _, _ , current_node_backward = frontier_backward.pop() 
            explored_backward.add(current_node_backward)
            # print(explored_backward)
            if explored_backward.intersection(explored_forward):   
                found_path = True
                frontier_forward_set = set([x[-1] for x in frontier_forward])
                intersection_nodes = list(explored_backward.intersection(explored_forward.union(frontier_forward_set)))
                intersections_cost = []
                for node in intersection_nodes:
                    if node is start:
                        #the cost to the from start to start is 0
                        cost = branch_backward[node][0]
                    elif node is goal:
                        cost = branch_forward[node][0]
                    else:
                        cost = branch_forward[node][0] + branch_backward[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node =intersection_nodes[intersection_node_index] 
                break

            if current_node_backward == goal:
                current_cost_backward = 0.0
            else:
                current_cost_backward = branch_backward[current_node_backward][0]

            # print(graph[current_node])
            for neighbour in sorted(graph.neighbors(current_node_backward)): #the queue is structured as (priority, counter,node)
                
                neighbour_cost = graph.get_edge_weight(current_node_backward,neighbour)
                
                cost_total_backward = current_cost_backward + neighbour_cost
                h = heuristic(graph,neighbour,start)
                f = cost_total_backward + h 
                if neighbour not in frontier_backward and neighbour not in explored_backward:
                    frontier_backward.append((f, neighbour))
                    branch_backward [neighbour] = (cost_total_backward, current_node_backward) #add the parent branch
                    
                elif neighbour in frontier_backward and cost_total_backward < branch_backward[neighbour][0]:
                    #how to remove while not knowing the counter number?
                    frontier_backward.append((f,neighbour))#is it okay to add without removing
                    branch_backward [neighbour] = (cost_total_backward, current_node_backward) #add the parent branch 
            #alternate to the backward search 
            forward_search = True 
            
    if found_path:
        path = []
        n = intersection_node
        path.append(n)
        #to avoid duplicate path, more check statement is added  (if the intersection is the start or end, it may result in duplicate nodes in the path) 
        if n != start:
            while branch_forward [n][1] != start:
                path.append(branch_forward[n][1])
                n = branch_forward[n][1]
            path.append(start)
            path.reverse()
        n = intersection_node
        if n != goal:
            while branch_backward[n][1] != goal:
                path.append(branch_backward[n][1])
                n = branch_backward[n][1]
            path.append(goal) #now path should contain path from intersection to goal 
        # print("start: {}, goal: {}".format(start,goal))
        # print(path)
        return path
    # TODO: finish this function!
    raise NotImplementedError
def bidirectional_a_star1(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    #search from the start
    frontier_forward = PriorityQueue()
    explored_forward = set(start)
    current_node_forward = None
    frontier_forward.append((0,start))
    branch_forward = {}
    #searh from the goal 
    frontier_backward = PriorityQueue()
    explored_backward = set(goal)
    current_node_backward = None
    frontier_backward.append((0,goal))
    branch_backward = {}

    #Flag to to alternate between search 
    forward_search = True
    #flag to check if path is found
    found_path = False
    #intersection nodes
    intersection_nodes = None
    intersection_node = None
    while frontier_forward or frontier_backward:
        if forward_search:
            _, _ , current_node_forward = frontier_forward.pop() 
            explored_forward.add(current_node_forward)
            if explored_backward.intersection(explored_forward):   
                found_path = True
                frontier_backward_set = set([x[-1] for x in frontier_backward])
                intersection_nodes = list(explored_forward.intersection(explored_backward.union(frontier_backward_set)))
                intersections_cost = []
                for node in intersection_nodes:
                    if node is start:
                        #the cost to the from start to start is 0
                        cost = branch_backward[node][0]
                    elif node is goal:
                        cost = branch_forward[node][0]
                    else:
                        cost = branch_forward[node][0] + branch_backward[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node =intersection_nodes[intersection_node_index] 
                
                break
            
            if current_node_forward == start:
                current_cost_forward = 0.0
            else:
                current_cost_forward = branch_forward[current_node_forward][0]
            

            
            
            # print(graph[current_node])
            for neighbour in sorted(graph.neighbors(current_node_forward)): #the queue is structured as (priority, counter,node)
                neighbour_cost = graph.get_edge_weight(current_node_forward,neighbour)
        
                cost_total_forward = current_cost_forward + neighbour_cost
                # print("cost for {} is: {}".format(neighbour,cost_total))
                h = heuristic(graph,neighbour,goal)
                f = cost_total_forward + h 
                if neighbour not in frontier_forward and neighbour not in explored_forward:
                    frontier_forward.append((f, neighbour))
                    branch_forward [neighbour] = (cost_total_forward, current_node_forward) #add the parent branch
                    
                elif neighbour in frontier_forward and cost_total_forward < branch_forward[neighbour][0]:
                    #how to remove while not knowing the counter number?
                    frontier_forward.append((f,neighbour))#is it okay to add without removing
                    branch_forward [neighbour] = (cost_total_forward, current_node_forward) #add the parent branch 
            #alternate to the backward search 
            forward_search = False 
        else:
            _, _ , current_node_backward = frontier_backward.pop() 
            explored_backward.add(current_node_backward)
            # print(explored_backward)
            if explored_backward.intersection(explored_forward):   
                found_path = True
                frontier_forward_set = set([x[-1] for x in frontier_forward])
                intersection_nodes = list(explored_backward.intersection(explored_forward.union(frontier_forward_set)))
                intersections_cost = []
                for node in intersection_nodes:
                    if node is start:
                        #the cost to the from start to start is 0
                        cost = branch_backward[node][0]
                    elif node is goal:
                        cost = branch_forward[node][0]
                    else:
                        cost = branch_forward[node][0] + branch_backward[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node =intersection_nodes[intersection_node_index] 
                break

            if current_node_backward == goal:
                current_cost_backward = 0.0
            else:
                current_cost_backward = branch_backward[current_node_backward][0]

            # print(graph[current_node])
            for neighbour in sorted(graph.neighbors(current_node_backward)): #the queue is structured as (priority, counter,node)
                
                neighbour_cost = graph.get_edge_weight(current_node_backward,neighbour)
                
                cost_total_backward = current_cost_backward + neighbour_cost
                h = heuristic(graph,neighbour,start)
                f = cost_total_backward + h 
                if neighbour not in frontier_backward and neighbour not in explored_backward:
                    frontier_backward.append((f, neighbour))
                    branch_backward [neighbour] = (cost_total_backward, current_node_backward) #add the parent branch
                    
                elif neighbour in frontier_backward and cost_total_backward < branch_backward[neighbour][0]:
                    #how to remove while not knowing the counter number?
                    frontier_backward.append((f,neighbour))#is it okay to add without removing
                    branch_backward [neighbour] = (cost_total_backward, current_node_backward) #add the parent branch 
            #alternate to the backward search 
            forward_search = True 
            
    if found_path:
        path = []
        n = intersection_node
        path.append(n)
        #to avoid duplicate path, more check statement is added  (if the intersection is the start or end, it may result in duplicate nodes in the path) 
        if n != start:
            while branch_forward [n][1] != start:
                path.append(branch_forward[n][1])
                n = branch_forward[n][1]
            path.append(start)
            path.reverse()
        n = intersection_node
        if n != goal:
            while branch_backward[n][1] != goal:
                path.append(branch_backward[n][1])
                n = branch_backward[n][1]
            path.append(goal) #now path should contain path from intersection to goal 
        # print("start: {}, goal: {}".format(start,goal))
        # print(path)
        return path
def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    # print(" goals {}".format(goals))
    if len(set(goals)) ==1:
        return []
    elif  len(set(goals)) == 2:
        goals = list(set(goals))
        # print("found duplicate goals {}".format(goals))
        path = bidirectional_ucs1(graph,goals[0],goals[1])
        # print(" goals {} and path: {}".format(goals,path))
        return path
    
    #search from A 
    frontier_A = PriorityQueue()
    explored_A = set(goals[0])
    current_node_A= None
    frontier_A.append((0,goals[0]))
    branch_A = {}
    #searh from B 
    frontier_B = PriorityQueue()
    explored_B = set(goals[1])
    current_node_B = None
    frontier_B.append((0,goals[1]))
    branch_B = {}
    #search from C 
    frontier_C = PriorityQueue()
    explored_C = set(goals[2])
    current_node_C= None
    frontier_C.append((0,goals[2]))
    branch_C = {}

    #Flag to to alternate between searchs (0=A,1=B,2=C)
    branch2search= 0
    #flag to check if path is found
    found_path = False
    #intersection nodes
    intersection_nodes = None
    intersection_node_AB = None
    intersection_cost_AB = None
    intersection_node_AC = None
    intersection_cost_AC = None
    intersection_node_BC = None
    intersection_cost_BC = None
    #flag to stop exploring branches
    stop_A = False
    stop_B = False
    stop_C = False
    while True:
        
        if branch2search==0 and stop_A == False:
            _, _ , current_node_A = frontier_A.pop() 
            explored_A.add(current_node_A)
            if explored_A.intersection(explored_B) and intersection_node_AB is None:
                #finding the optimal path from A -- >B , B-->A
                frontier_B_set = set([x[-1] for x in frontier_B])
                intersection_nodes = list(explored_A.intersection(explored_B.union(frontier_B_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[0]:
                        #the cost to the from start to start is 0
                        cost = branch_B[node][0]
                    elif node is goals[1]:
                        cost = branch_A[node][0]
                    else:
                        cost = branch_A[node][0] + branch_B[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_AB =intersection_nodes[intersection_node_index]
                intersection_cost_AB = min(intersections_cost)
            if explored_A.intersection(explored_C) and intersection_node_AC is None:
                #finding the optimal path from A -- >C , B-->C
                frontier_C_set = set([x[-1] for x in frontier_C])
                intersection_nodes = list(explored_A.intersection(explored_C.union(frontier_C_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[0]:
                        #the cost to the from start to start is 0
                        cost = branch_C[node][0]
                    elif node is goals[2]:
                        cost = branch_A[node][0]
                    else:
                        cost = branch_A[node][0] + branch_C[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_AC =intersection_nodes[intersection_node_index]
                intersection_cost_AC = min(intersections_cost)
            if explored_A.intersection(explored_B) and explored_A.intersection(explored_C):
                #which means the optimal solution for AB and AC have been already calculated
                stop_A = True
                
                
            else:

                if current_node_A == goals[0]:
                    current_cost_A = 0.0
                else:
                    current_cost_A = branch_A[current_node_A][0]
                

                
                
                # print(graph[current_node])
                for neighbour in sorted(graph.neighbors(current_node_A)): #the queue is structured as (priority, counter,node)
                    neighbour_cost = graph.get_edge_weight(current_node_A,neighbour)
            
                    cost_total_A = current_cost_A + neighbour_cost
                    if neighbour not in frontier_A and neighbour not in explored_A:
                        frontier_A.append((cost_total_A, neighbour))
                        branch_A [neighbour] = (cost_total_A, current_node_A) #add the parent branch
                        
                    elif neighbour in frontier_A and cost_total_A < branch_A[neighbour][0]:
                        #how to remove while not knowing the counter number?
                        frontier_A.append((cost_total_A,neighbour))#is it okay to add without removing
                        branch_A [neighbour] = (cost_total_A, current_node_A) #add the parent branch 
                #alternate to the backward search 
             
        elif branch2search==1 and stop_B ==False:
            _, _ , current_node_B = frontier_B.pop() 
            explored_B.add(current_node_B)
            if explored_B.intersection(explored_A) and intersection_node_AB is None:
                #finding the optimal path from A -- >B , B-->A
                frontier_A_set = set([x[-1] for x in frontier_A])
                intersection_nodes = list(explored_B.intersection(explored_A.union(frontier_A_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[0]:
                        #the cost to the from start to start is 0
                        cost = branch_B[node][0]
                    elif node is goals[1]:
                        cost = branch_A[node][0]
                    else:
                        cost = branch_A[node][0] + branch_B[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_AB =intersection_nodes[intersection_node_index]
                intersection_cost_AB = min(intersections_cost)
            if explored_B.intersection(explored_C) and intersection_node_BC is None:
                #finding the optimal path from B -- >C , C-->B
                frontier_C_set = set([x[-1] for x in frontier_C])
                intersection_nodes = list(explored_B.intersection(explored_C.union(frontier_C_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[1]:
                        #the cost to the from start to start is 0
                        cost = branch_C[node][0]
                    elif node is goals[2]:
                        cost = branch_B[node][0]
                    else:
                        cost = branch_B[node][0] + branch_C[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_BC =intersection_nodes[intersection_node_index]
                intersection_cost_BC = min(intersections_cost)
            if explored_B.intersection(explored_C) and explored_B.intersection(explored_A):
                #which means the optimal solution for AB and AC have been already calculated
                stop_B = True

            else:    
            
                if current_node_B == goals[1]:
                    current_cost_B = 0.0
                else:
                    current_cost_B = branch_B[current_node_B][0]
                

                
                
                # print(graph[current_node])
                for neighbour in sorted(graph.neighbors(current_node_B)): #the queue is structured as (priority, counter,node)
                    neighbour_cost = graph.get_edge_weight(current_node_B,neighbour)
            
                    cost_total_B = current_cost_B + neighbour_cost
                    if neighbour not in frontier_B and neighbour not in explored_B:
                        frontier_B.append((cost_total_B, neighbour))
                        branch_B [neighbour] = (cost_total_B, current_node_B) #add the parent branch
                        
                    elif neighbour in frontier_B and cost_total_B < branch_B[neighbour][0]:
                        #how to remove while not knowing the counter number?
                        frontier_B.append((cost_total_B,neighbour))#is it okay to add without removing
                        branch_B [neighbour] = (cost_total_B, current_node_B) #add the parent branch 
            
        elif branch2search ==2 and stop_C == False:
            _, _ , current_node_C = frontier_C.pop() 
            explored_C.add(current_node_C)
            if explored_C.intersection(explored_A) and intersection_node_AC is None:
                #finding the optimal path from C -- >A , A-->C
                frontier_A_set = set([x[-1] for x in frontier_A])
                intersection_nodes = list(explored_C.intersection(explored_A.union(frontier_A_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[0]:
                        #the cost to the from start to start is 0
                        cost = branch_C[node][0]
                    elif node is goals[2]:
                        cost = branch_A[node][0]
                    else:
                        cost = branch_A[node][0] + branch_C[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_AC =intersection_nodes[intersection_node_index]
                intersection_cost_AC = min(intersections_cost)
            if explored_C.intersection(explored_B) and intersection_node_BC is None:
                #finding the optimal path from B -- >C , C-->B
                frontier_B_set = set([x[-1] for x in frontier_B])
                intersection_nodes = list(explored_C.intersection(explored_B.union(frontier_B_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[1]:
                        #the cost to the from start to start is 0
                        cost = branch_C[node][0]
                    elif node is goals[2]:
                        cost = branch_B[node][0]
                    else:
                        cost = branch_B[node][0] + branch_C[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_BC =intersection_nodes[intersection_node_index]
                intersection_cost_BC = min(intersections_cost)
            if explored_C.intersection(explored_A) and explored_C.intersection(explored_C):
                #which means the optimal solution for AB and AC have been already calculated
                stop_C = True
            
            else:
                if current_node_C == goals[2]:
                    current_cost_C = 0.0
                else:
                    current_cost_C = branch_C[current_node_C][0]
                
                

                
                
                # print(graph[current_node])
                for neighbour in sorted(graph.neighbors(current_node_C)): #the queue is structured as (priority, counter,node)
                    neighbour_cost = graph.get_edge_weight(current_node_C,neighbour)
            
                    cost_total_C = current_cost_C + neighbour_cost
                    if neighbour not in frontier_C and neighbour not in explored_C:
                        frontier_C.append((cost_total_C, neighbour))
                        branch_C [neighbour] = (cost_total_C, current_node_C) #add the parent branch
                        
                    elif neighbour in frontier_C and cost_total_C < branch_C[neighbour][0]:
                        #how to remove while not knowing the counter number?
                        frontier_C.append((cost_total_C,neighbour))#is it okay to add without removing
                        branch_C [neighbour] = (cost_total_C, current_node_C) #add the parent branch  

        if intersection_node_AC is not None and intersection_node_AB is not None and intersection_node_BC is not None:
            found_path = True    
            break
        else:
            #continue exploring with least cost node 
            costs = []
            if stop_A:
                costs.append(math.inf)
            else:
                costs.append(frontier_A.top()[0])
            if stop_B:
                costs.append(math.inf)
            else:
                costs.append(frontier_B.top()[0])
            if stop_C:
                costs.append(math.inf)
            else:
                costs.append(frontier_C.top()[0])
            branch2search = costs.index(min(costs))
    
    if found_path:
        # print("soluiton has been found")
        path = []
        # path_AB = find_path_bidir(goals[0],goals[1],intersection_node_AB,branch_A,branch_B)
        # path_AC = find_path_bidir(goals[0],goals[2],intersection_node_AC,branch_A,branch_C)
        # path_BC = find_path_bidir(goals[1],goals[2],intersection_node_BC,branch_B,branch_C)

        #There are 3 possible combination of paths: ABC=CBA= AB + BC , and ACB=BCA = AC + CB and BAC=CAB = BA + AC 
        paths_cost = []
        cost_ABC= intersection_cost_AB + intersection_cost_BC
        cost_ACB= intersection_cost_AC + intersection_cost_BC
        cost_BAC = intersection_cost_AB + intersection_cost_AC
        paths_cost.append(cost_ABC)
        paths_cost.append(cost_ACB)
        paths_cost.append(cost_BAC)
        # print(paths_cost)
        best_path = paths_cost.index(min(paths_cost))
        if best_path == 0: #ABC=CBA
            path_AB = find_path_bidir(goals[0],goals[1],intersection_node_AB,branch_A,branch_B)
            path += path_AB[:-1] #B will be included next 
            path_BC = find_path_bidir(goals[1],goals[2],intersection_node_BC,branch_B,branch_C)
            path += path_BC
        elif best_path ==1: # best path is ACB=BCA
            path_AC = find_path_bidir(goals[0],goals[2],intersection_node_AC,branch_A,branch_C)
            path += path_AC[:-1] # C will be inculded next
            path_CB = find_path_bidir(goals[2],goals[1],intersection_node_BC,branch_C,branch_B)
            path += path_CB
        elif best_path == 2: #best path is BAC=CAB 
            path_BA = find_path_bidir(goals[1],goals[0],intersection_node_AB,branch_B,branch_A)
            path += path_BA[:-1]
            path_AC = find_path_bidir(goals[0],goals[2],intersection_node_AC,branch_A,branch_C)
            path += path_AC
        # print(" goals {} and path: {}".format(goals,path))
        return path
    raise NotImplementedError

def find_path_bidir (start, goal, intersection_node, branch_forward, branch_backward):
    path = []
    n = intersection_node
    path.append(n)
    #to avoid duplicate path, more check statement is added  (if the intersection is the start or end, it may result in duplicate nodes in the path) 
    if n != start:
        while branch_forward [n][1] != start:
            path.append(branch_forward[n][1])
            n = branch_forward[n][1]
        path.append(start)
        path.reverse()
    n = intersection_node
    if n != goal:
        while branch_backward[n][1] != goal:
            path.append(branch_backward[n][1])
            n = branch_backward[n][1]
        path.append(goal) #now path should contain path from intersection to goal 
    return path 

def tridirectional_search1(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    # print(" goals {}".format(goals))
    if len(set(goals)) ==1:
        return []
    elif  len(set(goals)) == 2:
        goals = list(set(goals))
        # print("found duplicate goals {}".format(goals))
        return bidirectional_ucs1(graph,goals[0],goals[1])
    
    #search from A 
    frontier_A = PriorityQueue()
    explored_A = set(goals[0])
    current_node_A= None
    frontier_A.append((0,goals[0]))
    branch_A = {}
    #searh from B 
    frontier_B = PriorityQueue()
    explored_B = set(goals[1])
    current_node_B = None
    frontier_B.append((0,goals[1]))
    branch_B = {}
    #search from C 
    frontier_C = PriorityQueue()
    explored_C = set(goals[2])
    current_node_C= None
    frontier_C.append((0,goals[2]))
    branch_C = {}

    #Flag to to alternate between searchs (0=A,1=B,2=C)
    branch2search= 0
    #flag to check if path is found
    found_path = False
    #intersection nodes
    intersection_nodes = None
    intersection_node = None
    while True:
        if (explored_A.intersection(explored_B)).intersection(explored_C) :   
                found_path = True
                frontier_A_set = set([x[-1] for x in frontier_A])
                frontier_B_set = set([x[-1] for x in frontier_B])
                frontier_C_set = set([x[-1] for x in frontier_C])
                branch_A_union = explored_A.union(frontier_A_set)
                branch_B_union = explored_B.union(frontier_B_set)
                branch_C_union = explored_C.union(frontier_C_set)
                frontier_B_set = set([x[-1] for x in frontier_B])
                intersection_nodes = list(branch_B_union.intersection(branch_A_union.intersection(branch_C_union))) 
                break
        
        if (branch2search%3)==0:
            _, _ , current_node_A = frontier_A.pop() 
            explored_A.add(current_node_A)
            
            
            if current_node_A == goals[0]:
                current_cost_A = 0.0
            else:
                current_cost_A = branch_A[current_node_A][0]
            

            
            
            # print(graph[current_node])
            for neighbour in sorted(graph.neighbors(current_node_A)): #the queue is structured as (priority, counter,node)
                neighbour_cost = graph.get_edge_weight(current_node_A,neighbour)
        
                cost_total_A = current_cost_A + neighbour_cost
                if neighbour not in frontier_A and neighbour not in explored_A:
                    frontier_A.append((cost_total_A, neighbour))
                    branch_A [neighbour] = (cost_total_A, current_node_A) #add the parent branch
                    
                elif neighbour in frontier_A and cost_total_A < branch_A[neighbour][0]:
                    #how to remove while not knowing the counter number?
                    frontier_A.append((cost_total_A,neighbour))#is it okay to add without removing
                    branch_A [neighbour] = (cost_total_A, current_node_A) #add the parent branch 
            #alternate to the backward search 
            branch2search += 1 
        elif (branch2search%3)==1:
            _, _ , current_node_B = frontier_B.pop() 
            explored_B.add(current_node_B)
            
            
            if current_node_B == goals[1]:
                current_cost_B = 0.0
            else:
                current_cost_B = branch_B[current_node_B][0]
            

            
            
            # print(graph[current_node])
            for neighbour in sorted(graph.neighbors(current_node_B)): #the queue is structured as (priority, counter,node)
                neighbour_cost = graph.get_edge_weight(current_node_B,neighbour)
        
                cost_total_B = current_cost_B + neighbour_cost
                if neighbour not in frontier_B and neighbour not in explored_B:
                    frontier_B.append((cost_total_B, neighbour))
                    branch_B [neighbour] = (cost_total_B, current_node_B) #add the parent branch
                    
                elif neighbour in frontier_B and cost_total_B < branch_B[neighbour][0]:
                    #how to remove while not knowing the counter number?
                    frontier_B.append((cost_total_B,neighbour))#is it okay to add without removing
                    branch_B [neighbour] = (cost_total_B, current_node_B) #add the parent branch 
            branch2search += 1
        elif (branch2search%3) ==2:
            _, _ , current_node_C = frontier_C.pop() 
            explored_C.add(current_node_C)
            
            if current_node_C == goals[2]:
                current_cost_C = 0.0
            else:
                current_cost_C = branch_C[current_node_C][0]
            

            
            
            # print(graph[current_node])
            for neighbour in sorted(graph.neighbors(current_node_C)): #the queue is structured as (priority, counter,node)
                neighbour_cost = graph.get_edge_weight(current_node_C,neighbour)
        
                cost_total_C = current_cost_C + neighbour_cost
                if neighbour not in frontier_C and neighbour not in explored_C:
                    frontier_C.append((cost_total_C, neighbour))
                    branch_C [neighbour] = (cost_total_C, current_node_C) #add the parent branch
                    
                elif neighbour in frontier_C and cost_total_C < branch_C[neighbour][0]:
                    #how to remove while not knowing the counter number?
                    frontier_C.append((cost_total_C,neighbour))#is it okay to add without removing
                    branch_C [neighbour] = (cost_total_C, current_node_C) #add the parent branch  
            branch2search += 1
    
    if found_path:
        paths = []
        costs = []
        for node in intersection_nodes:
            path , cost = find_path_tri(graph,branch_A,branch_B,branch_C,explored_A,explored_B ,goals, node)
            paths.append(path)
            costs.append(cost)
        
        path = paths[costs.index(min(costs))]   
            
        print(intersection_nodes)
        print(paths)
        print(costs)
        return path
    raise NotImplementedError

def find_path_tri(graph, branch_A,branch_B,branch_C,explored_A, explored_B,goals, intersection_node):
    path = []
    if intersection_node == goals[0]:
        n = intersection_node
        path.append(n)
        #to avoid duplicate path, more check statement is added  (if the intersection is the start or end, it may result in duplicate nodes in the path)
        if n != goals[1]: #btw, this must be false
            while branch_B [n][1] != goals[1]:
                path.append(branch_B[n][1])
                n = branch_B[n][1]
            path.append(goals[1])
            path.reverse()
        n = intersection_node
        if n != goals[2]:
            while branch_C[n][1] != goals[2]:
                path.append(branch_C[n][1])
                n = branch_C[n][1]
            path.append(goals[2]) #now path should contain path from intersection to goal 
    elif intersection_node == goals [1]:
        n = intersection_node
        path.append(n)
        #to avoid duplicate path, more check statement is added  (if the intersection is the start or end, it may result in duplicate nodes in the path)
        if n != goals[0]: #btw, this must be false
            while branch_A [n][1] != goals[0]:
                path.append(branch_A[n][1])
                n = branch_A[n][1]
            path.append(goals[0])
            path.reverse()
        n = intersection_node
        if n != goals[2]:
            while branch_C[n][1] != goals[2]:
                path.append(branch_C[n][1])
                n = branch_C[n][1]
            path.append(goals[2]) #now path should contain path from intersection to goal 
    elif intersection_node == goals[2]:
        n = intersection_node
        path.append(n)
        #to avoid duplicate path, more check statement is added  (if the intersection is the start or end, it may result in duplicate nodes in the path)
        if n != goals[0]: #btw, this must be false
            while branch_A [n][1] != goals[0]:
                path.append(branch_A[n][1])
                n = branch_A[n][1]
            path.append(goals[0])
            path.reverse()
        n = intersection_node
        if n != goals[1]:
            while branch_B[n][1] != goals[1]:
                path.append(branch_B[n][1])
                n = branch_B[n][1]
            path.append(goals[1]) #now path should contain path from intersection to goal
    else:
        #go from A to B directly 
        if goals[1] in explored_A:
            n = goals[1]
            path_temp = []
            while branch_A [n][1] != goals[0]:
                path_temp.append(branch_A[n][1])
                n = branch_A[n][1]
            path_temp.append(goals[0])
            path_temp.reverse()
            path += path_temp
            path.append(goals[1])

        else:
            #from A to intersection
            n = intersection_node
            path.append(n)
            while branch_A [n][1] != goals[0]:
                path.append(branch_A[n][1])
                n = branch_A[n][1]
            path.append(goals[0])
            path.reverse()
            #from intersection to B
            n = intersection_node
            path_temp = []
            if n != goals[1]:
                while branch_B[n][1] != goals[1]:
                    path_temp.append(branch_B[n][1])
                    n = branch_B[n][1]
                #path_tem will have full from intersectino to B (exclusive of the intersection and goal[2])
                path += path_temp
                path.append(goals[1]) #now path should be from A to B
        
        #check if you can go from B to C directly 
        if goals[2] in explored_B:
            n = goals[2]
            path_temp = []
            while branch_B [n][1] != goals[1]:
                path_temp.append(branch_B[n][1])
                n = branch_B[n][1]
            path_temp.reverse()
            path += path_temp
            path.append(goals[2])
        else:    
            #from B to intersection
            path_temp.reverse()
            path += path_temp 
            path.append(intersection_node) # now path should be: A-->intersection-->B --> intersection 
            n = intersection_node 

            #from intersection to C
            while branch_C [n][1] != goals[2]:
                path.append(branch_C[n][1])
                n = branch_C[n][1]
            path.append(goals[2])

    cost = 0 
    for i in range(len(path)-1):
        cost += graph.get_edge_weight(path[i],path[i+1])
        
    return path, cost 

def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    '''
    The main idea of this tri-A* is that it start with the min heuristic from both goals but once two goal intersects, the focus sololy on the remaining goal.
    '''
    if len(set(goals)) ==1:
        return []
    elif  len(set(goals)) == 2:
        goals = list(set(goals))
        # print("found duplicate goals {}".format(goals))
        path = bidirectional_a_star1(graph,goals[0],goals[1])
        # print("tri-A*: goals {} and path: {}".format(goals,path))
        # print(path)
        return path
    
    #search from A 
    frontier_A = PriorityQueue()
    explored_A = set(goals[0])
    current_node_A= None
    frontier_A.append((0,goals[0]))
    branch_A = {}
    #searh from B 
    frontier_B = PriorityQueue()
    explored_B = set(goals[1])
    current_node_B = None
    frontier_B.append((0,goals[1]))
    branch_B = {}
    #search from C 
    frontier_C = PriorityQueue()
    explored_C = set(goals[2])
    current_node_C= None
    frontier_C.append((0,goals[2]))
    branch_C = {}

    #Flag to to alternate between searchs (0=A,1=B,2=C)
    branch2search= 0
    #flag to check if path is found
    found_path = False
    #intersection nodes
    intersection_nodes = None
    intersection_node_AB = None
    intersection_cost_AB = None
    intersection_node_AC = None
    intersection_cost_AC = None
    intersection_node_BC = None
    intersection_cost_BC = None
    #flag to stop exploring branches
    stop_A = False
    stop_B = False
    stop_C = False
    while True:
        
        if branch2search==0 and stop_A == False:
            _, _ , current_node_A = frontier_A.pop() 
            # print("A: ",current_node_A)
            explored_A.add(current_node_A)
            if explored_A.intersection(explored_B) and intersection_node_AB is None:
                #finding the optimal path from A -- >B , B-->A
                frontier_B_set = set([x[-1] for x in frontier_B])
                intersection_nodes = list(explored_A.intersection(explored_B.union(frontier_B_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[0]:
                        #the cost to the from start to start is 0
                        cost = branch_B[node][0]
                    elif node is goals[1]:
                        cost = branch_A[node][0]
                    else:
                        cost = branch_A[node][0] + branch_B[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_AB =intersection_nodes[intersection_node_index]
                intersection_cost_AB = min(intersections_cost)
                #update froniter A and B to find C 
                frontier_A_temp = PriorityQueue()
                frontier_B_temp = PriorityQueue()
                for node in frontier_A:
                    cost = branch_A[node[-1]][0] + heuristic(graph,node[-1],goals[2])
                    frontier_A_temp.append((cost,node[-1]))
                frontier_A = frontier_A_temp
                for node in frontier_B:
                    cost = branch_B[node[-1]][0] + heuristic(graph,node[-1],goals[2])
                    frontier_B_temp.append((cost,node[-1]))
                frontier_B = frontier_B_temp
            if explored_A.intersection(explored_C) and intersection_node_AC is None:
                #finding the optimal path from A -- >C , B-->C
                frontier_C_set = set([x[-1] for x in frontier_C])
                intersection_nodes = list(explored_A.intersection(explored_C.union(frontier_C_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[0]:
                        #the cost to the from start to start is 0
                        cost = branch_C[node][0]
                    elif node is goals[2]:
                        cost = branch_A[node][0]
                    else:
                        cost = branch_A[node][0] + branch_C[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_AC =intersection_nodes[intersection_node_index]
                intersection_cost_AC = min(intersections_cost)
                # stop_A = True
                #update froniter A and C so that they focus on B
                frontier_A_temp = PriorityQueue()
                frontier_C_temp = PriorityQueue()
                for node in frontier_A:
                    cost = branch_A[node[-1]][0] + heuristic(graph,node[-1],goals[1])
                    frontier_A_temp.append((cost,node[-1]))
                frontier_A = frontier_A_temp
                for node in frontier_C:
                    cost = branch_C[node[-1]][0] + heuristic(graph,node[-1],goals[1])
                    frontier_C_temp.append((cost,node[-1]))
                frontier_C = frontier_C_temp
            if explored_A.intersection(explored_B) and explored_A.intersection(explored_C):
                #which means the optimal solution for AB and AC have been already calculated
                stop_A = True
                
                
            else:

                if current_node_A == goals[0]:
                    current_cost_A = 0.0
                else:
                    current_cost_A = branch_A[current_node_A][0]
                
                # print(graph[current_node])
                for neighbour in sorted(graph.neighbors(current_node_A)): #the queue is structured as (priority, counter,node)
                    neighbour_cost = graph.get_edge_weight(current_node_A,neighbour)
                    cost_total_A = current_cost_A + neighbour_cost
                    if intersection_node_AB is not None:
                        h = heuristic(graph,neighbour,goals[2])
                    elif intersection_node_AC is not None:
                        h = heuristic(graph,neighbour,goals[1])
                    else:
                        h2 = heuristic(graph,neighbour,goals[2])
                        h1 = heuristic(graph,neighbour,goals[1])
                        if h1 >= h2:
                            h = h2
                        else:
                            h = h1
                        # h = h1 + h2
                    f = cost_total_A + h 
                    if neighbour not in frontier_A and neighbour not in explored_A:
                        frontier_A.append((f, neighbour))
                        branch_A [neighbour] = (cost_total_A, current_node_A) #add the parent branch
                        
                    elif neighbour in frontier_A and cost_total_A < branch_A[neighbour][0]:
                        #how to remove while not knowing the counter number?
                        frontier_A.append((f,neighbour))#is it okay to add without removing
                        branch_A [neighbour] = (cost_total_A, current_node_A) #add the parent branch 
                #alternate to the backward search 
             
        elif branch2search==1 and stop_B ==False:
            _, _ , current_node_B = frontier_B.pop() 
            # print("B: ",current_node_B)
            explored_B.add(current_node_B)
            if explored_B.intersection(explored_A) and intersection_node_AB is None:
                #finding the optimal path from A -- >B , B-->A
                frontier_A_set = set([x[-1] for x in frontier_A])
                intersection_nodes = list(explored_B.intersection(explored_A.union(frontier_A_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[0]:
                        #the cost to the from start to start is 0
                        cost = branch_B[node][0]
                    elif node is goals[1]:
                        cost = branch_A[node][0]
                    else:
                        cost = branch_A[node][0] + branch_B[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_AB =intersection_nodes[intersection_node_index]
                intersection_cost_AB = min(intersections_cost)
                # stop_B = True

                #update froniter A and B so that they focus on C
                frontier_A_temp = PriorityQueue()
                frontier_B_temp = PriorityQueue()
                for node in frontier_A:
                    cost = branch_A[node[-1]][0] + heuristic(graph,node[-1],goals[2])
                    frontier_A_temp.append((cost,node[-1]))
                frontier_A = frontier_A_temp
                for node in frontier_B:
                    cost = branch_B[node[-1]][0] + heuristic(graph,node[-1],goals[2])
                    frontier_B_temp.append((cost,node[-1]))
                frontier_B = frontier_B_temp

            if explored_B.intersection(explored_C) and intersection_node_BC is None:
                #finding the optimal path from B -- >C , C-->B
                frontier_C_set = set([x[-1] for x in frontier_C])
                intersection_nodes = list(explored_B.intersection(explored_C.union(frontier_C_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[1]:
                        #the cost to the from start to start is 0
                        cost = branch_C[node][0]
                    elif node is goals[2]:
                        cost = branch_B[node][0]
                    else:
                        cost = branch_B[node][0] + branch_C[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_BC =intersection_nodes[intersection_node_index]
                intersection_cost_BC = min(intersections_cost)
                # stop_B = True
                #update froniter B and C so that they focus on A
                frontier_B_temp = PriorityQueue()
                frontier_C_temp = PriorityQueue()
                for node in frontier_B:
                    cost = branch_B[node[-1]][0] + heuristic(graph,node[-1],goals[0])
                    frontier_B_temp.append((cost,node[-1]))
                frontier_B = frontier_B_temp
                for node in frontier_C:
                    cost = branch_C[node[-1]][0] + heuristic(graph,node[-1],goals[0])
                    frontier_C_temp.append((cost,node[-1]))
                frontier_C = frontier_C_temp
            if explored_B.intersection(explored_C) and explored_B.intersection(explored_A):
                #which means the optimal solution for AB and AC have been already calculated
                stop_B = True

            else:    
            
                if current_node_B == goals[1]:
                    current_cost_B = 0.0
                else:
                    current_cost_B = branch_B[current_node_B][0]
                

                
                
                # print(graph[current_node])
                for neighbour in sorted(graph.neighbors(current_node_B)): #the queue is structured as (priority, counter,node)
                    neighbour_cost = graph.get_edge_weight(current_node_B,neighbour)
                    cost_total_B = current_cost_B + neighbour_cost
                    if intersection_node_AB is not None:
                        h = heuristic(graph,neighbour,goals[2])
                    elif intersection_node_BC is not None:
                        h = heuristic(graph,neighbour,goals[0])
                    else:
                        h2 = heuristic(graph,neighbour,goals[2])
                        h1 = heuristic(graph,neighbour,goals[0])
                        if h1 >= h2:
                            h = h2
                        else:
                            h = h1
                        # h = h1 + h2
                    f = cost_total_B + h 
                    if neighbour not in frontier_B and neighbour not in explored_B:
                        frontier_B.append((f, neighbour))
                        branch_B [neighbour] = (cost_total_B, current_node_B) #add the parent branch
                        
                    elif neighbour in frontier_B and cost_total_B < branch_B[neighbour][0]:
                        #how to remove while not knowing the counter number?
                        frontier_B.append((f,neighbour))#is it okay to add without removing
                        branch_B [neighbour] = (cost_total_B, current_node_B) #add the parent branch 
            
        elif branch2search ==2 and stop_C == False:
            _, _ , current_node_C = frontier_C.pop() 
            # print("C: ",current_node_C)
            explored_C.add(current_node_C)
            if explored_C.intersection(explored_A) and intersection_node_AC is None:
                #finding the optimal path from C -- >A , A-->C
                frontier_A_set = set([x[-1] for x in frontier_A])
                intersection_nodes = list(explored_C.intersection(explored_A.union(frontier_A_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[0]:
                        #the cost to the from start to start is 0
                        cost = branch_C[node][0]
                    elif node is goals[2]:
                        cost = branch_A[node][0]
                    else:
                        cost = branch_A[node][0] + branch_C[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_AC =intersection_nodes[intersection_node_index]
                intersection_cost_AC = min(intersections_cost)
                # stop_C = True
                #update froniter A and C so that they focus on B
                frontier_A_temp = PriorityQueue()
                frontier_C_temp = PriorityQueue()
                for node in frontier_A:
                    cost = branch_A[node[-1]][0] + heuristic(graph,node[-1],goals[1])
                    frontier_A_temp.append((cost,node[-1]))
                frontier_A = frontier_A_temp
                for node in frontier_C:
                    cost = branch_C[node[-1]][0] + heuristic(graph,node[-1],goals[1])
                    frontier_C_temp.append((cost,node[-1]))
                frontier_C = frontier_C_temp
            if explored_C.intersection(explored_B) and intersection_node_BC is None:
                #finding the optimal path from B -- >C , C-->B
                frontier_B_set = set([x[-1] for x in frontier_B])
                intersection_nodes = list(explored_C.intersection(explored_B.union(frontier_B_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[1]:
                        #the cost to the from start to start is 0
                        cost = branch_C[node][0]
                    elif node is goals[2]:
                        cost = branch_B[node][0]
                    else:
                        cost = branch_B[node][0] + branch_C[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_BC =intersection_nodes[intersection_node_index]
                intersection_cost_BC = min(intersections_cost)
                # stop_C = True
                #update froniter B and C so that they focus on A
                frontier_B_temp = PriorityQueue()
                frontier_C_temp = PriorityQueue()
                for node in frontier_B:
                    cost = branch_B[node[-1]][0] + heuristic(graph,node[-1],goals[0])
                    frontier_B_temp.append((cost,node[-1]))
                frontier_B = frontier_B_temp
                for node in frontier_C:
                    cost = branch_C[node[-1]][0] + heuristic(graph,node[-1],goals[0])
                    frontier_C_temp.append((cost,node[-1]))
                frontier_C = frontier_C_temp
            if explored_C.intersection(explored_A) and explored_C.intersection(explored_C):
                #which means the optimal solution for AB and AC have been already calculated
                stop_C = True
            
            else:
                if current_node_C == goals[2]:
                    current_cost_C = 0.0
                else:
                    current_cost_C = branch_C[current_node_C][0]
                
                

                
                
                # print(graph[current_node])
                for neighbour in sorted(graph.neighbors(current_node_C)): #the queue is structured as (priority, counter,node)
                    neighbour_cost = graph.get_edge_weight(current_node_C,neighbour)
                    cost_total_C = current_cost_C + neighbour_cost
                    if intersection_node_AC is not None:
                        h = heuristic(graph,neighbour,goals[1])
                    elif intersection_node_BC is not None:
                        h = heuristic(graph,neighbour,goals[0])
                    else:
                        h2 = heuristic(graph,neighbour,goals[1])
                        h1 = heuristic(graph,neighbour,goals[0])
                        if h1 >= h2:
                            h = h2
                        else:
                            h = h1
                        # h = h1 + h2
                    f = cost_total_C + h
                    if neighbour not in frontier_C and neighbour not in explored_C:
                        frontier_C.append((f, neighbour))
                        branch_C [neighbour] = (cost_total_C, current_node_C) #add the parent branch
                        
                    elif neighbour in frontier_C and cost_total_C < branch_C[neighbour][0]:
                        #how to remove while not knowing the counter number?
                        frontier_C.append((f,neighbour))#is it okay to add without removing
                        branch_C [neighbour] = (cost_total_C, current_node_C) #add the parent branch  


        if intersection_node_AC is not None and intersection_node_AB is not None and intersection_node_BC is not None:
            found_path = True    
            break
        else:
            #continue exploring with least cost node 
            costs = []
            if stop_A:
                costs.append(math.inf)
            else:
                costs.append(frontier_A.top()[0])
            if stop_B:
                costs.append(math.inf)
            else:
                costs.append(frontier_B.top()[0])
            if stop_C:
                costs.append(math.inf)
            else:
                costs.append(frontier_C.top()[0])
            
            # print(costs)
            branch2search = costs.index(min(costs))
    
    if found_path:
        # print("soluiton has been found")
        path = []
        # path_AB = find_path_bidir(goals[0],goals[1],intersection_node_AB,branch_A,branch_B)
        # path_AC = find_path_bidir(goals[0],goals[2],intersection_node_AC,branch_A,branch_C)
        # path_BC = find_path_bidir(goals[1],goals[2],intersection_node_BC,branch_B,branch_C)

        #There are 3 possible combination of paths: ABC=CBA= AB + BC , and ACB=BCA = AC + CB and BAC=CAB = BA + AC 
        paths_cost = []
        cost_ABC= intersection_cost_AB + intersection_cost_BC
        cost_ACB= intersection_cost_AC + intersection_cost_BC
        cost_BAC = intersection_cost_AB + intersection_cost_AC
        paths_cost.append(cost_ABC)
        paths_cost.append(cost_ACB)
        paths_cost.append(cost_BAC)
        # print(paths_cost)
        best_path = paths_cost.index(min(paths_cost))
        if best_path == 0: #ABC=CBA
            path_AB = find_path_bidir(goals[0],goals[1],intersection_node_AB,branch_A,branch_B)
            path += path_AB[:-1] #B will be included next 
            path_BC = find_path_bidir(goals[1],goals[2],intersection_node_BC,branch_B,branch_C)
            path += path_BC
        elif best_path ==1: # best path is ACB=BCA
            path_AC = find_path_bidir(goals[0],goals[2],intersection_node_AC,branch_A,branch_C)
            path += path_AC[:-1] # C will be inculded next
            path_CB = find_path_bidir(goals[2],goals[1],intersection_node_BC,branch_C,branch_B)
            path += path_CB
        elif best_path == 2: #best path is BAC=CAB 
            path_BA = find_path_bidir(goals[1],goals[0],intersection_node_AB,branch_B,branch_A)
            path += path_BA[:-1]     
            path_AC = find_path_bidir(goals[0],goals[2],intersection_node_AC,branch_A,branch_C)
            path += path_AC
        # print(" tri-A*: goals {} and path: {}".format(goals,path))
        return path
    raise NotImplementedError
def tridirectional_upgraded2(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    '''
    The main idea of this tri-A* is that it start with the min heuristic from both goals but once two goal intersects, the focus sololy on the remaining goal.
    '''
    if len(set(goals)) ==1:
        return []
    elif  len(set(goals)) == 2:
        goals = list(set(goals))
        # print("found duplicate goals {}".format(goals))
        path = bidirectional_a_star1(graph,goals[0],goals[1])
        # print("tri-A*: goals {} and path: {}".format(goals,path))
        # print(path)
        return path
    
    #search from A 
    frontier_A = PriorityQueue()
    explored_A = set(goals[0])
    current_node_A= None
    frontier_A.append((0,goals[0]))
    branch_A = {}
    #searh from B 
    frontier_B = PriorityQueue()
    explored_B = set(goals[1])
    current_node_B = None
    frontier_B.append((0,goals[1]))
    branch_B = {}
    #search from C 
    frontier_C = PriorityQueue()
    explored_C = set(goals[2])
    current_node_C= None
    frontier_C.append((0,goals[2]))
    branch_C = {}

    #Flag to to alternate between searchs (0=A,1=B,2=C)
    branch2search= 0
    #flag to check if path is found
    found_path = False
    #intersection nodes
    intersection_nodes = None
    intersection_node_AB = None
    intersection_cost_AB = None
    intersection_node_AC = None
    intersection_cost_AC = None
    intersection_node_BC = None
    intersection_cost_BC = None
    #flag to stop exploring branches
    stop_A = False
    stop_B = False
    stop_C = False
    while True:
        
        if branch2search==0 and stop_A == False:
            _, _ , current_node_A = frontier_A.pop() 
            # print("A: ",current_node_A)
            explored_A.add(current_node_A)
            if explored_A.intersection(explored_B) and intersection_node_AB is None:
                #finding the optimal path from A -- >B , B-->A
                frontier_B_set = set([x[-1] for x in frontier_B])
                intersection_nodes = list(explored_A.intersection(explored_B.union(frontier_B_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[0]:
                        #the cost to the from start to start is 0
                        cost = branch_B[node][0]
                    elif node is goals[1]:
                        cost = branch_A[node][0]
                    else:
                        cost = branch_A[node][0] + branch_B[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_AB =intersection_nodes[intersection_node_index]
                intersection_cost_AB = min(intersections_cost)
                #update froniter A and B to find C 
                frontier_A_temp = PriorityQueue()
                frontier_B_temp = PriorityQueue()
                for node in frontier_A:
                    cost = branch_A[node[-1]][0] + heuristic(graph,node[-1],goals[2])
                    frontier_A_temp.append((cost,node[-1]))
                frontier_A = frontier_A_temp
                for node in frontier_B:
                    cost = branch_B[node[-1]][0] + heuristic(graph,node[-1],goals[2])
                    frontier_B_temp.append((cost,node[-1]))
                frontier_B = frontier_B_temp
            if explored_A.intersection(explored_C) and intersection_node_AC is None:
                #finding the optimal path from A -- >C , B-->C
                frontier_C_set = set([x[-1] for x in frontier_C])
                intersection_nodes = list(explored_A.intersection(explored_C.union(frontier_C_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[0]:
                        #the cost to the from start to start is 0
                        cost = branch_C[node][0]
                    elif node is goals[2]:
                        cost = branch_A[node][0]
                    else:
                        cost = branch_A[node][0] + branch_C[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_AC =intersection_nodes[intersection_node_index]
                intersection_cost_AC = min(intersections_cost)
                # stop_A = True
                #update froniter A and C so that they focus on B
                frontier_A_temp = PriorityQueue()
                frontier_C_temp = PriorityQueue()
                for node in frontier_A:
                    cost = branch_A[node[-1]][0] + heuristic(graph,node[-1],goals[1])
                    frontier_A_temp.append((cost,node[-1]))
                frontier_A = frontier_A_temp
                for node in frontier_C:
                    cost = branch_C[node[-1]][0] + heuristic(graph,node[-1],goals[1])
                    frontier_C_temp.append((cost,node[-1]))
                frontier_C = frontier_C_temp
            if explored_A.intersection(explored_B) and explored_A.intersection(explored_C):
                #which means the optimal solution for AB and AC have been already calculated
                stop_A = True
                
                
            else:

                if current_node_A == goals[0]:
                    current_cost_A = 0.0
                else:
                    current_cost_A = branch_A[current_node_A][0]
                
                # print(graph[current_node])
                for neighbour in sorted(graph.neighbors(current_node_A)): #the queue is structured as (priority, counter,node)
                    neighbour_cost = graph.get_edge_weight(current_node_A,neighbour)
                    cost_total_A = current_cost_A + neighbour_cost
                    if intersection_node_AB is not None:
                        h = heuristic(graph,neighbour,goals[2])
                    elif intersection_node_AC is not None:
                        h = heuristic(graph,neighbour,goals[1])
                    else:
                        h2 = heuristic(graph,neighbour,goals[2])
                        h1 = heuristic(graph,neighbour,goals[1])
                        if h1 >= h2:
                            h = h2
                        else:
                            h = h1
                        # h = h1 + h2
                    f = cost_total_A + h 
                    if neighbour not in frontier_A and neighbour not in explored_A:
                        frontier_A.append((f, neighbour))
                        branch_A [neighbour] = (cost_total_A, current_node_A) #add the parent branch
                        
                    elif neighbour in frontier_A and cost_total_A < branch_A[neighbour][0]:
                        #how to remove while not knowing the counter number?
                        frontier_A.append((f,neighbour))#is it okay to add without removing
                        branch_A [neighbour] = (cost_total_A, current_node_A) #add the parent branch 
                #alternate to the backward search 
             
        elif branch2search==1 and stop_B ==False:
            _, _ , current_node_B = frontier_B.pop() 
            # print("B: ",current_node_B)
            explored_B.add(current_node_B)
            if explored_B.intersection(explored_A) and intersection_node_AB is None:
                #finding the optimal path from A -- >B , B-->A
                frontier_A_set = set([x[-1] for x in frontier_A])
                intersection_nodes = list(explored_B.intersection(explored_A.union(frontier_A_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[0]:
                        #the cost to the from start to start is 0
                        cost = branch_B[node][0]
                    elif node is goals[1]:
                        cost = branch_A[node][0]
                    else:
                        cost = branch_A[node][0] + branch_B[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_AB =intersection_nodes[intersection_node_index]
                intersection_cost_AB = min(intersections_cost)
                # stop_B = True

                #update froniter A and B so that they focus on C
                frontier_A_temp = PriorityQueue()
                frontier_B_temp = PriorityQueue()
                for node in frontier_A:
                    cost = branch_A[node[-1]][0] + heuristic(graph,node[-1],goals[2])
                    frontier_A_temp.append((cost,node[-1]))
                frontier_A = frontier_A_temp
                for node in frontier_B:
                    cost = branch_B[node[-1]][0] + heuristic(graph,node[-1],goals[2])
                    frontier_B_temp.append((cost,node[-1]))
                frontier_B = frontier_B_temp

            if explored_B.intersection(explored_C) and intersection_node_BC is None:
                #finding the optimal path from B -- >C , C-->B
                frontier_C_set = set([x[-1] for x in frontier_C])
                intersection_nodes = list(explored_B.intersection(explored_C.union(frontier_C_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[1]:
                        #the cost to the from start to start is 0
                        cost = branch_C[node][0]
                    elif node is goals[2]:
                        cost = branch_B[node][0]
                    else:
                        cost = branch_B[node][0] + branch_C[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_BC =intersection_nodes[intersection_node_index]
                intersection_cost_BC = min(intersections_cost)
                # stop_B = True
                #update froniter B and C so that they focus on A
                frontier_B_temp = PriorityQueue()
                frontier_C_temp = PriorityQueue()
                for node in frontier_B:
                    cost = branch_B[node[-1]][0] + heuristic(graph,node[-1],goals[0])
                    frontier_B_temp.append((cost,node[-1]))
                frontier_B = frontier_B_temp
                for node in frontier_C:
                    cost = branch_C[node[-1]][0] + heuristic(graph,node[-1],goals[0])
                    frontier_C_temp.append((cost,node[-1]))
                frontier_C = frontier_C_temp
            if explored_B.intersection(explored_C) and explored_B.intersection(explored_A):
                #which means the optimal solution for AB and AC have been already calculated
                stop_B = True

            else:    
            
                if current_node_B == goals[1]:
                    current_cost_B = 0.0
                else:
                    current_cost_B = branch_B[current_node_B][0]
                

                
                
                # print(graph[current_node])
                for neighbour in sorted(graph.neighbors(current_node_B)): #the queue is structured as (priority, counter,node)
                    neighbour_cost = graph.get_edge_weight(current_node_B,neighbour)
                    cost_total_B = current_cost_B + neighbour_cost
                    if intersection_node_AB is not None:
                        h = heuristic(graph,neighbour,goals[2])
                    elif intersection_node_BC is not None:
                        h = heuristic(graph,neighbour,goals[0])
                    else:
                        h2 = heuristic(graph,neighbour,goals[2])
                        h1 = heuristic(graph,neighbour,goals[0])
                        if h1 >= h2:
                            h = h2
                        else:
                            h = h1
                        # h = h1 + h2
                    f = cost_total_B + h 
                    if neighbour not in frontier_B and neighbour not in explored_B:
                        frontier_B.append((f, neighbour))
                        branch_B [neighbour] = (cost_total_B, current_node_B) #add the parent branch
                        
                    elif neighbour in frontier_B and cost_total_B < branch_B[neighbour][0]:
                        #how to remove while not knowing the counter number?
                        frontier_B.append((f,neighbour))#is it okay to add without removing
                        branch_B [neighbour] = (cost_total_B, current_node_B) #add the parent branch 
            
        elif branch2search ==2 and stop_C == False:
            _, _ , current_node_C = frontier_C.pop() 
            # print("C: ",current_node_C)
            explored_C.add(current_node_C)
            if explored_C.intersection(explored_A) and intersection_node_AC is None:
                #finding the optimal path from C -- >A , A-->C
                frontier_A_set = set([x[-1] for x in frontier_A])
                intersection_nodes = list(explored_C.intersection(explored_A.union(frontier_A_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[0]:
                        #the cost to the from start to start is 0
                        cost = branch_C[node][0]
                    elif node is goals[2]:
                        cost = branch_A[node][0]
                    else:
                        cost = branch_A[node][0] + branch_C[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_AC =intersection_nodes[intersection_node_index]
                intersection_cost_AC = min(intersections_cost)
                # stop_C = True
                #update froniter A and C so that they focus on B
                frontier_A_temp = PriorityQueue()
                frontier_C_temp = PriorityQueue()
                for node in frontier_A:
                    cost = branch_A[node[-1]][0] + heuristic(graph,node[-1],goals[1])
                    frontier_A_temp.append((cost,node[-1]))
                frontier_A = frontier_A_temp
                for node in frontier_C:
                    cost = branch_C[node[-1]][0] + heuristic(graph,node[-1],goals[1])
                    frontier_C_temp.append((cost,node[-1]))
                frontier_C = frontier_C_temp
            if explored_C.intersection(explored_B) and intersection_node_BC is None:
                #finding the optimal path from B -- >C , C-->B
                frontier_B_set = set([x[-1] for x in frontier_B])
                intersection_nodes = list(explored_C.intersection(explored_B.union(frontier_B_set)))
                intersections_cost = []
                # print(branch_forward)
                for node in intersection_nodes:
                    if node is goals[1]:
                        #the cost to the from start to start is 0
                        cost = branch_C[node][0]
                    elif node is goals[2]:
                        cost = branch_B[node][0]
                    else:
                        cost = branch_B[node][0] + branch_C[node][0]
                    intersections_cost.append(cost)
                
                intersection_node_index = intersections_cost.index(min(intersections_cost))
                intersection_node_BC =intersection_nodes[intersection_node_index]
                intersection_cost_BC = min(intersections_cost)
                # stop_C = True
                #update froniter B and C so that they focus on A
                frontier_B_temp = PriorityQueue()
                frontier_C_temp = PriorityQueue()
                for node in frontier_B:
                    cost = branch_B[node[-1]][0] + heuristic(graph,node[-1],goals[0])
                    frontier_B_temp.append((cost,node[-1]))
                frontier_B = frontier_B_temp
                for node in frontier_C:
                    cost = branch_C[node[-1]][0] + heuristic(graph,node[-1],goals[0])
                    frontier_C_temp.append((cost,node[-1]))
                frontier_C = frontier_C_temp
            if explored_C.intersection(explored_A) and explored_C.intersection(explored_C):
                #which means the optimal solution for AB and AC have been already calculated
                stop_C = True
            
            else:
                if current_node_C == goals[2]:
                    current_cost_C = 0.0
                else:
                    current_cost_C = branch_C[current_node_C][0]
                
                

                
                
                # print(graph[current_node])
                for neighbour in sorted(graph.neighbors(current_node_C)): #the queue is structured as (priority, counter,node)
                    neighbour_cost = graph.get_edge_weight(current_node_C,neighbour)
                    cost_total_C = current_cost_C + neighbour_cost
                    if intersection_node_AC is not None:
                        h = heuristic(graph,neighbour,goals[1])
                    elif intersection_node_BC is not None:
                        h = heuristic(graph,neighbour,goals[0])
                    else:
                        h2 = heuristic(graph,neighbour,goals[1])
                        h1 = heuristic(graph,neighbour,goals[0])
                        if h1 >= h2:
                            h = h2
                        else:
                            h = h1
                        # h = h1 + h2
                    f = cost_total_C + h
                    if neighbour not in frontier_C and neighbour not in explored_C:
                        frontier_C.append((f, neighbour))
                        branch_C [neighbour] = (cost_total_C, current_node_C) #add the parent branch
                        
                    elif neighbour in frontier_C and cost_total_C < branch_C[neighbour][0]:
                        #how to remove while not knowing the counter number?
                        frontier_C.append((f,neighbour))#is it okay to add without removing
                        branch_C [neighbour] = (cost_total_C, current_node_C) #add the parent branch  


        if (intersection_node_AC is not None and intersection_node_AB) or (intersection_node_AC is not None and intersection_node_BC)  or  (intersection_node_BC is not None and intersection_node_AB):
            found_path = True    
            break
        else:
            #continue exploring with least cost node 
            costs = []
            if stop_A:
                costs.append(math.inf)
            else:
                costs.append(frontier_A.top()[0])
            if stop_B:
                costs.append(math.inf)
            else:
                costs.append(frontier_B.top()[0])
            if stop_C:
                costs.append(math.inf)
            else:
                costs.append(frontier_C.top()[0])
            
            # print(costs)
            branch2search = costs.index(min(costs))
    
    if found_path:
        # print("soluiton has been found")
        path = []
        

        #There are 3 possible combination of paths: ABC=CBA= AB + BC , and ACB=BCA = AC + CB and BAC=CAB = BA + AC 
        
        if intersection_node_BC is not None and intersection_node_AB == 0: #ABC=CBA
            path_AB = find_path_bidir(goals[0],goals[1],intersection_node_AB,branch_A,branch_B)
            path += path_AB[:-1] #B will be included next 
            path_BC = find_path_bidir(goals[1],goals[2],intersection_node_BC,branch_B,branch_C)
            path += path_BC
        elif intersection_node_AC is not None and intersection_node_BC: # best path is ACB=BCA
            path_AC = find_path_bidir(goals[0],goals[2],intersection_node_AC,branch_A,branch_C)
            path += path_AC[:-1] # C will be inculded next
            path_CB = find_path_bidir(goals[2],goals[1],intersection_node_BC,branch_C,branch_B)
            path += path_CB
        elif intersection_node_AC is not None and intersection_node_AB: #best path is BAC=CAB 
            path_BA = find_path_bidir(goals[1],goals[0],intersection_node_AB,branch_B,branch_A)
            path += path_BA[:-1]     
            path_AC = find_path_bidir(goals[0],goals[2],intersection_node_AC,branch_A,branch_C)
            path += path_AC
        # print(" tri-A*: goals {} and path: {}".format(goals,path))
        return path

def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Ali Alrasheed"
    raise NotImplementedError


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
