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
    best_intersection_node = None
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
                
                best_intersection_node_index = intersections_cost.index(min(intersections_cost))
                best_intersection_node =intersection_nodes[best_intersection_node_index] 
                
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
                
                best_intersection_node_index = intersections_cost.index(min(intersections_cost))
                best_intersection_node =intersection_nodes[best_intersection_node_index] 
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
        n = best_intersection_node
        path.append(n)
        if n != start:
            while branch_forward [n][1] != start:
                path.append(branch_forward[n][1])
                n = branch_forward[n][1]
            path.append(start)
            path.reverse()
        n = best_intersection_node
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
    best_intersection_node = None
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
                
                best_intersection_node_index = intersections_cost.index(min(intersections_cost))
                best_intersection_node =intersection_nodes[best_intersection_node_index] 
                
                break
            
            if current_node_forward == start:
                current_cost_forward = 0.0
            else:
                current_cost_forward = branch_forward[current_node_forward][0]
            

            
            
            # print(graph[current_node])
            for neighbour in sorted(graph.neighbors(current_node_forward)): #the queue is structured as (priority, counter,node)
                neighbour_cost = graph.get_edge_weight(current_cost_forward,neighbour)
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
                
                best_intersection_node_index = intersections_cost.index(min(intersections_cost))
                best_intersection_node =intersection_nodes[best_intersection_node_index] 
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
        n = best_intersection_node
        path.append(n)
        if n != start:
            while branch_forward [n][1] != start:
                path.append(branch_forward[n][1])
                n = branch_forward[n][1]
            path.append(start)
            path.reverse()
        n = best_intersection_node
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
    raise NotImplementedError


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
    raise NotImplementedError


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
