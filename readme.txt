UTA ID:1002059166
Full Name: Sravan Chandaka  

Programming language Used:Python 


To run the code:

1.Open the terminal and go to the path where the expense_8_puzzle.py file is saved 
2.In case of mine it is C:\Users\chand\Projects\AI (make sure start.txt, goal.txt is present)
3.Run the command as expense_8_puzzle.py start.txt goal.txt <method> <dump>, Here methods that can be used are(bfs,ucs,dfs,dls,ids,greedy,a*) and dump can be True or False.
4.If <method> and <dump> has no input given it takes default values as a* and false respectively.



Brief explaination of code:
1.Program based on 8puzzle program.
1.initial state, goal state, method and dump file inputs are taken from command line respectively (expense_8_puzzle.py start.txt goal.txt dls false).
2.initial and goal state inputs are converted to list using changelist() function.
3.get_blank_pos() function is used to get the postion of 0 in the 3D array.
4.indexucs() that takes a 2D list state representing the current state of a game board, and returns the position of the blank space
5.get_ucspath() that takes a 2D list state representing the current state of a game board, and returns a list of valid actions that can be taken to move the blank space (represented by the value 0) in the game board.
6.successordfs() function is used to generate the successors. Along with other required data
7.get_ucsstate that takes a 2D list state representing the current state of a game board, and a string action representing a valid action that can be taken to move the blank space in the game board. 
8.Search startgies used are Breadth-First Search(BFS), Greedy Search , Depth-First Search(DFS), Depth-limit Search(DLS), Uniform-Cost Search(UCS), Iterative-Deepening Search(IDS) and A* Search.
9.This function get_ucspath takes in a 3x3 state and returns a list of valid actions that can be taken from that state. It first finds the location of the blank tile in the state and then determines which directions are valid based on the location of the blank tile. If the blank tile is not on the top row, then the 'Up' action is valid. If the blank tile is not on the bottom row, then the 'Down' action is valid. If the blank tile is not on the leftmost column, then the 'Left' action is valid. If the blank tile is not on the rightmost column, then the 'Right' action is valid. 
10.For Greedy and A* Search heuristic function is used in the program it is defined as manhattan-distance() which calculates the heuristic to count the number of squares required to move each tile to its desired location based on the number of moves and cost of tile.
11.When the code is run, following information is displayed when the search is completed
	Nodes Popped: 
	Nodes expanded: 
	Nodes Genereated: 
	Max Fringe Size: 
	Solution Found at depth, cost, steps or moves
12.In case if the search is incomplete, it will display the No Solution found
13.While running the program if the dump value is true it will create a trace file and prints all the information into it.
14.In my code greedysearch that performs a search using the greedy best-first algorithm to find a path from a given start state to a goal state. It takes the start and goal states as input and returns various metrics about the search, including the number of nodes popped from the fringe, the number of nodes expanded, the maximum size of the fringe, the depth of the goal node, the cost of the path to the goal node, and the sequence of actions taken to reach the goal node.
15. A* search algorithm for finding the optimal path from a given start state to a goal state, using a priority queue to store nodes with a cost function based on the sum of the actual cost and a heuristic function. The code also includes logging functionality to write log entries to a file.
16. In iterative Deepening Search (IDS) algorithm it performs a series of depth-limited searches incrementally, increasing the depth limit with each iteration until a solution is found.
17. The depth-limited search (DLS) algorithm that limits the depth of the search tree, by exploring all possible paths at a given depth before moving on to the next depth level.




