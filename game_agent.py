#! /usr/bin/env python
#**********************************************************************
#* Name: game_agent (Isolation)                                       *
#*                                                                    *
#* Function: This module serves as an implementation of a game agent  *
#* for the board game Isolation. This agent is realized with the      *
#* implementation of the CustomPlayer defined herein. An instance of  *
#* this class represents an Isolation player that, given a Board      *
#* representation, is able to perform depth-first-search of a game    *
#* tree using adversarial search concepts to determine a best move    *
#* within a specified time limit.                                     *
#* The CustomPlayer instance is initialized with a maximum search     *
#* depth, a timeout value, a evaluation score function, a flag        *
#* indicating whether to make use of Iterative Deepening, and an      *
#* indicator of whether to use the Minimax or AlphaBeta algorithms    *
#* during search.                                                     *
#* This module features three difference evaluation score functions   *
#* which may be swapped in to the custom_score() function for actual  *
#* use by the game playing agent.                                     *
#*                                                                    *
#* Usage: Import this module to make use of the CustomPlayer class    *
#*                                                                    *
#* Written:  03/16/2017  James Damgar (Based on Udacity AIND content) *
#* Modified: 03/20/2017  JED  Added additional heuristic functions    *
#*                                                                    *
#**********************************************************************

"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Return our current favorite evaluation function
    #return simple_score(game, player)
    #return central_score(game, player)
    return partition_score(game, player)
    

def simple_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This function represents a "simple" evaluation score function which uses the
    following to score the state of the board for a player:
       
       - If the score of the game is positive or negative infinity based on "utility":
          - Then we've reached an end-game state
          - Return +inf for a maximizing player and -inf for a minimizing player
       - Otherwise, the score is the number of available moves for the current
         player minus 2 times the number of available moves for the opponent

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # First, see if we've reached an end-game situation
    #   +inf means this game state is a win for the maximizing player
    #   -inf means this game state is a loss for the minimizing player
    util = game.utility(player)
    
    # If we're at an endgame, then that's the heuristic score for this node
    if util != 0:
       return util
       
    # Otherwise, the heuristic is the difference in available moves between
    # the current player and the opposition
    return float(len(game.get_legal_moves(player)) - 2.0 * len(game.get_legal_moves(game.get_opponent(player))))
    
    
def central_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This function represents a somewhat more involved evaluation score function that
    takes into account how centrally-located the player is in addition to the factors
    which go into the simple_score() function. Score is determined as follows:
       
       - If the score of the game is positive or negative infinity based on "utility":
          - Then we've reached an end-game state
          - Return +inf for a maximizing player and -inf for a minimizing player
       - Otherwise, the score is the number of available moves for the current
         player minus 2 times the number of available moves for the opponent minus how far
         away the current player is from the center of the board

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # First, see if we've reached an end-game situation
    #   +inf means this game state is a win for the current player
    #   -inf means this game state is a loss for the current player
    util = game.utility(player)
    
    # If we're at an endgame, then that's the heuristic score for this node
    if util != 0:
       return util
       
    # Otherwise, the heuristic is the difference in available moves between
    # the current player and the opposition
    return float(len(game.get_legal_moves(player)) - 2.0 * len(game.get_legal_moves(game.get_opponent(player)))) - board_distance(game, player)
    
    
def partition_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This function represents a more complex evaluation function that takes into account
    whether there is a "partition" present on the board. If there is a partition, this
    means that each player is effectively on an "island" of squares and cannot reach the
    other player. We first check to see if there is a partition. If there is, then if a
    player has a greater number of contiguous squares on their "island" than the opponent,
    then that player should win. If the number of squares is tied, then the player whose
    turn it is will lose. If none of these apply, use the simple_score() heuristic. Mentioned
    earlier. If a player has a partition advantage, return the appropriate value (+/- inf).
    

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # First, see if we've reached an end-game situation
    #   +inf means this game state is a win for the current player
    #   -inf means this game state is a loss for the current player
    util = game.utility(player)
    
    # If we're at an endgame, then that's the heuristic score for this node
    if util != 0:
       return util
       
    # Next, check for a partition on the board.
    # Partitions are only possible if we have a certain number of moves that have occurred.
    if ( game.move_count >= 2 * game.height ) or ( game.move_count >= 2 * game.width ):
    
        # Grab the set of blank spaces and each player's position
        blank_spaces      = game.get_blank_spaces()
        player_location   = game.get_player_location(player)
        opponent_location = game.get_player_location(game.get_opponent(player))
        
        # Find all partitions on the game board as lines where each is a list of the form: list<(int, int)>
        partition_lines = find_partitions(game.width, game.height, blank_spaces)
        player_contig   = -1
        opponent_contig = -1
        for line in partition_lines:
        
            # Check to see if players are on either side of this partition line
            partitioned = False
            if line[0][0] == line[1][0]:
                # ROW-line : Row indexes match across line
                # See if player row locations differ and are separated by this line
                if player_location[0] != opponent_location[0] and \
                   ( ( player_location[0] > line[0][0] and opponent_location[0] < line[0][0] ) or \
                     ( player_location[0] < line[0][0] and opponent_location[0] > line[0][0] ) ):
                     
                    # Players are on either side of this partition!
                    # Count contiguous squares for each player if it hasn't already been done.
                    partitioned = True
                    if player_contig == -1:
                       player_contig = count_contig(player_location, blank_spaces)
                    if opponent_contig == -1:
                       opponent_contig = count_contig(opponent_location, blank_spaces)
            elif line[0][1] == line[1][1]:
                # COLUMN-line : Column indexes match across line
                # See if player row locations differ and are separated by this line
                if player_location[1] != opponent_location[1] and \
                   ( ( player_location[1] > line[0][1] and opponent_location[1] < line[0][1] ) or \
                     ( player_location[1] < line[0][1] and opponent_location[1] > line[0][1] ) ):
                     
                    # Players are on either side of this partition!
                    # Count contiguous squares for each player if it hasn't already been done.
                    partitioned = True
                    if player_contig == -1:
                       player_contig = count_contig(player_location, blank_spaces)
                    if opponent_contig == -1:
                       opponent_contig = count_contig(opponent_location, blank_spaces)
                      
            # If this line counts as a partition, we should be able to determine a winner
            if partitioned == True:
                # If the contiguous space for the current player is greater than the opponent,
                # then the current player should win
                if player_contig > opponent_contig:
                   return float("inf")
                else:
                    # Else if there's less contiguous space or a tie in space, the current player
                    # should most likely lose
                    return float("-inf")
                    

    # Otherwise, the heuristic is the difference in available moves between
    # the current player and the opposition
    return float(len(game.get_legal_moves(player)) - 2.0 * len(game.get_legal_moves(game.get_opponent(player))))
    
    
def find_partitions(width, height, blank_spaces):
    """Given the width and height of a game board along with a set of the "blank"
    spaces on the board, determine if there is a partition present on the board.
    One of the following two conditions apply for a partition as estimated here:
       (1) A double band of non-blank (used) spaces on the board. For example:
                    XX
                    XX
                    XX
                    XX
       (2) A single band of non-blank (used) spaces on the board, along with
           a "cross" alternating down the sides. For example:
                   XXX
                    X
                   XXX
                    X
                   XXX
                   
    Return a list of all such partitions present either horizontally or virtically.
    The elements returns in the list are lines that represent dividing lines of the partition.

    Parameters
    ----------
    width : integer
        Width of an Isolation game board
        
    height : integer
        Height of an Isolation game board

    blank_spaces : list<(int, int)>
        A list of integer pairs representing blank spaces on the Isolation game board

    Returns
    -------
    list<list<(int, int)>>
        List of lines representing partitioning lines
    """
    partition_lines = []
    
    # ROWS
    
    # For each horizontal row other than the ends, check for straight lines
    row_lines = []
    for r in range(1,(height-1)):
        current_line = []
        full_line    = True
        for c in range(0,width):
            if (r,c) in blank_spaces:
                full_line = False
                break;
            current_line.append((r,c))
        if full_line == True:
            row_lines.append(current_line)
    
    # Check for row lines which are adjacent forming a partition and add a dividing line to represent them
    adjacent_lines = [(lineA, lineB) for lineA in row_lines for lineB in row_lines if lineB[0][0] == lineA[0][0]+1 ]
    for adj_lines in adjacent_lines:
        partition_lines.append(adj_lines[0])
    
    # For each row line, check for cross pattern partitions
    for line in row_lines:
        cross_columns      = []
        still_possible     = True
        r                  = line[0][0]
        for c in range(0,width):
            # Check above and below for a cross
            if not (r-1,c) in blank_spaces and not (r+1,c) in blank_spaces:
                cross_columns.append(c)
                
        # If no crosses found, give up
        if len(cross_columns) == 0:
            break        
               
        # If cross columns were found, make sure they were spaced 2 units apart
        if still_possible == True:
            for c in range(cross_columns[0], width, 2):
               if not c in cross_columns:
                   still_possible = False
                   break
                   
        # If we've passed all of the checks. Then add the line
        if still_possible == True:
            partition_lines.append(line)
            
    # COLUMNS
    
    # For each virtical column other than the ends, check for straight lines
    col_lines = []
    for c in range(1,(width-1)):
        current_line = []
        full_line    = True
        for r in range(0,height):
            if (r,c) in blank_spaces:
                full_line = False
                break;
            current_line.append((r,c))
        if full_line == True:
            col_lines.append(current_line)
    
    # Check for row lines which are adjacent forming a partition and add a dividing line to represent them
    adjacent_lines = [(lineA ,lineB) for lineA in col_lines for lineB in col_lines if lineB[0][1] == lineA[0][1]+1 ]
    for adj_lines in adjacent_lines:
        partition_lines.append(adj_lines[0])
    
    # For each row line, check for cross pattern partitions
    for line in col_lines:
        cross_rows         = []
        still_possible     = True
        c                  = line[0][1]
        for r in range(0,height):
            # Check to left and right for a cross
            if not (r,c-1) in blank_spaces and not (r,c+1) in blank_spaces:
                cross_rows.append(r)
                
        # If no crosses found, give up
        if len(cross_rows) == 0:
            break        
               
        # If cross columns were found, make sure they were spaced 2 units apart
        if still_possible == True:
            for r in range(cross_rows[0], height, 2):
               if not r in cross_rows:
                   still_possible = False
                   break
                   
        # If we've passed all of the checks. Then add the line
        if still_possible == True:
            partition_lines.append(line)
            
    return partition_lines
    
    
def count_contig(player_location, blank_spaces):
    """Given a player location and the overall set of blank spaces on the Isolation
    board, count the number of contiguous spaces the player has around them, excluding
    diagonals. Here we perform a breadth-first search of game board, from the position of
    the player outwards to count.

    Parameters
    ----------
    player_location : (int, int)
        Tuple coordinates for row, column location of the player
        
    blank_spaces : list<(int, int)>
        List of all of the blank spaces on the game board

    Returns
    -------
    int
        Count of the contiguous spaces available to the player
    """
    frontier         = [player_location]
    spaces_visited   = [player_location]
    while frontier:
        space = frontier.pop(0)
        
        # Only add blank spaces around this one we haven't visited already
        
        # Check up
        if (space[0]-1, space[1]) in blank_spaces and not (space[0]-1, space[1]) in spaces_visited:
            frontier.append((space[0]-1, space[1]))
            spaces_visited.append((space[0]-1, space[1]))
        
        # Check down
        if (space[0]+1, space[1]) in blank_spaces and not (space[0]+1, space[1]) in spaces_visited:
            frontier.append((space[0]+1, space[1]))
            spaces_visited.append((space[0]+1, space[1]))
        
        # Check left
        if (space[0], space[1]-1) in blank_spaces and not (space[0], space[1]-1) in spaces_visited:
            frontier.append((space[0], space[1]-1))
            spaces_visited.append((space[0], space[1]-1))
        
        # Check right
        if (space[0]-1, space[1]+1) in blank_spaces and not (space[0]-1, space[1]+1) in spaces_visited:
            frontier.append((space[0]-1, space[1]+1))
            spaces_visited.append((space[0]-1, space[1]+1))
        
    
    # Return the count of spaces, minus our starting point
    return len(spaces_visited) - 1
    
    
    
def board_distance(game, player):
    """Calculate the approximate distance between the player and the center
    of the game board. 

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        An approximate distance between the player and the center of the board
    """
    center_x = game.width / 2.0
    center_y = game.height / 2.0
    player_x, player_y = game.get_player_location(player)
    dist_x = center_x - player_x
    dist_y = center_y - player_y
    return math.sqrt( math.pow(dist_x, 2) + math.pow(dist_y, 2) )


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth    = search_depth
        self.iterative       = iterative
        self.score           = score_fn
        self.method          = method
        self.time_left       = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        # Keep a record of how much time we have left to go
        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        
        # No move so far
        move = (-1, -1)
        
        # Return if there are no legal moves to attempt
        if len(legal_moves) == 0:
            return move
            
        # Move to the center, if possible (as an optimal place to start as a player)
        if (int(game.height/2), int(game.width/2)) in legal_moves:
            move = (int(game.height/2), int(game.width/2))
        else:
            move = legal_moves[0]

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            
            # Note that we can employ iterative deepening here to progressively search
            # greater depths of the game tree
            if self.iterative == True:
                # Keep increasing depths 
                d = 1
                while True:
                    if self.time_left() < self.TIMER_THRESHOLD:
                        raise Timeout()
                    if self.method == 'minimax':
                        score, move = self.minimax(game, d, True)
                        # Check if we've reached endgame
                        if score == float("inf") or score == float("-inf"):
                            break
                    elif self.method == 'alphabeta':
                        score, move = self.alphabeta(game, d, float("-inf"), float("inf"), True)
                        # Check if we've reached endgame
                        if score == float("inf") or score == float("-inf"):
                            break
                    d = d+1
            else:
                if self.method == 'minimax':
                    score, move = self.minimax(game, self.search_depth, True)
                elif self.method == 'alphabeta':
                    score, move = self.alphabeta(game, self.search_depth, float("-inf"), float("inf"), True)

        except Timeout:
            # Return the best move we've found so far or an insurance move
            return move

        # Return the best move from the last completed search iteration
        return move

        
    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        # Raise an exception if we've run out of time without an answer
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Evaluate the score of the current board for potential use
        # Keep track of our best move choice.
        if maximizing_player:
            best_score = float("-inf")
        else:
            best_score = float("inf")
        best_move      = (-1,-1)
        current_score  = best_score
        current_move   = best_move
            
        # Get the set of legal moves for the current player
        legal_moves = game.get_legal_moves()
            
        # If, they're aren't any legal moves, then we're at an endgame
        if len(legal_moves) == 0:
            return best_score, (-1,-1)
            
        # If we've reached our max depth, evaluate the game board as per
        # the current player
        if depth == 0:
            return self.score(game, self), (-1,-1)
            
        # If the depth is 1, evaluate all children
        if depth == 1:
            for current_move in legal_moves:
                game_copy     = game.forecast_move(current_move)  # Get the counter to increment
                current_score = self.score(game_copy, self)
                # Update our best choice, if necessary
                if maximizing_player and current_score > best_score:
                    best_score = current_score
                    best_move  = current_move
                elif not maximizing_player and current_score < best_score:
                    best_score = current_score
                    best_move  = current_move
                # Shortcut if we've found the best score possible
                if best_score == float("inf") or best_score == float("-inf"):
                    return best_score, best_move
            return best_score, best_move


        # Iterate over every legal move in depth-search fashion
        for current_move in legal_moves:
            # Copy the game state as if the move occurred and make a recursive call
            game_copy = game.forecast_move(current_move)  # Get the counter to increment
            next_max_player = True
            if maximizing_player == True:
               next_max_player = False
            current_score, junk_move = self.minimax(game_copy, depth-1, next_max_player)
            # Update our best choice, if necessary
            if maximizing_player and current_score > best_score:
                best_score = current_score
                best_move  = current_move
            elif not maximizing_player and current_score < best_score:
                best_score = current_score
                best_move  = current_move
            # Shortcut if we've found the best score possible
            if best_score == float("inf") or best_score == float("-inf"):
                return best_score, best_move
            
        # Return the best score and move found
        return best_score, best_move
        

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        # Raise an exception if we've run out of time without an answer
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Evaluate the score of the current board for potential use
        # Keep track of our best move choice.
        if maximizing_player:
            best_score = float("-inf")
        else:
            best_score = float("inf")
        best_move      = (-1,-1)
        current_score  = best_score
        current_move   = best_move
            
        # Get the set of legal moves for the current player
        legal_moves = game.get_legal_moves()
            
        # If, they're aren't any legal moves, then we're at an endgame
        if len(legal_moves) == 0:
            return best_score, best_move
            
        # If we've reached our max depth, evaluate the game board as per
        # the current player
        if depth == 0:
            return self.score(game, self), (-1,-1)      

        # If the depth is 1, evaluate all children
        if depth == 1:
            for current_move in legal_moves:
                game_copy     = game.forecast_move(current_move)  # Get the counter to increment
                current_score = self.score(game_copy, self)
                # Update our best choice, if necessary
                if maximizing_player and current_score > best_score:
                    best_score = current_score
                    best_move  = current_move
                elif not maximizing_player and current_score < best_score:
                    best_score = current_score
                    best_move  = current_move
                # Shortcut if we've found the best score possible
                if best_score == float("inf") or best_score == float("-inf"):
                    break
                
                # Decide if we need to perform a cut-off.
                # If we're a maximizing player:
                #   - If the current move taken from the minimizing child is
                #     higher than anything we've seen so far on the path back to the root,
                #     we don't need to proceed further.
                # If we're a minimizing player:
                #   - If the current move taken from the maximizing child is
                #     lower than anything we've seen so far on the path back to the root,
                #     we don't need to proceed further.
                if maximizing_player:
                    alpha = max(alpha, current_score)
                    if alpha >= beta:
                        # Cutoff
                        break
                else:
                    beta = min(beta, current_score)
                    if beta <= alpha:
                        # Cutoff
                        break
                        
            # We're at our max depth, return a result
            return best_score, best_move
    
        # Iterate over every legal move in depth-search fashion if there's still depth left to go
        for current_move in legal_moves:
            # Copy the game state as if the move occurred and make a recursive call
            game_copy = game.forecast_move(current_move)  # Get the counter to increment
            next_max_player = True
            if maximizing_player == True:
               next_max_player = False
            current_score, junk_move = self.alphabeta(game_copy, depth-1, alpha, beta, next_max_player)

            # Update our best choice, if necessary
            if maximizing_player and current_score > best_score:
                best_score = current_score
                best_move  = current_move
            elif not maximizing_player and current_score < best_score:
                best_score = current_score
                best_move  = current_move
            # Shortcut if we've found the best score possible
            if best_score == float("inf") or best_score == float("-inf"):
                break
                
            # Decide if we need to perform a cut-off.
            # If we're a maximizing player:
            #   - If the current move taken from the minimizing child is
            #     higher than anything we've seen so far on the path back to the root,
            #     we don't need to proceed further.
            # If we're a minimizing player:
            #   - If the current move taken from the maximizing child is
            #     lower than anything we've seen so far on the path back to the root,
            #     we don't need to proceed further.
            if maximizing_player:
                alpha = max(alpha, current_score)
                if alpha >= beta:
                    # Cutoff
                    break
            else:
                beta = min(beta, current_score)
                if beta <= alpha:
                    # Cutoff
                    break
            
        # Return the best score and move found
        return best_score, best_move
