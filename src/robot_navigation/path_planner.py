#!/usr/bin/env python3

import heapq
import math
import rospy
import numpy as np

class PathPlanner:
    def __init__(self, grid_manager, safety_distance=1, smoothing_weight=0.5, turning_cost=0.3):
        self.grid_manager = grid_manager
        self.safety_distance = safety_distance
        self.smoothing_weight = smoothing_weight
        self.turning_cost = turning_cost
        self.right_turn_only = False  # New parameter for right-turn-only mode
        self.verbose = False

    def inflate_obstacles(self, occupancy_grid):
        """Inflate obstacles by safety distance."""
        if occupancy_grid is None:
            rospy.logwarn("No occupancy grid available for obstacle inflation")
            return None
        
        inflated = np.copy(occupancy_grid)
        safety_dist_cells = int(self.safety_distance)
        
        obstacle_indices = np.where(occupancy_grid == 1)
        
        if len(obstacle_indices) < 2 or len(obstacle_indices[0]) == 0:
            rospy.logwarn("No obstacles found in occupancy grid")
            return inflated
        
        for i, j in zip(obstacle_indices[0], obstacle_indices[1]):
            for dx in range(-safety_dist_cells, safety_dist_cells + 1):
                for dy in range(-safety_dist_cells, safety_dist_cells + 1):
                    new_i, new_j = i + dx, j + dy
                    if (0 <= new_i < self.grid_manager.grid_size and 
                        0 <= new_j < self.grid_manager.grid_size):
                        if math.sqrt(dx**2 + dy**2) <= self.safety_distance:
                            inflated[new_i, new_j] = 1
        
        return inflated
        

    def astar(self, goal):
        """A* pathfinding algorithm with direction costs."""
        # start position in the middle of the grid
        start = self.grid_manager.grid_center
        
        # convert goal to int
        goal = int(goal[0]), int(goal[1])
        
        if not self._validate_positions(start, goal):
            return []

        # Convert the list of lists to numpy array for proper inflation
        occupancy_grid = np.array(self.grid_manager.occupancy_grid, dtype=np.int8)
        self.inflated_grid = self.inflate_obstacles(occupancy_grid)
        if self.inflated_grid is None:
            return []

        # Add debug logging
        if self.verbose:
            rospy.loginfo(f"Starting A* search from {start} to {goal}")
        
        frontier = []
        heapq.heappush(frontier, (0, start, None))
        came_from = {start: None}
        cost_so_far = {start: 0}
        directions = {start: None}

        while frontier:
            current_cost, current, prev_direction = heapq.heappop(frontier)
            
            if current == goal:
                break

            for next_node in self._get_neighbors(current, prev_direction):
                next_pos = next_node['pos']
                new_cost = cost_so_far[current] + next_node['cost']

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self._heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos, next_node['direction']))
                    came_from[next_pos] = current
                    directions[next_pos] = next_node['direction']

        if goal not in came_from:
            if self.verbose:
                rospy.logwarn("No path found to goal")
            return []

        path = self._reconstruct_path(came_from, start, goal)
        smoothed_path = self._smooth_path(path)
        if self.verbose:
            rospy.loginfo(f"Path found with {len(smoothed_path)} points")
        return smoothed_path

    def _validate_positions(self, start, goal):
        """Validate start and goal positions."""
        if not (0 <= start[0] < self.grid_manager.grid_size and 
                0 <= start[1] < self.grid_manager.grid_size):
            rospy.logwarn(f"Invalid start position: {start}")
            return False
            
        if not (0 <= goal[0] < self.grid_manager.grid_size and 
                0 <= goal[1] < self.grid_manager.grid_size):
            rospy.logwarn(f"Invalid goal position: {goal}")
            return False
            
        return True

    def _get_neighbors(self, pos, prev_direction=None):
        """Return list of neighboring grid cells with direction costs."""
        x, y = pos
        neighbors = []
        
        # Generate directions in 16 angles
        num_angles = 16
        for i in range(num_angles):
            angle = 2 * math.pi * i / num_angles
            step_size = 1.0
            dx = step_size * math.cos(angle)
            dy = step_size * math.sin(angle)
            
            new_x = int(round(x + dx))
            new_y = int(round(y + dy))
            
            if (0 <= new_x < self.grid_manager.grid_size and 
                0 <= new_y < self.grid_manager.grid_size and 
                not self.inflated_grid[new_x][new_y]):
                
                direction_cost = 0
                if prev_direction is not None:
                    # Calculate angle change
                    current_angle = angle
                    prev_angle = math.atan2(prev_direction[1], prev_direction[0])
                    angle_diff = current_angle - prev_angle
                    
                    # Normalize angle difference to [-pi, pi]
                    while angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
                    while angle_diff < -math.pi:
                        angle_diff += 2 * math.pi
                    
                    if self.right_turn_only:
                        # Heavily penalize left turns
                        if angle_diff < 0:  # Left turn
                            direction_cost = 1000.0  # High cost to discourage left turns
                        else:  # Right turn
                            direction_cost = 2.0 * abs(angle_diff)
                    else:
                        direction_cost = 2.0 * abs(angle_diff)
                
                # Base movement cost
                movement_cost = math.sqrt(dx*dx + dy*dy)
                total_cost = movement_cost + direction_cost
                
                neighbors.append({
                    'pos': (new_x, new_y),
                    'direction': (dx, dy),
                    'angle': angle,
                    'cost': total_cost
                })
        
        # Sort neighbors by angle difference
        if prev_direction is not None:
            target_angle = math.atan2(y - self.grid_manager.grid_center[1], 
                                    x - self.grid_manager.grid_center[0])
            
            def neighbor_priority(n):
                angle_diff = n['angle'] - target_angle
                # Normalize to [-pi, pi]
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                if self.right_turn_only and angle_diff < 0:
                    # Heavily penalize left turns in sorting
                    return (1000.0, abs(angle_diff))
                return (abs(angle_diff), 0)
            
            neighbors.sort(key=neighbor_priority)
        
        return neighbors

    def _heuristic(self, a, b):
        """Estimate distance between two points with rotation preference."""
        # Calculate direct distance
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        direct_distance = math.sqrt(dx*dx + dy*dy)
        
        # Calculate angle from current position to goal
        current_angle = math.atan2(dy, dx)
        
        # Add rotation cost to heuristic
        # This encourages the path planner to consider rotation cost
        rotation_cost = abs(current_angle) * 0.5
        
        return direct_distance + rotation_cost

    def _reconstruct_path(self, came_from, start, goal):
        """Reconstruct path from came_from dict."""
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def _smooth_path(self, path):
        """Apply path smoothing to reduce unnecessary turns."""
        if len(path) <= 2:
            return path
            
        smoothed = path.copy()
        change = True
        while change:
            change = False
            for i in range(1, len(smoothed) - 1):
                old_pos = smoothed[i]
                new_x = int(smoothed[i-1][0] * (1-self.smoothing_weight) + 
                          smoothed[i+1][0] * self.smoothing_weight)
                new_y = int(smoothed[i-1][1] * (1-self.smoothing_weight) + 
                          smoothed[i+1][1] * self.smoothing_weight)
                
                if (0 <= new_x < self.grid_manager.grid_size and 
                    0 <= new_y < self.grid_manager.grid_size and 
                    not self.inflated_grid[new_x][new_y] and
                    self._is_path_clear((smoothed[i-1], (new_x, new_y))) and
                    self._is_path_clear(((new_x, new_y), smoothed[i+1]))):
                    
                    smoothed[i] = (new_x, new_y)
                    if old_pos != smoothed[i]:
                        change = True
        
        return smoothed

    def _is_path_clear(self, path_segment):
        """Check if a straight line between two points is collision-free."""
        start, end = path_segment
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if self.inflated_grid[x][y]:
                return False
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
                
        return True 