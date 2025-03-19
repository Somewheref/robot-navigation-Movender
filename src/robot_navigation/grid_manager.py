#!/usr/bin/env python3

import numpy as np
from time import time
import rospy
import math

class GridManager:
    def __init__(self, grid_size=10, grid_resolution=0.2, cell_persistence=0.2):
        self.GRID_RESOLUTION = grid_resolution
        self.grid_size = int(grid_size / grid_resolution)
        self.cell_persistence = cell_persistence
        
        # Initialize regular grids
        self.occupancy_grid = [[False for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.grid_timestamps = [[0.0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.inflated_grid = None
        
        # Initialize debug large map (10x bigger)
        self.debug_grid_size = grid_size * 10
        self.debug_map = [[False for _ in range(self.debug_grid_size)] for _ in range(self.debug_grid_size)]
        self.debug_timestamps = [[0.0 for _ in range(self.debug_grid_size)] for _ in range(self.debug_grid_size)]
        
        # Current view window for debug mode (now as floats)
        self.view_center_x = float(self.debug_grid_size // 2)
        self.view_center_y = float(self.debug_grid_size // 2)
        
        # Clear zone parameters
        self.clear_zone_size = 5
        self.clear_zone_offset = -3   # avoid the robot itself
        self.grid_center = self.world_to_grid(0, 0)
        
        self.last_update_pose = None
        self.verbose = False
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        grid_x = int(x / self.GRID_RESOLUTION + self.grid_size / 2)
        grid_y = int(y / self.GRID_RESOLUTION + self.grid_size / 2)
        return (grid_x, grid_y)
    
    def world_to_grid_float(self, world_x, world_y):
        """Convert world coordinates to floating-point grid coordinates."""
        # Convert to grid coordinates without rounding
        grid_x = world_x / self.GRID_RESOLUTION + self.grid_center[0]
        grid_y = world_y / self.GRID_RESOLUTION + self.grid_center[1]
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates with interpolation."""
        # Convert to world coordinates with proper scaling
        world_x = (grid_x - self.grid_center[0]) * self.GRID_RESOLUTION
        world_y = (grid_y - self.grid_center[1]) * self.GRID_RESOLUTION
        return world_x, world_y
    
    def update_last_pose(self, current_pose):
        """Update the last pose."""
        # Add safety check for None
        if current_pose is None:
            return
        
        # log the difference between the current pose and the last pose
        if self.last_update_pose is not None:
            delta_x = current_pose['x'] - self.last_update_pose['x']
            delta_y = current_pose['y'] - self.last_update_pose['y']
            delta_yaw = current_pose['yaw'] - self.last_update_pose['yaw']
            if self.verbose:
                rospy.loginfo(f"Delta pose: ({delta_x:.3f}, {delta_y:.3f}, {delta_yaw:.3f})")
            self.transform_grid(current_pose, self.last_update_pose)
            
        self.last_update_pose = current_pose.copy()

    def update_occupancy_grid(self, point, current_pose=None, use_slam=False):
        """Update grid with new laser point."""
        # if self.last_update_pose is not None and use_slam and current_pose is not None:
        #     # Calculate the angular difference between poses
        #     delta_yaw = current_pose['yaw'] - self.last_update_pose['yaw']
        #     # Normalize angle to [-π, π]
        #     delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
            
        #     delta_x = current_pose['x'] - self.last_update_pose['x']
        #     delta_y = current_pose['y'] - self.last_update_pose['y']

        #     # Transform point using the relative angle
        #     cos_yaw = np.cos(delta_yaw)
        #     sin_yaw = np.sin(delta_yaw)
            
        #     # Transform point from robot frame to odom frame using relative transformation
        #     map_x = point['x'] * cos_yaw - point['y'] * sin_yaw + delta_x
        #     map_y = point['x'] * sin_yaw + point['y'] * cos_yaw + delta_y
        # else:
        #     map_x = point['x']
        #     map_y = point['y']
            
        map_x = point['x']
        map_y = point['y']
            
        grid_x, grid_y = self.world_to_grid(map_x, map_y)
        
        if (0 <= grid_x < self.grid_size and 
            0 <= grid_y < self.grid_size):
            self.occupancy_grid[grid_x][grid_y] = True
            self.grid_timestamps[grid_x][grid_y] = time()
        
        self.maintain_clear_zone()
            
        # log the number of occupied cells
        # occupied_cells = sum(row.count(True) for row in self.occupancy_grid)
        # rospy.loginfo(f"Grid has {occupied_cells} occupied cells")

    def maintain_clear_zone(self):
        """Keep an area around the robot clear of obstacles."""
        center_x = self.grid_center[0]
        center_y = self.grid_center[1]
        
        # Clear zone in the occupancy grid
        # offset only applies to the x axis
        for dx in range(-self.clear_zone_size + self.clear_zone_offset, self.clear_zone_size + self.clear_zone_offset + 1):
            for dy in range(-self.clear_zone_size, self.clear_zone_size + 1):
                x = center_x + dx
                y = center_y + dy
                if (0 <= x < self.grid_size and 
                    0 <= y < self.grid_size):
                    self.occupancy_grid[x][y] = False
                    self.grid_timestamps[x][y] = 0.0
        
        # Clear zone in the debug map if it exists
        for dx in range(-self.clear_zone_size, self.clear_zone_size + 1):
            for dy in range(-self.clear_zone_size, self.clear_zone_size + 1):
                x = center_x + dx
                y = center_y + dy
                if (0 <= x < self.debug_grid_size and 
                    0 <= y < self.debug_grid_size):
                    self.debug_map[x][y] = False
                    self.debug_timestamps[x][y] = 0.0

    def update_grid_persistence(self):
        """Remove old obstacles based on persistence time."""
        current_time = time()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.occupancy_grid[x][y]:
                    if current_time - self.grid_timestamps[x][y] > self.cell_persistence:
                        self.occupancy_grid[x][y] = False
                        self.grid_timestamps[x][y] = 0.0

    def transform_point_world(self, x, y, yaw, dx, dy, dyaw, center_x=0, center_y=0):
        """Transform a point in world coordinates using rigid body transformation.
        
        Args:
            x, y: Point coordinates to transform
            yaw: Initial orientation of the point
            dx, dy: Translation
            dyaw: Change in rotation angle in radians
            center_x, center_y: Center of rotation (optional)
        
        Returns:
            tuple: (new_x, new_y) transformed coordinates
        """
        # Normalize rotation angle to [-π, π]
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        
        # Translate to origin (relative to rotation center)
        rel_x = x - center_x
        rel_y = y - center_y
        
        # Calculate current angle and radius (polar coordinates)
        current_angle = np.arctan2(rel_y, rel_x)
        radius = np.sqrt(rel_x * rel_x + rel_y * rel_y)
        
        # Apply rotation
        new_angle = current_angle + dyaw
        
        # Convert back to cartesian coordinates
        new_x = radius * np.cos(new_angle) - dx + center_x
        new_y = radius * np.sin(new_angle) - dy + center_y
        
        return new_x, new_y

    def transform_grid(self, current_pose, last_update_pose):
        """Transform occupancy grid based on robot motion."""
        if not current_pose or not last_update_pose:
            return

        # Calculate transformation parameters
        dx = current_pose['x'] - last_update_pose['x']
        dy = current_pose['y'] - last_update_pose['y']
        dyaw = current_pose['yaw'] - last_update_pose['yaw']
        
        new_grid = [[False for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        new_timestamps = [[0.0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        center_x = 0
        center_y = 0
        yaw = current_pose['yaw']
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.occupancy_grid[x][y]:
                    # Convert grid to world coordinates
                    world_x, world_y = self.grid_to_world(x, y)
                    
                    # Apply transformation using the reusable function
                    new_world_x, new_world_y = self.transform_point_world(
                        world_x, world_y, yaw, dx, dy, dyaw, center_x, center_y
                    )
                    
                    # Convert back to grid coordinates
                    new_grid_x, new_grid_y = self.world_to_grid(new_world_x, new_world_y)
                    
                    if (0 <= new_grid_x < self.grid_size and 
                        0 <= new_grid_y < self.grid_size):
                        new_grid[new_grid_x][new_grid_y] = True
                        new_timestamps[new_grid_x][new_grid_y] = self.grid_timestamps[x][y]
        
        self.occupancy_grid = new_grid
        self.grid_timestamps = new_timestamps

    def create_debug_map(self, density=0.1, cluster_size=3, cluster_probability=0.3):
        """Generate a random obstacle map for debugging purposes."""
        # Reset the debug grid
        self.debug_map = [[False for _ in range(self.debug_grid_size)] for _ in range(self.debug_grid_size)]
        self.debug_timestamps = [[0.0 for _ in range(self.debug_grid_size)] for _ in range(self.debug_grid_size)]
        current_time = time()
        
        # First pass: Generate initial random obstacles in the large map
        for x in range(self.debug_grid_size):
            for y in range(self.debug_grid_size):
                if np.random.random() < density:
                    self.debug_map[x][y] = True
                    self.debug_timestamps[x][y] = current_time
        
        # Second pass: Grow clusters
        for _ in range(cluster_size):
            temp_grid = [row[:] for row in self.debug_map]
            for x in range(self.debug_grid_size):
                for y in range(self.debug_grid_size):
                    if self.debug_map[x][y]:
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                new_x, new_y = x + dx, y + dy
                                if (0 <= new_x < self.debug_grid_size and 
                                    0 <= new_y < self.debug_grid_size and 
                                    not temp_grid[new_x][new_y] and
                                    np.random.random() < cluster_probability):
                                    temp_grid[new_x][new_y] = True
                                    self.debug_timestamps[new_x][new_y] = current_time
            self.debug_map = temp_grid
        
        # Update the visible portion
        self.update_visible_grid()
        
        self.maintain_clear_zone()
        
    def is_in_clear_zone(self, x, y):
        """Check if a point in grid coordinates is in the clear zone.
        
        Args:
            x, y: Grid coordinates to check
            
        Returns:
            bool: True if point is in clear zone
        """
        # Get relative position to grid center
        rel_x = x
        rel_y = y
        
        # Check if point is within the shifted square bounds
        return (rel_x <= self.clear_zone_size + self.clear_zone_offset and 
                rel_x >= -self.clear_zone_size + self.clear_zone_offset and 
                rel_y <= self.clear_zone_size and 
                rel_y >= -self.clear_zone_size)


    def update_visible_grid(self):
        """Update the visible portion of the debug map."""
        # Convert view center to integers only for array indexing
        start_x = int(self.view_center_x - self.grid_center[0])
        start_y = int(self.view_center_y - self.grid_center[1])
        
        # Reset the visible grid
        self.occupancy_grid = [[False for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.grid_timestamps = [[0.0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Copy the visible portion from debug map
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                debug_x = start_x + i
                debug_y = start_y + j
                if (0 <= debug_x < self.debug_grid_size and 
                    0 <= debug_y < self.debug_grid_size):
                    self.occupancy_grid[i][j] = self.debug_map[debug_x][debug_y]
                    self.grid_timestamps[i][j] = self.debug_timestamps[debug_x][debug_y]

    def update_debug_view_position(self, dx, dy, dtheta):
        """Update the view center based on robot movement and rotation."""
        # Convert world displacement to grid cells (maintain floating point precision)
        grid_dx = dx / self.GRID_RESOLUTION
        grid_dy = dy / self.GRID_RESOLUTION
        
        # Update view center with bounds checking (keep as float)
        self.view_center_x = max(float(self.grid_size // 2), 
                               min(float(self.debug_grid_size - self.grid_size // 2),
                                   self.view_center_x + grid_dx))
        self.view_center_y = max(float(self.grid_size // 2),
                               min(float(self.debug_grid_size - self.grid_size // 2),
                                   self.view_center_y + grid_dy))
        
        rospy.logdebug(f"Updated view center to ({self.view_center_x}, {self.view_center_y})")
        
        # Create temporary grid for rotation
        temp_grid = [[False for _ in range(self.debug_grid_size)] for _ in range(self.debug_grid_size)]
        temp_timestamps = [[0.0 for _ in range(self.debug_grid_size)] for _ in range(self.debug_grid_size)]
        
        # Calculate rotation matrix components
        cos_theta = np.cos(-dtheta)
        sin_theta = np.sin(-dtheta)
        center_x = self.view_center_x  # Use floating point center
        center_y = self.view_center_y
        
        # Rotate the entire debug map around the center
        for x in range(self.debug_grid_size):
            for y in range(self.debug_grid_size):
                # Translate to origin
                rel_x = x - center_x
                rel_y = y - center_y
                
                # Rotate
                rot_x = rel_x * cos_theta - rel_y * sin_theta
                rot_y = rel_x * sin_theta + rel_y * cos_theta
                
                # Translate back and apply translation
                new_x = int(rot_x + center_x)  # Convert to int only for array indexing
                new_y = int(rot_y + center_y)
                
                # Check bounds and copy values
                if (0 <= new_x < self.debug_grid_size and 
                    0 <= new_y < self.debug_grid_size):
                    temp_grid[x][y] = self.debug_map[new_x][new_y]
                    temp_timestamps[x][y] = self.debug_timestamps[new_x][new_y]
        
        # Update the debug maps
        self.debug_map = temp_grid
        self.debug_timestamps = temp_timestamps
        
        # Update the visible portion
        self.update_visible_grid()

