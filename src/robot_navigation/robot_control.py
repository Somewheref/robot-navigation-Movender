#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
import math

class RobotController:
    def __init__(self, grid_manager):
        """Initialize the robot controller."""
        self.grid_manager = grid_manager
        self.cmd_vel_pub = rospy.Publisher('/RosAria/cmd_vel', Twist, queue_size=1)
        self.target_pub = rospy.Publisher('/current_target', Marker, queue_size=1)
        self.max_speed = 0.5
        self.freeze_state = False
        self.is_rotating = False
        self.right_turn_only = False  # New parameter for right-turn-only mode
        
        # PID controller parameters
        self.Kp = 0.5  # Proportional gain
        self.Ki = 0.01  # Integral gain
        self.Kd = 0.1  # Derivative gain
        
        # PID state variables
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.max_integral = 1.0  # Anti-windup limit
        
        self.verbose = False
        self.emergency_stop_distance = 1
        
        self.idle = True
        self.emergency_stoped = False
        
        rospy.loginfo("Publishing velocity commands to /RosAria/cmd_vel")

    def pid_control(self, error):
        """Apply PID control to the angular error."""
        current_time = rospy.Time.now()
        if self.last_time is None:
            self.last_time = current_time
            self.prev_error = error
            return self.Kp * error
        
        # Calculate time difference
        dt = (current_time - self.last_time).to_sec()
        if dt == 0:
            return self.Kp * error
        
        # Calculate integral term with anti-windup
        self.integral += error * dt
        self.integral = max(min(self.integral, self.max_integral), -self.max_integral)
        
        # Calculate derivative term (with low-pass filter)
        derivative = (error - self.prev_error) / dt
        alpha = 0.1  # Low-pass filter coefficient
        derivative = alpha * derivative + (1 - alpha) * self.prev_error
        
        # Calculate control output
        output = (self.Kp * error + 
                 self.Ki * self.integral + 
                 self.Kd * derivative)
        
        # Update state
        self.prev_error = error
        self.last_time = current_time
        
        rospy.logdebug(f"PID components - P: {self.Kp * error:.3f}, " +
                      f"I: {self.Ki * self.integral:.3f}, " +
                      f"D: {self.Kd * derivative:.3f}")
        
        return output

    def publish_cmd_vel(self, linear_speed, angular_speed):
        """Publish velocity command with safety checks."""
        cmd_vel = Twist()
        
        linear_speed = max(min(linear_speed, self.max_speed), -self.max_speed)
        angular_speed = max(min(angular_speed, 1.0), -1.0)
        
        cmd_vel.linear.x = linear_speed
        cmd_vel.angular.z = angular_speed
        
        self.cmd_vel_pub.publish(cmd_vel)

    def publish_target(self, x, y):
        """Publish visualization marker for current target."""
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.target_pub.publish(marker)

    def execute_path(self, current_path, grid_to_world):
        """Execute the planned path."""      
        if not current_path or len(current_path) < 2:
            self.set_freeze_state(True)
            if self.verbose:
                rospy.loginfo("No path to execute, setting velocity to 0")
            self.reset_pid()
            self.idle = True
            return 0.0, 0.0, 0.0

        self.idle = False
        
        # Get next waypoint and its direction
        current = current_path[0]
        target = current_path[1]
        
        # Calculate actual direction vector
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        target_angle = math.atan2(dy, dx)
        
        # Convert to world coordinates
        target_world = grid_to_world(target[0], target[1])
        
        # Calculate relative position to target
        dx_world = target_world[0]
        dy_world = target_world[1]
        distance = math.sqrt(dx_world*dx_world + dy_world*dy_world)
        
        # Use the actual angle for control
        angular_error = target_angle

        # Normalize the error to [-pi, pi]
        angular_error = ((angular_error + math.pi) % (2 * math.pi)) - math.pi
        
        if self.verbose:
            rospy.loginfo(f"Target: ({dx_world:.2f}, {dy_world:.2f}), Distance: {distance:.2f}, " +
                         f"Angle: {math.degrees(target_angle):.1f}°, Error: {math.degrees(angular_error):.1f}°")
        
        # Apply PID control for angular velocity
        angular_speed = self.pid_control(angular_error)
        angular_speed = max(min(angular_speed, 0.5), -0.5)
        
        if self.right_turn_only and angular_speed > 0:
            angular_speed = -angular_speed
        
        # Smooth linear speed based on angle error
        linear_speed = 0.0
        if abs(angular_error) < 0.3:
            linear_speed = 0.2 * (1 - abs(angular_error) / 0.3)
            self.is_rotating = False
        else:
            self.is_rotating = True
        
        # Update path when waypoint is reached
        if distance < 0.2 and abs(angular_error) < 0.1:
            current_path.pop(0)
            self.reset_pid()
            if self.verbose:
                rospy.loginfo("Waypoint reached, moving to next point")
            
            if len(current_path) < 2:
                self.set_freeze_state(True)
                if self.verbose:
                    rospy.loginfo("End of path reached")
                return 0.0, 0.0, 0.0
        
        self.set_freeze_state(False)
        self.set_velocity(linear_speed, angular_speed)
        self.publish_target(dx_world, dy_world)
        
        return dx_world, dy_world, distance

    def reset_pid(self):
        """Reset PID controller state."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def set_freeze_state(self, freeze_state):
        """Set the freeze state of the robot."""
        self.freeze_state = freeze_state
        if freeze_state:
            self.set_velocity()
        if self.verbose:
            rospy.loginfo(f"Freeze state set to {self.freeze_state}")
        
    
    def set_velocity(self, linear_speed = 0, angular_speed = 0):
        """Set the velocity of the robot."""
        if self.freeze_state:
            self.publish_cmd_vel(0, 0)
        else:
            self.publish_cmd_vel(linear_speed, angular_speed)
            
    def emergency_stop(self):
        """Emergency stop the robot."""
        # self.publish_cmd_vel(0, 0)
        # self.set_freeze_state(True)
        self.emergency_stoped = True

    def check_emergency_stop(self):
        """Check if the robot should stop due to an obstacle in emergency stop distance."""
        # Convert emergency stop distance to grid cells
        stop_cells = int(self.emergency_stop_distance / self.grid_manager.GRID_RESOLUTION)
        
        # Check cells in front of the robot (positive x in robot frame)
        for dx in range(0, stop_cells + 1):  # Only check forward
            for dy in range(-stop_cells, stop_cells + 1):
                grid_x = self.grid_manager.grid_center[0] + dx
                grid_y = self.grid_manager.grid_center[1] + dy
                
                # Skip if outside grid bounds
                if (grid_x < 0 or grid_x >= self.grid_manager.grid_size or 
                    grid_y < 0 or grid_y >= self.grid_manager.grid_size):
                    continue
                    
                
                # Check if cell is occupied and within emergency stop radius
                if self.grid_manager.occupancy_grid[grid_x][grid_y]:
                    # Convert to world coordinates for distance check
                    world_x, world_y = self.grid_manager.grid_to_world(grid_x, grid_y)
                    distance = math.sqrt(world_x**2 + world_y**2)
                    
                    if distance < self.emergency_stop_distance:
                        rospy.logwarn(f"Emergency stop! Obstacle at {distance:.2f}m")
                        return True
        
        return False

    def reset_emergency_stop(self):
        """Reset the emergency stop state of the robot."""
        self.emergency_stoped = False
