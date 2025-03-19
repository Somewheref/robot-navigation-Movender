#!/usr/bin/env python3

import rospy
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
import tf
import tf.transformations
import numpy as np

class DebugPosePublisher:
    def __init__(self, grid_manager):
        # rospy.init_node('debug_pose_publisher')
        
        # Publisher for odometry data
        self.grid_manager = grid_manager
        self.odom_pub = rospy.Publisher('/RosAria/pose', Odometry, queue_size=50)
        
        # Initialize pose
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        
        # Initialize velocities
        self.vx = 0.0
        self.vy = 0.0
        self.vth = 0.0
        
        # Get parameters
        # self.update_rate = rospy.get_param('~update_rate', 10.0)  # Hz
        # self.noise_std = rospy.get_param('~noise_std', 0.001)  # Standard deviation for noise
        
        self.update_rate = 10.0  # Hz
        self.noise_std = 0.001  # Standard deviation for noise
        
        # Start timer for publishing pose
        self.timer = rospy.Timer(rospy.Duration(1.0/self.update_rate), self.timer_callback)
        
        # Subscribe to velocity commands (if you want to move the robot based on cmd_vel)
        self.cmd_vel_sub = rospy.Subscriber('/RosAria/cmd_vel', Twist, self.cmd_vel_callback)
        
        rospy.loginfo("Debug pose publisher initialized")

    def cmd_vel_callback(self, msg):
        """Update velocities based on received command"""
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.vth = msg.angular.z

    def timer_callback(self, event):
        """Publish simulated pose at regular intervals"""
        current_time = rospy.Time.now()
        
        # Compute dt
        try:
            dt = (current_time - self.last_time).to_sec()
        except AttributeError:
            dt = 1.0/self.update_rate
        self.last_time = current_time
        
        # Store previous position and orientation for calculating changes
        prev_x = self.x
        prev_y = self.y
        prev_th = self.th
        
        # Update pose based on velocities
        delta_x = (self.vx * math.cos(self.th) - self.vy * math.sin(self.th)) * dt
        delta_y = (self.vx * math.sin(self.th) + self.vy * math.cos(self.th)) * dt
        delta_th = self.vth * dt
        
        self.x += delta_x
        self.y += delta_y
        self.th += delta_th
        
        # Calculate total displacement in world coordinates
        dx_world = self.x - prev_x
        dy_world = self.y - prev_y
        dth = self.th - prev_th
        
        # Transform displacement from world frame to robot's local frame
        cos_yaw = math.cos(-prev_th)  # Use previous theta for inverse transform
        sin_yaw = math.sin(-prev_th)
        dx = dx_world * cos_yaw - dy_world * sin_yaw
        dy = dx_world * sin_yaw + dy_world * cos_yaw
        
        # Update the grid manager's view position (if available)
        try:
            if self.grid_manager:
                self.grid_manager.update_debug_view_position(dx, dy, dth)
        except:
            pass
        
        # Create quaternion from yaw
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, self.th)
        
        # Create and publish odometry message
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        
        # Set the position
        odom.pose.pose = Pose(Point(self.x, self.y, 0.), Quaternion(*odom_quat))
        
        # Set the velocity
        odom.twist.twist = Twist(Vector3(self.vx, self.vy, 0), Vector3(0, 0, self.vth))
        
        # Publish the message
        self.odom_pub.publish(odom)
        
        rospy.loginfo(f"Publishing pose at {self.x}, {self.y}, {self.th}")
